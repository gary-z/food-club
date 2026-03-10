mod pirates;

use pirates::GameData;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const DAYS_GRID: u64 = 500_000;
const DAYS_FINAL: u64 = 20_000_000;

// --- Hypothesis configs ---

#[derive(Debug, Clone, Copy)]
enum Hypothesis {
    /// H4: weight penalty every course, extra for allergies, fixed fav bonus
    H4 {
        max_weight: u32,
        max_effect: u32,
        fav_bonus: f64,
    },
    /// H5: H4 base + weight-scaled fav dice (heavier = more gain from favorites)
    H5 {
        max_weight: u32,
        max_effect: u32,
        max_fav_effect: u32,
    },
    /// H6: H4 base + weight-based variance on initial life
    H6 {
        max_weight: u32,
        max_effect: u32,
        fav_bonus: f64,
        weight_var_div: u32,
    },
}

impl std::fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hypothesis::H4 { max_weight, max_effect, fav_bonus } =>
                write!(f, "H4 max_w={max_weight} max_e={max_effect} fav={fav_bonus}"),
            Hypothesis::H5 { max_weight, max_effect, max_fav_effect } =>
                write!(f, "H5 max_w={max_weight} max_e={max_effect} max_fav_e={max_fav_effect}"),
            Hypothesis::H6 { max_weight, max_effect, fav_bonus, weight_var_div } =>
                write!(f, "H6 max_w={max_weight} max_e={max_effect} fav={fav_bonus} var_div={weight_var_div}"),
        }
    }
}

// dice(1, n) = uniform int in [1, n]; returns 0 if n == 0
fn roll(rng: &mut impl Rng, n: u32) -> i64 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) as i64 }
}

fn weight_offset(pirate_weight: u32, max_weight: u32, max_effect: u32) -> u32 {
    if pirate_weight >= max_weight {
        return 0;
    }
    ((max_weight - pirate_weight) / 2).min(max_effect)
}

const MIN_PIRATE_WEIGHT: u32 = 112; // Young Sproggie

fn pirate_life(pirate: &pirates::Pirate, courses: &[usize], hyp: Hypothesis, rng: &mut impl Rng) -> i64 {
    let wo = weight_offset(pirate.weight, hyp_max_weight(hyp), hyp_max_effect(hyp));
    let mut life = pirate.strength as i64;

    // Allergies apply to all allergy courses; favorites only apply if NOT also an allergy
    match hyp {
        Hypothesis::H4 { fav_bonus, .. } => {
            for &course in courses {
                life -= roll(rng, wo);
                if pirate.allergy_courses.contains(&course) {
                    life -= roll(rng, wo);
                } else if pirate.favorite_courses.contains(&course) {
                    life += fav_bonus as i64;
                }
            }
        }
        Hypothesis::H5 { max_fav_effect, .. } => {
            let fav_wo = ((pirate.weight.saturating_sub(MIN_PIRATE_WEIGHT)) / 2).min(max_fav_effect);
            for &course in courses {
                life -= roll(rng, wo);
                if pirate.allergy_courses.contains(&course) {
                    life -= roll(rng, wo);
                } else if pirate.favorite_courses.contains(&course) {
                    life += roll(rng, fav_wo);
                }
            }
        }
        Hypothesis::H6 { fav_bonus, weight_var_div, .. } => {
            life += roll(rng, pirate.weight / weight_var_div);
            for &course in courses {
                life -= roll(rng, wo);
                if pirate.allergy_courses.contains(&course) {
                    life -= roll(rng, wo);
                } else if pirate.favorite_courses.contains(&course) {
                    life += fav_bonus as i64;
                }
            }
        }
    }
    life
}

fn hyp_max_weight(h: Hypothesis) -> u32 {
    match h {
        Hypothesis::H4 { max_weight, .. } => max_weight,
        Hypothesis::H5 { max_weight, .. } => max_weight,
        Hypothesis::H6 { max_weight, .. } => max_weight,
    }
}

fn hyp_max_effect(h: Hypothesis) -> u32 {
    match h {
        Hypothesis::H4 { max_effect, .. } => max_effect,
        Hypothesis::H5 { max_effect, .. } => max_effect,
        Hypothesis::H6 { max_effect, .. } => max_effect,
    }
}

fn simulate_chunk(data: &GameData, hyp: Hypothesis, days: u64, seed: u64) -> HashMap<String, u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut wins: HashMap<String, u64> = HashMap::new();
    let n = data.pirates.len();
    let nc = data.num_courses();
    let mut pirate_order: Vec<usize> = (0..n).collect();

    for _ in 0..days {
        pirate_order.shuffle(&mut rng);
        let course_indices: Vec<usize> = sample(&mut rng, nc, 10).into_vec();

        for group in pirate_order.chunks(4) {
            // Compute scores, handle ties with uniform random choice
            let scores: Vec<(usize, i64)> = group
                .iter()
                .map(|&pi| {
                    let life = pirate_life(&data.pirates[pi], &course_indices, hyp, &mut rng);
                    (pi, life)
                })
                .collect();
            let max_score = scores.iter().map(|&(_, s)| s).max().unwrap();
            let tied: Vec<usize> = scores
                .iter()
                .filter(|&&(_, s)| s == max_score)
                .map(|&(pi, _)| pi)
                .collect();
            let winner_idx = tied[rng.gen_range(0..tied.len())];
            *wins.entry(data.pirates[winner_idx].name.clone()).or_insert(0) += 1;
        }
    }
    wins
}

fn log_ratio_avg(wins: &HashMap<String, u64>, total_days: u64, data: &GameData) -> f64 {
    let total: f64 = data
        .pirates
        .iter()
        .map(|p| {
            let sim = (wins.get(&p.name).copied().unwrap_or(0) as f64 / total_days as f64).max(1e-9);
            (sim / p.win_rate).ln().abs()
        })
        .sum();
    total / data.pirates.len() as f64
}

fn run_grid(data: &Arc<GameData>, configs: Vec<Hypothesis>, days: u64, label: &str) -> Hypothesis {
    println!("\n=== {label} ({} configs @ {days} days) ===", configs.len());

    let mut results: Vec<(f64, Hypothesis)> = configs
        .into_par_iter()
        .enumerate()
        .map(|(i, hyp)| {
            let n_threads = 4u64; // sub-threads per config
            let chunk = days / n_threads;
            let all_wins: Vec<HashMap<String, u64>> = (0..n_threads)
                .map(|t| simulate_chunk(data, hyp, chunk, i as u64 * 97 + t * 13 + 7))
                .collect();
            let mut total_wins: HashMap<String, u64> = HashMap::new();
            for w in all_wins {
                for (name, count) in w {
                    *total_wins.entry(name).or_insert(0) += count;
                }
            }
            let err = log_ratio_avg(&total_wins, chunk * n_threads, data);
            (err, hyp)
        })
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (err, hyp) in results.iter().take(5) {
        println!("  {hyp}  ->  {err:.4}");
    }
    results[0].1
}

fn run_final(data: &Arc<GameData>, hyp: Hypothesis, days: u64) {
    let n_threads = rayon::current_num_threads() as u64;
    let chunk = days / n_threads;
    let total = chunk * n_threads;

    let all_wins: Vec<HashMap<String, u64>> = (0..n_threads)
        .into_par_iter()
        .map(|i| simulate_chunk(data, hyp, chunk, i * 1337 + 99))
        .collect();

    let mut total_wins: HashMap<String, u64> = HashMap::new();
    for w in all_wins {
        for (name, count) in w {
            *total_wins.entry(name).or_insert(0) += count;
        }
    }

    println!("\n--- {hyp} (final {days} days) ---");
    let mut pirates_sorted = data.pirates.clone();
    pirates_sorted.sort_by(|a, b| a.win_rate.partial_cmp(&b.win_rate).unwrap());
    for p in &pirates_sorted {
        let sim = total_wins.get(&p.name).copied().unwrap_or(0) as f64 / total as f64;
        println!("  hist={:.3}  sim={:.3}  {}", p.win_rate, sim, p.name);
    }
    println!("  Log ratio avg: {:.4}", log_ratio_avg(&total_wins, total, data));
}

fn main() {
    let json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = Arc::new(GameData::load(&json));

    // H4 refined grid
    let h4_configs: Vec<Hypothesis> = [230u32, 240, 250, 260, 280, 300]
        .iter()
        .flat_map(|&mw| [10u32, 12, 15, 18, 20].iter().flat_map(move |&me|
            [1.0f64, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 25.0].iter().map(move |&fb|
                Hypothesis::H4 { max_weight: mw, max_effect: me, fav_bonus: fb }
            )
        ))
        .collect();
    let best_h4 = run_grid(&data, h4_configs, DAYS_GRID, "H4: All courses + allergy extra + fixed fav");

    // H5: weight-scaled fav bonus
    let h5_configs: Vec<Hypothesis> = [230u32, 250, 300]
        .iter()
        .flat_map(|&mw| [10u32, 15, 20].iter().flat_map(move |&me|
            [5u32, 10, 15, 20, 25, 30].iter().map(move |&mfe|
                Hypothesis::H5 { max_weight: mw, max_effect: me, max_fav_effect: mfe }
            )
        ))
        .collect();
    let best_h5 = run_grid(&data, h5_configs, DAYS_GRID, "H5: H4 base + weight-scaled fav dice");

    // H6: weight-based initial life variance
    let h6_configs: Vec<Hypothesis> = [240u32, 250, 260]
        .iter()
        .flat_map(|&mw| [12u32, 15, 18].iter().flat_map(move |&me|
            [1.0f64, 2.0, 3.0].iter().flat_map(move |&fb|
                [5u32, 8, 10, 15, 20].iter().map(move |&wvd|
                    Hypothesis::H6 { max_weight: mw, max_effect: me, fav_bonus: fb, weight_var_div: wvd }
                )
            )
        ))
        .collect();
    let best_h6 = run_grid(&data, h6_configs, DAYS_GRID, "H6: H4 base + weight variance on initial life");

    println!("\n\n=== FINAL RUNS ({DAYS_FINAL} days) ===");
    run_final(&data, best_h4, DAYS_FINAL);
    run_final(&data, best_h5, DAYS_FINAL);
    run_final(&data, best_h6, DAYS_FINAL);
}
