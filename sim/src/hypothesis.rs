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
    /// H7: One bulk roll per category (fav + allergy only), no per-course loop.
    ///     life = strength + roll(n_fav * fav_param) - roll(n_allergy * wo)
    ///     Matches the "3 rolls" intuition: two category rolls + no regular penalty.
    H7 {
        max_weight: u32,
        max_effect: u32,
        fav_param: u32,
    },
    /// H8: H7 + one bulk roll for regular (non-fav, non-allergy) courses.
    ///     life = strength + roll(n_fav * fav_param) - roll(n_allergy * wo) - roll(n_reg * reg_param)
    ///     All three category types represented by exactly one roll each.
    H8 {
        max_weight: u32,
        max_effect: u32,
        fav_param: u32,
        reg_param: u32,
    },
    /// H9: Like Python's 3-roll model but each roll has a category-specific scale.
    ///     All rolls share (base - strength) as the center, matching Python's structure.
    ///     fav_roll:     upper = base - strength - n_fav * fav_scale
    ///     allergy_roll: upper = base - strength + n_allergy * allergy_scale
    ///     normal_roll:  upper = base - strength
    H9 {
        base: u32,
        fav_scale: u32,
        allergy_scale: u32,
    },
    /// H10: 3-roll multiplicative. Each category roll's upper is scaled by a per-course factor.
    ///     fav_roll:     upper = (base - strength) * (fav_pct/100)^n_fav    (<100 = faster with more favs)
    ///     allergy_roll: upper = (base - strength) * (allergy_pct/100)^n_allergy (>100 = slower)
    ///     normal_roll:  upper = (base - strength)
    ///     Motivated by Orvinn's non-linear win rate spike at many favorites.
    H10 {
        base: u32,
        fav_pct: u32,      // e.g. 85 means 0.85x per favorite
        allergy_pct: u32,  // e.g. 130 means 1.30x per allergy
    },
    /// H11: Python-style (all 3 rolls share one upper) but food effects are multiplicative.
    ///     upper = (base - strength) * (fav_pct/100)^n_fav * (allergy_pct/100)^n_allergy
    H11 {
        base: u32,
        fav_pct: u32,
        allergy_pct: u32,
    },
}

impl std::fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hypothesis::H7 { max_weight, max_effect, fav_param } =>
                write!(f, "H7 max_w={max_weight} max_e={max_effect} fav_p={fav_param}"),
            Hypothesis::H8 { max_weight, max_effect, fav_param, reg_param } =>
                write!(f, "H8 max_w={max_weight} max_e={max_effect} fav_p={fav_param} reg_p={reg_param}"),
            Hypothesis::H9 { base, fav_scale, allergy_scale } =>
                write!(f, "H9 base={base} fav_s={fav_scale} all_s={allergy_scale}"),
            Hypothesis::H10 { base, fav_pct, allergy_pct } =>
                write!(f, "H10 base={base} fav%={fav_pct} all%={allergy_pct}"),
            Hypothesis::H11 { base, fav_pct, allergy_pct } =>
                write!(f, "H11 base={base} fav%={fav_pct} all%={allergy_pct}"),
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

// Returns (n_fav, n_allergy, n_regular) for a pirate given the arena courses.
// Favorites exclude courses that are also allergies (matching Python logic).
fn course_counts(pirate: &pirates::Pirate, courses: &[usize]) -> (u32, u32, u32) {
    let n_allergy = courses.iter().filter(|&&c| pirate.allergy_courses.contains(&c)).count() as u32;
    let n_fav = courses
        .iter()
        .filter(|&&c| pirate.favorite_courses.contains(&c) && !pirate.allergy_courses.contains(&c))
        .count() as u32;
    let n_reg = 10 - n_allergy - n_fav;
    (n_fav, n_allergy, n_reg)
}

fn pirate_life(pirate: &pirates::Pirate, courses: &[usize], hyp: Hypothesis, rng: &mut impl Rng) -> i64 {
    match hyp {
        Hypothesis::H7 { max_weight, max_effect, fav_param } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            pirate.strength as i64
                + roll(rng, n_fav * fav_param)
                - roll(rng, n_allergy * wo)
        }
        Hypothesis::H8 { max_weight, max_effect, fav_param, reg_param } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let (n_fav, n_allergy, n_reg) = course_counts(pirate, courses);
            pirate.strength as i64
                + roll(rng, n_fav * fav_param)
                - roll(rng, n_allergy * wo)
                - roll(rng, n_reg * reg_param)
        }
        Hypothesis::H9 { base, fav_scale, allergy_scale } => {
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let center = base as i64 - pirate.strength as i64;
            let fav_u = (center - n_fav as i64 * fav_scale as i64).max(1) as u32;
            let all_u = (center + n_allergy as i64 * allergy_scale as i64).max(1) as u32;
            let normal_u = center.max(1) as u32;
            -roll(rng, fav_u) - roll(rng, all_u) - roll(rng, normal_u)
        }
        Hypothesis::H10 { base, fav_pct, allergy_pct } => {
            // Multiplicative per-category rolls. Each roll's upper scales exponentially with count.
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let center = (base as f64 - pirate.strength as f64).max(1.0);
            let fav_u = (center * (fav_pct as f64 / 100.0).powi(n_fav as i32)).max(1.0) as u32;
            let all_u = (center * (allergy_pct as f64 / 100.0).powi(n_allergy as i32)).max(1.0) as u32;
            let normal_u = center as u32;
            -roll(rng, fav_u) - roll(rng, all_u) - roll(rng, normal_u)
        }
        Hypothesis::H11 { base, fav_pct, allergy_pct } => {
            // All 3 rolls share one multiplicative upper bound combining all food effects.
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let center = (base as f64 - pirate.strength as f64).max(1.0);
            let upper = (center
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)
                * (allergy_pct as f64 / 100.0).powi(n_allergy as i32))
                .max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
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

    // H9 reference (best additive 3-roll from last run)
    let h9_configs: Vec<Hypothesis> = [100u32, 105, 110, 113, 120, 130]
        .iter()
        .flat_map(|&base| [1u32, 2, 3, 4, 5, 6, 8, 10].iter().flat_map(move |&fs|
            [1u32, 2, 3, 4, 5, 6, 8, 10].iter().map(move |&as_|
                Hypothesis::H9 { base, fav_scale: fs, allergy_scale: as_ }
            )
        ))
        .collect();
    let best_h9 = run_grid(&data, h9_configs, DAYS_GRID, "H9: additive 3-roll (reference)");

    // H10: multiplicative 3-roll (separate bounds per category)
    let h10_configs: Vec<Hypothesis> = [100u32, 105, 110, 113, 120]
        .iter()
        .flat_map(|&base| [70u32, 75, 80, 85, 88, 90, 92, 95].iter().flat_map(move |&fp|
            [105u32, 110, 115, 120, 130, 140, 150].iter().map(move |&ap|
                Hypothesis::H10 { base, fav_pct: fp, allergy_pct: ap }
            )
        ))
        .collect();
    let best_h10 = run_grid(&data, h10_configs, DAYS_GRID, "H10: multiplicative 3-roll (per-category)");

    // H11: multiplicative single-bound (Python-style but multiplied)
    let h11_configs: Vec<Hypothesis> = [100u32, 105, 110, 113, 120]
        .iter()
        .flat_map(|&base| [70u32, 75, 80, 85, 88, 90, 92, 95].iter().flat_map(move |&fp|
            [105u32, 110, 115, 120, 130, 140, 150].iter().map(move |&ap|
                Hypothesis::H11 { base, fav_pct: fp, allergy_pct: ap }
            )
        ))
        .collect();
    let best_h11 = run_grid(&data, h11_configs, DAYS_GRID, "H11: multiplicative single-bound (Python-style)");

    println!("\n\n=== FINAL RUNS ({DAYS_FINAL} days) ===");
    run_final(&data, best_h9, DAYS_FINAL);
    run_final(&data, best_h10, DAYS_FINAL);
    run_final(&data, best_h11, DAYS_FINAL);
}

// --- old H7/H8 entry point kept below for reference ---
fn _old_main() {
    let json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = Arc::new(GameData::load(&json));

    // H7: one bulk roll for favs + one for allergies (no regular penalty)
    let h7_configs: Vec<Hypothesis> = [220u32, 230, 250, 270, 300]
        .iter()
        .flat_map(|&mw| [10u32, 15, 20, 25, 30].iter().flat_map(move |&me|
            [3u32, 5, 8, 10, 15, 20, 30].iter().map(move |&fp|
                Hypothesis::H7 { max_weight: mw, max_effect: me, fav_param: fp }
            )
        ))
        .collect();
    let best_h7 = run_grid(&data, h7_configs, DAYS_GRID, "H7: bulk fav roll + bulk allergy roll");

    // H8: H7 + bulk roll for regular courses (one roll per category type = 3 total)
    let h8_configs: Vec<Hypothesis> = [220u32, 230, 250, 270, 300]
        .iter()
        .flat_map(|&mw| [10u32, 15, 20, 25].iter().flat_map(move |&me|
            [3u32, 5, 8, 10, 15].iter().flat_map(move |&fp|
                [1u32, 2, 3, 5, 8].iter().map(move |&rp|
                    Hypothesis::H8 { max_weight: mw, max_effect: me, fav_param: fp, reg_param: rp }
                )
            )
        ))
        .collect();
    let best_h8 = run_grid(&data, h8_configs, DAYS_GRID, "H8: 3 bulk rolls (fav + allergy + regular)");

    // H9: Python-style 3 rolls with category-specific scales, all anchored on (base - strength)
    // Python baseline equivalent: base=113, fav_scale=3, allergy_scale=3
    // (Python applies fav/allergy adjustments equally across all 3 rolls via effective_strength)
    let h9_configs: Vec<Hypothesis> = [100u32, 105, 110, 113, 120, 130]
        .iter()
        .flat_map(|&base| [1u32, 2, 3, 4, 5, 6, 8, 10].iter().flat_map(move |&fs|
            [1u32, 2, 3, 4, 5, 6, 8, 10].iter().map(move |&as_|
                Hypothesis::H9 { base, fav_scale: fs, allergy_scale: as_ }
            )
        ))
        .collect();
    let best_h9 = run_grid(&data, h9_configs, DAYS_GRID, "H9: 3 rolls with category-specific upper bounds");

    println!("\n\n=== FINAL RUNS ({DAYS_FINAL} days) ===");
    run_final(&data, best_h7, DAYS_FINAL);
    run_final(&data, best_h8, DAYS_FINAL);
    run_final(&data, best_h9, DAYS_FINAL);
}
