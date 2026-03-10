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
    /// H11: Python-style (all 3 rolls share one upper) but food effects are multiplicative.
    ///     upper = (base - strength) * (fav_pct/100)^n_fav * (allergy_pct/100)^n_allergy
    H11 {
        base: u32,
        fav_pct: u32,
        allergy_pct: u32,
    },
    /// H18: Two-phase — code leak per allergy, then H11 time on remaining life.
    ///     Phase 1: life = strength; for each allergy: life -= roll(wo)
    ///     Phase 2: upper = max(1, base - life) * fav_pct^n_fav
    ///     score = -(roll(upper) + roll(upper) + roll(upper))
    ///     Allergies handled by code leak (weight-dependent), favs by time model.
    H18 {
        max_weight: u32,
        max_effect: u32,
        base: u32,
        fav_pct: u32,  // < 100: favs speed up eating time
    },
    /// H19: H18 + zero-variable exception.
    ///     When n_fav == 0 AND n_allergy == 0, pirate gets a random life bonus
    ///     of roll(zv_bonus) before the time model. Prevents pure strength contests.
    H19 {
        max_weight: u32,
        max_effect: u32,
        base: u32,
        fav_pct: u32,
        zv_bonus: u32,  // bonus roll upper when 0 favs and 0 allergies
    },
}

impl std::fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hypothesis::H11 { base, fav_pct, allergy_pct } =>
                write!(f, "H11 base={base} fav%={fav_pct} all%={allergy_pct}"),
            Hypothesis::H18 { max_weight, max_effect, base, fav_pct } =>
                write!(f, "H18 max_w={max_weight} max_e={max_effect} base={base} fav%={fav_pct}"),
            Hypothesis::H19 { max_weight, max_effect, base, fav_pct, zv_bonus } =>
                write!(f, "H19 max_w={max_weight} max_e={max_effect} base={base} fav%={fav_pct} zv={zv_bonus}"),
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

fn pirate_score(pirate: &pirates::Pirate, courses: &[usize], hyp: Hypothesis, rng: &mut impl Rng) -> i64 {
    match hyp {
        Hypothesis::H11 { base, fav_pct, allergy_pct } => {
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let center = (base as f64 - pirate.strength as f64).max(1.0);
            let upper = (center
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)
                * (allergy_pct as f64 / 100.0).powi(n_allergy as i32))
                .max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
        Hypothesis::H18 { max_weight, max_effect, base, fav_pct } => {
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64;
            for _ in 0..n_allergy {
                life -= roll(rng, wo);
            }
            let upper = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32))
                .max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
        Hypothesis::H19 { max_weight, max_effect, base, fav_pct, zv_bonus } => {
            let (n_fav, n_allergy, _) = course_counts(pirate, courses);
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64;
            for _ in 0..n_allergy {
                life -= roll(rng, wo);
            }
            // Zero-variable exception: if no favs AND no allergies, add random bonus
            if n_fav == 0 && n_allergy == 0 {
                life += roll(rng, zv_bonus);
            }
            let upper = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32))
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
            let scores: Vec<(usize, i64)> = group
                .iter()
                .map(|&pi| {
                    let s = pirate_score(&data.pirates[pi], &course_indices, hyp, &mut rng);
                    (pi, s)
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
            let n_threads = 4u64;
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

    // H11 reference (best known: base=110, fav%=92, all%=115 → 0.0709)
    let h11_ref = Hypothesis::H11 { base: 110, fav_pct: 92, allergy_pct: 115 };

    // H18 reference (best known: max_w=221, max_e=10, base=103, fav%=91 → 0.048)
    let h18_ref = Hypothesis::H18 { max_weight: 221, max_effect: 10, base: 103, fav_pct: 91 };

    // H19: H18 + zero-variable exception.
    // When n_fav=0 AND n_allergy=0, pirate gets roll(zv_bonus) added to life.
    // Search around H18's best params with varying zv_bonus.
    let h19_configs: Vec<Hypothesis> = [220u32, 221, 222, 223]
        .iter()
        .flat_map(|&mw| [8u32, 9, 10, 11, 12].iter().flat_map(move |&me|
            [101u32, 102, 103, 104, 105].iter().flat_map(move |&base|
                [89u32, 90, 91, 92, 93].iter().flat_map(move |&fp|
                    [0u32, 3, 5, 8, 10, 15, 20, 25, 30].iter().map(move |&zv|
                        Hypothesis::H19 { max_weight: mw, max_effect: me, base, fav_pct: fp, zv_bonus: zv }
                    )
                )
            )
        ))
        .collect();
    let best_h19 = run_grid(&data, h19_configs, DAYS_GRID,
        "H19: H18 + zero-variable exception");

    // Verification: run H19 best with different seeds at high day count
    let h19_best = Hypothesis::H19 { max_weight: 221, max_effect: 10, base: 103, fav_pct: 91, zv_bonus: 8 };

    println!("\n\n=== FINAL RUNS ({DAYS_FINAL} days) ===");
    run_final(&data, h11_ref, DAYS_FINAL);
    run_final(&data, h18_ref, DAYS_FINAL);
    run_final(&data, h19_best, DAYS_FINAL);

    // Re-run H19 best with different seeds to verify stability
    println!("\n\n=== VERIFICATION RUNS (H19 best, 3x {DAYS_FINAL} days, different seeds) ===");
    for seed_offset in [1000u64, 2000, 3000] {
        let n_threads = rayon::current_num_threads() as u64;
        let chunk = DAYS_FINAL / n_threads;
        let total = chunk * n_threads;
        let all_wins: Vec<HashMap<String, u64>> = (0..n_threads)
            .into_par_iter()
            .map(|i| simulate_chunk(&data, h19_best, chunk, i * 1337 + seed_offset))
            .collect();
        let mut total_wins: HashMap<String, u64> = HashMap::new();
        for w in all_wins {
            for (name, count) in w {
                *total_wins.entry(name).or_insert(0) += count;
            }
        }
        let err = log_ratio_avg(&total_wins, total, &data);
        println!("  seed_offset={seed_offset}: log ratio = {err:.4}");
    }
}
