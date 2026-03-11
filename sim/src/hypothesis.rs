mod pirates;

use pirates::{GameData, HistMatch, load_historical_matches};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const PMF_SAMPLES: u64 = 50_000;
const SCORE_MIN: i64 = -500;
const SCORE_RANGE: usize = 501; // scores from -500 to 0

// --- Hypothesis configs ---

#[derive(Debug, Clone, Copy)]
enum Hypothesis {
    H19 { max_weight: u32, max_effect: u32, base: u32, fav_pct: u32, zv_bonus: u32 },
    H19PosLife { max_weight: u32, max_effect: u32, base: u32, fav_pct: u32, zv_bonus: u32, pos_bonus: u32 },
    H19PosMul { max_weight: u32, max_effect: u32, base: u32, fav_pct: u32, zv_bonus: u32, pos_pct: u32 },
    H19PosRoll { max_weight: u32, max_effect: u32, base: u32, fav_pct: u32, zv_bonus: u32, pos_bonus: u32 },
}

impl std::fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hypothesis::H19 { base, fav_pct, zv_bonus, .. } =>
                write!(f, "H19 b={base} f%={fav_pct} zv={zv_bonus}"),
            Hypothesis::H19PosLife { base, fav_pct, zv_bonus, pos_bonus, .. } =>
                write!(f, "PosLife b={base} f%={fav_pct} zv={zv_bonus} pb={pos_bonus}"),
            Hypothesis::H19PosMul { base, fav_pct, zv_bonus, pos_pct, .. } =>
                write!(f, "PosMul b={base} f%={fav_pct} zv={zv_bonus} pp={pos_pct}"),
            Hypothesis::H19PosRoll { base, fav_pct, zv_bonus, pos_bonus, .. } =>
                write!(f, "PosRoll b={base} f%={fav_pct} zv={zv_bonus} pb={pos_bonus}"),
        }
    }
}

fn roll(rng: &mut impl Rng, n: u32) -> i64 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) as i64 }
}

fn weight_offset(pirate_weight: u32, max_weight: u32, max_effect: u32) -> u32 {
    if pirate_weight >= max_weight { return 0; }
    ((max_weight - pirate_weight) / 2).min(max_effect)
}

fn course_counts(pirate: &pirates::Pirate, courses: &[usize]) -> (u32, u32) {
    let n_allergy = courses.iter().filter(|&&c| pirate.allergy_courses.contains(&c)).count() as u32;
    let n_fav = courses.iter()
        .filter(|&&c| pirate.favorite_courses.contains(&c) && !pirate.allergy_courses.contains(&c))
        .count() as u32;
    (n_fav, n_allergy)
}

fn pirate_score(pirate: &pirates::Pirate, n_fav: u32, n_allergy: u32, pos: u32, hyp: Hypothesis, rng: &mut impl Rng) -> i64 {
    match hyp {
        Hypothesis::H19 { max_weight, max_effect, base, fav_pct, zv_bonus } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64;
            for _ in 0..n_allergy { life -= roll(rng, wo); }
            if n_fav == 0 && n_allergy == 0 { life += roll(rng, zv_bonus); }
            let upper = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)).max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
        Hypothesis::H19PosLife { max_weight, max_effect, base, fav_pct, zv_bonus, pos_bonus } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64 + (pos * pos_bonus) as i64;
            for _ in 0..n_allergy { life -= roll(rng, wo); }
            if n_fav == 0 && n_allergy == 0 { life += roll(rng, zv_bonus); }
            let upper = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)).max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
        Hypothesis::H19PosMul { max_weight, max_effect, base, fav_pct, zv_bonus, pos_pct } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64;
            for _ in 0..n_allergy { life -= roll(rng, wo); }
            if n_fav == 0 && n_allergy == 0 { life += roll(rng, zv_bonus); }
            let upper_base = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)).max(1.0);
            let pos_mul = (100.0 - pos as f64 * pos_pct as f64) / 100.0;
            let upper = (upper_base * pos_mul).max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
        Hypothesis::H19PosRoll { max_weight, max_effect, base, fav_pct, zv_bonus, pos_bonus } => {
            let wo = weight_offset(pirate.weight, max_weight, max_effect);
            let mut life = pirate.strength as i64;
            life += roll(rng, pos * pos_bonus);
            for _ in 0..n_allergy { life -= roll(rng, wo); }
            if n_fav == 0 && n_allergy == 0 { life += roll(rng, zv_bonus); }
            let upper = ((base as i64 - life).max(1) as f64
                * (fav_pct as f64 / 100.0).powi(n_fav as i32)).max(1.0) as u32;
            -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
        }
    }
}

// --- Precomputed score PMF ---

#[derive(Clone)]
struct ScoreDist {
    pmf: Vec<f64>,  // pmf[i] = P(score = SCORE_MIN + i)
    cdf: Vec<f64>,  // cdf[i] = P(score <= SCORE_MIN + i)
}

impl ScoreDist {
    fn from_samples(samples: &[i64]) -> Self {
        let mut pmf = vec![0.0; SCORE_RANGE];
        let n = samples.len() as f64;
        for &s in samples {
            let idx = (s - SCORE_MIN) as usize;
            if idx < SCORE_RANGE { pmf[idx] += 1.0 / n; }
        }
        let mut cdf = vec![0.0; SCORE_RANGE];
        cdf[0] = pmf[0];
        for i in 1..SCORE_RANGE { cdf[i] = cdf[i-1] + pmf[i]; }
        ScoreDist { pmf, cdf }
    }
}

/// Key for precomputed PMF cache: (pirate_index, n_fav, n_allergy, position)
type PmfKey = (usize, u32, u32, u32);

/// Build all needed PMFs for a hypothesis.
fn build_pmf_cache(
    data: &GameData,
    matches: &[HistMatch],
    hyp: Hypothesis,
    n_samples: u64,
) -> HashMap<PmfKey, ScoreDist> {
    // Collect all unique keys needed
    let mut keys: std::collections::HashSet<PmfKey> = std::collections::HashSet::new();
    for m in matches {
        for (pos, &pi) in m.pirate_indices.iter().enumerate() {
            let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
            keys.insert((pi, nf, na, pos as u32));
        }
    }

    let keys_vec: Vec<PmfKey> = keys.into_iter().collect();
    let results: Vec<(PmfKey, ScoreDist)> = keys_vec
        .par_iter()
        .map(|&key| {
            let (pi, nf, na, pos) = key;
            let mut rng = SmallRng::seed_from_u64(pi as u64 * 1000 + nf as u64 * 100 + na as u64 * 10 + pos as u64);
            let samples: Vec<i64> = (0..n_samples)
                .map(|_| pirate_score(&data.pirates[pi], nf, na, pos, hyp, &mut rng))
                .collect();
            (key, ScoreDist::from_samples(&samples))
        })
        .collect();

    results.into_iter().collect()
}

/// Compute P(pirate at position `idx` wins) from 4 score distributions.
fn win_probability(idx: usize, dists: &[&ScoreDist; 4]) -> f64 {
    let mut p_win = 0.0;
    let opponents: Vec<usize> = (0..4).filter(|&j| j != idx).collect();

    for s_idx in 0..SCORE_RANGE {
        let p_i = dists[idx].pmf[s_idx];
        if p_i < 1e-12 { continue; }

        // P(opponent j has score < s) = CDF(s-1)
        // P(opponent j has score = s) = PMF(s)
        let p_lt: [f64; 3] = [
            if s_idx > 0 { dists[opponents[0]].cdf[s_idx - 1] } else { 0.0 },
            if s_idx > 0 { dists[opponents[1]].cdf[s_idx - 1] } else { 0.0 },
            if s_idx > 0 { dists[opponents[2]].cdf[s_idx - 1] } else { 0.0 },
        ];
        let p_eq: [f64; 3] = [
            dists[opponents[0]].pmf[s_idx],
            dists[opponents[1]].pmf[s_idx],
            dists[opponents[2]].pmf[s_idx],
        ];

        // Enumerate all 8 subsets of opponents that tie at score s
        for mask in 0u32..8 {
            let n_tied = mask.count_ones() + 1; // +1 for pirate idx
            let mut prob = 1.0 / n_tied as f64;
            for bit in 0..3 {
                if mask & (1 << bit) != 0 {
                    prob *= p_eq[bit];
                } else {
                    prob *= p_lt[bit];
                }
            }
            p_win += p_i * prob;
        }
    }
    p_win
}

/// Evaluate hypothesis on historical matches using precomputed PMFs.
/// Returns average log-likelihood (higher = better).
fn match_log_likelihood(
    data: &GameData,
    matches: &[HistMatch],
    hyp: Hypothesis,
    n_samples: u64,
) -> f64 {
    let cache = build_pmf_cache(data, matches, hyp, n_samples);

    let mut total_ll = 0.0;
    for m in matches {
        let keys: Vec<PmfKey> = m.pirate_indices.iter().enumerate()
            .map(|(pos, &pi)| {
                let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
                (pi, nf, na, pos as u32)
            }).collect();
        let dists: [&ScoreDist; 4] = [
            &cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]],
        ];
        let p_winner = win_probability(m.winner_pos, &dists);
        total_ll += p_winner.max(1e-6).ln();
    }

    total_ll / matches.len() as f64
}

fn run_grid(
    data: &Arc<GameData>,
    train: &[HistMatch],
    configs: Vec<Hypothesis>,
    label: &str,
) -> Hypothesis {
    println!("\n=== {label} ({} configs) ===", configs.len());

    let mut results: Vec<(f64, Hypothesis)> = configs
        .into_par_iter()
        .map(|hyp| {
            let ll = match_log_likelihood(data, train, hyp, PMF_SAMPLES);
            (ll, hyp)
        })
        .collect();

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (ll, hyp) in results.iter().take(10) {
        println!("  {hyp}  ->  ll={ll:.5}");
    }
    results[0].1
}

fn run_eval(data: &Arc<GameData>, matches: &[HistMatch], label: &str, hyp: Hypothesis) {
    let ll = match_log_likelihood(data, matches, hyp, PMF_SAMPLES * 4);
    println!("  {hyp} [{label}]: ll={ll:.5}  (uniform={:.5})", (0.25f64).ln());
}

/// Monte Carlo validation: directly simulate arenas and compare win probs to PMF-based.
fn validate_pmf(data: &GameData, matches: &[HistMatch], hyp: Hypothesis) {
    let n_mc = 200_000u64;
    let n_pmf = 200_000u64;
    let n_check = 20.min(matches.len());

    let cache = build_pmf_cache(data, &matches[..n_check], hyp, n_pmf);

    println!("\n=== PMF VALIDATION ({n_check} matches, {n_mc} MC sims each) ===");
    println!("  {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "match", "mc_p0", "pmf_p0", "mc_p1", "pmf_p1", "mc_p2", "pmf_p2", "mc_p3", "pmf_p3");

    let mut max_err = 0.0f64;
    for (mi, m) in matches[..n_check].iter().enumerate() {
        // PMF-based probabilities
        let keys: Vec<PmfKey> = m.pirate_indices.iter().enumerate()
            .map(|(pos, &pi)| {
                let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
                (pi, nf, na, pos as u32)
            }).collect();
        let dists: [&ScoreDist; 4] = [
            &cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]],
        ];
        let pmf_probs: Vec<f64> = (0..4).map(|i| win_probability(i, &dists)).collect();

        // Monte Carlo: simulate full arena
        let mut mc_wins = [0u64; 4];
        let mut rng = SmallRng::seed_from_u64(42 + mi as u64);
        for _ in 0..n_mc {
            let scores: Vec<i64> = m.pirate_indices.iter().enumerate()
                .map(|(pos, &pi)| {
                    let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
                    pirate_score(&data.pirates[pi], nf, na, pos as u32, hyp, &mut rng)
                }).collect();
            // Find winner (highest score, tie = random among tied)
            let max_s = *scores.iter().max().unwrap();
            let tied: Vec<usize> = scores.iter().enumerate()
                .filter(|(_, &s)| s == max_s).map(|(i, _)| i).collect();
            let winner = tied[rng.gen_range(0..tied.len())];
            mc_wins[winner] += 1;
        }
        let mc_probs: Vec<f64> = mc_wins.iter().map(|&w| w as f64 / n_mc as f64).collect();

        for i in 0..4 {
            max_err = max_err.max((mc_probs[i] - pmf_probs[i]).abs());
        }

        println!("  {:>4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                 mi,
                 mc_probs[0], pmf_probs[0], mc_probs[1], pmf_probs[1],
                 mc_probs[2], pmf_probs[2], mc_probs[3], pmf_probs[3]);
    }
    let pmf_sum: f64 = {
        let m = &matches[0];
        let keys: Vec<PmfKey> = m.pirate_indices.iter().enumerate()
            .map(|(pos, &pi)| {
                let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
                (pi, nf, na, pos as u32)
            }).collect();
        let dists: [&ScoreDist; 4] = [
            &cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]],
        ];
        (0..4).map(|i| win_probability(i, &dists)).sum()
    };
    println!("  Max |MC - PMF| error: {max_err:.4}");
    println!("  PMF prob sum (match 0): {pmf_sum:.6} (should be ~1.0)");
}

fn main() {
    let pirates_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = Arc::new(GameData::load(&pirates_json));

    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let all_days = load_historical_matches(&data, &hist_json);

    let n_days = all_days.len();
    let split = (n_days as f64 * 0.8) as usize;
    let train: Vec<HistMatch> = all_days[..split].iter().flat_map(|d| d.iter().cloned()).collect();
    let test: Vec<HistMatch> = all_days[split..].iter().flat_map(|d| d.iter().cloned()).collect();
    println!("Data: {} days, train={} matches, test={} matches", n_days, train.len(), test.len());

    // Validate PMF approach against Monte Carlo
    let h19_check = Hypothesis::H19 { max_weight: 221, max_effect: 10, base: 103, fav_pct: 91, zv_bonus: 8 };
    validate_pmf(&data, &train, h19_check);
    let pos_check = Hypothesis::H19PosMul { max_weight: 221, max_effect: 10, base: 110, fav_pct: 93, zv_bonus: 4, pos_pct: 7 };
    validate_pmf(&data, &train, pos_check);

    let h19 = Hypothesis::H19 { max_weight: 221, max_effect: 10, base: 103, fav_pct: 91, zv_bonus: 8 };

    // === Wave 1: Broad search across position variants + base ===
    let mut w1 = vec![h19];

    for pb in 1..=15 {
        for base in (103..=145).step_by(3) {
            w1.push(Hypothesis::H19PosLife {
                max_weight: 221, max_effect: 10, base, fav_pct: 91, zv_bonus: 8, pos_bonus: pb,
            });
        }
    }
    for pp in (2..=30).step_by(2) {
        for base in (95..=115).step_by(4) {
            w1.push(Hypothesis::H19PosMul {
                max_weight: 221, max_effect: 10, base, fav_pct: 91, zv_bonus: 8, pos_pct: pp,
            });
        }
    }
    for pb in 1..=15 {
        for base in (103..=145).step_by(3) {
            w1.push(Hypothesis::H19PosRoll {
                max_weight: 221, max_effect: 10, base, fav_pct: 91, zv_bonus: 8, pos_bonus: pb,
            });
        }
    }

    let best1 = run_grid(&data, &train, w1, "Wave 1: Broad position search");

    // === Wave 2: Refine winner ===
    let mut w2 = Vec::new();
    match best1 {
        Hypothesis::H19PosLife { base: bb, pos_bonus: bpb, .. } => {
            for base in (bb.saturating_sub(4))..=(bb+4) {
                for fav_pct in 88..=94 {
                    for pb in (bpb.saturating_sub(2))..=(bpb+2) {
                        for zv in [4, 6, 8, 10, 12] {
                            for me in [8, 10, 12] {
                                w2.push(Hypothesis::H19PosLife {
                                    max_weight: 221, max_effect: me, base, fav_pct, zv_bonus: zv, pos_bonus: pb,
                                });
                            }
                        }
                    }
                }
            }
        }
        Hypothesis::H19PosMul { base: bb, pos_pct: bpp, .. } => {
            for base in (bb.saturating_sub(3))..=(bb+3) {
                for fav_pct in 88..=94 {
                    for pp in (bpp.saturating_sub(3))..=(bpp+3) {
                        for zv in [4, 6, 8, 10, 12] {
                            w2.push(Hypothesis::H19PosMul {
                                max_weight: 221, max_effect: 10, base, fav_pct, zv_bonus: zv, pos_pct: pp,
                            });
                        }
                    }
                }
            }
        }
        Hypothesis::H19PosRoll { base: bb, pos_bonus: bpb, .. } => {
            for base in (bb.saturating_sub(4))..=(bb+4) {
                for fav_pct in 88..=94 {
                    for pb in (bpb.saturating_sub(2))..=(bpb+2) {
                        for zv in [4, 6, 8, 10, 12] {
                            for me in [8, 10, 12] {
                                w2.push(Hypothesis::H19PosRoll {
                                    max_weight: 221, max_effect: me, base, fav_pct, zv_bonus: zv, pos_bonus: pb,
                                });
                            }
                        }
                    }
                }
            }
        }
        _ => {
            println!("Baseline won wave 1.");
            run_eval(&data, &test, "TEST", h19);
            return;
        }
    }

    println!("\nWave 2: {} configs", w2.len());
    let best2 = run_grid(&data, &train, w2, "Wave 2: Refinement");

    // === Final evaluation ===
    println!("\n========== FINAL EVALUATION ==========");
    run_eval(&data, &train, "TRAIN", h19);
    run_eval(&data, &train, "TRAIN", best2);
    run_eval(&data, &test, "TEST", h19);
    run_eval(&data, &test, "TEST", best2);
}
