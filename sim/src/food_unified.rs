mod pirates;

use pirates::{GameData, Pirate, load_historical_matches};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::collections::HashMap;

const MAX_WEIGHT: u32 = 221;
const N_ROLLS: u32 = 4;
const PMF_SAMPLES: u32 = 500_000; // MC samples to build each PMF
const MAX_SCORE: usize = 40;      // max possible quantized score

fn roll(rng: &mut impl Rng, n: u32) -> u32 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) }
}

/// Build PMF of quantized eating time for given params via MC
fn build_pmf(strength: u32, weight: u32, nf: u32, na: u32,
             base: u32, fav_div: u32, divisor: u32, max_effect: u32,
             seed: u64) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut counts = vec![0u32; MAX_SCORE + 1];

    let wo = if weight >= MAX_WEIGHT { 0 } else { ((MAX_WEIGHT - weight) / 2).min(max_effect) };

    for _ in 0..PMF_SAMPLES {
        let mut s = strength;
        for _ in 0..na {
            s = s.saturating_sub(roll(&mut rng, wo));
        }
        let mut upper = if base > s { base - s } else { 1 };
        let reduction = if fav_div > 0 { upper / fav_div } else { 0 };
        upper = upper.saturating_sub(nf * reduction).max(1);
        let mut time = 0u32;
        for _ in 0..N_ROLLS {
            time += roll(&mut rng, upper);
        }
        let score = (time / divisor) as usize;
        if score <= MAX_SCORE {
            counts[score] += 1;
        }
    }

    counts.iter().map(|&c| c as f64 / PMF_SAMPLES as f64).collect()
}

/// Compute win probability for each position given 4 PMFs.
/// Lowest score wins; ties go to later position.
fn win_probs_from_pmfs(pmfs: &[&Vec<f64>; 4]) -> [f64; 4] {
    let mut probs = [0.0f64; 4];
    let len = MAX_SCORE + 1;

    // For each possible score combination, find the winner
    // Optimization: use CDF to avoid iterating over all 4D combos
    // P(pirate i wins) = sum over s: P(i scores s) * P(all j<i score > s) * P(all j>i score >= s)
    // Wait, ties go to LATER position, so:
    // P(i wins with score s) = P(i=s) * prod_{j<i} P(j>s) * prod_{j>i} P(j>=s)
    // But that's not quite right either since we need j>i to score > s (not >=) if ties go to later...
    // Actually: ties go to later position means if i and j both score s and j>i, then j wins.
    // So i wins with score s iff:
    //   - all j != i have score > s, OR
    //   - all j with score == s have index < i (impossible since later wins)
    // More precisely: i wins iff score_i == min AND i is the LAST index with that min score.
    // So: i wins with score s iff score_i = s AND for all j > i: score_j > s AND for all j < i: score_j >= s
    // Wait no. If ties go to later position: among all pirates with the minimum score,
    // the one at the latest position wins.
    // So: i wins with score s iff score_i = s AND for all j: score_j >= s, AND for all j > i: score_j > s.
    // i.e., i is the last one with the minimum score. So no one after i ties.

    // Precompute CDFs: P(score > s) and P(score >= s) for each pirate
    // P(score > s) = 1 - CDF(s) = 1 - sum_{t=0}^{s} pmf[t]
    // P(score >= s) = 1 - CDF(s-1) = 1 - sum_{t=0}^{s-1} pmf[t]
    let mut cdf = vec![vec![0.0f64; len]; 4];
    for p in 0..4 {
        cdf[p][0] = pmfs[p][0];
        for s in 1..len {
            cdf[p][s] = cdf[p][s-1] + pmfs[p][s];
        }
    }

    // P(score > s) = 1 - cdf[s]
    // P(score >= s) = if s == 0 { 1.0 } else { 1 - cdf[s-1] }

    for s in 0..len {
        for i in 0..4 {
            let p_i_eq_s = pmfs[i][s];
            if p_i_eq_s < 1e-15 { continue; }

            let mut prob = p_i_eq_s;
            for j in 0..4 {
                if j == i { continue; }
                if j < i {
                    // j must score >= s (j scoring s is ok since i is later and wins tie)
                    // Wait, no. If j < i and j scores s too, then both have min score s,
                    // but i > j so i is later and wins the tie. So j scoring s is fine for i.
                    // P(score_j >= s)
                    let p = if s == 0 { 1.0 } else { 1.0 - cdf[j][s-1] };
                    prob *= p;
                } else {
                    // j > i, j must score > s (if j scores s, j would win the tie, not i)
                    let p = 1.0 - cdf[j][s];
                    prob *= p;
                }
            }
            probs[i] += prob;
        }
    }

    probs
}

fn course_counts(pirate: &Pirate, course_indices: &[usize]) -> (u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    for &c in course_indices {
        let is_fav = pirate.favorite_courses.contains(&c);
        let is_allergy = pirate.allergy_courses.contains(&c);
        match (is_fav, is_allergy) {
            (true, true)   => { na += 1; }
            (true, false)  => { nf += 1; }
            (false, true)  => { na += 1; }
            (false, false) => {}
        }
    }
    (nf, na)
}

struct FlatMatch {
    pirate_indices: [usize; 4],
    course_indices: Vec<usize>,
    winner_pos: usize,
    is_train: bool,
}

fn main() {
    let pj = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = GameData::load(&pj);
    let hj = std::fs::read_to_string("../historical_matches.json").expect("historical_matches.json not found");
    let hist = load_historical_matches(&data, &hj);

    let n_train = (hist.len() as f64 * 0.8) as usize;

    let matches: Vec<FlatMatch> = hist.iter().enumerate()
        .flat_map(|(day_idx, day)| {
            day.iter().enumerate().map(move |(_arena_idx, m)| {
                FlatMatch {
                    pirate_indices: m.pirate_indices,
                    course_indices: m.course_indices.clone(),
                    winner_pos: m.winner_pos,
                    is_train: day_idx < n_train,
                }
            })
        })
        .collect();

    // Precompute (pirate_idx, nf, na) for each match position
    struct MatchKey {
        keys: [(usize, u32, u32); 4], // (pirate_idx, nf, na) per position
        winner_pos: usize,
        is_train: bool,
    }

    let match_keys: Vec<MatchKey> = matches.iter().map(|m| {
        let mut keys = [(0usize, 0u32, 0u32); 4];
        for pos in 0..4 {
            let pi = m.pirate_indices[pos];
            let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
            keys[pos] = (pi, nf, na);
        }
        MatchKey { keys, winner_pos: m.winner_pos, is_train: m.is_train }
    }).collect();

    // Collect all unique (pirate_idx, nf, na) tuples
    let mut unique_keys: Vec<(usize, u32, u32)> = Vec::new();
    let mut key_set: HashMap<(usize, u32, u32), usize> = HashMap::new();
    for mk in &match_keys {
        for &k in &mk.keys {
            if !key_set.contains_key(&k) {
                key_set.insert(k, unique_keys.len());
                unique_keys.push(k);
            }
        }
    }

    // Map match keys to indices into unique_keys
    struct MatchIdx {
        key_indices: [usize; 4],
        winner_pos: usize,
        is_train: bool,
    }
    let match_idxs: Vec<MatchIdx> = match_keys.iter().map(|mk| {
        MatchIdx {
            key_indices: [
                key_set[&mk.keys[0]],
                key_set[&mk.keys[1]],
                key_set[&mk.keys[2]],
                key_set[&mk.keys[3]],
            ],
            winner_pos: mk.winner_pos,
            is_train: mk.is_train,
        }
    }).collect();

    println!("Matches: {} ({} train, {} test)",
        match_idxs.len(),
        match_idxs.iter().filter(|m| m.is_train).count(),
        match_idxs.iter().filter(|m| !m.is_train).count());
    println!("Unique (pirate, nf, na) tuples: {}", unique_keys.len());
    println!("PMF samples per tuple: {PMF_SAMPLES}");
    println!();

    // Grid search ranges
    let bases: Vec<u32> = (109..=115).collect();
    let fav_divs: Vec<u32> = (13..=17).collect();
    let divisors: Vec<u32> = (12..=16).collect();
    let max_effects: Vec<u32> = (5..=9).collect();

    let total_combos = bases.len() * fav_divs.len() * divisors.len() * max_effects.len();
    println!("Grid: base={:?} fd={:?} div={:?} me={:?}", bases, fav_divs, divisors, max_effects);
    println!("Total combinations: {}", total_combos);
    println!();

    let mut results: Vec<(u32, u32, u32, u32, f64, f64)> = Vec::new();

    for (combo_idx, &base) in bases.iter().enumerate() {
        for &fav_div in &fav_divs {
            for &divisor in &divisors {
                for &max_effect in &max_effects {
                    // Build all PMFs in parallel
                    let pmfs: Vec<Vec<f64>> = unique_keys.par_iter().enumerate().map(|(idx, &(pi, nf, na))| {
                        let p = &data.pirates[pi];
                        build_pmf(p.strength, p.weight, nf, na,
                                  base, fav_div, divisor, max_effect,
                                  idx as u64 * 7919 + 42)
                    }).collect();

                    // Evaluate LL across all matches
                    let (train_ll, train_n, test_ll, test_n): (f64, u64, f64, u64) =
                        match_idxs.par_iter().map(|m| {
                            let p = [
                                &pmfs[m.key_indices[0]],
                                &pmfs[m.key_indices[1]],
                                &pmfs[m.key_indices[2]],
                                &pmfs[m.key_indices[3]],
                            ];
                            let probs = win_probs_from_pmfs(&p);
                            let ll = probs[m.winner_pos].max(1e-10).ln();
                            if m.is_train {
                                (ll, 1u64, 0.0, 0u64)
                            } else {
                                (0.0, 0u64, ll, 1u64)
                            }
                        }).reduce(|| (0.0, 0, 0.0, 0), |a, b| (a.0+b.0, a.1+b.1, a.2+b.2, a.3+b.3));

                    let trl = train_ll / train_n as f64;
                    let tel = test_ll / test_n as f64;
                    results.push((base, fav_div, divisor, max_effect, trl, tel));
                }
            }
        }
        eprintln!("  base={} done ({}/{})", base, (combo_idx+1) * fav_divs.len() * divisors.len() * max_effects.len(), total_combos);
    }

    // Sort by train LL descending
    results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    println!("{:>5} {:>4} {:>4} {:>4} {:>10} {:>10}", "base", "fd", "div", "me", "train_LL", "test_LL");
    println!("{}", "-".repeat(45));
    for (b, fd, d, me, train, test) in results.iter().take(30) {
        let marker = if *b == 112 && *fd == 15 && *d == 14 && *me == 7 { " <--" } else { "" };
        println!("{:>5} {:>4} {:>4} {:>4} {:>10.5} {:>10.5}{}", b, fd, d, me, train, test, marker);
    }

    println!();
    if let Some(pos) = results.iter().position(|(b, fd, d, me, _, _)| *b == 112 && *fd == 15 && *d == 14 && *me == 7) {
        let r = &results[pos];
        println!("Current baseline rank: {}/{} (train_LL={:.5}, test_LL={:.5})",
            pos + 1, results.len(), r.4, r.5);
    }

    // Also sort by test LL and show top 10
    results.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());
    println!();
    println!("=== TOP 10 BY TEST LL ===");
    println!("{:>5} {:>4} {:>4} {:>4} {:>10} {:>10}", "base", "fd", "div", "me", "train_LL", "test_LL");
    println!("{}", "-".repeat(45));
    for (b, fd, d, me, train, test) in results.iter().take(10) {
        let marker = if *b == 112 && *fd == 15 && *d == 14 && *me == 7 { " <--" } else { "" };
        println!("{:>5} {:>4} {:>4} {:>4} {:>10.5} {:>10.5}{}", b, fd, d, me, train, test, marker);
    }
}
