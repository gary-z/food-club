mod pirates;

use pirates::{GameData, HistMatch, load_historical_matches};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Best model params
const BASE: u32 = 112;
const FAV_DIV: u32 = 15;
const N_ROLLS: u32 = 4;
const DIVISOR: u32 = 14;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 7;

const PMF_SAMPLES: u64 = 100_000;
// Score is positive (eating time). Range 0..SCORE_MAX covers all possible values.
const SCORE_MAX: usize = 500;
const SCORE_RANGE: usize = SCORE_MAX + 1;

fn roll(rng: &mut impl Rng, n: u32) -> u32 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) }
}

fn course_counts(pirate: &pirates::Pirate, courses: &[usize]) -> (u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    for &c in courses {
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

/// Compute a pirate's eating time (score). Lower = faster = better.
///
/// 1. Allergies reduce effective strength: strength -= rand(1, weight_offset) per allergy
/// 2. Upper (die size) = base - effective_strength  (weaker -> bigger die -> slower)
/// 3. Favorites shrink the die: upper -= nf * floor(upper / fav_div)
/// 4. Roll n_rolls dice of size upper, sum them: eating_time = sum(rand(1, upper))
/// 5. Quantize: score = floor(eating_time / divisor)
fn pirate_score(p: &pirates::Pirate, nf: u32, na: u32, rng: &mut impl Rng) -> u32 {
    // Allergy damage: reduce effective strength
    let wo = if p.weight >= MAX_WEIGHT { 0 } else { ((MAX_WEIGHT - p.weight) / 2).min(MAX_EFFECT) };
    let mut strength = p.strength;
    for _ in 0..na {
        strength = strength.saturating_sub(roll(rng, wo));
    }

    // Die size: weaker pirates roll bigger dice (slower)
    let mut upper = if BASE > strength { BASE - strength } else { 1 };

    // Favorites shrink the die (eat faster)
    let reduction = upper / FAV_DIV;
    upper = upper.saturating_sub(nf * reduction).max(1);

    // Roll dice: total eating time
    let mut time = 0u32;
    for _ in 0..N_ROLLS {
        time += roll(rng, upper);
    }

    // Quantize
    time / DIVISOR
}

#[derive(Clone)]
struct ScoreDist {
    pmf: Vec<f64>,  // pmf[s] = P(score == s)
    cdf: Vec<f64>,  // cdf[s] = P(score <= s)
}

impl ScoreDist {
    fn from_samples(samples: &[u32]) -> Self {
        let mut pmf = vec![0.0; SCORE_RANGE];
        let n = samples.len() as f64;
        for &s in samples {
            let idx = s as usize;
            if idx < SCORE_RANGE { pmf[idx] += 1.0 / n; }
        }
        let mut cdf = vec![0.0; SCORE_RANGE];
        cdf[0] = pmf[0];
        for i in 1..SCORE_RANGE { cdf[i] = cdf[i-1] + pmf[i]; }
        ScoreDist { pmf, cdf }
    }
}

type ScoreKey = (usize, u32, u32); // (pirate_index, n_fav, n_allergy)

fn build_cache(data: &GameData, matches: &[HistMatch]) -> HashMap<ScoreKey, ScoreDist> {
    let mut keys: HashSet<ScoreKey> = HashSet::new();
    for m in matches {
        for &pi in &m.pirate_indices {
            let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
            keys.insert((pi, nf, na));
        }
    }
    let keys_vec: Vec<ScoreKey> = keys.into_iter().collect();
    keys_vec.par_iter().map(|&key| {
        let (pi, nf, na) = key;
        let mut rng = SmallRng::seed_from_u64(pi as u64 * 1000 + nf as u64 * 100 + na as u64);
        let samples: Vec<u32> = (0..PMF_SAMPLES)
            .map(|_| pirate_score(&data.pirates[pi], nf, na, &mut rng))
            .collect();
        (key, ScoreDist::from_samples(&samples))
    }).collect()
}

/// P(pirate at position idx wins). Lowest score wins; ties go to later position.
fn win_prob(idx: usize, dists: &[&ScoreDist; 4]) -> f64 {
    let opponents: Vec<usize> = (0..4).filter(|&j| j != idx).collect();
    let mut p_win = 0.0;

    for s in 0..SCORE_RANGE {
        let p_i = dists[idx].pmf[s];
        if p_i < 1e-12 { continue; }

        // P(opponent j has score > s) = 1 - cdf[s]  (opponent is slower -> we beat them)
        let p_gt: [f64; 3] = [
            1.0 - dists[opponents[0]].cdf[s],
            1.0 - dists[opponents[1]].cdf[s],
            1.0 - dists[opponents[2]].cdf[s],
        ];
        let p_eq: [f64; 3] = [
            dists[opponents[0]].pmf[s],
            dists[opponents[1]].pmf[s],
            dists[opponents[2]].pmf[s],
        ];

        // Enumerate which opponents tie with us (mask) vs are strictly slower (rest)
        for mask in 0u32..8 {
            // Among tied pirates, later position wins
            let mut max_tied_pos = idx;
            for bit in 0..3 {
                if mask & (1 << bit) != 0 && opponents[bit] > max_tied_pos {
                    max_tied_pos = opponents[bit];
                }
            }
            let wins_tie = if max_tied_pos == idx { 1.0 } else { 0.0 };

            let mut prob = wins_tie;
            for bit in 0..3 {
                if mask & (1 << bit) != 0 { prob *= p_eq[bit]; }
                else { prob *= p_gt[bit]; }
            }
            p_win += p_i * prob;
        }
    }
    p_win
}

fn match_log_likelihood(data: &GameData, matches: &[HistMatch]) -> f64 {
    let cache = build_cache(data, matches);
    let mut total_ll = 0.0;
    for m in matches {
        let keys: Vec<ScoreKey> = m.pirate_indices.iter()
            .map(|&pi| {
                let (nf, na) = course_counts(&data.pirates[pi], &m.course_indices);
                (pi, nf, na)
            }).collect();
        let dists: [&ScoreDist; 4] = [
            &cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]],
        ];
        let p = win_prob(m.winner_pos, &dists);
        total_ll += p.max(1e-6).ln();
    }
    total_ll / matches.len() as f64
}

// --- H2H analysis ---
#[derive(Deserialize)]
struct HistPirate { name: String, #[allow(dead_code)] odds: u32 }
#[derive(Deserialize)]
struct HistArena { pirates: Vec<HistPirate>, winner: String, foods: Vec<String>, #[allow(dead_code)] arena_name: String }

fn h2h_and_pirate_fit(
    data: &GameData, all_matches: &[HistMatch], historical: &[Vec<HistArena>],
    cache: &HashMap<ScoreKey, ScoreDist>,
    target_indices: &[usize],
) {
    let course_map = data.course_name_to_index();
    let mut h2h: HashMap<(usize, usize), (u32, u32, f64, u32)> = HashMap::new();
    for day_arenas in historical {
        for arena in day_arenas {
            let arena_pis: Vec<usize> = arena.pirates.iter()
                .map(|hp| data.pirates.iter().position(|p| p.name == hp.name).unwrap())
                .collect();
            let course_indices: Vec<usize> = arena.foods.iter()
                .filter_map(|f| course_map.get(f.as_str()).copied()).collect();
            for i in 0..target_indices.len() {
                for j in (i+1)..target_indices.len() {
                    let pi_a = target_indices[i];
                    let pi_b = target_indices[j];
                    let pos_a = arena_pis.iter().position(|&p| p == pi_a);
                    let pos_b = arena_pis.iter().position(|&p| p == pi_b);
                    if let (Some(pa), Some(pb)) = (pos_a, pos_b) {
                        let a_won = arena.pirates[pa].name == arena.winner;
                        let b_won = arena.pirates[pb].name == arena.winner;
                        if a_won || b_won {
                            let keys: Vec<ScoreKey> = arena_pis.iter()
                                .map(|&pi| {
                                    let (nf, na) = course_counts(&data.pirates[pi], &course_indices);
                                    (pi, nf, na)
                                }).collect();
                            if keys.iter().all(|k| cache.contains_key(k)) {
                                let dists: [&ScoreDist; 4] = [
                                    &cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]],
                                ];
                                let prob_a = win_prob(pa, &dists);
                                let prob_b = win_prob(pb, &dists);
                                let cond_a = prob_a / (prob_a + prob_b);
                                let entry = h2h.entry((pi_a, pi_b)).or_insert((0, 0, 0.0, 0));
                                if a_won { entry.0 += 1; }
                                if b_won { entry.1 += 1; }
                                entry.2 += cond_a;
                                entry.3 += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("  {:<22} {:<22} {:>5} {:>5} {:>7} {:>7} {:>7}",
        "Pirate A", "Pirate B", "A win", "B win", "A h%", "A m%", "Gap");
    for i in 0..target_indices.len() {
        for j in (i+1)..target_indices.len() {
            let pi_a = target_indices[i];
            let pi_b = target_indices[j];
            if let Some(&(aw, bw, ms, cnt)) = h2h.get(&(pi_a, pi_b)) {
                let tot = aw + bw;
                let hr = aw as f64 / tot as f64;
                let mr = ms / cnt as f64;
                println!("  {:<22} {:<22} {:>5} {:>5} {:>6.1}% {:>6.1}% {:>+6.1}",
                    &data.pirates[pi_a].name[..22.min(data.pirates[pi_a].name.len())],
                    &data.pirates[pi_b].name[..22.min(data.pirates[pi_b].name.len())],
                    aw, bw, hr*100.0, mr*100.0, (hr-mr)*100.0);
            }
        }
    }

    println!("\n  {:<22} {:>7} {:>7} {:>7}", "Pirate", "Hist%", "Model%", "Gap");
    for &pi in target_indices {
        let mut hw = 0u32;
        let mut mps = 0.0f64;
        let mut nm = 0u32;
        for m in all_matches {
            if let Some(pos) = m.pirate_indices.iter().position(|&p| p == pi) {
                let keys: Vec<ScoreKey> = m.pirate_indices.iter()
                    .map(|&p| { let (nf, na) = course_counts(&data.pirates[p], &m.course_indices); (p, nf, na) }).collect();
                if keys.iter().all(|k| cache.contains_key(k)) {
                    let dists: [&ScoreDist; 4] = [&cache[&keys[0]], &cache[&keys[1]], &cache[&keys[2]], &cache[&keys[3]]];
                    mps += win_prob(pos, &dists);
                    if m.winner_pos == pos { hw += 1; }
                    nm += 1;
                }
            }
        }
        let hr = hw as f64 / nm as f64;
        let mr = mps / nm as f64;
        println!("  {:<22} {:>6.1}% {:>6.1}% {:>+6.1}",
            &data.pirates[pi].name[..22.min(data.pirates[pi].name.len())], hr*100.0, mr*100.0, (hr-mr)*100.0);
    }
}

fn main() {
    let pj = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = Arc::new(GameData::load(&pj));
    let hj = std::fs::read_to_string("../historical_matches.json").expect("historical_matches.json not found");
    let all_days = load_historical_matches(&data, &hj);
    let historical: Vec<Vec<HistArena>> = serde_json::from_str(&hj).expect("parse");

    let n_days = all_days.len();
    let split = (n_days as f64 * 0.8) as usize;
    let train: Vec<HistMatch> = all_days[..split].iter().flat_map(|d| d.iter().cloned()).collect();
    let test: Vec<HistMatch> = all_days[split..].iter().flat_map(|d| d.iter().cloned()).collect();
    let all_matches: Vec<HistMatch> = all_days.iter().flat_map(|d| d.iter().cloned()).collect();

    println!("Model: b={BASE} bulk_fd={FAV_DIV} r={N_ROLLS} d={DIVISOR} me={MAX_EFFECT}");
    println!("Data: {} days, train={}, test={}, total={}\n",
        n_days, train.len(), test.len(), all_matches.len());

    let train_ll = match_log_likelihood(&data, &train);
    let test_ll = match_log_likelihood(&data, &test);
    println!("Train LL: {train_ll:.5}");
    println!("Test  LL: {test_ll:.5}\n");

    let target_names = ["Gooblah the Grarrl", "Buck Cutlass", "Franchisco Corvallio",
                        "Fairfax the Deckhand", "Stuff-A-Roo", "Orvinn the First Mate"];
    let target_indices: Vec<usize> = target_names.iter()
        .map(|name| data.pirates.iter().position(|p| p.name == *name).unwrap())
        .collect();

    println!("=== H2H ANALYSIS ===");
    let cache = build_cache(&data, &all_matches);
    h2h_and_pirate_fit(&data, &all_matches, &historical, &cache, &target_indices);
}
