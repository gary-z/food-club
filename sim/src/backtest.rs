mod pirates;

use pirates::{GameData, Pirate};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// Best model params (b=111 with life_adj)
const BASE: u32 = 111;
const FAV_DIV: u32 = 15;
const N_ROLLS: u32 = 4;
const DIVISOR: u32 = 14;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 7;
const MAX_PAYOUT: u32 = 60;
const SIM_ITERATIONS: u32 = 50_000;

// Per-pirate strength adjustments (indexed by pirate order in pirates.json)
// [Scurvy, Young+1, Orvinn+6, Lucky-1, Edmund+1, PegLeg, Bonnie, Puffo-1,
//  Stuff+1, Squire+1, Crossblades-2, Stripey, Ned, Fairfax+1, Gooblah-1,
//  Franchisco, Federismo, Blackbeard-1, Buck, Tailhook+1]
const STR_ADJ: [i32; 20] = [0, 1, 6, -1, 1, 0, 0, -1, 1, 1, -2, 0, 0, 1, -1, 0, 0, -1, 0, 1];

fn roll(rng: &mut impl Rng, n: u32) -> u32 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) }
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

/// Compute eating time (lower = faster = better).
fn eating_time(pirate: &Pirate, course_indices: &[usize], life_adj: i32, rng: &mut impl Rng) -> u32 {
    let (nf, na) = course_counts(pirate, course_indices);

    // Life starts at strength, allergy damage reduces it
    let wo = if pirate.weight >= MAX_WEIGHT { 0 } else { ((MAX_WEIGHT - pirate.weight) / 2).min(MAX_EFFECT) };
    let mut life = (pirate.strength as i32 + life_adj).max(0) as u32;
    for _ in 0..na {
        life = life.saturating_sub(roll(rng, wo));
    }

    // Die size: lower life = bigger die = slower
    let mut upper = if BASE > life { BASE - life } else { 1 };

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

/// Compute win probabilities via MC simulation. Lowest time wins; ties go to later position.
fn arena_win_probs(
    pirates: &[&Pirate],
    course_indices: &[usize],
    life_adjs: &[i32],
    iterations: u32,
    seed: u64,
) -> HashMap<String, f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut wins: HashMap<String, u32> = HashMap::new();
    for p in pirates {
        wins.insert(p.name.clone(), 0);
    }

    for _ in 0..iterations {
        let times: Vec<u32> = pirates.iter().enumerate()
            .map(|(i, p)| eating_time(p, course_indices, life_adjs[i], &mut rng))
            .collect();

        // Lowest time wins. Ties: later position wins.
        let min_time = *times.iter().min().unwrap();
        let mut winner_pos = 0;
        for (pos, &t) in times.iter().enumerate() {
            if t == min_time {
                winner_pos = pos;
            }
        }
        *wins.get_mut(&pirates[winner_pos].name).unwrap() += 1;
    }

    wins.into_iter()
        .map(|(name, count)| (name, count as f64 / iterations as f64))
        .collect()
}

// --- Historical data parsing ---

#[derive(Deserialize, Clone)]
struct HistPirate {
    name: String,
    odds: u32,
    #[serde(default)]
    current_odds: Option<u32>,
}

#[derive(Deserialize)]
struct HistArena {
    #[allow(dead_code)]
    arena_name: String,
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
}

struct Bet {
    pirate_names: Vec<String>,
    win_probability: f64,
    payout: u32,
}

fn make_bets_top_ev(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
) -> Vec<Bet> {
    let n = arenas.len();
    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            let mut win_prob = 1.0;
            let mut payout = 1u32;
            let mut pirate_names = Vec::with_capacity(arena_indices.len());

            for (j, &ai) in arena_indices.iter().enumerate() {
                let pirate = &pirate_options[j][combo_indices[j]];
                win_prob *= arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                payout = (payout * pirate.odds).min(MAX_PAYOUT);
                pirate_names.push(pirate.name.clone());
            }

            possible_bets.push(Bet {
                pirate_names,
                win_probability: win_prob,
                payout,
            });

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= pirate_options[j].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Only bet on single-arena odds=2 pirates, ranked by model probability.
fn make_bets_only_2s(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
) -> Vec<Bet> {
    let mut bets: Vec<Bet> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        for pirate in &arena.pirates {
            if pirate.odds == 2 {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                bets.push(Bet {
                    pirate_names: vec![pirate.name.clone()],
                    win_probability: prob,
                    payout: 2,
                });
            }
        }
    }
    // Sort by probability (highest first = best EV at fixed payout)
    bets.sort_by(|a, b| b.win_probability.partial_cmp(&a.win_probability).unwrap());
    bets.truncate(10);
    bets
}

/// Only bet on odds=2 pirates where model says p > threshold.
fn make_bets_2s_filtered(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_prob: f64,
) -> Vec<Bet> {
    let mut bets: Vec<Bet> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        for pirate in &arena.pirates {
            if pirate.odds == 2 {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                if prob >= min_prob {
                    bets.push(Bet {
                        pirate_names: vec![pirate.name.clone()],
                        win_probability: prob,
                        payout: 2,
                    });
                }
            }
        }
    }
    bets.sort_by(|a, b| b.win_probability.partial_cmp(&a.win_probability).unwrap());
    bets.truncate(10);
    bets
}

/// Top-10 EV but excluding any bet that contains an odds=2 pirate.
fn make_bets_no_2s(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
) -> Vec<Bet> {
    let n = arenas.len();
    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            // Skip if any pirate has odds=2
            let has_2 = arena_indices.iter().enumerate().any(|(j, &ai)| {
                pirate_options[j][combo_indices[j]].odds == 2
            });

            if !has_2 {
                let mut win_prob = 1.0;
                let mut payout = 1u32;
                let mut pirate_names = Vec::with_capacity(arena_indices.len());

                for (j, &ai) in arena_indices.iter().enumerate() {
                    let pirate = &pirate_options[j][combo_indices[j]];
                    win_prob *= arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                    payout = (payout * pirate.odds).min(MAX_PAYOUT);
                    pirate_names.push(pirate.name.clone());
                }

                possible_bets.push(Bet {
                    pirate_names,
                    win_probability: win_prob,
                    payout,
                });
            }

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= pirate_options[j].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Parlays using only odds=2 pirates. Always produce 10 bets.
/// Enumerate all combinations of odds=2 pirates across arenas (singles + parlays),
/// rank by EV, take top 10.
fn make_bets_2s_parlays(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
) -> Vec<Bet> {
    // Collect odds=2 pirates per arena: (arena_index, pirate_name, prob, odds)
    let mut twos_by_arena: Vec<Vec<(usize, String, f64)>> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        let mut arena_twos = Vec::new();
        for pirate in &arena.pirates {
            if pirate.odds == 2 {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                arena_twos.push((ai, pirate.name.clone(), prob));
            }
        }
        if !arena_twos.is_empty() {
            twos_by_arena.push(arena_twos);
        }
    }

    let n = twos_by_arena.len();
    if n == 0 {
        return Vec::new();
    }

    let mut possible_bets: Vec<Bet> = Vec::new();

    // Enumerate all subsets of arenas that have odds=2 pirates
    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        // For each subset, enumerate all combinations of odds=2 pirates
        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            let mut win_prob = 1.0;
            let mut payout = 1u32;
            let mut pirate_names = Vec::with_capacity(arena_indices.len());

            for (j, &ai) in arena_indices.iter().enumerate() {
                let (_, ref name, prob) = twos_by_arena[ai][combo_indices[j]];
                win_prob *= prob;
                payout = (payout * 2).min(MAX_PAYOUT);
                pirate_names.push(name.clone());
            }

            possible_bets.push(Bet {
                pirate_names,
                win_probability: win_prob,
                payout,
            });

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= twos_by_arena[arena_indices[j]].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Filtered 2:1 parlays: only use odds=2 pirates where model p >= min_prob,
/// only keep bets with EV >= 1.0, take up to 10 best by EV.
fn make_bets_2s_parlays_filtered(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_prob: f64,
) -> Vec<Bet> {
    // Collect odds=2 pirates per arena, filtered by min_prob
    let mut twos_by_arena: Vec<Vec<(usize, String, f64)>> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        let mut arena_twos = Vec::new();
        for pirate in &arena.pirates {
            if pirate.odds == 2 {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                if prob >= min_prob {
                    arena_twos.push((ai, pirate.name.clone(), prob));
                }
            }
        }
        if !arena_twos.is_empty() {
            twos_by_arena.push(arena_twos);
        }
    }

    let n = twos_by_arena.len();
    if n == 0 {
        return Vec::new();
    }

    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            let mut win_prob = 1.0;
            let mut payout = 1u32;
            let mut pirate_names = Vec::with_capacity(arena_indices.len());

            for (j, &ai) in arena_indices.iter().enumerate() {
                let (_, ref name, prob) = twos_by_arena[ai][combo_indices[j]];
                win_prob *= prob;
                payout = (payout * 2).min(MAX_PAYOUT);
                pirate_names.push(name.clone());
            }

            let ev = win_prob * payout as f64;
            if ev >= 1.0 {
                possible_bets.push(Bet {
                    pirate_names,
                    win_probability: win_prob,
                    payout,
                });
            }

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= twos_by_arena[arena_indices[j]].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Anchor on good 2:1 pirates, pad with any other pirates to boost payout.
/// Every bet must include at least one odds=2 pirate with model p >= min_prob.
/// Non-2:1 pirates can be added freely to multiply payout.
/// Only keep bets with EV >= 1.0. Take top 10 by EV.
fn make_bets_anchored_2s(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_prob: f64,
) -> Vec<Bet> {
    let n = arenas.len();

    // Which arenas have a "good" 2:1 pirate?
    let mut good_2_arenas: Vec<bool> = vec![false; n];
    for (ai, arena) in arenas.iter().enumerate() {
        for pirate in &arena.pirates {
            if pirate.odds == 2 {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                if prob >= min_prob {
                    good_2_arenas[ai] = true;
                }
            }
        }
    }

    // Need at least one arena with a good 2:1
    if !good_2_arenas.iter().any(|&x| x) {
        return Vec::new();
    }

    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        // Must include at least one arena that has a good 2:1
        if !arena_indices.iter().any(|&i| good_2_arenas[i]) {
            continue;
        }

        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            // Check: at least one pirate in this combo must be a good 2:1
            let has_good_2 = arena_indices.iter().enumerate().any(|(j, &ai)| {
                let pirate = &pirate_options[j][combo_indices[j]];
                pirate.odds == 2 && {
                    let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                    prob >= min_prob
                }
            });

            if has_good_2 {
                let mut win_prob = 1.0;
                let mut payout = 1u32;
                let mut pirate_names = Vec::with_capacity(arena_indices.len());

                for (j, &ai) in arena_indices.iter().enumerate() {
                    let pirate = &pirate_options[j][combo_indices[j]];
                    win_prob *= arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                    payout = (payout * pirate.odds).min(MAX_PAYOUT);
                    pirate_names.push(pirate.name.clone());
                }

                let ev = win_prob * payout as f64;
                if ev >= 1.0 {
                    possible_bets.push(Bet {
                        pirate_names,
                        win_probability: win_prob,
                        payout,
                    });
                }
            }

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= pirate_options[j].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Effective payout: current_odds if available, else opening odds.
fn eff_odds(pirate: &HistPirate) -> u32 {
    pirate.current_odds.unwrap_or(pirate.odds)
}

/// Combined current-odds exploitation strategy.
/// A pirate is a "+EV anchor" if:
///   (a) opening_odds=2 and model p >= min_2s_prob, OR
///   (b) current_odds >= opening_odds + min_jump (guaranteed +EV by opening bound)
/// Every bet must contain at least one anchor. All payouts use current_odds.
/// Only keep bets with EV >= 1.0. Top 10 by EV.
fn make_bets_current_exploit(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_2s_prob: f64,
    min_jump: u32,
) -> Vec<Bet> {
    let n = arenas.len();

    // Precompute: is each pirate an anchor?
    let mut has_anchor = vec![false; n]; // does this arena have at least one anchor?
    let mut pirate_is_anchor: Vec<Vec<bool>> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        let mut arena_anchors = Vec::new();
        for pirate in &arena.pirates {
            let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
            let cur = eff_odds(pirate);
            let is_anc = (pirate.odds == 2 && prob >= min_2s_prob)
                || (cur >= pirate.odds + min_jump);
            arena_anchors.push(is_anc);
            if is_anc { has_anchor[ai] = true; }
        }
        pirate_is_anchor.push(arena_anchors);
    }

    if !has_anchor.iter().any(|&x| x) {
        return Vec::new();
    }

    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        // Must include at least one arena that has an anchor
        if !arena_indices.iter().any(|&i| has_anchor[i]) {
            continue;
        }

        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            // Check: at least one pirate in this combo is an anchor
            let combo_has_anchor = arena_indices.iter().enumerate().any(|(j, &ai)| {
                pirate_is_anchor[ai][combo_indices[j]]
            });

            if combo_has_anchor {
                let mut win_prob = 1.0;
                let mut payout = 1u32;
                let mut pirate_names = Vec::with_capacity(arena_indices.len());

                for (j, &ai) in arena_indices.iter().enumerate() {
                    let pirate = &pirate_options[j][combo_indices[j]];
                    win_prob *= arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                    payout = (payout * eff_odds(pirate)).min(MAX_PAYOUT);
                    pirate_names.push(pirate.name.clone());
                }

                let ev = win_prob * payout as f64;
                if ev >= 1.0 {
                    possible_bets.push(Bet {
                        pirate_names,
                        win_probability: win_prob,
                        payout,
                    });
                }
            }

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= pirate_options[j].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Same as make_bets_current_exploit but uses opening-odds-bounded probability
/// floor (1/(opening+1)) for jump pirates instead of model probability.
/// For 2:1 anchors, still uses model (opening bound is too wide: p in (1/3, 1)).
/// For non-anchor pirates, uses model probability.
fn make_bets_current_exploit_floor(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_2s_prob: f64,
    min_jump: u32,
) -> Vec<Bet> {
    let n = arenas.len();

    let mut has_anchor = vec![false; n];
    let mut pirate_is_anchor: Vec<Vec<bool>> = Vec::new();
    let mut pirate_is_jump: Vec<Vec<bool>> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        let mut arena_anchors = Vec::new();
        let mut arena_jumps = Vec::new();
        for pirate in &arena.pirates {
            let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
            let cur = eff_odds(pirate);
            let is_jump = cur >= pirate.odds + min_jump;
            let is_anc = (pirate.odds == 2 && prob >= min_2s_prob) || is_jump;
            arena_anchors.push(is_anc);
            arena_jumps.push(is_jump);
            if is_anc { has_anchor[ai] = true; }
        }
        pirate_is_anchor.push(arena_anchors);
        pirate_is_jump.push(arena_jumps);
    }

    if !has_anchor.iter().any(|&x| x) {
        return Vec::new();
    }

    let mut possible_bets: Vec<Bet> = Vec::new();

    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        if !arena_indices.iter().any(|&i| has_anchor[i]) {
            continue;
        }

        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        let mut combo_indices = vec![0usize; arena_indices.len()];
        loop {
            let combo_has_anchor = arena_indices.iter().enumerate().any(|(j, &ai)| {
                pirate_is_anchor[ai][combo_indices[j]]
            });

            if combo_has_anchor {
                let mut win_prob = 1.0;
                let mut payout = 1u32;
                let mut pirate_names = Vec::with_capacity(arena_indices.len());

                for (j, &ai) in arena_indices.iter().enumerate() {
                    let pirate = &pirate_options[j][combo_indices[j]];
                    let pi = combo_indices[j];

                    // For jump pirates: use floor probability 1/(opening+1)
                    // For others: use model probability
                    let prob = if pirate_is_jump[ai][pi] {
                        1.0 / (pirate.odds as f64 + 1.0)
                    } else {
                        arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0)
                    };

                    win_prob *= prob;
                    payout = (payout * eff_odds(pirate)).min(MAX_PAYOUT);
                    pirate_names.push(pirate.name.clone());
                }

                let ev = win_prob * payout as f64;
                if ev >= 1.0 {
                    possible_bets.push(Bet {
                        pirate_names,
                        win_probability: win_prob,
                        payout,
                    });
                }
            }

            let mut carry = true;
            for j in (0..combo_indices.len()).rev() {
                if carry {
                    combo_indices[j] += 1;
                    if combo_indices[j] >= pirate_options[j].len() {
                        combo_indices[j] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }
    }

    possible_bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    possible_bets.truncate(10);
    possible_bets
}

/// Singles-only on pirates where current_odds > opening_odds (for comparison).
fn make_bets_current_jump_singles(
    arena_probs: &[HashMap<String, f64>],
    arenas: &[HistArena],
    min_jump: u32,
) -> Vec<Bet> {
    let mut bets: Vec<Bet> = Vec::new();
    for (ai, arena) in arenas.iter().enumerate() {
        for pirate in &arena.pirates {
            let cur = eff_odds(pirate);
            if cur >= pirate.odds + min_jump {
                let prob = arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                let ev = prob * cur as f64;
                if ev >= 1.0 {
                    bets.push(Bet {
                        pirate_names: vec![pirate.name.clone()],
                        win_probability: prob,
                        payout: cur,
                    });
                }
            }
        }
    }
    bets.sort_by(|a, b| {
        let ev_a = a.win_probability * a.payout as f64;
        let ev_b = b.win_probability * b.payout as f64;
        ev_b.partial_cmp(&ev_a).unwrap()
    });
    bets.truncate(10);
    bets
}

fn bet_won(bet: &Bet, arenas: &[HistArena]) -> bool {
    let winners: std::collections::HashSet<&str> = arenas.iter()
        .map(|a| a.winner.as_str())
        .collect();
    bet.pirate_names.iter().all(|name| winners.contains(name.as_str()))
}

struct StrategyResult {
    total_wagered: u32,
    total_payout: u32,
    total_ev: f64,
    num_days: u32,
    num_bets: u32,
    day_profits: Vec<i32>, // per-day net profit for variance calc
}

impl StrategyResult {
    fn new() -> Self {
        StrategyResult {
            total_wagered: 0, total_payout: 0, total_ev: 0.0,
            num_days: 0, num_bets: 0, day_profits: Vec::new(),
        }
    }

    fn add_day(&mut self, bets: &[Bet], arenas: &[HistArena]) {
        if bets.is_empty() { return; }
        self.num_days += 1;
        let wagered = bets.len() as u32;
        self.num_bets += wagered;
        self.total_wagered += wagered;

        let mut day_payout = 0u32;
        for b in bets {
            self.total_ev += b.win_probability * b.payout as f64;
            if bet_won(b, arenas) {
                day_payout += b.payout;
            }
        }
        self.total_payout += day_payout;
        self.day_profits.push(day_payout as i32 - wagered as i32);
    }

    fn print(&self, label: &str) {
        if self.num_days == 0 {
            println!("  {:<40} no data", label);
            return;
        }
        let roi = (self.total_payout as f64 - self.total_wagered as f64)
            / self.total_wagered as f64 * 100.0;
        let avg_ev_per_bet = self.total_ev / self.num_bets as f64;
        let avg_bets_per_day = self.num_bets as f64 / self.num_days as f64;

        // Per-bet ROI standard error
        let profit = self.total_payout as f64 - self.total_wagered as f64;
        let avg_profit_per_bet = profit / self.num_bets as f64;

        // Compute variance of per-bet returns
        // We approximate using per-day profits scaled by bets per day
        let mean_day_profit = self.day_profits.iter().map(|&x| x as f64).sum::<f64>()
            / self.day_profits.len() as f64;
        let var_day = self.day_profits.iter()
            .map(|&x| (x as f64 - mean_day_profit).powi(2))
            .sum::<f64>() / (self.day_profits.len() as f64 - 1.0);
        let se_day = (var_day / self.day_profits.len() as f64).sqrt();
        let se_roi = se_day / avg_bets_per_day * 100.0; // as % of wager

        let profit_total = self.total_payout as i32 - self.total_wagered as i32;

        println!("  {:<40} {:>5} days {:>6} bets | ROI {:>+6.1}% +/- {:.1}% | EV/bet {:.3} | profit {:>+6}",
            label, self.num_days, self.num_bets, roi, se_roi * 1.96, avg_ev_per_bet, profit_total);
    }
}

fn main() {
    let game_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let game_data = GameData::load(&game_json);
    let course_map = game_data.course_name_to_index();

    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let historical: Vec<Vec<HistArena>> = serde_json::from_str(&hist_json)
        .expect("Failed to parse historical_matches.json");

    println!("Loaded {} historical days", historical.len());
    println!("Model: b={BASE} bulk_fd={FAV_DIV} r={N_ROLLS} d={DIVISOR} me={MAX_EFFECT}");
    println!("Sim iterations per arena: {SIM_ITERATIONS}");
    println!();

    // Build pirate index lookup for life_adj
    let pirate_index: HashMap<&str, usize> = game_data.pirates.iter().enumerate()
        .map(|(i, p)| (p.name.as_str(), i))
        .collect();

    // Compute arena probs for all days in parallel
    let day_probs: Vec<Vec<HashMap<String, f64>>> = historical.par_iter().enumerate().map(|(day_idx, day_arenas)| {
        day_arenas.iter().enumerate().map(|(arena_idx, arena)| {
            let arena_pirates: Vec<&Pirate> = arena.pirates.iter()
                .map(|hp| game_data.pirate_by_name(&hp.name)
                    .unwrap_or_else(|| panic!("Unknown pirate: {}", hp.name)))
                .collect();
            let course_indices: Vec<usize> = arena.foods.iter()
                .filter_map(|food| course_map.get(food.as_str()).copied())
                .collect();
            let adjs: Vec<i32> = arena.pirates.iter()
                .map(|hp| {
                    let idx = *pirate_index.get(hp.name.as_str()).unwrap_or(&0);
                    if idx < STR_ADJ.len() { STR_ADJ[idx] } else { 0 }
                })
                .collect();
            arena_win_probs(&arena_pirates, &course_indices, &adjs, SIM_ITERATIONS,
                day_idx as u64 * 17 + arena_idx as u64 * 31 + 42)
        }).collect()
    }).collect();

    // Run strategies
    let mut strat_top_ev = StrategyResult::new();
    let mut strat_anc_55 = StrategyResult::new();
    let mut strat_comb_55_j1 = StrategyResult::new();

    for (day_arenas, probs) in historical.iter().zip(day_probs.iter()) {
        strat_top_ev.add_day(&make_bets_top_ev(probs, day_arenas), day_arenas);
        strat_anc_55.add_day(&make_bets_anchored_2s(probs, day_arenas, 0.55), day_arenas);
        strat_comb_55_j1.add_day(&make_bets_current_exploit(probs, day_arenas, 0.55, 1), day_arenas);
    }

    println!("=== STRATEGY COMPARISON (b={BASE}, life_adj applied) ===\n");
    println!("  --- Pure model EV (opening odds payout) ---");
    strat_top_ev.print("Top-10 EV (model_p * odds)");
    strat_anc_55.print("Anchor 2:1 p>=0.55 + any, EV>=1");

    println!();
    println!("  --- Combined: current_odds payout ---");
    strat_comb_55_j1.print("2:1 p>=0.55 OR jump>=1, cur odds");
}
