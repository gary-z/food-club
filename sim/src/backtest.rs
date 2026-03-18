mod pirates;

use pirates::{GameData, Pirate};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// Model 4: Iterative Fav + Allergy-After (best hand-rolled, modern LL=-1.06314)
const BASE: u32 = 120;
const FAV_DIV: u32 = 16;
const N_ROLLS: u32 = 6;
const DIVISOR: u32 = 22;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 6;
const MAX_PAYOUT: u32 = 60;

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

// ==================== PMF-based exact probability engine ====================

/// PMF of sum of `n` dice, each uniform on {1, ..., d}.
fn dice_sum_pmf(n: u32, d: u32) -> Vec<f64> {
    if d == 0 || n == 0 { return vec![1.0]; }
    let max = (n * d) as usize;
    let inv_d = 1.0 / d as f64;
    let mut pmf = vec![0.0; max + 1];
    for k in 1..=(d as usize) { pmf[k] = inv_d; }
    for _ in 1..n {
        let mut new = vec![0.0; max + 1];
        let mut s = 0.0;
        for k in 0..=max {
            if k >= 1 { s += pmf[k - 1]; }
            if k > d as usize { s -= pmf[k - d as usize - 1]; }
            new[k] = s * inv_d;
        }
        pmf = new;
    }
    pmf
}

/// Compute a pirate's quantized score PMF for Model 4.
/// Iterative fav, allergy-after, floor quantization, later-wins tiebreak.
fn pirate_score_pmf(pirate: &Pirate, course_indices: &[usize], roll_table: &[Vec<f64>]) -> Vec<f64> {
    let (nf, na) = course_counts(pirate, course_indices);

    let raw_wo = MAX_WEIGHT.saturating_sub(pirate.weight.min(MAX_WEIGHT)) / 2;
    let wo = if MAX_EFFECT > 0 { raw_wo.min(MAX_EFFECT) } else { raw_wo };

    // Allergy damage PMF (sum of na dice each uniform 1..wo)
    let dmg_pmf: Vec<f64> = if na > 0 && wo > 0 {
        dice_sum_pmf(na, wo)
    } else {
        vec![1.0]
    };

    let max_raw_score = (N_ROLLS as usize) * (roll_table.len() - 1);
    let mut raw_pmf = vec![0.0; max_raw_score + 1];

    for (dmg_val, &dp) in dmg_pmf.iter().enumerate() {
        if dp < 1e-15 { continue; }

        // Die size from strength
        let mut upper = if BASE > pirate.strength { BASE - pirate.strength } else { 1 }.max(1);

        // Iterative fav reduction
        for _ in 0..nf {
            let red = upper / FAV_DIV;
            upper = upper.saturating_sub(red).max(1);
        }

        // Allergy damage AFTER fav: increases the die
        upper += dmg_val as u32;
        upper = upper.max(1);

        if (upper as usize) < roll_table.len() {
            let rpmf = &roll_table[upper as usize];
            for (k, &rp) in rpmf.iter().enumerate() {
                if rp > 0.0 && k < raw_pmf.len() {
                    raw_pmf[k] += dp * rp;
                }
            }
        }
    }

    // Floor quantization by DIVISOR
    let max_q = max_raw_score / DIVISOR as usize;
    let mut qpmf = vec![0.0; max_q + 1];
    for (k, &pr) in raw_pmf.iter().enumerate() {
        if pr < 1e-15 { continue; }
        let qk = k / DIVISOR as usize;
        if qk <= max_q { qpmf[qk] += pr; }
    }
    qpmf
}

/// Compute win probabilities from 4 independent score PMFs. Later position wins ties.
fn win_probs_from_pmfs(pmfs: [&[f64]; 4]) -> [f64; 4] {
    let max_t = pmfs.iter().map(|p| p.len()).max().unwrap_or(1);
    // Survival functions: P(score > t)
    let surv: [Vec<f64>; 4] = std::array::from_fn(|i| {
        let mut s = vec![0.0; max_t + 1];
        let mut acc = 0.0;
        for t in (0..pmfs[i].len()).rev() {
            s[t] = acc;
            acc += pmfs[i][t];
        }
        s
    });
    let f = |i: usize, t: usize| -> f64 {
        if t < pmfs[i].len() { pmfs[i][t] } else { 0.0 }
    };
    let s = |i: usize, t: usize| -> f64 {
        if t < surv[i].len() { surv[i][t] } else { 0.0 }
    };
    // g(i,t) = P(score_i >= t) = P(score_i > t-1)
    let g = |i: usize, t: usize| -> f64 {
        if t == 0 { 1.0 } else { s(i, t - 1) }
    };

    let mut probs = [0.0f64; 4];
    for t in 0..max_t {
        // Later position wins ties (tiebreak=0)
        probs[3] += f(3,t) * g(0,t) * g(1,t) * g(2,t);
        probs[2] += f(2,t) * g(0,t) * g(1,t) * s(3,t);
        probs[1] += f(1,t) * g(0,t) * s(2,t) * s(3,t);
        probs[0] += f(0,t) * s(1,t) * s(2,t) * s(3,t);
    }
    probs
}

/// Compute exact win probabilities for an arena via PMF convolution.
fn arena_win_probs(
    pirates: &[&Pirate],
    course_indices: &[usize],
    roll_table: &[Vec<f64>],
) -> HashMap<String, f64> {
    let pmfs: Vec<Vec<f64>> = pirates.iter()
        .map(|p| pirate_score_pmf(p, course_indices, roll_table))
        .collect();
    let pmf_refs: [&[f64]; 4] = [&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]];
    let probs = win_probs_from_pmfs(pmf_refs);
    let mut result = HashMap::new();
    for (i, p) in pirates.iter().enumerate() {
        result.insert(p.name.clone(), probs[i]);
    }
    result
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
    #[serde(default)]
    legacy: bool,
}

#[derive(Clone)]
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

// Probability interval implied by house odds N = max(2, min(13, floor(1/p)))
fn odds_prob_bounds(odds: u32) -> (f64, f64) {
    match odds {
        2 => (1.0 / 3.0, 1.0),
        13 => (0.0, 1.0 / 13.0),
        n => (1.0 / (n as f64 + 1.0), 1.0 / n as f64),
    }
}

fn clamp_and_redistribute(probs: &[f64; 4], intervals: &[(f64, f64); 4]) -> [f64; 4] {
    let mut p = *probs;
    let mut fixed = [false; 4];
    for _ in 0..20 {
        let mut changed = false;
        for i in 0..4 {
            if fixed[i] { continue; }
            if p[i] < intervals[i].0 {
                p[i] = intervals[i].0;
                fixed[i] = true;
                changed = true;
            } else if p[i] > intervals[i].1 {
                p[i] = intervals[i].1;
                fixed[i] = true;
                changed = true;
            }
        }
        let fixed_sum: f64 = (0..4).filter(|&i| fixed[i]).map(|i| p[i]).sum();
        let free_idx: Vec<usize> = (0..4).filter(|&i| !fixed[i]).collect();
        let free_sum: f64 = free_idx.iter().map(|&i| p[i]).sum();
        if !free_idx.is_empty() && free_sum > 0.0 {
            let target = 1.0 - fixed_sum;
            let scale = target / free_sum;
            for &i in &free_idx {
                p[i] *= scale;
            }
        }
        if !changed { break; }
    }
    p
}

/// Compute clamped win probabilities for an arena using odds intervals.
fn arena_win_probs_clamped(
    pirates: &[&Pirate],
    course_indices: &[usize],
    roll_table: &[Vec<f64>],
    opening_odds: &[u32],
) -> HashMap<String, f64> {
    let pmfs: Vec<Vec<f64>> = pirates.iter()
        .map(|p| pirate_score_pmf(p, course_indices, roll_table))
        .collect();
    let pmf_refs: [&[f64]; 4] = [&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]];
    let raw_probs = win_probs_from_pmfs(pmf_refs);

    let intervals: [(f64, f64); 4] = std::array::from_fn(|i| odds_prob_bounds(opening_odds[i]));
    let clamped = clamp_and_redistribute(&raw_probs, &intervals);

    let mut result = HashMap::new();
    for (i, p) in pirates.iter().enumerate() {
        result.insert(p.name.clone(), clamped[i]);
    }
    result
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


fn bet_won(bet: &Bet, arenas: &[HistArena]) -> bool {
    let winners: std::collections::HashSet<&str> = arenas.iter()
        .map(|a| a.winner.as_str())
        .collect();
    bet.pirate_names.iter().all(|name| winners.contains(name.as_str()))
}

struct StrategyResult {
    total_wagered: u32,
    total_payout: u32,
    total_model_ev: f64,
    num_days: u32,
    num_bets: u32,
    bust_days: u32,
    day_profits: Vec<i32>, // per-day net profit for variance calc
}

impl StrategyResult {
    fn new() -> Self {
        StrategyResult {
            total_wagered: 0, total_payout: 0, total_model_ev: 0.0,
            num_days: 0, num_bets: 0, bust_days: 0, day_profits: Vec::new(),
        }
    }

    fn add_day(&mut self, bets: &[Bet], arenas: &[HistArena], arena_probs: &[HashMap<String, f64>]) {
        if bets.is_empty() { return; }
        self.num_days += 1;
        let wagered = bets.len() as u32;
        self.num_bets += wagered;
        self.total_wagered += wagered;

        let mut day_payout = 0u32;
        for b in bets {
            // Compute model-based EV using actual model probabilities
            let model_win_prob: f64 = b.pirate_names.iter().map(|name| {
                arena_probs.iter()
                    .filter_map(|ap| ap.get(name))
                    .next()
                    .copied()
                    .unwrap_or(0.0)
            }).product();
            self.total_model_ev += model_win_prob * b.payout as f64;
            if bet_won(b, arenas) {
                day_payout += b.payout;
            }
        }
        self.total_payout += day_payout;
        if day_payout == 0 { self.bust_days += 1; }
        self.day_profits.push(day_payout as i32 - wagered as i32);
    }

    fn print(&self, label: &str) {
        if self.num_days == 0 {
            println!("  {:<40} no data", label);
            return;
        }
        let roi = (self.total_payout as f64 - self.total_wagered as f64)
            / self.total_wagered as f64 * 100.0;
        let expected_roi = (self.total_model_ev - self.total_wagered as f64)
            / self.total_wagered as f64 * 100.0;
        let avg_bets_per_day = self.num_bets as f64 / self.num_days as f64;

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

        let bust_pct = self.bust_days as f64 / self.num_days as f64 * 100.0;

        println!("  {:<40} {:>5} days {:>6} bets | expected {:>+6.1}% | actual {:>+6.1}% +/- {:.1}% | bust {:.1}% | profit {:>+6}",
            label, self.num_days, self.num_bets, expected_roi, roi, se_roi * 1.96, bust_pct, profit_total);
    }
}

fn main() {
    let game_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let game_data = GameData::load(&game_json);
    let course_map = game_data.course_name_to_index();

    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let all_days: Vec<Vec<HistArena>> = serde_json::from_str(&hist_json)
        .expect("Failed to parse historical_matches.json");

    // Filter to modern data only (post-legacy PHP upgrade)
    let historical: Vec<Vec<HistArena>> = all_days.into_iter()
        .filter(|day| day.first().map_or(false, |a| !a.legacy))
        .collect();

    let total_arenas: usize = historical.iter().map(|d| d.len()).sum();
    println!("Modern data: {} days, {} arenas", historical.len(), total_arenas);
    println!("Model 4: b={BASE} iter_fd={FAV_DIV} r={N_ROLLS} d={DIVISOR} me={MAX_EFFECT} allergy_after");
    println!("Using exact PMF engine (no MC noise)");
    println!();

    // Precompute roll table: roll_table[d] = PMF of sum of N_ROLLS dice each 1..d
    // Max possible upper bound: BASE + MAX_EFFECT * max_na (generous upper bound ~150)
    let max_upper = 200;
    let roll_table: Vec<Vec<f64>> = (0..=max_upper).map(|d| dice_sum_pmf(N_ROLLS, d as u32)).collect();

    // Compute arena probs for all days in parallel (clamped by odds intervals)
    let day_probs: Vec<Vec<HashMap<String, f64>>> = historical.par_iter().map(|day_arenas| {
        day_arenas.iter().map(|arena| {
            let arena_pirates: Vec<&Pirate> = arena.pirates.iter()
                .map(|hp| game_data.pirate_by_name(&hp.name)
                    .unwrap_or_else(|| panic!("Unknown pirate: {}", hp.name)))
                .collect();
            let course_indices: Vec<usize> = arena.foods.iter()
                .filter_map(|food| course_map.get(food.as_str()).copied())
                .collect();
            let opening_odds: Vec<u32> = arena.pirates.iter().map(|hp| hp.odds).collect();
            arena_win_probs_clamped(&arena_pirates, &course_indices, &roll_table, &opening_odds)
        }).collect()
    }).collect();

    // Run strategy and analyze worst-pirate inclusion
    let mut strat = StrategyResult::new();
    let mut worst_pirate_bets = 0u32;
    let mut total_selections = 0u32; // total (bet, arena) selections
    let mut worst_rank_counts = [0u32; 4]; // rank 0=best, 3=worst

    for (i, day_arenas) in historical.iter().enumerate() {
        let bets = make_bets_current_exploit_floor(&day_probs[i], day_arenas, 0.55, 1);
        strat.add_day(&bets, day_arenas, &day_probs[i]);

        // For each arena, rank pirates by model probability
        let mut arena_rankings: Vec<Vec<(String, usize)>> = Vec::new(); // name -> rank (0=best)
        for (ai, arena) in day_arenas.iter().enumerate() {
            let mut pirate_probs: Vec<(String, f64)> = arena.pirates.iter().map(|hp| {
                let prob = day_probs[i][ai].get(&hp.name).copied().unwrap_or(0.0);
                (hp.name.clone(), prob)
            }).collect();
            pirate_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let rankings: Vec<(String, usize)> = pirate_probs.iter().enumerate()
                .map(|(rank, (name, _))| (name.clone(), rank)).collect();
            arena_rankings.push(rankings);
        }

        for bet in &bets {
            for name in &bet.pirate_names {
                for (ai, rankings) in arena_rankings.iter().enumerate() {
                    if let Some((_, rank)) = rankings.iter().find(|(n, _)| n == name) {
                        worst_rank_counts[*rank] += 1;
                        total_selections += 1;
                        if *rank == 3 {
                            worst_pirate_bets += 1;
                        }
                    }
                }
            }
        }
    }

    println!("=== STRATEGY (Model 4, odds-clamped, floor for jumped) ===\n");
    strat.print("floor for jumped, model rest");

    println!("\n=== PIRATE RANK IN SELECTED BETS ===\n");
    println!("  Total pirate selections: {}", total_selections);
    for rank in 0..4 {
        let pct = worst_rank_counts[rank] as f64 / total_selections as f64 * 100.0;
        let label = match rank { 0 => "best", 1 => "2nd", 2 => "3rd", 3 => "worst", _ => "" };
        println!("  Rank {} ({}):  {:>5} ({:.1}%)", rank + 1, label, worst_rank_counts[rank], pct);
    }
}
