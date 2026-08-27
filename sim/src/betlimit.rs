// Bet-limit / wager-size analysis.
//
// Neopets caps a single bet's winnings at 1,000,000 NP, so wagering w on a bet
// with total odds O pays min(w*O, 1e6) and the expected profit is
//
//     EV(w) = p * min(w*O, 1e6) - w
//
// which rises linearly until w*O hits the cap and falls at slope -1 after it.
// The profit-maximising wager is therefore min(L, floor(1e6/O)) for a bet limit
// L, and "wager less than the limit" is optimal exactly when O > 1e6/L.
//
// This binary sweeps L and compares two policies over the historical rounds:
//   fixed - the deployed strategy: wager the full limit on every bet, and cap
//           the payout multiplier considered at floor(1e6/L)
//   free  - pick the wager per bet: w_i = min(L, floor(1e6/O_i)), rank by the
//           resulting absolute expected profit
// Both take the top 10 bets under their own objective.

mod pirates;

use pirates::{GameData, Pirate};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// Model 4: Iterative Fav + Allergy-After (modern LL=-1.06314)
const BASE: u32 = 120;
const FAV_DIV: u32 = 16;
const N_ROLLS: u32 = 6;
const DIVISOR: u32 = 22;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 6;

const WIN_CAP: f64 = 1_000_000.0;
const MAX_BETS: usize = 10;

// Strategy parameters, matching make_bets_current_exploit_floor / generateBets.
const MIN_2S_PROB: f64 = 0.55;
const MIN_JUMP: u32 = 1;

fn course_counts(pirate: &Pirate, course_indices: &[usize]) -> (u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    for &c in course_indices {
        let is_fav = pirate.favorite_courses.contains(&c);
        let is_allergy = pirate.allergy_courses.contains(&c);
        match (is_fav, is_allergy) {
            (true, true) => na += 1,
            (true, false) => nf += 1,
            (false, true) => na += 1,
            (false, false) => {}
        }
    }
    (nf, na)
}

/// PMF of sum of `n` dice, each uniform on {1, ..., d}.
fn dice_sum_pmf(n: u32, d: u32) -> Vec<f64> {
    if d == 0 || n == 0 {
        return vec![1.0];
    }
    let max = (n * d) as usize;
    let inv_d = 1.0 / d as f64;
    let mut pmf = vec![0.0; max + 1];
    for k in 1..=(d as usize) {
        pmf[k] = inv_d;
    }
    for _ in 1..n {
        let mut new = vec![0.0; max + 1];
        let mut s = 0.0;
        for k in 0..=max {
            if k >= 1 {
                s += pmf[k - 1];
            }
            if k > d as usize {
                s -= pmf[k - d as usize - 1];
            }
            new[k] = s * inv_d;
        }
        pmf = new;
    }
    pmf
}

/// Die size for a pirate after strength and `nf` favourite-course reductions.
fn fav_adjusted_upper(pirate: &Pirate, nf: u32) -> u32 {
    let mut upper = if BASE > pirate.strength { BASE - pirate.strength } else { 1 }.max(1);
    for _ in 0..nf {
        let red = upper / FAV_DIV;
        upper = upper.saturating_sub(red).max(1);
    }
    upper
}

fn weight_offset(pirate: &Pirate) -> u32 {
    let raw = MAX_WEIGHT.saturating_sub(pirate.weight.min(MAX_WEIGHT)) / 2;
    if MAX_EFFECT > 0 { raw.min(MAX_EFFECT) } else { raw }
}

fn pirate_score_pmf(pirate: &Pirate, course_indices: &[usize], roll_table: &[Vec<f64>]) -> Vec<f64> {
    let (nf, na) = course_counts(pirate, course_indices);
    let wo = weight_offset(pirate);
    let dmg_pmf: Vec<f64> = if na > 0 && wo > 0 { dice_sum_pmf(na, wo) } else { vec![1.0] };

    let max_raw_score = (N_ROLLS as usize) * (roll_table.len() - 1);
    let mut raw_pmf = vec![0.0; max_raw_score + 1];

    for (dmg_val, &dp) in dmg_pmf.iter().enumerate() {
        if dp < 1e-15 {
            continue;
        }
        let mut upper = fav_adjusted_upper(pirate, nf);
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

    let max_q = max_raw_score / DIVISOR as usize;
    let mut qpmf = vec![0.0; max_q + 1];
    for (k, &pr) in raw_pmf.iter().enumerate() {
        if pr < 1e-15 {
            continue;
        }
        let qk = k / DIVISOR as usize;
        if qk <= max_q {
            qpmf[qk] += pr;
        }
    }
    qpmf
}

fn win_probs_from_pmfs(pmfs: [&[f64]; 4]) -> [f64; 4] {
    let max_t = pmfs.iter().map(|p| p.len()).max().unwrap_or(1);
    let surv: [Vec<f64>; 4] = std::array::from_fn(|i| {
        let mut s = vec![0.0; max_t + 1];
        let mut acc = 0.0;
        for t in (0..pmfs[i].len()).rev() {
            s[t] = acc;
            acc += pmfs[i][t];
        }
        s
    });
    let f = |i: usize, t: usize| -> f64 { if t < pmfs[i].len() { pmfs[i][t] } else { 0.0 } };
    let s = |i: usize, t: usize| -> f64 { if t < surv[i].len() { surv[i][t] } else { 0.0 } };
    let g = |i: usize, t: usize| -> f64 { if t == 0 { 1.0 } else { s(i, t - 1) } };

    let mut probs = [0.0f64; 4];
    for t in 0..max_t {
        probs[3] += f(3, t) * g(0, t) * g(1, t) * g(2, t);
        probs[2] += f(2, t) * g(0, t) * g(1, t) * s(3, t);
        probs[1] += f(1, t) * g(0, t) * s(2, t) * s(3, t);
        probs[0] += f(0, t) * s(1, t) * s(2, t) * s(3, t);
    }
    probs
}

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
            if fixed[i] {
                continue;
            }
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
            let scale = (1.0 - fixed_sum) / free_sum;
            for &i in &free_idx {
                p[i] *= scale;
            }
        }
        if !changed {
            break;
        }
    }
    p
}

// --- Historical data ---

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
    #[allow(dead_code)]
    winner: String,
    #[serde(default)]
    legacy: bool,
}

fn eff_odds(p: &HistPirate) -> u32 {
    p.current_odds.unwrap_or(p.odds)
}

/// One candidate bet: win probability and the true (uncapped) total odds.
#[derive(Clone, Copy)]
struct Cand {
    p: f64,
    odds: u32,
    legs: u8,
}

/// Enumerate every bet the deployed anchored strategy would consider.
/// `use_floor` mirrors make_bets_current_exploit_floor: jumped pirates get the
/// opening-odds probability floor 1/(opening+1) instead of the model estimate.
fn candidates(arenas: &[HistArena], probs: &[[f64; 4]], use_floor: bool) -> Vec<Cand> {
    let n = arenas.len();
    let mut is_jump = vec![[false; 4]; n];
    let mut is_anchor = vec![[false; 4]; n];
    for (ai, arena) in arenas.iter().enumerate() {
        for (pi, pirate) in arena.pirates.iter().enumerate() {
            let jump = eff_odds(pirate) >= pirate.odds + MIN_JUMP;
            is_jump[ai][pi] = jump;
            is_anchor[ai][pi] =
                (pirate.odds == 2 && probs[ai][pi] >= MIN_2S_PROB) || jump;
        }
    }

    let mut out = Vec::with_capacity(3125);
    // Base-5 digits: 0 = arena not used, 1..=4 = pirate index + 1.
    let total = 5usize.pow(n as u32);
    for code in 1..total {
        let mut c = code;
        let mut p = 1.0f64;
        let mut odds: u32 = 1;
        let mut legs = 0u8;
        let mut has_anchor = false;
        for ai in 0..n {
            let d = c % 5;
            c /= 5;
            if d == 0 {
                continue;
            }
            let pi = d - 1;
            let pirate = &arenas[ai].pirates[pi];
            let leg_p = if use_floor && is_jump[ai][pi] {
                1.0 / (pirate.odds as f64 + 1.0)
            } else {
                probs[ai][pi]
            };
            p *= leg_p;
            odds = odds.saturating_mul(eff_odds(pirate));
            legs += 1;
            if is_anchor[ai][pi] {
                has_anchor = true;
            }
        }
        if legs == 0 || !has_anchor {
            continue;
        }
        out.push(Cand { p, odds, legs });
    }
    out
}

/// Top-`MAX_BETS` selection by a key, returning the chosen indices.
fn top_k<F: Fn(&Cand) -> f64>(cands: &[Cand], key: F) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = cands
        .iter()
        .enumerate()
        .map(|(i, c)| (key(c), i))
        .filter(|(v, _)| *v > 0.0)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(MAX_BETS);
    scored.into_iter().map(|(_, i)| i).collect()
}

/// Best wager for a bet at limit L: the payout cap makes anything above
/// floor(1e6/O) pure loss.
fn best_wager(odds: u32, limit: f64) -> f64 {
    (WIN_CAP / odds as f64).floor().min(limit).max(0.0)
}

fn ev_at(c: &Cand, w: f64) -> f64 {
    c.p * (w * c.odds as f64).min(WIN_CAP) - w
}

#[derive(Default, Clone)]
struct Agg {
    days: u32,
    free_ev: f64,
    free_wagered: f64,
    free_bets: u32,
    free_sublimit_bets: u32,
    free_sublimit_days: u32,
    free_wager_frac_sum: f64,
    free_odds_sum: f64,
    fixed_ev: f64,
    fixed_wagered: f64,
    fixed_bets: u32,
    fixed_wasted_bets: u32, // full-limit bets whose payout is clipped by the cap
    fixed_waste: f64,       // NP of wager that buys no extra payout
    top_odds_sum: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mc_mode = args.iter().any(|a| a == "--mc");
    let use_model_probs = args.iter().any(|a| a == "--model-probs");

    let game_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let game_data = GameData::load(&game_json);
    let course_map = game_data.course_name_to_index();

    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let all_days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse historical_matches.json");
    let historical: Vec<Vec<HistArena>> = all_days
        .into_iter()
        .filter(|day| day.first().map_or(false, |a| !a.legacy))
        .collect();

    let max_upper = 200;
    let roll_table: Vec<Vec<f64>> =
        (0..=max_upper).map(|d| dice_sum_pmf(N_ROLLS, d as u32)).collect();

    // Clamped model probabilities per day/arena/pirate.
    let day_probs: Vec<Vec<[f64; 4]>> = historical
        .par_iter()
        .map(|day| {
            day.iter()
                .map(|arena| {
                    let ps: Vec<&Pirate> = arena
                        .pirates
                        .iter()
                        .map(|hp| game_data.pirate_by_name(&hp.name).expect("unknown pirate"))
                        .collect();
                    let ci: Vec<usize> = arena
                        .foods
                        .iter()
                        .filter_map(|f| course_map.get(f.as_str()).copied())
                        .collect();
                    let pmfs: Vec<Vec<f64>> =
                        ps.iter().map(|p| pirate_score_pmf(p, &ci, &roll_table)).collect();
                    let raw = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]]);
                    let intervals: [(f64, f64); 4] =
                        std::array::from_fn(|i| odds_prob_bounds(arena.pirates[i].odds));
                    clamp_and_redistribute(&raw, &intervals)
                })
                .collect()
        })
        .collect();

    println!("Modern data: {} days", historical.len());
    println!(
        "Leg probabilities: {}",
        if use_model_probs { "model for all legs" } else { "opening-odds floor for jumped legs, model otherwise" }
    );
    println!();

    if mc_mode {
        run_sanity_checks(&historical, &day_probs, &game_data, &course_map, &roll_table, !use_model_probs);
        return;
    }

    if args.iter().any(|a| a == "--detail") {
        let ls: Vec<f64> = args
            .iter()
            .skip_while(|a| *a != "--detail")
            .skip(1)
            .filter_map(|a| a.parse::<f64>().ok())
            .collect();
        let ls = if ls.is_empty() { vec![16_792.0] } else { ls };
        run_detail(&historical, &day_probs, !use_model_probs, &ls);
        return;
    }

    let all_cands: Vec<Vec<Cand>> = historical
        .par_iter()
        .zip(day_probs.par_iter())
        .map(|(day, probs)| candidates(day, probs, !use_model_probs))
        .collect();

    // Bet limit grid: geometric, plus the values that matter today.
    let mut limits: Vec<f64> = Vec::new();
    let mut l = 20.0f64;
    while l <= 500_000.0 {
        limits.push(l.round());
        l *= 1.05;
    }
    for extra in [50.0, 100.0, 500.0, 1000.0, 5000.0, 10_000.0, 16_570.0, 16_792.0, 500_000.0] {
        limits.push(extra);
    }
    limits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    limits.dedup();

    let results: Vec<(f64, Agg)> = limits
        .par_iter()
        .map(|&limit| {
            let mut agg = Agg::default();
            let cap_ratio = (WIN_CAP / limit).floor().max(1.0);
            for cands in all_cands.iter() {
                if cands.is_empty() {
                    continue;
                }
                agg.days += 1;

                // free: optimal wager per bet, ranked by absolute expected profit
                let free_idx = top_k(cands, |c| ev_at(c, best_wager(c.odds, limit)));
                let mut day_sublimit = 0u32;
                for (rank, &i) in free_idx.iter().enumerate() {
                    let c = cands[i];
                    let w = best_wager(c.odds, limit);
                    agg.free_ev += ev_at(&c, w);
                    agg.free_wagered += w;
                    agg.free_bets += 1;
                    agg.free_wager_frac_sum += w / limit;
                    agg.free_odds_sum += c.odds as f64;
                    if w < limit {
                        agg.free_sublimit_bets += 1;
                        day_sublimit += 1;
                    }
                    if rank == 0 {
                        agg.top_odds_sum += c.odds as f64;
                    }
                }
                if day_sublimit > 0 {
                    agg.free_sublimit_days += 1;
                }

                // fixed: the deployed policy - full limit on every bet, payout
                // multiplier capped at floor(1e6 / limit) when ranking
                let fixed_idx = top_k(cands, |c| {
                    c.p * (c.odds as f64).min(cap_ratio) - 1.0
                });
                for &i in fixed_idx.iter() {
                    let c = cands[i];
                    agg.fixed_ev += ev_at(&c, limit);
                    agg.fixed_wagered += limit;
                    agg.fixed_bets += 1;
                    let needed = best_wager(c.odds, limit);
                    if needed < limit {
                        agg.fixed_wasted_bets += 1;
                        agg.fixed_waste += limit - needed;
                    }
                }
            }
            (limit, agg)
        })
        .collect();

    println!("{:>9} {:>7} {:>12} {:>12} {:>7} {:>9} {:>9} {:>8} {:>10}",
        "limit", "cap_mul", "EV_free/day", "EV_fix/day", "gain%", "sub<L/day", "days_sub%", "avg w/L", "top_odds");
    for (limit, a) in &results {
        if a.days == 0 { continue; }
        let d = a.days as f64;
        let cap_ratio = (WIN_CAP / limit).floor().max(1.0);
        let gain = if a.fixed_ev > 0.0 { (a.free_ev / a.fixed_ev - 1.0) * 100.0 } else { f64::NAN };
        println!("{:>9.0} {:>7.0} {:>12.0} {:>12.0} {:>7.2} {:>9.2} {:>9.1} {:>8.3} {:>10.0}",
            limit,
            cap_ratio,
            a.free_ev / d,
            a.fixed_ev / d,
            gain,
            a.free_sublimit_bets as f64 / d,
            a.free_sublimit_days as f64 / d * 100.0,
            a.free_wager_frac_sum / a.free_bets as f64,
            a.top_odds_sum / d,
        );
    }

    // First limit at which sub-limit wagering appears / matters.
    println!();
    for &(thresh, label) in &[(0.0, "any sub-limit bet is optimal"),
                              (0.5, "gain from sizing >= 0.5%"),
                              (1.0, "gain from sizing >= 1%"),
                              (5.0, "gain from sizing >= 5%"),
                              (25.0, "gain from sizing >= 25%")] {
        let hit = results.iter().find(|(_, a)| {
            let g = if a.fixed_ev > 0.0 { (a.free_ev / a.fixed_ev - 1.0) * 100.0 } else { 0.0 };
            if thresh == 0.0 { a.free_sublimit_bets > 0 } else { g >= thresh }
        });
        match hit {
            Some((limit, _)) => println!("  first limit where {:<32}: {:>9.0} NP", label, limit),
            None => println!("  first limit where {:<32}: never within grid", label),
        }
    }
}

// ==================== sanity checks ====================

// splitmix64: the low bits of a hand-rolled xorshift are far too structured for
// `% sides` dice, and that shows up as a huge PMF-vs-MC gap.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn roll(state: &mut u64, sides: u32) -> u32 {
    // Rejection sampling so `% sides` introduces no modulo bias.
    let s = sides as u64;
    let limit = u64::MAX - (u64::MAX % s);
    loop {
        let r = splitmix(state);
        if r < limit {
            return (r % s) as u32 + 1;
        }
    }
}

/// Monte-carlo the model's dice process directly, as an independent check on
/// the PMF convolution used everywhere else.
fn mc_arena_probs(
    pirates: &[&Pirate],
    course_indices: &[usize],
    trials: u32,
    seed: u64,
) -> [f64; 4] {
    let mut state = seed | 1;
    let mut wins = [0u32; 4];
    let params: Vec<(u32, u32, u32)> = pirates
        .iter()
        .map(|p| {
            let (nf, na) = course_counts(p, course_indices);
            (fav_adjusted_upper(p, nf), na, weight_offset(p))
        })
        .collect();
    for _ in 0..trials {
        // Lower score wins: strength and favourite courses shrink the die, and
        // win_probs_from_pmfs gives pirate i the win when every other pirate
        // scores at or above it.
        let mut best = i64::MAX;
        let mut winner = 0usize;
        for (i, &(base_upper, na, wo)) in params.iter().enumerate() {
            let mut upper = base_upper;
            if wo > 0 {
                for _ in 0..na {
                    upper += roll(&mut state, wo);
                }
            }
            let mut score = 0u32;
            for _ in 0..N_ROLLS {
                score += roll(&mut state, upper);
            }
            let q = (score / DIVISOR) as i64;
            // later position wins ties
            if q <= best {
                best = q;
                winner = i;
            }
        }
        wins[winner] += 1;
    }
    std::array::from_fn(|i| wins[i] as f64 / trials as f64)
}

fn run_sanity_checks(
    historical: &[Vec<HistArena>],
    day_probs: &[Vec<[f64; 4]>],
    game_data: &GameData,
    course_map: &HashMap<&str, usize>,
    roll_table: &[Vec<f64>],
    use_floor: bool,
) {
    // 1. PMF win probabilities vs. a direct simulation of the dice process.
    println!("=== CHECK 1: PMF arena probabilities vs monte carlo ===");
    let trials = 2_000_000u32;
    let mut worst_z: f64 = 0.0;
    for (di, day) in historical.iter().enumerate().take(6) {
        for (ai, arena) in day.iter().enumerate() {
            let ps: Vec<&Pirate> = arena
                .pirates
                .iter()
                .map(|hp| game_data.pirate_by_name(&hp.name).unwrap())
                .collect();
            let ci: Vec<usize> = arena
                .foods
                .iter()
                .filter_map(|f| course_map.get(f.as_str()).copied())
                .collect();
            let pmfs: Vec<Vec<f64>> =
                ps.iter().map(|p| pirate_score_pmf(p, &ci, roll_table)).collect();
            let exact = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]]);
            let mc = mc_arena_probs(&ps, &ci, trials, 0x9E3779B97F4A7C15u64 ^ ((di * 5 + ai) as u64));
            for i in 0..4 {
                let se = (exact[i] * (1.0 - exact[i]) / trials as f64).sqrt().max(1e-12);
                let z = (mc[i] - exact[i]) / se;
                if z.abs() > worst_z.abs() {
                    worst_z = z;
                }
            }
            if di == 0 {
                println!(
                    "  day0 arena{}: exact [{:.4} {:.4} {:.4} {:.4}]  mc [{:.4} {:.4} {:.4} {:.4}]",
                    ai, exact[0], exact[1], exact[2], exact[3], mc[0], mc[1], mc[2], mc[3]
                );
            }
        }
    }
    println!("  worst z over 30 arenas x 4 pirates at {} trials: {:.2}", trials, worst_z);

    // 2. Analytic portfolio EV vs. simulated play of the same portfolio.
    println!();
    println!("=== CHECK 2: analytic portfolio EV vs simulated play ===");
    for &limit in &[1_000.0f64, 16_570.0, 100_000.0] {
        let mut analytic = 0.0f64;
        let mut simulated = 0.0f64;
        let sims = 20_000u32;
        let mut state = 0xDEADBEEFCAFEu64 ^ (limit as u64);
        let days = historical.len().min(200);
        for di in 0..days {
            let day = &historical[di];
            let probs = &day_probs[di];
            let cands = candidates(day, probs, use_floor);
            if cands.is_empty() {
                continue;
            }
            // Re-derive the chosen bets with their leg identities so a simulated
            // day can be scored: enumerate the same base-5 codes.
            let idx = top_k(&cands, |c| ev_at(c, best_wager(c.odds, limit)));
            let codes = candidate_codes(day, probs, use_floor);
            let chosen: Vec<(usize, f64)> = idx
                .iter()
                .map(|&i| (codes[i], best_wager(cands[i].odds, limit)))
                .collect();
            for &i in &idx {
                analytic += ev_at(&cands[i], best_wager(cands[i].odds, limit));
            }
            // Simulate winners from the (clamped) model probabilities.
            let mut day_profit = 0.0f64;
            for _ in 0..sims {
                let mut winners = [0usize; 5];
                for (ai, p) in probs.iter().enumerate() {
                    let u = (splitmix(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                    let mut acc = 0.0;
                    let mut w = 3usize;
                    for k in 0..4 {
                        acc += p[k];
                        if u < acc {
                            w = k;
                            break;
                        }
                    }
                    winners[ai] = w;
                }
                for &(code, wager) in &chosen {
                    let mut c = code;
                    let mut hit = true;
                    let mut odds = 1u32;
                    for ai in 0..day.len() {
                        let d = c % 5;
                        c /= 5;
                        if d == 0 {
                            continue;
                        }
                        if winners[ai] != d - 1 {
                            hit = false;
                        }
                        odds = odds.saturating_mul(eff_odds(&day[ai].pirates[d - 1]));
                    }
                    day_profit += if hit { (wager * odds as f64).min(WIN_CAP) - wager } else { -wager };
                }
            }
            simulated += day_profit / sims as f64;
        }
        println!(
            "  limit {:>7.0}: analytic EV {:>14.1}   simulated {:>14.1}   rel diff {:+.4}%",
            limit,
            analytic,
            simulated,
            (simulated / analytic - 1.0) * 100.0
        );
    }
    println!();
    println!("  (check 2 uses model probabilities for the draw, so it validates the");
    println!("   min(w*O, 1e6) payout accounting and the top-10 selection, not the model)");
}

/// The base-5 codes for candidates, in the same order `candidates` emits them.
fn candidate_codes(arenas: &[HistArena], probs: &[[f64; 4]], _use_floor: bool) -> Vec<usize> {
    let n = arenas.len();
    let mut is_anchor = vec![[false; 4]; n];
    for (ai, arena) in arenas.iter().enumerate() {
        for (pi, pirate) in arena.pirates.iter().enumerate() {
            let jump = eff_odds(pirate) >= pirate.odds + MIN_JUMP;
            is_anchor[ai][pi] = (pirate.odds == 2 && probs[ai][pi] >= MIN_2S_PROB) || jump;
        }
    }
    let mut out = Vec::new();
    let total = 5usize.pow(n as u32);
    for code in 1..total {
        let mut c = code;
        let mut legs = 0;
        let mut has_anchor = false;
        for ai in 0..n {
            let d = c % 5;
            c /= 5;
            if d == 0 {
                continue;
            }
            legs += 1;
            if is_anchor[ai][d - 1] {
                has_anchor = true;
            }
        }
        if legs == 0 || !has_anchor {
            continue;
        }
        out.push(code);
    }
    out
}

// ==================== detail + bankroll modes ====================

/// The 1024-outcome profit distribution of a portfolio on one day.
/// Outcome probabilities use the model estimates (not the conservative floor
/// the selector uses), so this is the honest distribution of the day's result.
fn profit_distribution(
    arenas: &[HistArena],
    probs: &[[f64; 4]],
    chosen: &[(usize, f64)], // (base-5 code, wager)
) -> Vec<(f64, f64)> {
    let n = arenas.len();
    let mut out = Vec::with_capacity(4usize.pow(n as u32));
    for o in 0..4usize.pow(n as u32) {
        let mut winners = [0usize; 5];
        let mut q = o;
        let mut prob = 1.0f64;
        for ai in 0..n {
            let w = q % 4;
            q /= 4;
            winners[ai] = w;
            prob *= probs[ai][w];
        }
        if prob <= 0.0 {
            continue;
        }
        let mut profit = 0.0f64;
        for &(code, wager) in chosen {
            let mut c = code;
            let mut hit = true;
            let mut odds = 1u32;
            for ai in 0..n {
                let d = c % 5;
                c /= 5;
                if d == 0 {
                    continue;
                }
                if winners[ai] != d - 1 {
                    hit = false;
                }
                odds = odds.saturating_mul(eff_odds(&arenas[ai].pirates[d - 1]));
            }
            profit += if hit { (wager * odds as f64).min(WIN_CAP) - wager } else { 0.0 } - wager;
        }
        out.push((prob, profit));
    }
    out
}

/// Smallest bankroll at which wagering the full EV-optimal size is also
/// log-optimal: the s=1 first-order condition E[X / (B + X)] >= 0.
fn kelly_bankroll(dist: &[(f64, f64)], total_wagered: f64) -> f64 {
    let deriv = |b: f64| -> f64 { dist.iter().map(|&(p, x)| p * x / (b + x)).sum::<f64>() };
    let mut lo = total_wagered * 1.000_001; // log is undefined at or below a total loss
    if deriv(lo) >= 0.0 {
        return lo;
    }
    // E[X/(B+X)] -> E[X]/B as B grows, so a portfolio with non-positive model EV
    // has no bankroll large enough: full-size wagering is never log-optimal.
    let mean: f64 = dist.iter().map(|&(p, x)| p * x).sum();
    if mean <= 0.0 {
        return f64::INFINITY;
    }
    let mut hi = lo.max(1.0);
    for _ in 0..400 {
        hi *= 2.0;
        if deriv(hi) >= 0.0 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if deriv(mid) >= 0.0 { hi = mid } else { lo = mid }
    }
    hi
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

fn run_detail(
    historical: &[Vec<HistArena>],
    day_probs: &[Vec<[f64; 4]>],
    use_floor: bool,
    limits: &[f64],
) {
    for &limit in limits {
        let per_day: Vec<(usize, usize, f64, f64, f64, f64, f64, f64)> = historical
            .par_iter()
            .zip(day_probs.par_iter())
            .filter_map(|(day, probs)| {
                let cands = candidates(day, probs, use_floor);
                if cands.is_empty() {
                    return None;
                }
                let codes = candidate_codes(day, probs, use_floor);
                let idx = top_k(&cands, |c| ev_at(c, best_wager(c.odds, limit)));
                if idx.is_empty() {
                    return None;
                }
                let chosen: Vec<(usize, f64)> = idx
                    .iter()
                    .map(|&i| (codes[i], best_wager(cands[i].odds, limit)))
                    .collect();
                let n_sub = chosen.iter().filter(|&&(_, w)| w < limit).count();
                let wagered: f64 = chosen.iter().map(|&(_, w)| w).sum();
                let min_w = chosen.iter().map(|&(_, w)| w).fold(f64::INFINITY, f64::min);
                let ev: f64 = idx.iter().map(|&i| ev_at(&cands[i], best_wager(cands[i].odds, limit))).sum();
                let max_odds = idx.iter().map(|&i| cands[i].odds).max().unwrap() as f64;
                let dist = profit_distribution(day, probs, &chosen);
                let bust: f64 = dist.iter().filter(|&&(_, x)| x < 0.0).map(|&(p, _)| p).sum();
                let bankroll = kelly_bankroll(&dist, wagered);
                Some((chosen.len(), n_sub, wagered, min_w, ev, max_odds, bust, bankroll))
            })
            .collect();

        let d = per_day.len() as f64;
        let bets: f64 = per_day.iter().map(|x| x.0 as f64).sum();
        let subs: f64 = per_day.iter().map(|x| x.1 as f64).sum();
        let wagered: f64 = per_day.iter().map(|x| x.2).sum();
        let ev: f64 = per_day.iter().map(|x| x.4).sum();
        let mut banks: Vec<f64> = per_day.iter().map(|x| x.7).collect();
        banks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_inf = banks.iter().filter(|b| !b.is_finite()).count();
        let mut minws: Vec<f64> = per_day.iter().map(|x| x.3).collect();
        minws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bust: f64 = per_day.iter().map(|x| x.6).sum::<f64>() / d;

        println!("--- bet limit {:.0} NP (payout cap binds above odds {:.0}) ---",
            limit, (WIN_CAP / limit).floor());
        println!("  bets/day {:.2} | wagered/day {:.0} | EV/day {:.0} ({:+.1}% ROI) | P(day loses money) {:.1}%",
            bets / d, wagered / d, ev / d, ev / wagered * 100.0, bust * 100.0);
        println!("  bets wagered below the limit: {:.2}/day ({:.1}% of bets); wager as % of limit: {:.1}%",
            subs / d, subs / bets * 100.0, wagered / (bets * limit) * 100.0);
        println!("  smallest wager in the day's slate: median {:.0} NP ({:.1}% of limit), 10th pct {:.0} NP",
            percentile(&minws, 0.5), percentile(&minws, 0.5) / limit * 100.0, percentile(&minws, 0.10));
        let fmt = |v: f64| -> String {
            if v.is_finite() { format!("{:.0} NP ({:.1}x daily wager)", v, v / (wagered / d)) }
            else { "no bankroll suffices".to_string() }
        };
        println!("  bankroll for full-size wagering to also be log-optimal: median {}, 75th pct {}, 90th pct {}",
            fmt(percentile(&banks, 0.5)), fmt(percentile(&banks, 0.75)), fmt(percentile(&banks, 0.9)));
        println!("  days whose slate has negative EV under the model (floor-selected, so never log-optimal at any size): {:.1}%",
            n_inf as f64 / d * 100.0);
        println!();
    }
}
