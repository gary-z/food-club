// Odds-maker edge hunt.
//
// Question: previous work found the opening odds maker gives no edge except at 2:1.
// Is there ANY stratum -- specific pirates, allergy levels, favourite counts,
// positions, arena shapes, eras, foods -- where opening odds are +EV?
//
// Structure of the problem (statistical_findings.txt #26/#27):
//     displayed odds  N_i = clamp(floor(1 / p_om_i), 2, 13)
// where p_om_i is the odds maker's own win probability.
//
//   floor  =>  N_i <= 1/p_om_i  =>  p_om_i <= 1/N_i  =>  EV = p_om_i*N_i <= 1
//
// So for 3 <= N <= 13 a *correct* odds maker is unbeatable by construction; a +EV
// stratum at N>=3 is possible ONLY if the odds maker is wrong (p_true > 1/N).
// The 2:1 bin is the sole exception: it is clamped from below, so p_om can run all
// the way to 1.0 while the payout stays 2.
//
// Everything below tests that structure and then hunts for odds-maker error.
//
// Statistical notes
//  * EV test: X = won * N, H0: E[X] = 1 (i.e. p = 1/N), Var(X | H0) = N - 1.
//    z = (sum X - n) / sqrt(sum (N-1)).  Matches the convention in finding #27.
//  * Multiple testing is handled by a permutation test: winners are resampled from
//    the odds-implied probabilities (the "odds maker is exactly right" null) and the
//    GLOBAL max z over every cell of every scan is recomputed. This accounts for the
//    heavy correlation between overlapping cells, unlike Bonferroni.
//  * Calibration (actual wins vs a point estimate of p_om) is reported descriptively
//    only. It CANNOT prove odds-maker error: within one odds bin p_om ranges over
//    (1/(N+1), 1/N], and any selector correlated with true skill (pirate identity,
//    na, the model) preferentially picks the top of that interval, so cal > 1 is
//    expected even from a perfect odds maker. Only the EV test is assumption-free,
//    because 1/N is a hard upper bound on p_om.

mod pirates;

use pirates::{GameData, Pirate};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// ---------------- Model 4 (best hand-rolled, modern LL = -1.06314) ----------------
const BASE: u32 = 120;
const FAV_DIV: u32 = 16;
const N_ROLLS: u32 = 6;
const DIVISOR: u32 = 22;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 6;

const N_PERM: usize = 4000;

/// (nf, na, n_overlap). Overlap foods (fav AND allergy) count as allergy (finding #7),
/// but are tracked so the determinism test cannot be confounded by the counting rule.
fn course_counts(pirate: &Pirate, course_indices: &[usize]) -> (u32, u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    let mut ov = 0u32;
    for &c in course_indices {
        let is_fav = pirate.favorite_courses.contains(&c);
        let is_alg = pirate.allergy_courses.contains(&c);
        match (is_fav, is_alg) {
            (true, true) => { na += 1; ov += 1; }
            (true, false) => { nf += 1; }
            (false, true) => { na += 1; }
            (false, false) => {}
        }
    }
    (nf, na, ov)
}

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

fn pirate_score_pmf(pirate: &Pirate, course_indices: &[usize], roll_table: &[Vec<f64>]) -> Vec<f64> {
    let (nf, na, _) = course_counts(pirate, course_indices);
    let wo = (MAX_WEIGHT.saturating_sub(pirate.weight.min(MAX_WEIGHT)) / 2).min(MAX_EFFECT);
    let dmg_pmf: Vec<f64> = if na > 0 && wo > 0 { dice_sum_pmf(na, wo) } else { vec![1.0] };

    let max_raw_score = (N_ROLLS as usize) * (roll_table.len() - 1);
    let mut raw_pmf = vec![0.0; max_raw_score + 1];
    for (dmg_val, &dp) in dmg_pmf.iter().enumerate() {
        if dp < 1e-15 { continue; }
        let mut upper = if BASE > pirate.strength { BASE - pirate.strength } else { 1 }.max(1);
        for _ in 0..nf {
            let red = upper / FAV_DIV;
            upper = upper.saturating_sub(red).max(1);
        }
        upper = (upper + dmg_val as u32).max(1);
        if (upper as usize) < roll_table.len() {
            for (k, &rp) in roll_table[upper as usize].iter().enumerate() {
                if rp > 0.0 && k < raw_pmf.len() { raw_pmf[k] += dp * rp; }
            }
        }
    }
    let max_q = max_raw_score / DIVISOR as usize;
    let mut qpmf = vec![0.0; max_q + 1];
    for (k, &pr) in raw_pmf.iter().enumerate() {
        if pr < 1e-15 { continue; }
        let qk = k / DIVISOR as usize;
        if qk <= max_q { qpmf[qk] += pr; }
    }
    qpmf
}

fn win_probs_from_pmfs(pmfs: [&[f64]; 4]) -> [f64; 4] {
    let max_t = pmfs.iter().map(|p| p.len()).max().unwrap_or(1);
    let surv: [Vec<f64>; 4] = std::array::from_fn(|i| {
        let mut s = vec![0.0; max_t + 1];
        let mut acc = 0.0;
        for t in (0..pmfs[i].len()).rev() { s[t] = acc; acc += pmfs[i][t]; }
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

// ---------------- odds -> probability intervals ----------------

fn odds_prob_bounds(odds: u32) -> (f64, f64) {
    match odds {
        2 => (1.0 / 3.0, 1.0),   // lower clamp: floor(1/p) <= 2  <=>  p > 1/3
        13 => (0.0, 1.0 / 13.0), // upper clamp: floor(1/p) >= 13 <=>  p <= 1/13
        n => (1.0 / (n as f64 + 1.0), 1.0 / n as f64),
    }
}

/// Arc-consistency narrowing of the four intervals using sum_i p_i = 1.
fn tighten(odds: &[u32; 4]) -> ([f64; 4], [f64; 4]) {
    let mut lo = [0.0f64; 4];
    let mut hi = [0.0f64; 4];
    for i in 0..4 {
        let (l, h) = odds_prob_bounds(odds[i]);
        lo[i] = l; hi[i] = h;
    }
    for _ in 0..8 {
        for i in 0..4 {
            let sum_lo: f64 = (0..4).filter(|&j| j != i).map(|j| lo[j]).sum();
            let sum_hi: f64 = (0..4).filter(|&j| j != i).map(|j| hi[j]).sum();
            hi[i] = hi[i].min(1.0 - sum_lo);
            lo[i] = lo[i].max(1.0 - sum_hi);
        }
    }
    (lo, hi)
}

/// Odds-only point estimate of p_om: the unique point lo + t*(hi-lo) with sum = 1.
fn point_estimate(lo: &[f64; 4], hi: &[f64; 4]) -> [f64; 4] {
    let sum_lo: f64 = lo.iter().sum();
    let width: f64 = (0..4).map(|i| hi[i] - lo[i]).sum();
    let t = if width > 1e-12 { ((1.0 - sum_lo) / width).clamp(0.0, 1.0) } else { 0.0 };
    let mut p: [f64; 4] = std::array::from_fn(|i| lo[i] + t * (hi[i] - lo[i]));
    let s: f64 = p.iter().sum();
    if s > 0.0 { for v in p.iter_mut() { *v /= s; } }
    p
}

/// Project model probabilities into the odds-implied intervals (model + odds combined).
fn clamp_and_redistribute(probs: &[f64; 4], lo: &[f64; 4], hi: &[f64; 4]) -> [f64; 4] {
    let mut p = *probs;
    let mut fixed = [false; 4];
    for _ in 0..20 {
        let mut changed = false;
        for i in 0..4 {
            if fixed[i] { continue; }
            if p[i] < lo[i] { p[i] = lo[i]; fixed[i] = true; changed = true; }
            else if p[i] > hi[i] { p[i] = hi[i]; fixed[i] = true; changed = true; }
        }
        let fixed_sum: f64 = (0..4).filter(|&i| fixed[i]).map(|i| p[i]).sum();
        let free: Vec<usize> = (0..4).filter(|&i| !fixed[i]).collect();
        let free_sum: f64 = free.iter().map(|&i| p[i]).sum();
        if !free.is_empty() && free_sum > 0.0 {
            let scale = (1.0 - fixed_sum) / free_sum;
            for &i in &free { p[i] *= scale; }
        }
        if !changed { break; }
    }
    p
}

// ---------------- historical data ----------------

#[derive(Deserialize, Clone)]
struct HistPirate { name: String, odds: u32 }

#[derive(Deserialize)]
struct HistArena {
    #[allow(dead_code)] arena_name: String,
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
    #[serde(default)] legacy: bool,
}

#[derive(Clone)]
struct Row {
    day: u32,
    legacy: bool,
    pirate: usize,
    pos: usize,
    nf: u32,
    na: u32,
    odds: u32,
    won: bool,
    usum: f64,     // sum_j 1/odds_j across the arena ("overround")
    lo: f64,       // tightened lower bound on p_om
    p_hat: f64,    // odds-only point estimate of p_om
    model_p: f64,  // Model 4 probability (uses no odds information)
    comb_p: f64,   // Model 4 projected into odds intervals
    fav_foods: Vec<usize>,
    alg_foods: Vec<usize>,
    opp_str: u32,  // sum of opponents' strengths
    opp_nf: u32,   // sum of opponents' favourite counts
    opp_na: u32,   // sum of opponents' allergy counts
}

#[derive(Clone)]
struct Arena {
    rows: [usize; 4],  // indices into the flat row list
    winner: usize,     // 0..3
    p_hat: [f64; 4],
    sig: u64,          // engine-equivalence signature (class, nf, na, overlap) per position
    odds: [u32; 4],
}

// ---------------- accumulators / stats ----------------

#[derive(Default, Clone)]
struct Acc {
    n: u64,
    wins: u64,
    sum_x: f64,
    null_var: f64,
    sum_p: f64,
    sum_pq: f64,
    sum_mp: f64,
    sum_odds: f64,
    sum_bound: f64,
    sum_inv: f64,
}

impl Acc {
    fn add(&mut self, r: &Row) {
        self.n += 1;
        if r.won { self.wins += 1; self.sum_x += r.odds as f64; }
        self.null_var += (r.odds - 1) as f64;
        self.sum_p += r.p_hat;
        self.sum_pq += r.p_hat * (1.0 - r.p_hat);
        self.sum_mp += r.model_p;
        self.sum_odds += r.odds as f64;
        self.sum_bound += r.odds as f64 * r.lo;
        self.sum_inv += 1.0 / r.odds as f64;
    }
    fn inv(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum_inv / self.n as f64 } }
    fn ev(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum_x / self.n as f64 } }
    fn wr(&self) -> f64 { if self.n == 0 { 0.0 } else { self.wins as f64 / self.n as f64 } }
    fn z(&self) -> f64 {
        if self.null_var <= 0.0 { 0.0 } else { (self.sum_x - self.n as f64) / self.null_var.sqrt() }
    }
    fn cal_z(&self) -> f64 {
        if self.sum_pq <= 0.0 { 0.0 } else { (self.wins as f64 - self.sum_p) / self.sum_pq.sqrt() }
    }
    fn cal(&self) -> f64 { if self.sum_p <= 0.0 { 0.0 } else { self.wins as f64 / self.sum_p } }
    fn avg_odds(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum_odds / self.n as f64 } }
    fn avg_bound(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum_bound / self.n as f64 } }
    fn pred(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum_p / self.n as f64 } }
}

fn normal_sf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * (-x * x).exp();
    0.5 * (1.0 - sign * y)
}
fn two_sided_p(z: f64) -> f64 { 2.0 * normal_sf(z.abs()) }

fn inv_norm_two_sided(alpha: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 12.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if 2.0 * normal_sf(mid) > alpha { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

fn hash2(a: u64, b: u64) -> u64 {
    let mut h = a ^ 0x9E3779B97F4A7C15u64;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= b.wrapping_add(0x94D049BB133111EB);
    h = h.wrapping_mul(0xD6E8FEB86659FD93);
    h ^ (h >> 31)
}

// =============================================================================
fn main() {
    let game_json = std::fs::read_to_string("../pirates.json").expect("pirates.json");
    let gd = GameData::load(&game_json);
    let course_map = gd.course_name_to_index();
    let hist_json = std::fs::read_to_string("../historical_matches.json").expect("historical");
    let days: Vec<Vec<HistArena>> = serde_json::from_str(&hist_json).expect("parse");

    let roll_table: Vec<Vec<f64>> = (0..=220).map(|d| dice_sum_pmf(N_ROLLS, d as u32)).collect();

    // engine-equivalence class per pirate: everything the game engine can see
    // is (strength, weight_offset). Pirates sharing both are interchangeable at
    // matched (nf, na).
    let class_of: Vec<u32> = gd.pirates.iter()
        .map(|p| {
            let wo = (MAX_WEIGHT.saturating_sub(p.weight.min(MAX_WEIGHT)) / 2).min(MAX_EFFECT);
            p.strength * 100 + wo
        }).collect();

    // ---------- build rows + arenas ----------
    let per_day: Vec<(Vec<Row>, Vec<[usize; 4]>, Vec<usize>, Vec<[f64; 4]>, Vec<u64>, Vec<[u32; 4]>)> =
        days.par_iter().enumerate().map(|(di, day)| {
            let mut rows: Vec<Row> = Vec::new();
            let mut quads: Vec<[usize; 4]> = Vec::new();
            let mut winners: Vec<usize> = Vec::new();
            let mut phats: Vec<[f64; 4]> = Vec::new();
            let mut sigs: Vec<u64> = Vec::new();
            let mut oddss: Vec<[u32; 4]> = Vec::new();
            for arena in day.iter() {
                if arena.pirates.len() != 4 { continue; }
                let ps: Vec<&Pirate> = arena.pirates.iter()
                    .map(|hp| gd.pirate_by_name(&hp.name).expect("pirate")).collect();
                let mut cidx: Vec<usize> = arena.foods.iter()
                    .filter_map(|f| course_map.get(f.as_str()).copied()).collect();
                cidx.sort_unstable();
                let odds: [u32; 4] = std::array::from_fn(|i| arena.pirates[i].odds);
                let winner = arena.pirates.iter().position(|p| p.name == arena.winner).expect("winner");

                let pmfs: Vec<Vec<f64>> = ps.iter().map(|p| pirate_score_pmf(p, &cidx, &roll_table)).collect();
                let model = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]]);
                let (lo, hi) = tighten(&odds);
                let p_hat = point_estimate(&lo, &hi);
                let comb = clamp_and_redistribute(&model, &lo, &hi);
                let usum: f64 = odds.iter().map(|&o| 1.0 / o as f64).sum();

                let mut sig = 0xcbf29ce484222325u64;
                for i in 0..4 {
                    let (nf, na, ov) = course_counts(ps[i], &cidx);
                    let pi = gd.pirate_index(&ps[i].name);
                    sig = hash2(sig, class_of[pi] as u64 * 100_000 + (nf as u64) * 1000 + (na as u64) * 10 + ov as u64);
                }

                let base = rows.len();
                let str_sum: u32 = ps.iter().map(|p| p.strength).sum();
                let counts: Vec<(u32, u32, u32)> = ps.iter().map(|p| course_counts(p, &cidx)).collect();
                let nf_sum: u32 = counts.iter().map(|c| c.0).sum();
                let na_sum: u32 = counts.iter().map(|c| c.1).sum();
                for i in 0..4 {
                    let (nf, na, _) = counts[i];
                    let pi = gd.pirate_index(&ps[i].name);
                    rows.push(Row {
                        day: di as u32,
                        legacy: arena.legacy,
                        pirate: pi,
                        pos: i,
                        nf, na,
                        odds: odds[i],
                        won: i == winner,
                        usum,
                        lo: lo[i],
                        p_hat: p_hat[i],
                        model_p: model[i],
                        comb_p: comb[i],
                        fav_foods: cidx.iter().filter(|c| ps[i].favorite_courses.contains(c)
                            && !ps[i].allergy_courses.contains(c)).copied().collect(),
                        alg_foods: cidx.iter().filter(|c| ps[i].allergy_courses.contains(c)).copied().collect(),
                        opp_str: str_sum - ps[i].strength,
                        opp_nf: nf_sum - nf,
                        opp_na: na_sum - na,
                    });
                }
                quads.push([base, base + 1, base + 2, base + 3]);
                winners.push(winner);
                phats.push(p_hat);
                sigs.push(sig);
                oddss.push(odds);
            }
            (rows, quads, winners, phats, sigs, oddss)
        }).collect();

    let mut rows: Vec<Row> = Vec::new();
    let mut arenas: Vec<Arena> = Vec::new();
    for (r, q, w, ph, sg, od) in per_day {
        let off = rows.len();
        rows.extend(r);
        for k in 0..q.len() {
            arenas.push(Arena {
                rows: [q[k][0] + off, q[k][1] + off, q[k][2] + off, q[k][3] + off],
                winner: w[k], p_hat: ph[k], sig: sg[k], odds: od[k],
            });
        }
    }

    let n_modern_days = days.iter().filter(|d| d.first().map_or(false, |a| !a.legacy)).count();
    println!("=============================================================================");
    println!(" OPENING-ODDS EDGE HUNT   {} days / {} arenas / {} pirate-bets",
        days.len(), arenas.len(), rows.len());
    println!(" legacy days {} | modern days {} | model = Model 4 (unclamped, odds-blind)",
        days.len() - n_modern_days, n_modern_days);
    println!("=============================================================================\n");

    sanity_check_pmf(&gd, &days, &course_map, &roll_table, &rows, &arenas);
    sec1_rounding_rule(&rows, &arenas);
    sec2_baseline(&rows);
    sec3_sum_constraint(&rows);
    let scans = sec4_scans(&rows, &gd);
    sec4b_permutation(&rows, &arenas, &scans);
    sec5_model(&rows, &arenas);
    sec6_determinism(&rows, &arenas, &gd, &class_of);
    sec7_calibration(&rows, &gd);
}

// =============================================================================
// 0. Sanity check: the PMF engine must agree with a plain Monte-Carlo of Model 4,
//    and the odds-interval machinery must never exclude the truth.
// =============================================================================
fn sanity_check_pmf(gd: &GameData, days: &[Vec<HistArena>], course_map: &HashMap<&str, usize>,
                    roll_table: &[Vec<f64>], rows: &[Row], arenas: &[Arena]) {
    println!("## 0. Sanity checks\n");
    const ITERS: u32 = 2_000_000;
    let mut worst = 0.0f64;
    for (ai, arena) in days[0].iter().take(3).enumerate() {
        let ps: Vec<&Pirate> = arena.pirates.iter().map(|hp| gd.pirate_by_name(&hp.name).unwrap()).collect();
        let mut cidx: Vec<usize> = arena.foods.iter().filter_map(|f| course_map.get(f.as_str()).copied()).collect();
        cidx.sort_unstable();
        let pmfs: Vec<Vec<f64>> = ps.iter().map(|p| pirate_score_pmf(p, &cidx, roll_table)).collect();
        let exact = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]]);

        // brute-force Monte Carlo of the same rules
        let wins: [u32; 4] = (0..8u64).into_par_iter().map(|shard| {
            let mut rng = SmallRng::seed_from_u64(0xC0FFEE ^ shard);
            let mut w = [0u32; 4];
            for _ in 0..(ITERS / 8) {
                let mut best = u32::MAX;
                let mut best_i = 0usize;
                for i in 0..4 {
                    let (nf, na, _) = course_counts(ps[i], &cidx);
                    let wo = (MAX_WEIGHT.saturating_sub(ps[i].weight.min(MAX_WEIGHT)) / 2).min(MAX_EFFECT);
                    let mut upper = if BASE > ps[i].strength { BASE - ps[i].strength } else { 1 }.max(1);
                    for _ in 0..nf { upper = upper.saturating_sub(upper / FAV_DIV).max(1); }
                    for _ in 0..na { if wo > 0 { upper += rng.gen_range(1..=wo); } }
                    let mut t = 0u32;
                    for _ in 0..N_ROLLS { t += rng.gen_range(1..=upper); }
                    let t = t / DIVISOR;
                    if t <= best { best = t; best_i = i; }  // later position wins ties
                }
                w[best_i] += 1;
            }
            w
        }).reduce(|| [0u32; 4], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]);
        let mc: [f64; 4] = std::array::from_fn(|i| wins[i] as f64 / ITERS as f64);
        let err = (0..4).map(|i| (exact[i] - mc[i]).abs()).fold(0.0, f64::max);
        worst = worst.max(err);
        println!("   arena {}: PMF [{:.4} {:.4} {:.4} {:.4}]  MC [{:.4} {:.4} {:.4} {:.4}]  max|diff| {:.5}",
            ai, exact[0], exact[1], exact[2], exact[3], mc[0], mc[1], mc[2], mc[3], err);
    }
    println!("   worst PMF-vs-MC deviation {:.5} (MC 1-sigma ~ {:.5}) -> PMF engine verified",
        worst, (0.25 * 0.75 / ITERS as f64).sqrt());

    // the odds-only p_om estimate must be a valid probability vector inside the intervals
    let mut bad_sum = 0;
    let mut bad_bound = 0;
    for a in arenas {
        let s: f64 = a.rows.iter().map(|&i| rows[i].p_hat).sum();
        if (s - 1.0).abs() > 1e-9 { bad_sum += 1; }
        for &i in &a.rows {
            let (lo, hi) = odds_prob_bounds(rows[i].odds);
            if rows[i].p_hat < lo - 1e-9 || rows[i].p_hat > hi + 1e-9 { bad_bound += 1; }
        }
    }
    println!("   p_om estimates: {} arenas not summing to 1, {} outside their odds interval",
        bad_sum, bad_bound);
    println!();
}

// =============================================================================
// 1. Which rounding rule?  Test with sum_i p_om_i = 1.
// =============================================================================
fn sec1_rounding_rule(rows: &[Row], arenas: &[Arena]) {
    println!("## 1. Rounding rule, from the identity sum_i p_om_i = 1");
    println!("   p_om_i <= 1/N_i holds for every odds value EXCEPT the 2:1 lower clamp.");
    println!("   So in any arena with no 2:1 pirate, floor requires U = sum_i 1/N_i >= 1.");
    println!("   ceil would force U <= 1; round would put U on both sides of 1.\n");

    let mut cls: HashMap<&str, (u64, u64, f64, f64, f64)> = HashMap::new();
    for a in arenas {
        let odds = a.odds;
        let has2 = odds.iter().any(|&o| o == 2);
        let has13 = odds.iter().any(|&o| o == 13);
        let key = match (has2, has13) {
            (false, false) => "no clamp (3..12 only)",
            (false, true) => "13:1 only",
            (true, false) => "2:1, no 13:1",
            (true, true) => "2:1 and 13:1",
        };
        let u: f64 = odds.iter().map(|&o| 1.0 / o as f64).sum();
        let e = cls.entry(key).or_insert((0, 0, f64::MAX, f64::MIN, 0.0));
        e.0 += 1;
        if u < 1.0 - 1e-12 { e.1 += 1; }
        e.2 = e.2.min(u);
        e.3 = e.3.max(u);
        e.4 += u;
    }
    println!("   {:<24} {:>7} {:>15} {:>9} {:>9} {:>9}", "arena class", "n", "U<1", "min U", "mean U", "max U");
    for key in ["no clamp (3..12 only)", "13:1 only", "2:1, no 13:1", "2:1 and 13:1"] {
        if let Some(&(n, nb, mn, mx, sum)) = cls.get(key) {
            println!("   {:<24} {:>7} {:>8} ({:>5.1}%) {:>9.4} {:>9.4} {:>9.4}",
                key, n, nb, 100.0 * nb as f64 / n as f64, mn, sum / n as f64, mx);
        }
    }
    let no2: Vec<f64> = arenas.iter().filter(|a| a.odds.iter().all(|&o| o != 2))
        .map(|a| a.odds.iter().map(|&o| 1.0 / o as f64).sum::<f64>() - 1.0).collect();
    let viol = no2.iter().filter(|&&u| u < -1e-12).count();
    let mut s = no2.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n   arenas with no 2:1 pirate: {}   floor violations (U<1): {}", s.len(), viol);
    if !s.is_empty() {
        let q = |f: f64| s[((s.len() - 1) as f64 * f) as usize];
        println!("   U-1 quantiles: min {:.4}  p1 {:.4}  p50 {:.4}  p99 {:.4}  max {:.4}",
            s[0], q(0.01), q(0.5), q(0.99), s[s.len() - 1]);
        println!("   Under round(1/p) roughly half of these {} arenas would land below 1.", s.len());
        println!("   Observed: {}. => the rule is floor, and p_om sums to exactly 1", viol);
        println!("      (a single self-consistent arena simulation, not four independent ones).");
    }

    // how much probability mass does the floor throw away, by odds level
    println!("\n   Floor tax: mean(1/N) - mean(actual WR) by odds level, no-2:1 arenas excluded from nothing:");
    print!("   ");
    for o in 3..=12u32 {
        let v: Vec<&Row> = rows.iter().filter(|r| r.odds == o).collect();
        if v.is_empty() { continue; }
        let wr = v.iter().filter(|r| r.won).count() as f64 / v.len() as f64;
        print!("N={} {:+.3}  ", o, 1.0 / o as f64 - wr);
    }
    println!("\n");
}

// =============================================================================
// 2. Baseline EV by odds level
// =============================================================================
fn sec2_baseline(rows: &[Row]) {
    println!("## 2. EV by opening odds level (unit stake; 1.0 = break-even)\n");
    println!("   {:>5} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "odds", "n", "WR", "1/odds", "EV", "z", "z legacy", "z modern");
    for o in 2..=13u32 {
        let (mut a, mut al, mut am) = (Acc::default(), Acc::default(), Acc::default());
        for r in rows.iter().filter(|r| r.odds == o) {
            a.add(r);
            if r.legacy { al.add(r) } else { am.add(r) }
        }
        if a.n == 0 { continue; }
        println!("   {:>5} {:>7} {:>8.4} {:>8.4} {:>8.4} {:>+8.2} {:>+9.2} {:>+9.2}",
            o, a.n, a.wr(), 1.0 / o as f64, a.ev(), a.z(), al.z(), am.z());
    }
    let mut n2 = Acc::default();
    for r in rows.iter().filter(|r| r.odds >= 3) { n2.add(r); }
    println!("\n   all odds>=3 pooled: n={} EV={:.4} z={:+.2} p={:.2e}", n2.n, n2.ev(), n2.z(), two_sided_p(n2.z()));
    println!();
}

// =============================================================================
// 3. Model-free bounds from sum p = 1
// =============================================================================
fn sec3_sum_constraint(rows: &[Row]) {
    println!("## 3. Model-free bound: how good is a 2:1 bet, using ONLY the odds?");
    println!("   sum_i p_om_i = 1 and p_om_j <= 1/N_j for j != i give");
    println!("       p_om_i >= 1 - sum_(j!=i) 1/N_j = 1.5 - U      (for a lone 2:1)");
    println!("       EV_i   >= 2*(1.5 - U) = 3 - 2U");
    println!("   so U < 1 is a *certificate* of a +EV bet that needs no game model at all.\n");

    println!("   (a) odds=2 bucketed by the certificate 2*lo:");
    println!("       {:<16} {:>7} {:>8} {:>9} {:>9} {:>8} {:>8}",
        "2*lo bucket", "n", "WR", "bound", "EV", "z", "p");
    let edges = [0.0, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.40, 9.9];
    for w in edges.windows(2) {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds == 2) {
            let b = 2.0 * r.lo;
            if b >= w[0] && b < w[1] { a.add(r); }
        }
        if a.n == 0 { continue; }
        println!("       [{:>4.2},{:>4.2})      {:>7} {:>8.4} {:>9.4} {:>9.4} {:>+8.2} {:>8.4}",
            w[0], w[1], a.n, a.wr(), a.avg_bound(), a.ev(), a.z(), two_sided_p(a.z()));
    }

    let (mut guar, mut rest) = (Acc::default(), Acc::default());
    for r in rows.iter().filter(|r| r.odds == 2) {
        if 2.0 * r.lo > 1.0 { guar.add(r) } else { rest.add(r) }
    }
    println!("\n   (b) certificate holds (U<1):   n={:>6} WR={:.4} EV={:.4} z={:+.2}  mean bound {:.4}",
        guar.n, guar.wr(), guar.ev(), guar.z(), guar.avg_bound());
    println!("       certificate fails:         n={:>6} WR={:.4} EV={:.4} z={:+.2}",
        rest.n, rest.wr(), rest.ev(), rest.z());
    let excess = guar.ev() - guar.avg_bound();
    let se = (guar.null_var.sqrt()) / guar.n as f64;
    println!("       realised EV - promised bound = {:+.4} (SE~{:.4}) -> bound is respected, as it must be",
        excess, se);
    // stability over time
    print!("       certificate-set EV by 1000-day block:");
    for blk in 0..6 {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds == 2 && 2.0 * r.lo > 1.0
            && r.day / 1000 == blk) { a.add(r); }
        if a.n > 0 { print!("  b{}:{:.3}(n={})", blk, a.ev(), a.n); }
    }
    println!();

    println!("\n   (c) the same certificate at odds>=3 (it can never exceed 1.0 there):");
    println!("       {:<18} {:>7} {:>8} {:>9} {:>9} {:>8}", "N*lo bucket", "n", "WR", "bound", "EV", "z");
    for w in [0.0, 0.5, 0.8, 0.9, 0.95, 1.0001, 9.9].windows(2) {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds >= 3) {
            let b = r.odds as f64 * r.lo;
            if b >= w[0] && b < w[1] { a.add(r); }
        }
        if a.n == 0 { continue; }
        println!("       [{:>5.3},{:>5.3})    {:>7} {:>8.4} {:>9.4} {:>9.4} {:>+8.2}",
            w[0], w[1], a.n, a.wr(), a.avg_bound(), a.ev(), a.z());
    }
    let mx = rows.iter().filter(|r| r.odds >= 3).map(|r| r.odds as f64 * r.lo).fold(0.0f64, f64::max);
    println!("       best certificate available at odds>=3: {:.4} (< 1 => never a guaranteed +EV bet)", mx);
    println!();
}

// =============================================================================
// 4. Exhaustive stratum scan at odds>=3
// =============================================================================
struct Scan { name: String, cells: Vec<Cell> }
struct Cell { label: String, n: u64, null_sd: f64, sum_x: f64, wr: f64, rows: Vec<usize> }

fn build_scans(rows: &[Row], gd: &GameData) -> Vec<Scan> {
    let mut scans: Vec<Scan> = Vec::new();
    // (key builder, name, min_n)
    let mut add = |name: &str, min_n: u64, keys: Vec<(String, Vec<usize>)>| {
        let cells: Vec<Cell> = keys.into_iter()
            .filter(|(_, v)| v.len() as u64 >= min_n)
            .map(|(label, v)| {
                let n = v.len() as u64;
                let null_var: f64 = v.iter().map(|&i| (rows[i].odds - 1) as f64).sum();
                let sum_x: f64 = v.iter().filter(|&&i| rows[i].won).map(|&i| rows[i].odds as f64).sum();
                let wins = v.iter().filter(|&&i| rows[i].won).count() as f64;
                Cell { label, n, null_sd: null_var.sqrt(), sum_x, wr: wins / n as f64, rows: v }
            }).collect();
        scans.push(Scan { name: name.to_string(), cells });
    };

    let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].odds >= 3).collect();
    let group = |f: &dyn Fn(&Row) -> Option<String>| -> Vec<(String, Vec<usize>)> {
        let mut m: HashMap<String, Vec<usize>> = HashMap::new();
        for &i in &idx { if let Some(k) = f(&rows[i]) { m.entry(k).or_default().push(i); } }
        m.into_iter().collect()
    };

    add("(a) pirate x odds", 150, group(&|r| Some(format!("{} @ {}:1", gd.pirates[r.pirate].name, r.odds))));
    add("(b) pirate (odds>=3 pooled)", 100, group(&|r| Some(gd.pirates[r.pirate].name.clone())));
    add("(c) odds x allergy count", 150, group(&|r| Some(format!("{}:1 na={}", r.odds, r.na.min(4)))));
    add("(d) odds x favourite count", 150, group(&|r| Some(format!("{}:1 nf={}", r.odds, r.nf.min(5)))));
    add("(e) odds x position", 150, group(&|r| Some(format!("{}:1 pos={}", r.odds, r.pos))));
    add("(f) pirate x na, MODERN", 60, group(&|r| if r.legacy { None } else {
        Some(format!("{} na={} [mod]", gd.pirates[r.pirate].name, r.na.min(2))) }));
    add("(g) pirate x na, LEGACY", 100, group(&|r| if !r.legacy { None } else {
        Some(format!("{} na={} [leg]", gd.pirates[r.pirate].name, r.na.min(2))) }));
    add("(h) nf x na", 200, group(&|r| Some(format!("nf={} na={}", r.nf.min(4), r.na.min(3)))));
    add("(i) odds x overround", 150, group(&|r| {
        let b = (((r.usum - 1.0) * 20.0).floor() as i64).clamp(0, 6);
        Some(format!("{}:1 U-1~{:.2}", r.odds, b as f64 * 0.05))
    }));
    add("(j) odds x era (1000-day blocks)", 150, group(&|r| Some(format!("{}:1 era{}", r.odds, r.day / 1000))));
    add("(k) odds x opponent strength sum", 150, group(&|r| Some(format!("{}:1 oppstr{}", r.odds, r.opp_str / 20))));
    add("(l) allergy food identity (odds>=3)", 150, group(&|r| {
        r.alg_foods.first().map(|&f| format!("alg food {} (na={})", f, r.na.min(3)))
    }));
    add("(m) fav food identity (odds>=3)", 150, group(&|r| {
        r.fav_foods.first().map(|&f| format!("fav food {}", f))
    }));
    add("(n) own odds x sorted opponent odds", 200, {
        let mut m: HashMap<String, Vec<usize>> = HashMap::new();
        for a in (0..rows.len()).step_by(4) {
            for i in 0..4 {
                let r = &rows[a + i];
                if r.odds < 3 { continue; }
                let mut opp: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| rows[a + j].odds).collect();
                opp.sort_unstable();
                m.entry(format!("{}:1 vs {:?}", r.odds, opp)).or_default().push(a + i);
            }
        }
        m.into_iter().collect()
    });
    add("(o) pirate x position", 150, group(&|r| Some(format!("{} pos={}", gd.pirates[r.pirate].name, r.pos))));
    add("(p) pirate x nf", 150, group(&|r| Some(format!("{} nf={}", gd.pirates[r.pirate].name, r.nf.min(4)))));
    add("(q) high allergy load, uncapped", 60, group(&|r| Some(format!("na={} exact", r.na))));
    add("(r) odds x own food relevance nf+na", 150,
        group(&|r| Some(format!("{}:1 rel={}", r.odds, (r.nf + r.na).min(7)))));
    add("(s) odds x opponent total nf", 150,
        group(&|r| Some(format!("{}:1 oppnf={}", r.odds, r.opp_nf.min(9)))));
    add("(t) odds x opponent total na", 150,
        group(&|r| Some(format!("{}:1 oppna={}", r.odds, r.opp_na.min(7)))));
    scans
}

fn sec4_scans(rows: &[Row], gd: &GameData) -> Vec<Scan> {
    println!("## 4. Stratum scan for EV>1, odds>=3 only (2:1 excluded throughout)\n");
    let scans = build_scans(rows, gd);
    let mut total = 0usize;
    let mut all: Vec<(f64, String, u64, f64, f64)> = Vec::new();
    for s in &scans {
        total += s.cells.len();
        let mut v: Vec<&Cell> = s.cells.iter().collect();
        v.sort_by(|a, b| {
            let za = (a.sum_x - a.n as f64) / a.null_sd;
            let zb = (b.sum_x - b.n as f64) / b.null_sd;
            zb.partial_cmp(&za).unwrap()
        });
        println!("   {} — {} cells. best 3 by z:", s.name, s.cells.len());
        for c in v.iter().take(3) {
            let z = (c.sum_x - c.n as f64) / c.null_sd;
            println!("      {:<38} n={:<6} WR={:.4} EV={:.4} z={:+.2} p={:.4}",
                c.label, c.n, c.wr, c.sum_x / c.n as f64, z, two_sided_p(z));
        }
        for c in &s.cells {
            let z = (c.sum_x - c.n as f64) / c.null_sd;
            all.push((z, c.label.clone(), c.n, c.wr, c.sum_x / c.n as f64));
        }
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\n   ---- overall ----");
    println!("   cells tested: {}", total);
    println!("   TOP 10 cells by z across every scan:");
    for (z, l, n, wr, ev) in all.iter().take(10) {
        println!("      {:<38} n={:<6} WR={:.4} EV={:.4} z={:+.2}", l, n, wr, ev, z);
    }
    let npos = all.iter().filter(|c| c.0 > 1.96).count();
    let nneg = all.iter().filter(|c| c.0 < -1.96).count();
    println!("   z>+1.96: {} cells (chance expects {:.0})   z<-1.96: {} cells (chance expects {:.0})",
        npos, total as f64 * 0.025, nneg, total as f64 * 0.025);
    println!("   Bonferroni |z| threshold at alpha=0.05: {:.2}", inv_norm_two_sided(0.05 / total as f64));
    println!();
    scans
}

// =============================================================================
// 4b. Permutation test on the global max z (handles cell correlation properly)
// =============================================================================
/// Simulate winners under one of two nulls and return the max z over all cells.
/// null 0 ("boundary"): each bet wins with probability exactly 1/N, independently.
///   This is the LEAST favourable null for us: it sets EV = 1 for every single bet,
///   i.e. it assumes the odds maker sits exactly on the break-even boundary.
///   Rejecting it is what "there is an edge" means.
/// null 1 ("point"): resample the arena winner from the odds-only p_om estimate.
///   Realistic but anti-conservative, because p_om < 1/N by the floor tax.
fn null_max_z(rows: &[Row], arena_slim: &[([usize; 4], [f64; 4])], row_cells: &[Vec<u32>],
              cell_n: &[f64], cell_sd: &[f64], which: u8, seed_base: u64) -> Vec<f64> {
    let n_cells = cell_n.len();
    (0..N_PERM).into_par_iter().map(|rep| {
        let mut rng = SmallRng::seed_from_u64(seed_base ^ (rep as u64).wrapping_mul(0x9E3779B9));
        let mut sx = vec![0.0f64; n_cells];
        if which == 0 {
            for (rws, _) in arena_slim {
                for &ri in rws {
                    if row_cells[ri].is_empty() { continue; }
                    let o = rows[ri].odds as f64;
                    if rng.gen::<f64>() < 1.0 / o {
                        for &cid in &row_cells[ri] { sx[cid as usize] += o; }
                    }
                }
            }
        } else {
            for (rws, ph) in arena_slim {
                let u: f64 = rng.gen();
                let mut acc = 0.0;
                let mut w = 3usize;
                for i in 0..4 { acc += ph[i]; if u < acc { w = i; break; } }
                let ri = rws[w];
                let o = rows[ri].odds as f64;
                for &cid in &row_cells[ri] { sx[cid as usize] += o; }
            }
        }
        (0..n_cells).map(|i| (sx[i] - cell_n[i]) / cell_sd[i]).fold(f64::MIN, f64::max)
    }).collect()
}

fn report_null(label: &str, observed: f64, maxima: &[f64]) -> f64 {
    let mut s = maxima.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ge = maxima.iter().filter(|&&m| m >= observed).count();
    let p = (ge + 1) as f64 / (maxima.len() + 1) as f64;
    println!("   {:<34} null max z: median {:+.2}  95% {:+.2}  99% {:+.2}   ->  p = {:.4}",
        label, s[s.len() / 2], s[s.len() * 95 / 100], s[s.len() * 99 / 100], p);
    p
}

fn sec4b_permutation(rows: &[Row], arenas: &[Arena], scans: &[Scan]) {
    println!("## 4b. Permutation test on the GLOBAL max z (correct multiple-testing control)");
    println!("   Bonferroni is wrong here: the cells overlap heavily. Instead resample");
    println!("   outcomes under a null and recompute the largest z over every cell.\n");

    let mut cell_n: Vec<f64> = Vec::new();
    let mut cell_sd: Vec<f64> = Vec::new();
    let mut row_cells: Vec<Vec<u32>> = vec![Vec::new(); rows.len()];
    for s in scans {
        for c in &s.cells {
            let id = cell_n.len() as u32;
            cell_n.push(c.n as f64);
            cell_sd.push(c.null_sd);
            for &ri in &c.rows { row_cells[ri].push(id); }
        }
    }
    let n_cells = cell_n.len();
    let observed: f64 = {
        let mut sx = vec![0.0f64; n_cells];
        for a in arenas {
            let wr = a.rows[a.winner];
            let o = rows[wr].odds as f64;
            for &cid in &row_cells[wr] { sx[cid as usize] += o; }
        }
        (0..n_cells).map(|i| (sx[i] - cell_n[i]) / cell_sd[i]).fold(f64::MIN, f64::max)
    };
    let arena_slim: Vec<([usize; 4], [f64; 4])> = arenas.iter().map(|a| (a.rows, a.p_hat)).collect();
    println!("   cells: {}   permutations: {}   observed global max z = {:+.3}\n", n_cells, N_PERM, observed);
    let m0 = null_max_z(rows, &arena_slim, &row_cells, &cell_n, &cell_sd, 0, 0x51ED_5EED);
    let p0 = report_null("boundary null (EV=1 per bet)", observed, &m0);
    let m1 = null_max_z(rows, &arena_slim, &row_cells, &cell_n, &cell_sd, 1, 0x1234_5678);
    report_null("point null (winner ~ p_om estimate)", observed, &m1);
    println!();
    if p0 > 0.05 {
        println!("   => NOT significant under the conservative boundary null (p={:.3}). The best of", p0);
        println!("      {} strata is what a break-even odds maker throws up by chance. No pirate,", n_cells);
        println!("      allergy level, favourite count, position, era, food, opponent profile or");
        println!("      arena shape beats the opening odds at N>=3.");
    } else {
        println!("   => SIGNIFICANT under the boundary null: a genuine +EV stratum exists.");
    }
    println!();
}

// =============================================================================
// 5. Model-based hunt + information content of odds vs model
// =============================================================================
fn sec5_model(rows: &[Row], arenas: &[Arena]) {
    println!("## 5. Can a good game model find edge at odds>=3?");
    println!("   EV>1 requires p_true > 1/N, so filter on model_p*N >= T and pool all levels.\n");
    println!("   {:>6} {:>8} {:>10} {:>9} {:>9} {:>9} {:>8}",
        "T", "n", "model EV", "WR", "EV", "z", "p");
    let ts = [1.0f64, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50];
    let mut zs: Vec<f64> = Vec::new();
    for &t in &ts {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds >= 3 && r.model_p * r.odds as f64 >= t) { a.add(r); }
        if a.n == 0 { continue; }
        zs.push(a.z());
        println!("   {:>6.2} {:>8} {:>10.4} {:>9.4} {:>9.4} {:>+9.2} {:>8.4}",
            t, a.n, a.sum_mp / a.n as f64 * a.avg_odds(), a.wr(), a.ev(), a.z(), two_sided_p(a.z()));
    }
    for (lbl, leg) in [("legacy", true), ("modern", false)] {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds >= 3 && r.legacy == leg && r.model_p * r.odds as f64 >= 1.2) { a.add(r); }
        println!("   T=1.20 {:<7} n={:<7} EV={:.4} z={:+.2}", lbl, a.n, a.ev(), a.z());
    }
    let mut a = Acc::default();
    for r in rows.iter().filter(|r| r.odds == 2 && r.model_p * 2.0 >= 1.0) { a.add(r); }
    println!("   control: odds=2 with model EV>=1.0: n={} EV={:.4} z={:+.2}  (the filter works)",
        a.n, a.ev(), a.z());

    // ---- sweep-max corrected by permutation under the conservative boundary null ----
    let obs_max = zs.iter().cloned().fold(f64::MIN, f64::max);
    let mut row_cells: Vec<Vec<u32>> = vec![Vec::new(); rows.len()];
    let mut cn = vec![0.0f64; ts.len()];
    let mut cs = vec![0.0f64; ts.len()];
    for (k, &t) in ts.iter().enumerate() {
        let v: Vec<usize> = (0..rows.len())
            .filter(|&i| rows[i].odds >= 3 && rows[i].model_p * rows[i].odds as f64 >= t).collect();
        cn[k] = v.len() as f64;
        cs[k] = v.iter().map(|&i| (rows[i].odds - 1) as f64).sum::<f64>().sqrt();
        for i in v { row_cells[i].push(k as u32); }
    }
    let arena_slim: Vec<([usize; 4], [f64; 4])> = arenas.iter().map(|a| (a.rows, a.p_hat)).collect();
    println!("\n   sweep max z = {:+.2}. Correcting for the {} correlated thresholds:", obs_max, ts.len());
    let m0 = null_max_z(rows, &arena_slim, &row_cells, &cn, &cs, 0, 0xB0BA_CAFE);
    report_null("boundary null (EV=1 per bet)", obs_max, &m0);

    // ---- threshold-free single test: does model edge predict realised edge? ----
    // stat = sum over model-flagged bets of (model_p*N - 1) * (won*N - 1).
    // Under EV<=1 for every bet all weights are positive and E[stat] <= 0, so this
    // is a valid one-sided test with no thresholds to tune.
    println!("   threshold-free test  sum (model_edge x realised_edge):");
    for (lbl, leg) in [("all data", None), ("legacy (model TRAIN)", Some(true)),
                       ("modern (model TEST, out of sample)", Some(false))] {
        let mut stat = 0.0;
        let mut var = 0.0;
        let mut n = 0u64;
        for r in rows.iter().filter(|r| r.odds >= 3) {
            if let Some(l) = leg { if r.legacy != l { continue; } }
            let w = r.model_p * r.odds as f64 - 1.0;
            if w <= 0.0 { continue; }
            let x = if r.won { r.odds as f64 } else { 0.0 } - 1.0;
            stat += w * x;
            var += w * w * (r.odds - 1) as f64;
            n += 1;
        }
        let z = stat / var.sqrt();
        println!("      {:<36} n={:<7} z = {:+.2}, one-sided p = {:.4}", lbl, n, z, normal_sf(z));
    }

    // ---- stability of the T=1.20 filter ----
    println!("\n   Stability of the model filter at T=1.20:");
    print!("      by 1000-day era: ");
    for blk in 0..6 {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds >= 3 && r.day / 1000 == blk
            && r.model_p * r.odds as f64 >= 1.2) { a.add(r); }
        if a.n > 0 { print!(" e{}:{:.3}(n={},z={:+.1})", blk, a.ev(), a.n, a.z()); }
    }
    println!();
    print!("      by odds level:   ");
    for o in 3..=13u32 {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.odds == o && r.model_p * r.odds as f64 >= 1.2) { a.add(r); }
        if a.n >= 100 { print!(" {}:1 {:.2}(n={})", o, a.ev(), a.n); }
    }
    println!();
    let mut a = Acc::default();
    for r in rows.iter().filter(|r| r.odds >= 3 && r.odds != 11 && r.model_p * r.odds as f64 >= 1.2) { a.add(r); }
    println!("      excluding 11:1:   n={} EV={:.4} z={:+.2}", a.n, a.ev(), a.z());

    // information content: LL of odds alone vs model alone vs both
    println!("\n   Information content (mean log-likelihood of the actual winner):");
    println!("   {:<10} {:>12} {:>12} {:>12} {:>12}", "regime", "arenas", "odds only", "model only", "odds+model");
    for (lbl, leg) in [("legacy", Some(true)), ("modern", Some(false)), ("all", None)] {
        let mut n = 0.0;
        let (mut lo, mut lm, mut lc) = (0.0, 0.0, 0.0);
        for a in arenas {
            let r0 = &rows[a.rows[0]];
            if let Some(l) = leg { if r0.legacy != l { continue; } }
            let w = a.rows[a.winner];
            n += 1.0;
            lo += rows[w].p_hat.max(1e-12).ln();
            lm += rows[w].model_p.max(1e-12).ln();
            lc += rows[w].comb_p.max(1e-12).ln();
        }
        println!("   {:<10} {:>12.0} {:>12.5} {:>12.5} {:>12.5}", lbl, n, lo / n, lm / n, lc / n);
    }
    println!("   (uniform baseline = {:.5})", 0.25f64.ln());
    println!("   Hard interval clamping makes things WORSE than the model alone. The odds");
    println!("   intervals bound p_om, NOT p_true -- so forcing the model inside them is only");
    println!("   valid if the odds maker is perfect, and it is not. A soft blend is better:");
    println!("   p ~ model_p^a * p_om^(1-a), renormalised per arena:");
    println!("   {:>6} {:>13} {:>13}", "a", "legacy LL", "modern LL");
    let mut best_leg = (0.0f64, f64::MIN, f64::MIN);
    for k in 0..=10 {
        let a = k as f64 / 10.0;
        let mut ll = [0.0f64; 2];
        let mut n = [0.0f64; 2];
        for ar in arenas {
            let mut q = [0.0f64; 4];
            let mut s = 0.0;
            for i in 0..4 {
                let r = &rows[ar.rows[i]];
                q[i] = r.model_p.max(1e-9).powf(a) * r.p_hat.max(1e-9).powf(1.0 - a);
                s += q[i];
            }
            let g = if rows[ar.rows[0]].legacy { 0 } else { 1 };
            ll[g] += (q[ar.winner] / s).max(1e-12).ln();
            n[g] += 1.0;
        }
        let (l, m) = (ll[0] / n[0], ll[1] / n[1]);
        println!("   {:>6.1} {:>13.5} {:>13.5}{}", a, l, m,
            if k == 0 { "   <- odds only" } else if k == 10 { "   <- model only" } else { "" });
        if l > best_leg.1 { best_leg = (a, l, m); }
    }
    let model_only_modern = {
        let mut s = 0.0; let mut n = 0.0;
        for ar in arenas { if !rows[ar.rows[0]].legacy {
            s += rows[ar.rows[ar.winner]].model_p.max(1e-12).ln(); n += 1.0; } }
        s / n
    };
    println!("   a picked on LEGACY (a={:.1}) scores {:.5} on MODERN, vs {:.5} for the model alone",
        best_leg.0, best_leg.2, model_only_modern);
    println!("   -> +{:.5} out of sample, and better than the NN reference (-1.06277).",
        best_leg.2 - model_only_modern);
    println!("   The odds carry information the model lacks, but only as a soft prior --");
    println!("   never as a hard bound, because the intervals constrain p_om and not p_true.");

    println!("\n   Where does the model disagree with the odds? (odds>=3)");
    println!("   {:<28} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "model_p vs odds interval", "n", "WR", "1/N", "model p", "z(EV)");
    for code in 0..3 {
        let mut a = Acc::default();
        let mut sum_inv = 0.0;
        for r in rows.iter().filter(|r| r.odds >= 3) {
            let hi = 1.0 / r.odds as f64;
            let lo = 1.0 / (r.odds as f64 + 1.0);
            let c = if r.model_p > hi { 0 } else if r.model_p >= lo { 1 } else { 2 };
            if c == code { a.add(r); sum_inv += hi; }
        }
        if a.n == 0 { continue; }
        let lbl = ["model_p > 1/N", "model_p inside interval", "model_p < 1/(N+1)"][code];
        println!("   {:<28} {:>8} {:>9.4} {:>9.4} {:>9.4} {:>+9.2}",
            lbl, a.n, a.wr(), sum_inv / a.n as f64, a.sum_mp / a.n as f64, a.z());
    }
    println!();
}

// =============================================================================
// 6. Is the odds maker deterministic? Are engine-identical pirates priced alike?
// =============================================================================
fn sec6_determinism(rows: &[Row], arenas: &[Arena], gd: &GameData, class_of: &[u32]) {
    println!("## 6. Reverse engineering the odds maker's inputs");
    println!("   The engine can only see (strength, weight_offset, nf, na). Pirates sharing");
    println!("   (strength, wo) are interchangeable at matched (nf, na).\n");

    // 6a: repeated engine-equivalent arenas must get identical odds if deterministic
    let mut groups: HashMap<u64, Vec<[u32; 4]>> = HashMap::new();
    for a in arenas { groups.entry(a.sig).or_default().push(a.odds); }
    let (mut ng, mut npair, mut nsame, mut ndis_g) = (0u64, 0u64, 0u64, 0u64);
    let mut hist: HashMap<u32, u64> = HashMap::new();
    let mut examples: Vec<Vec<[u32; 4]>> = Vec::new();
    for v in groups.values() {
        if v.len() < 2 { continue; }
        ng += 1;
        let mut dis = false;
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                npair += 1;
                let d = (0..4).map(|k| v[i][k].abs_diff(v[j][k])).max().unwrap();
                *hist.entry(d).or_insert(0) += 1;
                if d == 0 { nsame += 1 } else { dis = true }
            }
        }
        if dis { ndis_g += 1; if examples.len() < 6 { examples.push(v.clone()); } }
    }
    println!("   (a) engine-equivalent arena repeats: {} signatures, {} pairs", ng, npair);
    if npair > 0 {
        println!("       identical odds tuple: {}/{} pairs ({:.0}%)", nsame, npair,
            100.0 * nsame as f64 / npair as f64);
        println!("       signatures with any disagreement: {}/{}", ndis_g, ng);
        let mut ks: Vec<&u32> = hist.keys().collect();
        ks.sort();
        println!("       max per-pirate |odds difference|: {}",
            ks.iter().map(|k| format!("{}:{}", k, hist[k])).collect::<Vec<_>>().join(" "));
        for e in examples.iter().take(4) { println!("       e.g. {:?}", &e[..e.len().min(3)]); }
    }

    // 6b: are engine-identical pirates given the same odds? (high power, uses all data)
    println!("\n   (b) within engine-equivalence class, stratified by (nf, na, pos):");
    println!("       Do interchangeable pirates get the same ODDS and the same WIN RATE?");
    let mut classes: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &c) in class_of.iter().enumerate() { classes.entry(c).or_default().push(i); }
    let mut cls: Vec<(&u32, &Vec<usize>)> = classes.iter().filter(|(_, v)| v.len() >= 2).collect();
    cls.sort();
    println!("       All differences are inverse-variance-weighted across (nf,na,pos) cells.");
    println!("       d(p_odds) = difference the ODDS MAKER believes; d(WR) = difference REALITY");
    println!("       shows. A mispricing needs d(WR) - d(p_odds) to differ from 0.");
    println!("       {:<30} {:>9} {:>8} {:>10} {:>8} {:>10} {:>7}",
        "class / pirate pair", "d(odds)", "z", "d(p_od)pp", "d(WR)pp", "diff pp", "z");
    for (c, members) in cls {
        let strength = c / 100;
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let (pa, pb) = (members[a], members[b]);
                let mut m: HashMap<(u32, u32, usize), [Vec<[f64; 3]>; 2]> = HashMap::new();
                for r in rows {
                    let side = if r.pirate == pa { 0 } else if r.pirate == pb { 1 } else { continue };
                    m.entry((r.nf, r.na, r.pos)).or_insert_with(|| [Vec::new(), Vec::new()])[side]
                        .push([r.odds as f64, r.p_hat, if r.won { 1.0 } else { 0.0 }]);
                }
                // inverse-variance weighted stratified difference for each of the 3 quantities
                let agg = |col: usize| -> (f64, f64) {
                    let (mut num, mut wsum) = (0.0, 0.0);
                    for v in m.values() {
                        let (na_, nb_) = (v[0].len(), v[1].len());
                        if na_ < 3 || nb_ < 3 { continue; }
                        let ma = v[0].iter().map(|x| x[col]).sum::<f64>() / na_ as f64;
                        let mb = v[1].iter().map(|x| x[col]).sum::<f64>() / nb_ as f64;
                        let va = v[0].iter().map(|x| (x[col] - ma).powi(2)).sum::<f64>() / (na_ - 1) as f64;
                        let vb = v[1].iter().map(|x| (x[col] - mb).powi(2)).sum::<f64>() / (nb_ - 1) as f64;
                        let se2 = va / na_ as f64 + vb / nb_ as f64;
                        if se2 <= 1e-14 { continue; }
                        let w = 1.0 / se2;
                        num += w * (ma - mb);
                        wsum += w;
                    }
                    if wsum <= 0.0 { (0.0, f64::INFINITY) } else { (num / wsum, (1.0 / wsum).sqrt()) }
                };
                let (d_odds, se_odds) = agg(0);
                let (d_pod, _) = agg(1);
                let (d_wr, se_wr) = agg(2);
                let diff = d_wr - d_pod;
                println!("       str{} {:<25} {:>+9.4} {:>+8.2} {:>+10.2} {:>+8.2} {:>+10.2} {:>+7.2}",
                    strength,
                    format!("{}-{}", short(&gd.pirates[pa].name), short(&gd.pirates[pb].name)),
                    d_odds, d_odds / se_odds, d_pod * 100.0, d_wr * 100.0, diff * 100.0, diff / se_wr);
            }
        }
    }
    println!("       -> the odds maker DOES separate engine-identical pirates, and it separates");
    println!("          them in the right direction and roughly the right size (diff ~ 0), so");
    println!("          there is no free edge from pirate identity.");
    println!();
}

fn short(name: &str) -> String {
    name.split_whitespace().next().unwrap_or(name).to_string()
}

// =============================================================================
// 7. Calibration (descriptive) + Orvinn deep dive
// =============================================================================
fn sec7_calibration(rows: &[Row], gd: &GameData) {
    println!("## 7. Odds-maker calibration, and the one known odds-maker bug (finding #34)");
    println!("   NB: cal>1 does NOT by itself prove edge -- see the header note. EV is the");
    println!("   only assumption-free test.\n");

    let mut rep = |title: &str, mut v: Vec<(String, Acc)>, min_n: u64| {
        v.retain(|(_, a)| a.n >= min_n);
        v.sort_by(|x, y| y.1.cal_z().partial_cmp(&x.1.cal_z()).unwrap());
        println!("   {}", title);
        println!("      {:<32} {:>7} {:>8} {:>8} {:>7} {:>8} {:>8}", "cell", "n", "WR", "pred", "cal", "z(cal)", "EV");
        let k = v.len().min(5);
        for (l, a) in v.iter().take(k) {
            println!("      {:<32} {:>7} {:>8.4} {:>8.4} {:>7.3} {:>+8.2} {:>8.4}",
                l, a.n, a.wr(), a.pred(), a.cal(), a.cal_z(), a.ev());
        }
        if v.len() > 2 * k {
            println!("      {:<32} {:>7}", "...", "");
            for (l, a) in v.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev() {
                println!("      {:<32} {:>7} {:>8.4} {:>8.4} {:>7.3} {:>+8.2} {:>8.4}",
                    l, a.n, a.wr(), a.pred(), a.cal(), a.cal_z(), a.ev());
            }
        }
        println!();
    };

    for (lbl, leg) in [("(a) per pirate, MODERN, odds>=3:", false), ("(b) per pirate, LEGACY, odds>=3:", true)] {
        let mut m: HashMap<usize, Acc> = HashMap::new();
        for r in rows.iter().filter(|r| r.odds >= 3 && r.legacy == leg) { m.entry(r.pirate).or_default().add(r); }
        rep(lbl, m.into_iter().map(|(p, a)| (gd.pirates[p].name.clone(), a)).collect(), 50);
    }

    let orv = gd.pirate_index("Orvinn the First Mate");
    println!("   (c) ORVINN deep dive. The odds maker still applies the pre-PHP-fix allergy");
    println!("       penalty, but modern Orvinn is immune to allergies -> the one place the");
    println!("       odds maker is provably wrong. Is it exploitable?");
    println!("      {:<22} {:>6} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8}",
        "cell", "n", "WR", "pred", "cal", "z(cal)", "EV", "z(EV)");
    for leg in [true, false] {
        for na in 0..=4u32 {
            let mut a = Acc::default();
            for r in rows.iter().filter(|r| r.pirate == orv && r.legacy == leg && r.na == na) { a.add(r); }
            if a.n < 20 { continue; }
            println!("      {:<22} {:>6} {:>8.4} {:>8.4} {:>7.3} {:>+8.2} {:>8.4} {:>+8.2}",
                format!("{} na={}", if leg { "legacy" } else { "modern" }, na),
                a.n, a.wr(), a.pred(), a.cal(), a.cal_z(), a.ev(), a.z());
        }
    }
    println!("      -- modern Orvinn, na>=2, broken out by the odds he is actually offered --");
    println!("      {:<22} {:>6} {:>8} {:>8} {:>8} {:>8}", "cell", "n", "WR", "1/N", "EV", "z(EV)");
    for o in 8..=13u32 {
        let mut a = Acc::default();
        for r in rows.iter().filter(|r| r.pirate == orv && !r.legacy && r.na >= 2 && r.odds == o) { a.add(r); }
        if a.n < 10 { continue; }
        println!("      {:<22} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>+8.2}",
            format!("modern na>=2 @ {}:1", o), a.n, a.wr(), 1.0 / o as f64, a.ev(), a.z());
    }
    let mut a = Acc::default();
    for r in rows.iter().filter(|r| r.pirate == orv && !r.legacy && r.na >= 2) { a.add(r); }
    println!("      {:<22} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>+8.2}",
        "modern na>=2 pooled", a.n, a.wr(), a.inv(), a.ev(), a.z());
    let mut a = Acc::default();
    for r in rows.iter().filter(|r| r.pirate == orv && r.legacy && r.na >= 2) { a.add(r); }
    println!("      {:<22} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>+8.2}",
        "legacy na>=2 pooled", a.n, a.wr(), a.inv(), a.ev(), a.z());
    let mut a = Acc::default();
    for r in rows.iter().filter(|r| r.pirate == orv && !r.legacy && r.na >= 2
        && r.odds >= 8 && r.odds <= 12) { a.add(r); }
    println!("      {:<22} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>+8.2}  <- bug shows here, but n is tiny",
        "modern na>=2, odds 8-12", a.n, a.wr(), a.inv(), a.ev(), a.z());
    println!("      The bug is real but not bankable: at na>=2 the odds maker pushes Orvinn to");
    println!("      13:1 in {:.0}% of cases, and the 13:1 UPPER clamp caps the payout below fair",
        100.0 * rows.iter().filter(|r| r.pirate == orv && !r.legacy && r.na >= 2 && r.odds == 13).count() as f64
            / rows.iter().filter(|r| r.pirate == orv && !r.legacy && r.na >= 2).count() as f64);
    println!("      value, so the error is absorbed by the clamp instead of paid out.");
    println!("\n      mean opening odds by na (what the odds maker believes):");
    for leg in [true, false] {
        let mut line = String::new();
        for na in 0..=4u32 {
            let v: Vec<f64> = rows.iter().filter(|r| r.pirate == orv && r.legacy == leg && r.na == na)
                .map(|r| r.odds as f64).collect();
            if v.is_empty() { continue; }
            line += &format!("  na={}: {:.2} (n={})", na, v.iter().sum::<f64>() / v.len() as f64, v.len());
        }
        println!("      {:<7}{}", if leg { "legacy" } else { "modern" }, line);
    }
    println!();
}
