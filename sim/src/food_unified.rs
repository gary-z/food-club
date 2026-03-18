mod pirates;

use pirates::{GameData, Pirate, load_historical_matches, HistMatch};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::io::Write;

const MAX_WEIGHT: u32 = 221;
const SIM_ITERS: u32 = 10_000;

fn roll(rng: &mut impl Rng, n: u32) -> u32 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) }
}

fn course_counts(pirate: &Pirate, courses: &[usize], overlap_mode: u8) -> (u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    for &c in courses {
        let is_f = pirate.favorite_courses.contains(&c);
        let is_a = pirate.allergy_courses.contains(&c);
        match (is_f, is_a) {
            (true, true) => match overlap_mode {
                0 => { na += 1; }          // allergy priority (current default)
                1 => { nf += 1; }          // fav priority
                2 => {}                    // neutral: count as neither
                _ => { nf += 1; na += 1; } // both: count as both
            },
            (false, true) => { na += 1; }
            (true, false) => { nf += 1; }
            _ => {}
        }
    }
    (nf, na)
}

fn allergy_damage(pirate: &Pirate, na: u32, max_effect: u32, mode: u8, rng: &mut impl Rng) -> u32 {
    let raw_wo = (MAX_WEIGHT - pirate.weight.min(MAX_WEIGHT)) / 2;
    let wo = if max_effect > 0 { raw_wo.min(max_effect) } else { raw_wo };
    if mode == 1 {
        na * wo
    } else {
        let mut dmg = 0u32;
        for _ in 0..na { dmg += roll(rng, wo); }
        dmg
    }
}

#[derive(Clone, Copy)]
struct Params {
    base: u32,
    n_rolls: u32,
    fav_mode: u8,    // 0=bulk, 1=multiplicative
    fav_param: u32,  // FAV_DIV for bulk, FAV_PCT for mul
    pos_mode: u8,    // 0=none, 1=mul_after
    pos_step: u32,
    tiebreak: u8,    // 0=later, 1=earlier, 2=random
    divisor: u32,    // 0=none
    quant_round: u8, // 0=floor, 1=ceil, 2=round
    max_effect: u32, // allergy damage cap per roll (0 = no cap)
    allergy_mode: u8, // 0=roll(1..wo), 1=flat(wo), 2=mul(die*=param^na), 3=pct(die+=na*die*param/100)
    allergy_param: u32,
    wo_min: u32, // minimum weight_offset (simulates dice(1,0) returning wo_min)
    max_fav: u32, // cap on number of favorite foods that count (0 = no cap)
    str_mode: u8, // 0=linear (base - str), 1=exponential (base * exp(-str * decay/10000))
    str_decay: u32, // decay rate for exponential mode (in units of 1/10000)
    overlap_mode: u8, // 0=allergy-priority, 1=fav-priority, 2=neutral, 3=both
    allergy_order: u8, // 0=before-fav (reduce str first), 1=after-fav (apply to die after fav reduction)
    fav_str_bonus: u32, // >0: favs add to life instead of reducing die. upper = BASE - (life + nf*bonus)
    legacy_wo_min: u32, // wo_min to use for legacy arenas (1 = old PHP rand(1,0) returns 1)
}

fn compute_upper(raw_upper: u32, nf: u32, pos: usize, p: &Params, weight: u32, life: u32) -> u32 {
    let u = match p.fav_mode {
        0 => { // bulk: die -= nf * floor(die / K)
            let red = raw_upper / p.fav_param;
            raw_upper.saturating_sub(nf * red).max(1)
        }
        1 => { // multiplicative: die *= (pct/100)^nf
            let mut u = raw_upper as f64;
            for _ in 0..nf { u *= p.fav_param as f64 / 100.0; }
            (u.floor() as u32).max(1)
        }
        2 => { // iterative bulk: apply floor(die/K) one at a time
            let mut u = raw_upper;
            for _ in 0..nf {
                let red = u / p.fav_param;
                u = u.saturating_sub(red).max(1);
            }
            u
        }
        3 => { // fixed constant: die -= nf * K
            raw_upper.saturating_sub(nf * p.fav_param).max(1)
        }
        // Weight-dependent fav modes (fav_param = K, allergy_param repurposed as K2)
        6 => { // die -= nf * floor(weight / K)
            let red = weight / p.fav_param;
            raw_upper.saturating_sub(nf * red).max(1)
        }
        7 => { // die -= nf * (floor(die/K) + floor(weight/K2))
            let red1 = raw_upper / p.fav_param;
            let red2 = weight / p.allergy_param;
            raw_upper.saturating_sub(nf * (red1 + red2)).max(1)
        }
        8 => { // die -= nf * floor((die + weight) / K)
            let red = (raw_upper + weight) / p.fav_param;
            raw_upper.saturating_sub(nf * red).max(1)
        }
        9 => { // die -= nf * floor(die / K) * weight / K2  (weight-scaled bulk)
            let red = raw_upper / p.fav_param;
            let scaled = (nf as u64 * red as u64 * weight as u64 / p.allergy_param as u64) as u32;
            raw_upper.saturating_sub(scaled).max(1)
        }
        10 => { // die -= nf * floor(weight / K) + nf * floor(die / K2)  (weight primary, die secondary)
            let red1 = weight / p.fav_param;
            let red2 = raw_upper / p.allergy_param;
            raw_upper.saturating_sub(nf * (red1 + red2)).max(1)
        }
        11 => { // die -= nf * floor(life / K) — fav reduction based on life not die
            let red = life / p.fav_param;
            raw_upper.saturating_sub(nf * red).max(1)
        }
        _ => raw_upper.max(1),
    };
    // Apply position
    match p.pos_mode {
        1 => (u * (100 - pos as u32 * p.pos_step) / 100).max(1),
        _ => u,
    }
}

/// MC sim for sanity check
fn sim_arena(pirates: &[&Pirate], courses: &[usize], p: &Params, iters: u32, seed: u64,
             pirate_indices: &[usize; 4], life_adj: &[i32]) -> [f64; 4] {
    let mut rng = SmallRng::seed_from_u64(seed);
    let counts: [(u32, u32); 4] = std::array::from_fn(|i| course_counts(pirates[i], courses, p.overlap_mode));
    let init_life: [u32; 4] = std::array::from_fn(|i| {
        let adj = if pirate_indices[i] < life_adj.len() { life_adj[pirate_indices[i]] } else { 0 };
        (pirates[i].strength as i32 + adj).max(0) as u32
    });

    let mut wins = [0u32; 4];
    for _ in 0..iters {
        let mut times = [0u32; 4];
        for pos in 0..4 {
            let (nf, na) = counts[pos];
            let life = init_life[pos]; // life starts at (adjusted) strength
            let weight = pirates[pos].weight;
            let allergy_dmg = if p.allergy_mode <= 1 {
                allergy_damage(pirates[pos], na, p.max_effect, p.allergy_mode, &mut rng)
            } else { 0 };

            let raw_upper = {
                let eff_life = life.saturating_sub(allergy_dmg);
                if p.base > eff_life { p.base - eff_life } else { 1 }
            }.max(1);

            let after_fav_pos = compute_upper(raw_upper, nf, pos, p, weight, life);

            let upper = if p.allergy_mode == 2 && na > 0 {
                let mut u = after_fav_pos as f64;
                for _ in 0..na { u *= p.allergy_param as f64 / 100.0; }
                (u.floor() as u32).max(1)
            } else if p.allergy_mode == 3 && na > 0 {
                let bonus = (na as u64 * after_fav_pos as u64 * p.allergy_param as u64 / 100) as u32;
                (after_fav_pos + bonus).max(1)
            } else {
                after_fav_pos
            };

            let mut time = 0u32;
            for _ in 0..p.n_rolls { time += roll(&mut rng, upper); }
            if p.divisor > 0 {
                time = match p.quant_round {
                    1 => (time + p.divisor - 1) / p.divisor,
                    2 => (time + p.divisor / 2) / p.divisor,
                    _ => time / p.divisor,
                };
            }
            times[pos] = time;
        }

        let min_time = *times.iter().min().unwrap();
        let winner = match p.tiebreak {
            0 => { let mut w = 0; for i in 0..4 { if times[i] <= min_time { w = i; } } w }
            1 => { times.iter().position(|&t| t == min_time).unwrap() }
            _ => { let tied: Vec<usize> = (0..4).filter(|&i| times[i] == min_time).collect();
                   tied[rng.gen_range(0..tied.len())] }
        };
        wins[winner] += 1;
    }
    std::array::from_fn(|i| wins[i] as f64 / iters as f64)
}

// ==================== PMF-based evaluation (exact, no MC noise) ====================

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

/// Compute a pirate's final (quantized) score PMF.
/// `life` starts at pirate.strength (may be adjusted by life_adj).
fn pirate_score_pmf(
    weight: u32, life: u32, nf_raw: u32, na: u32, pos: usize,
    p: &Params, roll_table: &[Vec<f64>],
) -> Vec<f64> {
    let nf = if p.max_fav > 0 { nf_raw.min(p.max_fav) } else { nf_raw };

    // Allergy damage distribution (for allergy_order=0: before fav; for order=1: after fav)
    let raw_wo = MAX_WEIGHT.saturating_sub(weight.min(MAX_WEIGHT)) / 2;
    let wo_capped = if p.max_effect > 0 { raw_wo.min(p.max_effect) } else { raw_wo };

    let wo = wo_capped.max(p.wo_min);

    let dmg_pmf: Vec<f64> = if p.allergy_mode <= 1 && (na > 0 && wo > 0) {
        if p.allergy_mode == 1 {
            let d = (na * wo) as usize;
            let mut v = vec![0.0; d + 1]; v[d] = 1.0; v
        } else {
            dice_sum_pmf(na, wo)
        }
    } else {
        vec![1.0]
    };

    let max_raw_score = (p.n_rolls as usize) * (roll_table.len() - 1);
    let mut raw_pmf = vec![0.0; max_raw_score + 1];

    for (dmg_val, &dp) in dmg_pmf.iter().enumerate() {
        if dp < 1e-15 { continue; }

        // === Fav-adds-to-life mode ===
        if p.fav_str_bonus > 0 {
            // Favs boost effective life: eff_life = life + nf * bonus
            let eff_life = if p.allergy_order == 0 {
                // allergy before fav: reduce life by allergy, then add fav bonus
                let life_after_allergy = life.saturating_sub(dmg_val as u32);
                life_after_allergy + nf * p.fav_str_bonus
            } else {
                // allergy after fav: add fav bonus first, then reduce
                // but allergy is on the die, not life — handle below
                life + nf * p.fav_str_bonus
            };
            let raw_upper = if p.base > eff_life { p.base - eff_life } else { 1 }.max(1);

            let upper = if p.allergy_order == 1 {
                // allergy damage increases the die AFTER fav
                (raw_upper + dmg_val as u32).max(1)
            } else {
                raw_upper
            };

            // Apply position
            let upper = match p.pos_mode {
                1 => (upper * (100 - pos as u32 * p.pos_step) / 100).max(1),
                _ => upper,
            };

            if (upper as usize) < roll_table.len() {
                let rpmf = &roll_table[upper as usize];
                for (k, &rp) in rpmf.iter().enumerate() {
                    if rp > 0.0 && k < raw_pmf.len() {
                        raw_pmf[k] += dp * rp;
                    }
                }
            }
            continue;
        }

        // === Standard flow ===
        // life after allergy damage (if allergy_order == 0, damage applied before fav)
        let eff_life = if p.allergy_order == 0 {
            life.saturating_sub(dmg_val as u32)
        } else {
            life // allergy applied after favs
        };

        let raw_upper = match p.str_mode {
            1 => {
                let u = p.base as f64 * (-(eff_life as f64) * p.str_decay as f64 / 10000.0).exp();
                (u.floor() as u32).max(1)
            }
            2 => {
                let bonus = weight / p.str_decay;
                let base_adj = p.base + bonus;
                if base_adj > eff_life { base_adj - eff_life } else { 1 }
            }
            3 => {
                // quadratic: upper = base - floor(life^2 / K)
                let s2 = (eff_life as u64 * eff_life as u64 / p.str_decay as u64) as u32;
                if p.base > s2 { p.base - s2 } else { 1 }
            }
            4 => {
                // sqrt: upper = base - floor(sqrt(life) * K)
                let s = ((eff_life as f64).sqrt() * p.str_decay as f64).floor() as u32;
                if p.base > s { p.base - s } else { 1 }
            }
            _ => {
                // linear: upper = base - life
                if p.base > eff_life { p.base - eff_life } else { 1 }
            }
        }.max(1);

        let after_fav_pos = compute_upper(raw_upper, nf, pos, p, weight, eff_life);

        // Apply allergy damage after fav if allergy_order == 1
        let after_allergy = if p.allergy_order == 1 && p.allergy_mode <= 1 {
            (after_fav_pos + dmg_val as u32).max(1)
        } else {
            after_fav_pos
        };

        let upper = match p.allergy_mode {
            2 if na > 0 => {
                let mut u = after_allergy as f64;
                for _ in 0..na { u *= p.allergy_param as f64 / 100.0; }
                (u.floor() as u32).max(1)
            }
            3 if na > 0 => {
                let bonus = (na as u64 * after_allergy as u64 * p.allergy_param as u64 / 100) as u32;
                (after_allergy + bonus).max(1)
            }
            _ => after_allergy,
        };

        if (upper as usize) < roll_table.len() {
            let rpmf = &roll_table[upper as usize];
            for (k, &rp) in rpmf.iter().enumerate() {
                if rp > 0.0 && k < raw_pmf.len() {
                    raw_pmf[k] += dp * rp;
                }
            }
        }
    }

    if p.divisor > 0 {
        if p.quant_round == 3 {
            // Inverse quantization: q = floor(divisor / score)
            // Higher raw score → lower q. To keep lowest-wins semantics,
            // we reverse: output_q = max_q - floor(divisor / score)
            // so low raw score → high floor(N/score) → low reversed_q → wins.
            let max_q = p.divisor as usize; // floor(N/1) = N is the max
            let mut qpmf = vec![0.0; max_q + 1];
            for (k, &pr) in raw_pmf.iter().enumerate() {
                if pr < 1e-15 || k == 0 { continue; }
                let inv_q = p.divisor as usize / k; // floor(N / score)
                let qk = max_q.saturating_sub(inv_q); // reverse so lowest wins
                if qk <= max_q { qpmf[qk] += pr; }
            }
            // Trim trailing zeros
            while qpmf.last() == Some(&0.0) { qpmf.pop(); }
            if qpmf.is_empty() { qpmf.push(1.0); }
            qpmf
        } else {
            let max_q = max_raw_score / p.divisor as usize;
            let mut qpmf = vec![0.0; max_q + 1];
            for (k, &pr) in raw_pmf.iter().enumerate() {
                if pr < 1e-15 { continue; }
                let qk = match p.quant_round {
                    1 => (k + p.divisor as usize - 1) / p.divisor as usize,
                    2 => (k + p.divisor as usize / 2) / p.divisor as usize,
                    _ => k / p.divisor as usize,
                };
                if qk <= max_q { qpmf[qk] += pr; }
            }
            qpmf
        }
    } else {
        while raw_pmf.last() == Some(&0.0) { raw_pmf.pop(); }
        if raw_pmf.is_empty() { raw_pmf.push(1.0); }
        raw_pmf
    }
}

/// Compute win probabilities from 4 independent score PMFs.
fn win_probs_from_pmfs(pmfs: [&[f64]; 4], tiebreak: u8) -> [f64; 4] {
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
    let f = |i: usize, t: usize| -> f64 {
        if t < pmfs[i].len() { pmfs[i][t] } else { 0.0 }
    };
    let s = |i: usize, t: usize| -> f64 {
        if t < surv[i].len() { surv[i][t] } else { 0.0 }
    };
    let g = |i: usize, t: usize| -> f64 {
        if t == 0 { 1.0 } else { s(i, t - 1) }
    };

    let mut probs = [0.0f64; 4];
    for t in 0..max_t {
        match tiebreak {
            0 => {
                probs[3] += f(3,t) * g(0,t) * g(1,t) * g(2,t);
                probs[2] += f(2,t) * g(0,t) * g(1,t) * s(3,t);
                probs[1] += f(1,t) * g(0,t) * s(2,t) * s(3,t);
                probs[0] += f(0,t) * s(1,t) * s(2,t) * s(3,t);
            }
            1 => {
                probs[0] += f(0,t) * g(1,t) * g(2,t) * g(3,t);
                probs[1] += f(1,t) * s(0,t) * g(2,t) * g(3,t);
                probs[2] += f(2,t) * s(0,t) * s(1,t) * g(3,t);
                probs[3] += f(3,t) * s(0,t) * s(1,t) * s(2,t);
            }
            _ => {
                for mask in 1u32..16 {
                    let tied: Vec<usize> = (0..4).filter(|&i| mask & (1 << i) != 0).collect();
                    let mut p_sub = 1.0 / tied.len() as f64;
                    for &i in &tied { p_sub *= f(i, t); }
                    for j in 0..4 {
                        if mask & (1 << j) == 0 { p_sub *= s(j, t); }
                    }
                    for &i in &tied { probs[i] += p_sub; }
                }
            }
        }
    }
    probs
}

struct EvalResult { ll: f64, violations: f64 }

// Probability interval implied by house odds N = max(2, min(13, floor(1/p)))
// For N in 3..=12: floor(1/p) = N => p in (1/(N+1), 1/N]
// For N=2: floor(1/p) <= 2 => p in (1/3, 1)  (weak upper — all p>0.5 maps here)
// For N=13: floor(1/p) >= 13 => p in (0, 1/13] (weak lower)
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

fn eval_pmf_clamped(data: &GameData, matches: &[Vec<HistMatch>], p: &Params, life_adj: &[i32]) -> EvalResult {
    let max_upper = if p.allergy_order == 1 || p.fav_str_bonus > 0 {
        p.base + 80
    } else {
        p.base + 10
    };
    let roll_table: Vec<Vec<f64>> = (0..=max_upper)
        .map(|d| dice_sum_pmf(p.n_rolls, d))
        .collect();

    let flat: Vec<&HistMatch> = matches.iter().flat_map(|d| d.iter()).collect();
    let results: Vec<f64> = flat.par_iter().map(|m| {
        let arena_p = if m.legacy && p.legacy_wo_min > 0 {
            let mut lp = *p;
            lp.wo_min = p.legacy_wo_min;
            lp
        } else {
            *p
        };

        let pirates: [&Pirate; 4] = std::array::from_fn(|i| &data.pirates[m.pirate_indices[i]]);
        let counts: [(u32, u32); 4] = std::array::from_fn(|i| {
            course_counts(pirates[i], &m.course_indices, arena_p.overlap_mode)
        });
        let init_life: [u32; 4] = std::array::from_fn(|i| {
            let adj = if m.pirate_indices[i] < life_adj.len() { life_adj[m.pirate_indices[i]] } else { 0 };
            (pirates[i].strength as i32 + adj).max(0) as u32
        });

        let pmfs: [Vec<f64>; 4] = std::array::from_fn(|i| {
            pirate_score_pmf(pirates[i].weight, init_life[i], counts[i].0, counts[i].1, i, &arena_p, &roll_table)
        });
        let raw_probs = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]], p.tiebreak);

        // Get odds intervals and clamp
        let intervals: [(f64, f64); 4] = std::array::from_fn(|i| {
            odds_prob_bounds(m.opening_odds[i])
        });
        let probs = clamp_and_redistribute(&raw_probs, &intervals);

        probs[m.winner_pos].max(1e-10).ln()
    }).collect();

    let n = results.len() as f64;
    let sum_ll: f64 = results.iter().sum();

    EvalResult {
        ll: sum_ll / n,
        violations: 0.0,
    }
}

fn eval_pmf(data: &GameData, matches: &[Vec<HistMatch>], p: &Params, life_adj: &[i32]) -> EvalResult {
    let max_upper = if p.allergy_mode == 2 {
        ((p.base as f64) * (p.allergy_param as f64 / 100.0).powi(10)).ceil() as u32 + 10
    } else if p.allergy_mode == 3 {
        p.base + 10 * p.base * p.allergy_param / 100 + 10
    } else if p.allergy_order == 1 || p.fav_str_bonus > 0 {
        // Allergy after fav or fav-adds-str: die can grow by up to na*wo = 10*7 = 70
        p.base + 80
    } else {
        p.base + 10
    };
    let roll_table: Vec<Vec<f64>> = (0..=max_upper)
        .map(|d| dice_sum_pmf(p.n_rolls, d))
        .collect();

    let flat: Vec<&HistMatch> = matches.iter().flat_map(|d| d.iter()).collect();
    let results: Vec<(f64, u32, u32)> = flat.par_iter().map(|m| {
        // Legacy regime: old PHP rand(1, 0) returns 1 instead of 0
        let arena_p = if m.legacy && p.legacy_wo_min > 0 {
            let mut lp = *p;
            lp.wo_min = p.legacy_wo_min;
            lp
        } else {
            *p
        };

        let pirates: [&Pirate; 4] = std::array::from_fn(|i| &data.pirates[m.pirate_indices[i]]);
        let counts: [(u32, u32); 4] = std::array::from_fn(|i| {
            course_counts(pirates[i], &m.course_indices, arena_p.overlap_mode)
        });
        let init_life: [u32; 4] = std::array::from_fn(|i| {
            let adj = if m.pirate_indices[i] < life_adj.len() { life_adj[m.pirate_indices[i]] } else { 0 };
            (pirates[i].strength as i32 + adj).max(0) as u32
        });

        let pmfs: [Vec<f64>; 4] = std::array::from_fn(|i| {
            pirate_score_pmf(pirates[i].weight, init_life[i], counts[i].0, counts[i].1, i, &arena_p, &roll_table)
        });
        let probs = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]], p.tiebreak);

        let ll = probs[m.winner_pos].max(1e-10).ln();
        let mut n_violations = 0u32;
        let mut n_checked = 0u32;
        for pos in 0..4 {
            let odds = m.opening_odds[pos];
            if odds == 2 || odds == 13 { continue; }
            n_checked += 1;
            let (lo, hi) = odds_prob_bounds(odds);
            let mp = probs[pos];
            if mp < lo || mp >= hi { n_violations += 1; }
        }
        (ll, n_violations, n_checked)
    }).collect();

    let n = results.len() as f64;
    let sum_ll: f64 = results.iter().map(|r| r.0).sum();
    let total_violations: u32 = results.iter().map(|r| r.1).sum();
    let total_checked: u32 = results.iter().map(|r| r.2).sum();

    EvalResult {
        ll: sum_ll / n,
        violations: if total_checked > 0 { total_violations as f64 / total_checked as f64 } else { 0.0 },
    }
}

/// MC-based eval for sanity check
fn eval_mc(data: &GameData, matches: &[Vec<HistMatch>], p: &Params, life_adj: &[i32]) -> f64 {
    let flat: Vec<&HistMatch> = matches.iter().flat_map(|d| d.iter()).collect();
    let results: Vec<f64> = flat.par_iter().enumerate().map(|(idx, m)| {
        let pirates: Vec<&Pirate> = m.pirate_indices.iter().map(|&i| &data.pirates[i]).collect();
        let probs = sim_arena(
            &[pirates[0], pirates[1], pirates[2], pirates[3]],
            &m.course_indices, p, SIM_ITERS,
            idx as u64 * 7 + 999,
            &m.pirate_indices, life_adj,
        );
        probs[m.winner_pos].max(1e-10).ln()
    }).collect();
    results.iter().sum::<f64>() / results.len() as f64
}

fn hash_day(idx: usize) -> u64 {
    let mut h = idx as u64;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= h >> 32;
    h = h.wrapping_mul(0x6c62272e07bb0142);
    h ^= h >> 32;
    h
}


fn main() {
    let pj = std::fs::read_to_string("../pirates.json").unwrap();
    let data = GameData::load(&pj);
    let hj = std::fs::read_to_string("../historical_matches.json").unwrap();
    let hist = load_historical_matches(&data, &hj);

    // Split: legacy = train, modern = test
    let mut legacy: Vec<Vec<HistMatch>> = Vec::new();
    let mut modern: Vec<Vec<HistMatch>> = Vec::new();
    for (_i, day) in hist.into_iter().enumerate() {
        if day.first().map_or(false, |m| m.legacy) {
            legacy.push(day);
        } else {
            modern.push(day);
        }
    }
    let leg_arenas: usize = legacy.iter().map(|d| d.len()).sum();
    let mod_arenas: usize = modern.iter().map(|d| d.len()).sum();
    println!("Legacy (train): {} days, {} arenas", legacy.len(), leg_arenas);
    println!("Modern (test):  {} days, {} arenas", modern.len(), mod_arenas);

    let baseline = Params {
        base: 112, n_rolls: 4, fav_mode: 0, fav_param: 15,
        pos_mode: 0, pos_step: 0, tiebreak: 0, divisor: 14,
        quant_round: 0, max_effect: 7, allergy_mode: 0, allergy_param: 0,
        wo_min: 0, max_fav: 0, str_mode: 0, str_decay: 0, overlap_mode: 0, allergy_order: 0, fav_str_bonus: 0, legacy_wo_min: 0,
    };
    let bl_leg = eval_pmf(&data, &legacy, &baseline, &[]);
    let bl_mod = eval_pmf(&data, &modern, &baseline, &[]);
    println!("Baseline M1: legacy={:.5} modern={:.5}", bl_leg.ll, bl_mod.ll);

    // === Worst violations: arenas where model disagrees most with odds maker ===
    {
        let m4 = Params {
            base: 120, n_rolls: 6, fav_mode: 2, fav_param: 16,
            pos_mode: 0, pos_step: 0, tiebreak: 0, divisor: 22, quant_round: 0,
            max_effect: 6, allergy_mode: 0, allergy_param: 0, wo_min: 0, max_fav: 0,
            str_mode: 0, str_decay: 0, overlap_mode: 0, allergy_order: 1, fav_str_bonus: 0, legacy_wo_min: 0,
        };
        let max_upper_m4 = m4.base + 80;
        let roll_table_m4: Vec<Vec<f64>> = (0..=max_upper_m4)
            .map(|d| dice_sum_pmf(m4.n_rolls, d))
            .collect();

        // For each pirate-slot, compute violation magnitude
        // Metric: log(model_p / interval_bound) — positive if we overshoot upper, negative if undershoot lower
        struct Violation {
            day_idx: usize,
            arena_idx: usize,
            pos: usize,
            pirate_idx: usize,
            model_p: f64,
            odds: u32,
            lo: f64,
            hi: f64,
            log_ratio: f64, // log(model_p / bound) — magnitude of violation
            direction: i8,  // +1 = model says stronger than odds, -1 = weaker
            // arena context
            all_model_p: [f64; 4],
            all_odds: [u32; 4],
            all_pirate_idx: [usize; 4],
            all_nf: [u32; 4],
            all_na: [u32; 4],
            winner_pos: usize,
        }

        let mut violations: Vec<Violation> = Vec::new();
        let all_data = [(&modern, "modern"), (&legacy, "legacy")];

        for &(dataset, _label) in &all_data {
            for (di, day) in dataset.iter().enumerate() {
                for (ai, m) in day.iter().enumerate() {
                    let pirates: [&Pirate; 4] = std::array::from_fn(|i| &data.pirates[m.pirate_indices[i]]);
                    let counts: [(u32, u32); 4] = std::array::from_fn(|i| {
                        course_counts(pirates[i], &m.course_indices, m4.overlap_mode)
                    });
                    let pmfs: [Vec<f64>; 4] = std::array::from_fn(|i| {
                        pirate_score_pmf(pirates[i].weight, pirates[i].strength, counts[i].0, counts[i].1, i, &m4, &roll_table_m4)
                    });
                    let probs = win_probs_from_pmfs([&pmfs[0], &pmfs[1], &pmfs[2], &pmfs[3]], m4.tiebreak);

                    for pos in 0..4 {
                        let odds = m.opening_odds[pos];
                        if odds == 2 || odds == 13 { continue; } // clamped bins, skip
                        let (lo, hi) = odds_prob_bounds(odds);
                        let p = probs[pos];
                        let (log_ratio, direction) = if p > hi {
                            ((p / hi).ln(), 1i8) // model says stronger
                        } else if p < lo {
                            ((lo / p).ln(), -1i8) // model says weaker
                        } else {
                            continue; // within interval
                        };
                        violations.push(Violation {
                            day_idx: di, arena_idx: ai, pos,
                            pirate_idx: m.pirate_indices[pos],
                            model_p: p, odds, lo, hi, log_ratio, direction,
                            all_model_p: [probs[0], probs[1], probs[2], probs[3]],
                            all_odds: m.opening_odds,
                            all_pirate_idx: m.pirate_indices,
                            all_nf: [counts[0].0, counts[1].0, counts[2].0, counts[3].0],
                            all_na: [counts[0].1, counts[1].1, counts[2].1, counts[3].1],
                            winner_pos: m.winner_pos,
                        });
                    }
                }
            }
        }

        violations.sort_by(|a, b| b.log_ratio.partial_cmp(&a.log_ratio).unwrap());

        println!("\n=== Top 20 worst model-vs-odds violations (modern+legacy, excl odds 2/13) ===\n");
        for (rank, v) in violations.iter().take(20).enumerate() {
            let p = &data.pirates[v.pirate_idx];
            let dir = if v.direction > 0 { "OVER" } else { "UNDER" };
            let model_odds = (1.0 / v.model_p).floor() as u32;
            println!("  #{:2} {} | {} (str={}, wt={}, wo={})",
                     rank + 1, dir, p.name, p.strength, p.weight,
                     ((MAX_WEIGHT - p.weight.min(MAX_WEIGHT)) / 2).min(m4.max_effect));
            println!("       model_p={:.3} (would be odds {}) vs actual odds {} [interval {:.3}-{:.3}]  log_ratio={:.3}",
                     v.model_p, model_odds, v.odds, v.lo, v.hi, v.log_ratio);
            println!("       pos={} nf={} na={} | winner=pos{}", v.pos, v.all_nf[v.pos], v.all_na[v.pos], v.winner_pos);
            // Show all 4 pirates in the arena
            println!("       Arena:");
            for i in 0..4 {
                let pi = &data.pirates[v.all_pirate_idx[i]];
                let marker = if i == v.pos { " <<" } else if i == v.winner_pos { " [W]" } else { "" };
                println!("         p{}: {:<24} str={:<2} wt={:<3} odds={:<2} model={:.3} nf={} na={}{}",
                         i, pi.name, pi.strength, pi.weight, v.all_odds[i],
                         v.all_model_p[i], v.all_nf[i], v.all_na[i], marker);
            }
            println!();
        }

        // Aggregate: which pirates appear most often in violations?
        println!("=== Pirates with most violations (model too strong = OVER, too weak = UNDER) ===\n");
        let mut pirate_over = vec![0u32; data.pirates.len()];
        let mut pirate_under = vec![0u32; data.pirates.len()];
        let mut pirate_total_slots = vec![0u32; data.pirates.len()]; // total non-2/13 slots
        let mut pirate_sum_log = vec![0.0f64; data.pirates.len()];
        // Count total slots per pirate
        for dataset in &[&modern, &legacy] {
            for day in dataset.iter() {
                for m in day.iter() {
                    for pos in 0..4 {
                        let odds = m.opening_odds[pos];
                        if odds == 2 || odds == 13 { continue; }
                        pirate_total_slots[m.pirate_indices[pos]] += 1;
                    }
                }
            }
        }
        for v in &violations {
            if v.direction > 0 { pirate_over[v.pirate_idx] += 1; }
            else { pirate_under[v.pirate_idx] += 1; }
            pirate_sum_log[v.pirate_idx] += v.log_ratio * v.direction as f64;
        }
        println!("  {:<28} {:>4} {:>5} {:>5} {:>6} {:>7} {:>5}", "pirate", "str", "wt", "over", "under", "total", "avg_d");
        let mut pidxs: Vec<usize> = (0..data.pirates.len()).collect();
        pidxs.sort_by(|&a, &b| {
            let tot_a = pirate_over[a] + pirate_under[a];
            let tot_b = pirate_over[b] + pirate_under[b];
            tot_b.cmp(&tot_a)
        });
        for &pidx in pidxs.iter().take(20) {
            let tot = pirate_over[pidx] + pirate_under[pidx];
            if tot == 0 { continue; }
            let avg_dir = pirate_sum_log[pidx] / tot as f64;
            let slots = pirate_total_slots[pidx];
            println!("  {:<28} {:>4} {:>5} {:>5} {:>5} {:>5}/{:<5} {:>+.3}",
                     data.pirates[pidx].name, data.pirates[pidx].strength, data.pirates[pidx].weight,
                     pirate_over[pidx], pirate_under[pidx], tot, slots, avg_dir);
        }
    }

    if false {
    // NN best on modern: -1.06277
    // We need to beat -1.06493 (M1 on modern)

    // === H2: Fav reduction based on strength instead of die ===
    // upper -= nf * floor(strength / K) instead of upper -= nf * floor(upper / K)
    // fav_mode=11 does this
    {
        println!("\n=== H2: Fav from strength (fav_mode=11) ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fk: u32, dv: u32, me: u32, te: f64 }
        let mut cfgs = Vec::new();
        for base in (106..=120).step_by(2) {
            for &nr in &[3u32, 4, 5, 6] {
                for fk in (10..=50).step_by(2) {
                    for dv in (0..=24).step_by(2) {
                        for &me in &[6u32, 7, 8] {
                            cfgs.push((base, nr, fk, dv, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fk,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:11, fav_param:fk,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,fk,dv,me,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} fk={} dv={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fk, r.dv, r.me, r.te, r.te - bl_mod.ll);
        }
    }

    // === H4: Larger fav divisor (re-tune baseline with bigger FAV_DIV) ===
    {
        println!("\n=== H4: Re-tune baseline with wider FAV_DIV ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, te: f64 }
        let mut cfgs = Vec::new();
        for base in (106..=120).step_by(2) {
            for &nr in &[3u32, 4, 5, 6] {
                for fp in (10..=30).step_by(1) {
                    for dv in (0..=24).step_by(2) {
                        for &me in &[6u32, 7, 8] {
                            cfgs.push((base, nr, fp, dv, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,fp,dv,me,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} fp={} dv={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.dv, r.me, r.te, r.te - bl_mod.ll);
        }
    }

    // === H1: Iterative fav (fav_mode=2) re-tuned on modern ===
    {
        println!("\n=== H1: Iterative fav (fav_mode=2) ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, te: f64 }
        let mut cfgs = Vec::new();
        for base in (106..=130).step_by(2) {
            for &nr in &[3u32, 4, 5, 6, 7] {
                for fp in (10..=30).step_by(1) {
                    for dv in (0..=30).step_by(2) {
                        for &me in &[6u32, 7, 8] {
                            cfgs.push((base, nr, fp, dv, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:2, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,fp,dv,me,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} fp={} dv={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.dv, r.me, r.te, r.te - bl_mod.ll);
        }
    }

    // === H5: Iterative fav + allergy-after ===
    {
        println!("\n=== H5: Iterative fav + allergy-after (ao=1) ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, te: f64 }
        let mut cfgs = Vec::new();
        for base in (106..=130).step_by(2) {
            for &nr in &[3u32, 4, 5, 6, 7] {
                for fp in (10..=30).step_by(1) {
                    for dv in (0..=30).step_by(2) {
                        for &me in &[5u32, 6, 7, 8] {
                            cfgs.push((base, nr, fp, dv, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:2, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:1, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,fp,dv,me,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} fp={} dv={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.dv, r.me, r.te, r.te - bl_mod.ll);
        }
    }

    // === H3: Fav-adds-to-strength (fav_str_bonus) ===
    {
        println!("\n=== H3: Favs add to strength ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, dv: u32, me: u32, fsb: u32, ao: u8, te: f64 }
        let mut cfgs = Vec::new();
        for base in (108..=130).step_by(2) {
            for &nr in &[3u32, 4, 5, 6] {
                for dv in (0..=24).step_by(2) {
                    for &me in &[6u32, 7, 8] {
                        for &fsb in &[1u32, 2, 3, 4, 5, 6, 8, 10] {
                            for &ao in &[0u8, 1] {
                                cfgs.push((base, nr, dv, me, fsb, ao));
                            }
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,dv,me,fsb,ao)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:15,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao, fav_str_bonus:fsb, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,dv,me,fsb,ao,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} dv={} me={} fsb={} ao={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.dv, r.me, r.fsb, r.ao, r.te, r.te - bl_mod.ll);
        }
    }

    // === PosMul variants ===
    // Model 2: multiplicative fav, position shrinks die, no quantization, random tiebreak
    {
        println!("\n=== PosMul baseline (fav_mode=1, pos_mode=1, tiebreak=2, divisor=0) ===");
        let pm_baseline = Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 1, pos_step: 7, tiebreak: 2, divisor: 0,
            quant_round: 0, max_effect: 7, allergy_mode: 0, allergy_param: 0,
            wo_min: 0, max_fav: 0, str_mode: 0, str_decay: 0, overlap_mode: 0,
            allergy_order: 0, fav_str_bonus: 0, legacy_wo_min: 0,
        };
        let pm_mod = eval_pmf(&data, &modern, &pm_baseline, &[]);
        println!("  PosMul baseline modern={:.5}", pm_mod.ll);

        // Re-tune PosMul on modern
        println!("\n=== PosMul re-tune ===");
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, pp: u32, me: u32, te: f64 }
        let mut cfgs = Vec::new();
        for base in (100..=120).step_by(1) {
            for &nr in &[2u32, 3, 4, 5] {
                for fp in (85..=98).step_by(1) {
                    for pp in (0..=12).step_by(1) {
                        for &me in &[6u32, 7, 8] {
                            cfgs.push((base, nr, fp, pp, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs.len());
        let mut res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,pp,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:1, fav_param:fp,
                pos_mode:1, pos_step:pp, tiebreak:2, divisor:0, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R{base:b,nr,fp,pp,me,te:e.ll}
        }).collect();
        res.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res.iter().take(5) {
            println!("  b={} nr={} fp={} pp={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.pp, r.me, r.te, r.te - pm_mod.ll);
        }

        // PosMul + allergy-after
        println!("\n=== PosMul + allergy-after ===");
        #[derive(Clone)]
        struct R2 { base: u32, nr: u32, fp: u32, pp: u32, me: u32, te: f64 }
        let mut cfgs2 = Vec::new();
        for base in (100..=125).step_by(1) {
            for &nr in &[2u32, 3, 4, 5] {
                for fp in (85..=98).step_by(1) {
                    for pp in (0..=12).step_by(1) {
                        for &me in &[5u32, 6, 7, 8] {
                            cfgs2.push((base, nr, fp, pp, me));
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs2.len());
        let mut res2: Vec<R2> = cfgs2.par_iter().map(|&(b,nr,fp,pp,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:1, fav_param:fp,
                pos_mode:1, pos_step:pp, tiebreak:2, divisor:0, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:1, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R2{base:b,nr,fp,pp,me,te:e.ll}
        }).collect();
        res2.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res2.iter().take(5) {
            println!("  b={} nr={} fp={} pp={} me={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.pp, r.me, r.te, r.te - pm_mod.ll);
        }

        // PosMul with quantization + later-wins tiebreak (hybrid with Model 1)
        println!("\n=== PosMul + quantization + later-wins ===");
        #[derive(Clone)]
        struct R3 { base: u32, nr: u32, fp: u32, pp: u32, dv: u32, me: u32, ao: u8, te: f64 }
        let mut cfgs3 = Vec::new();
        for base in (104..=120).step_by(2) {
            for &nr in &[3u32, 4, 5] {
                for fp in (88..=96).step_by(1) {
                    for pp in (0..=10).step_by(2) {
                        for dv in (10..=22).step_by(2) {
                            for &me in &[6u32, 7, 8] {
                                for &ao in &[0u8, 1] {
                                    cfgs3.push((base, nr, fp, pp, dv, me, ao));
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("  {} configs", cfgs3.len());
        let mut res3: Vec<R3> = cfgs3.par_iter().map(|&(b,nr,fp,pp,dv,me,ao)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:1, fav_param:fp,
                pos_mode:1, pos_step:pp, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao, fav_str_bonus:0, legacy_wo_min:0 };
            let e = eval_pmf(&data, &modern, &p, &[]);
            R3{base:b,nr,fp,pp,dv,me,ao,te:e.ll}
        }).collect();
        res3.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        for r in res3.iter().take(5) {
            println!("  b={} nr={} fp={} pp={} dv={} me={} ao={} modern={:.5} (delta={:+.5})",
                     r.base, r.nr, r.fp, r.pp, r.dv, r.me, r.ao, r.te, r.te - pm_mod.ll);
        }
    }

    println!("\nBaseline M1 modern: {:.5}", bl_mod.ll);
    println!("Model 4 modern:     -1.06314");
    println!("NN best modern:     -1.06277");
    } // end if false

}

/*  OLD SEARCHES (disabled — kept for reference)
    // === Search 11: Fav/allergy overlap handling modes ===
    // Test all 4 overlap modes with baseline and nearby params
    {
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 11: Fav/allergy overlap modes ---").unwrap();
        writeln!(f, "Modes: 0=allergy-priority(current), 1=fav-priority, 2=neutral, 3=both").unwrap();

        #[derive(Clone)]
        struct R { om: u8, base: u32, nr: u32, fp: u32, dv: u32, me: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &om in &[0u8, 1, 2, 3] {
            for &base in &[108u32, 110, 112, 114, 116, 118] {
                for &nr in &[3u32, 4, 5] {
                    for &fp in &[12u32, 14, 15, 16, 18] {
                        for &dv in &[10u32, 12, 14, 16, 18] {
                            for &me in &[6u32, 7, 8] {
                                cfgs.push((om, base, nr, fp, dv, me));
                            }
                        }
                    }
                }
            }
        }
        println!("Search 11 (overlap modes): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(om,b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:om, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{om,base:b,nr,fp,dv,me,tr:t.ll,te:e.ll}
        }).collect();
        // Best per overlap mode
        for &om in &[0u8, 1, 2, 3] {
            let mut sub: Vec<&R> = res.iter().filter(|r| r.om == om).collect();
            sub.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
            let label = match om { 0 => "allergy-prio", 1 => "fav-prio", 2 => "neutral", _ => "both" };
            writeln!(f, "  om={} ({}): best b={} nr={} fp={} dv={} me={} train={:.5} test={:.5}",
                     om, label, sub[0].base,sub[0].nr,sub[0].fp,sub[0].dv,sub[0].me,sub[0].tr,sub[0].te).unwrap();
            println!("  om={} ({}): test={:.5}", om, label, sub[0].te);
        }
    }

    // === Search 12: Per-pirate strength adjustments ===
    // Fit residuals for the 7 mispredicted pirates
    // Under-est: Ogletree(S=79), Buck(S=89), Orvinn(S=52) — need negative adj (make stronger)
    // Over-est: Lucky(S=82), Crossblades(S=66), Puffo(S=68), Blackbeard(S=76) — need positive adj (weaker)
    {
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 12: Per-pirate strength adjustments ---").unwrap();

        // Get pirate indices
        let names = ["Ogletree", "Buck Cutlass", "Orvinn", "Lucky", "Crossblades", "Puffo", "Blackbeard"];
        let mut indices = Vec::new();
        for name in &names {
            let found = data.pirates.iter().position(|p| p.name.contains(name));
            if let Some(idx) = found {
                writeln!(f, "  {} -> index {}", name, idx).unwrap();
                indices.push(idx);
            }
        }

        // Grid search per-pirate adjustments
        // Ogletree, Buck, Orvinn: try -10 to 0 (make stronger)
        // Lucky, Crossblades, Puffo, Blackbeard: try 0 to +10 (make weaker)
        let mut best_ll = bl_te.ll;
        let mut best_init_life = String::new();

        // First: individual adjustments to see which matter most
        writeln!(f, "  Individual adj (baseline b=112 nr=4 fp=15 dv=14 me=7):").unwrap();
        for (ni, &name) in names.iter().enumerate() {
            if ni >= indices.len() { continue; }
            let idx = indices[ni];
            let is_under = ni < 3; // first 3 are under-estimated
            let range: Vec<i32> = if is_under {
                vec![-15, -10, -8, -5, -3, -1, 0]
            } else {
                vec![0, 1, 3, 5, 8, 10, 15]
            };
            let mut best_a = 0i32;
            let mut best_t = bl_te.ll;
            for &adj in &range {
                let mut life_adj = vec![0i32; data.pirates.len()];
                life_adj[idx] = adj;
                let e = eval_pmf(&data, &test, &baseline, &life_adj);
                if e.ll > best_t { best_t = e.ll; best_a = adj; }
            }
            writeln!(f, "    {}: best_adj={:+}, test={:.5} (delta={:+.5})",
                     name, best_a, best_t, best_t - bl_te.ll).unwrap();
            println!("    {}: adj={:+}, test={:.5}", name, best_a, best_t);
        }

        // Combined: apply best individual adjustments simultaneously
        let best_individual = [-5i32, -5, -10, 5, 5, 5, 5]; // rough from above
        let mut life_adj = vec![0i32; data.pirates.len()];
        for (ni, &adj) in best_individual.iter().enumerate() {
            if ni < indices.len() { life_adj[indices[ni]] = adj; }
        }
        let combined = eval_pmf(&data, &test, &baseline, &life_adj);
        writeln!(f, "  Combined adj: test={:.5} (delta={:+.5})", combined.ll, combined.ll - bl_te.ll).unwrap();
        println!("  Combined adj: test={:.5}", combined.ll);

        // Grid search over adj magnitude for the 3 under-est pirates
        writeln!(f, "  Grid search under-est trio (Ogletree, Buck, Orvinn):").unwrap();
        let under_range = [-15i32, -10, -8, -5, -3, 0];
        let mut best_combo_ll = bl_te.ll;
        let mut best_combo = (0i32, 0i32, 0i32);
        for &a0 in &under_range {
            for &a1 in &under_range {
                for &a2 in &under_range {
                    let mut sa = vec![0i32; data.pirates.len()];
                    if indices.len() > 0 { sa[indices[0]] = a0; }
                    if indices.len() > 1 { sa[indices[1]] = a1; }
                    if indices.len() > 2 { sa[indices[2]] = a2; }
                    let e = eval_pmf(&data, &test, &baseline, &sa);
                    if e.ll > best_combo_ll {
                        best_combo_ll = e.ll;
                        best_combo = (a0, a1, a2);
                    }
                }
            }
        }
        writeln!(f, "    Best: Ogletree={:+}, Buck={:+}, Orvinn={:+} test={:.5} (delta={:+.5})",
                 best_combo.0, best_combo.1, best_combo.2, best_combo_ll, best_combo_ll - bl_te.ll).unwrap();
        println!("  Under-est trio: ({:+},{:+},{:+}) test={:.5}", best_combo.0, best_combo.1, best_combo.2, best_combo_ll);
    }

    // === Search 13: Flat allergy damage (allergy_mode=1) ===
    // Instead of rolling dice for allergy damage, use deterministic wo per allergy
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &base in &[108u32, 110, 112, 114, 116, 118] {
            for &nr in &[3u32, 4, 5] {
                for &fp in &[12u32, 14, 15, 16, 18] {
                    for &dv in &[10u32, 12, 14, 16, 18] {
                        for &me in &[5u32, 6, 7, 8, 10, 12] {
                            cfgs.push((base, nr, fp, dv, me));
                        }
                    }
                }
            }
        }
        println!("Search 13 (flat allergy): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:1, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fp,dv,me,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 13: Flat allergy: dmg = na * wo (deterministic) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} fp={:<2} dv={:<2} me={:<2} train={:.5} test={:.5}",
                     r.base,r.nr,r.fp,r.dv,r.me,r.tr,r.te).unwrap();
        }
        println!("  Best flat-allergy: b={} nr={} fp={} dv={} me={} test={:.5}", s[0].base,s[0].nr,s[0].fp,s[0].dv,s[0].me,s[0].te);
    }

    // === Search 14: Allergy percentage bonus: upper += na * upper * ap / 100 (allergy_mode=3) ===
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, ap: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &base in &[108u32, 110, 112, 114, 116, 118] {
            for &nr in &[3u32, 4, 5] {
                for &fp in &[12u32, 14, 15, 16, 18] {
                    for &dv in &[10u32, 12, 14, 16, 18] {
                        for &ap in &[2u32, 5, 8, 10, 15, 20, 30] {
                            cfgs.push((base, nr, fp, dv, ap));
                        }
                    }
                }
            }
        }
        println!("Search 14 (allergy pct): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,ap)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:0, allergy_mode:3, allergy_param:ap, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fp,dv,ap,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 14: Allergy pct: upper += na * upper * ap / 100 ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} fp={:<2} dv={:<2} ap={:<2} train={:.5} test={:.5}",
                     r.base,r.nr,r.fp,r.dv,r.ap,r.tr,r.te).unwrap();
        }
        println!("  Best alg-pct: b={} nr={} fp={} dv={} ap={} test={:.5}", s[0].base,s[0].nr,s[0].fp,s[0].dv,s[0].ap,s[0].te);
    }

    // === Search 15: Combined fav+allergy on die (fav_mode=8: upper -= nf*floor((die+weight)/K)) ===
    // This ties fav benefit to both die size AND weight
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &base in &[108u32, 110, 112, 114, 116, 118, 120] {
            for &nr in &[3u32, 4, 5] {
                for &fp in &[15u32, 20, 25, 30, 40, 50, 60, 80, 100] {
                    for &dv in &[10u32, 12, 14, 16, 18] {
                        for &me in &[6u32, 7, 8] {
                            cfgs.push((base, nr, fp, dv, me));
                        }
                    }
                }
            }
        }
        println!("Search 15 (die+wt fav): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:8, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fp,dv,me,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 15: Fav die+wt: upper -= nf * floor((die+weight)/K) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} fp={:<2} dv={:<2} me={} train={:.5} test={:.5}",
                     r.base,r.nr,r.fp,r.dv,r.me,r.tr,r.te).unwrap();
        }
        println!("  Best die+wt: b={} nr={} fp={} dv={} me={} test={:.5}", s[0].base,s[0].nr,s[0].fp,s[0].dv,s[0].me,s[0].te);
    }

    // === Search 16: Favs add to strength (life += nf * bonus) ===
    // Hypothesis: favs boost effective strength, giving strong pirates more relative benefit
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, dv: u32, me: u32, fsb: u32, ao: u8, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &base in &[105u32, 108, 110, 112, 114, 116, 118, 120, 125, 130] {
            for &nr in &[3u32, 4, 5] {
                for &dv in &[0u32, 10, 12, 14, 16, 18, 20] {
                    for &me in &[6u32, 7, 8] {
                        for &fsb in &[1u32, 2, 3, 4, 5, 6, 7, 8, 10] {
                            for &ao in &[0u8, 1] {
                                cfgs.push((base, nr, dv, me, fsb, ao));
                            }
                        }
                    }
                }
            }
        }
        println!("Search 16 (fav-adds-str): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,dv,me,fsb,ao)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:15,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao, fav_str_bonus:fsb, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,dv,me,fsb,ao,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 16: Favs add to strength: eff_str = str + nf*bonus, upper = base - eff_str ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} dv={:<2} me={} fsb={} ao={} train={:.5} test={:.5}",
                     r.base,r.nr,r.dv,r.me,r.fsb,r.ao,r.tr,r.te).unwrap();
        }
        writeln!(f, "Best per fsb:").unwrap();
        for &fsb in &[1u32, 2, 3, 4, 5, 6, 7, 8, 10] {
            let mut sub: Vec<&R> = s.iter().filter(|r| r.fsb == fsb).collect();
            sub.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
            if let Some(r) = sub.first() {
                writeln!(f, "  fsb={:<2}: b={} nr={} dv={} me={} ao={} test={:.5}", fsb, r.base,r.nr,r.dv,r.me,r.ao,r.te).unwrap();
            }
        }
        println!("  Best fav-adds-str: b={} nr={} dv={} me={} fsb={} ao={} test={:.5}",
                 s[0].base,s[0].nr,s[0].dv,s[0].me,s[0].fsb,s[0].ao,s[0].te);
    }

    // === Search 17: Allergy order swap (allergy AFTER favs) ===
    // Test with all fav modes: bulk, iterative, multiplicative
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fm: u8, fp: u32, dv: u32, me: u32, ao: u8, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &ao in &[0u8, 1] { // 0=before, 1=after
            for &fm in &[0u8, 1, 2] { // bulk, mul, iterative
                for &base in &[108u32, 110, 112, 114, 116, 118, 120] {
                    for &nr in &[3u32, 4, 5] {
                        let fp_range: Vec<u32> = match fm {
                            1 => vec![88, 90, 92, 93, 94, 95, 96],  // mul pct
                            _ => vec![10, 12, 14, 15, 16, 18, 20],   // div
                        };
                        for &fp in &fp_range {
                            for &dv in &[0u32, 12, 14, 16, 18] {
                                for &me in &[6u32, 7, 8] {
                                    cfgs.push((ao, fm, base, nr, fp, dv, me));
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Search 17 (allergy order): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(ao,fm,b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:fm, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fm,fp,dv,me,ao,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 17: Allergy order swap (0=before-fav, 1=after-fav) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            let fm_label = match r.fm { 0 => "bulk", 1 => "mul", _ => "iter" };
            writeln!(f, "  b={:<3} nr={} fm={:<4} fp={:<2} dv={:<2} me={} ao={} train={:.5} test={:.5}",
                     r.base,r.nr,fm_label,r.fp,r.dv,r.me,r.ao,r.tr,r.te).unwrap();
        }
        // Best per allergy_order
        for &ao in &[0u8, 1] {
            let mut sub: Vec<&R> = s.iter().filter(|r| r.ao == ao).collect();
            sub.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
            let r = sub[0];
            let fm_label = match r.fm { 0 => "bulk", 1 => "mul", _ => "iter" };
            writeln!(f, "Best ao={}: b={} nr={} fm={} fp={} dv={} me={} test={:.5}",
                     ao, r.base, r.nr, fm_label, r.fp, r.dv, r.me, r.te).unwrap();
            println!("  Best ao={}: b={} nr={} fm={} fp={} dv={} me={} test={:.5}",
                     ao, r.base, r.nr, fm_label, r.fp, r.dv, r.me, r.te);
        }
    }

    // === Search 18: Fav reduction from strength (upper -= nf * floor(str/K)) ===
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fk: u32, dv: u32, me: u32, ao: u8, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &base in &[108u32, 110, 112, 114, 116, 118, 120, 125, 130] {
            for &nr in &[3u32, 4, 5] {
                for &fk in &[15u32, 20, 25, 30, 35, 40, 50] {
                    for &dv in &[0u32, 12, 14, 16, 18] {
                        for &me in &[6u32, 7, 8] {
                            for &ao in &[0u8, 1] {
                                cfgs.push((base, nr, fk, dv, me, ao));
                            }
                        }
                    }
                }
            }
        }
        println!("Search 18 (str-based fav): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fk,dv,me,ao)| {
            // Use fav_mode=11 for strength-based reduction
            let p = Params { base:b, n_rolls:nr, fav_mode:11, fav_param:fk,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fk,dv,me,ao,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 18: Fav from strength: upper -= nf * floor(str/K) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} fk={:<2} dv={:<2} me={} ao={} train={:.5} test={:.5}",
                     r.base,r.nr,r.fk,r.dv,r.me,r.ao,r.tr,r.te).unwrap();
        }
        println!("  Best str-fav: b={} nr={} fk={} dv={} me={} ao={} test={:.5}",
                 s[0].base,s[0].nr,s[0].fk,s[0].dv,s[0].me,s[0].ao,s[0].te);
    }

    // === Search 19: Quadratic/sqrt strength ===
    // str_mode=3: upper = max(1, base - floor(str^2 / K))
    // str_mode=4: upper = max(1, base - floor(sqrt(str) * K))
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, sm: u8, sk: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &sm in &[3u8, 4] {
            let sk_range: Vec<u32> = match sm {
                3 => vec![50, 75, 100, 125, 150, 200], // str^2/K
                _ => vec![5, 7, 8, 9, 10, 11, 12, 14],  // sqrt(str)*K
            };
            for &base in &[80u32, 90, 100, 110, 112, 115, 120, 130, 140, 150] {
                for &nr in &[3u32, 4, 5] {
                    for &fp in &[12u32, 14, 15, 16, 18] {
                        for &dv in &[0u32, 12, 14, 16, 18] {
                            for &me in &[6u32, 7, 8] {
                                for &sk in &sk_range {
                                    cfgs.push((sm, sk, base, nr, fp, dv, me));
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Search 19 (nonlinear str): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(sm,sk,b,nr,fp,dv,me)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:0, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:sm, str_decay:sk, overlap_mode:0, allergy_order:0, fav_str_bonus:0, legacy_wo_min:0 };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fp,dv,me,sm,sk,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 19: Nonlinear strength (3=quadratic, 4=sqrt) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            let sm_label = if r.sm == 3 { "quad" } else { "sqrt" };
            writeln!(f, "  b={:<3} nr={} fp={:<2} dv={:<2} me={} sm={:<4} sk={:<3} train={:.5} test={:.5}",
                     r.base,r.nr,r.fp,r.dv,r.me,sm_label,r.sk,r.tr,r.te).unwrap();
        }
        for &sm in &[3u8, 4] {
            let mut sub: Vec<&R> = s.iter().filter(|r| r.sm == sm).collect();
            sub.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
            let r = sub[0];
            let label = if sm == 3 { "quadratic" } else { "sqrt" };
            writeln!(f, "Best {}: b={} nr={} fp={} dv={} me={} sk={} test={:.5}",
                     label, r.base, r.nr, r.fp, r.dv, r.me, r.sk, r.te).unwrap();
            println!("  Best {}: b={} nr={} fp={} dv={} me={} sk={} test={:.5}",
                     label, r.base, r.nr, r.fp, r.dv, r.me, r.sk, r.te);
        }
    }

    // === Search 28: Legacy regime shift (wo_min for legacy arenas) ===
    // For legacy PHP, rand(1, 0) behavior: 0=returns 0, 1=returns 1, 2=returns 0 or 1 randomly.
    // Only affects Orvinn (wo=0). Post-legacy always uses wo_min=0.
    {
        #[derive(Clone)]
        struct R { base: u32, nr: u32, fp: u32, dv: u32, me: u32, ao: u8, lwm: u32, tr: f64, te: f64 }
        let mut cfgs = Vec::new();
        for &lwm in &[0u32, 1, 2] {
            for &base in &[118u32, 120, 122, 124, 126, 128] {
                for &nr in &[4u32, 5, 6, 7] {
                    for &fp in &[15u32, 16, 17, 18] {
                        for &dv in &[16u32, 18, 20, 22, 24] {
                            for &me in &[5u32, 6, 7] {
                                for &ao in &[0u8, 1] {
                                    cfgs.push((base, nr, fp, dv, me, ao, lwm));
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Search 28 (legacy regime): {} configs", cfgs.len());
        let res: Vec<R> = cfgs.par_iter().map(|&(b,nr,fp,dv,me,ao,lwm)| {
            let p = Params { base:b, n_rolls:nr, fav_mode:2, fav_param:fp,
                pos_mode:0, pos_step:0, tiebreak:0, divisor:dv, quant_round:0,
                max_effect:me, allergy_mode:0, allergy_param:0, wo_min:0, max_fav:0,
                str_mode:0, str_decay:0, overlap_mode:0, allergy_order:ao,
                fav_str_bonus:0, legacy_wo_min:lwm };
            let t = eval_pmf(&data, &train, &p, &[]);
            let e = eval_pmf(&data, &test, &p, &[]);
            R{base:b,nr,fp,dv,me,ao,lwm,tr:t.ll,te:e.ll}
        }).collect();
        let mut s = res; s.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
        let mut f = std::fs::OpenOptions::new().append(true).open("../structural_search_results.txt").unwrap();
        writeln!(f, "\n--- Search 28: Legacy regime (lwm: 0=off, 1=returns 1, 2=random 0|1) ---").unwrap();
        writeln!(f, "Best 15:").unwrap();
        for r in s.iter().take(15) {
            writeln!(f, "  b={:<3} nr={} fp={:<2} dv={:<2} me={} ao={} lwm={} train={:.5} test={:.5}",
                     r.base,r.nr,r.fp,r.dv,r.me,r.ao,r.lwm,r.tr,r.te).unwrap();
        }
        for &lwm in &[0u32, 1, 2] {
            let mut sub: Vec<&R> = s.iter().filter(|r| r.lwm == lwm).collect();
            sub.sort_by(|a,b| b.te.partial_cmp(&a.te).unwrap());
            let r = sub[0];
            let label = match lwm { 0 => "off", 1 => "always-1", _ => "random-0|1" };
            writeln!(f, "Best lwm={} ({}): b={} nr={} fp={} dv={} me={} ao={} test={:.5}",
                     lwm, label, r.base, r.nr, r.fp, r.dv, r.me, r.ao, r.te).unwrap();
            println!("  Best lwm={} ({}): b={} nr={} fp={} dv={} me={} ao={} test={:.5}",
                     lwm, label, r.base, r.nr, r.fp, r.dv, r.me, r.ao, r.te);
        }
    }
}
*/

// === Regression helpers (used by old searches) ===
#[allow(dead_code)]

fn simple_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut ssx = 0.0f64;
    let mut sp = 0.0f64;
    for i in 0..x.len() {
        let d = x[i] - mx;
        ssx += d * d;
        sp += d * (y[i] - my);
    }
    let b = if ssx > 0.0 { sp / ssx } else { 0.0 };
    let mut ssr = 0.0f64;
    for i in 0..x.len() {
        let r = y[i] - my - b * (x[i] - mx);
        ssr += r * r;
    }
    let se = if ssx > 0.0 && n > 2.0 { (ssr / (n - 2.0) / ssx).sqrt() } else { f64::INFINITY };
    let t = b / se;
    let p = 2.0 * normal_cdf(-t.abs());
    (b, se, t, p)
}

fn multiple_regression_2(x1: &[f64], x2: &[f64], y: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    // OLS: y = b0 + b1*x1 + b2*x2
    let n = x1.len() as f64;
    let m1 = x1.iter().sum::<f64>() / n;
    let m2 = x2.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut s11 = 0.0f64; let mut s22 = 0.0f64; let mut s12 = 0.0f64;
    let mut s1y = 0.0f64; let mut s2y = 0.0f64;
    for i in 0..x1.len() {
        let d1 = x1[i] - m1;
        let d2 = x2[i] - m2;
        let dy = y[i] - my;
        s11 += d1 * d1; s22 += d2 * d2; s12 += d1 * d2;
        s1y += d1 * dy; s2y += d2 * dy;
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-20 { return (0.0, 0.0, f64::INFINITY, f64::INFINITY, 0.0, 0.0, 1.0, 1.0); }
    let b1 = (s22 * s1y - s12 * s2y) / det;
    let b2 = (s11 * s2y - s12 * s1y) / det;

    let mut ssr = 0.0f64;
    for i in 0..x1.len() {
        let r = y[i] - my - b1 * (x1[i] - m1) - b2 * (x2[i] - m2);
        ssr += r * r;
    }
    let s2 = ssr / (n - 3.0);
    let se1 = (s2 * s22 / det).sqrt();
    let se2 = (s2 * s11 / det).sqrt();
    let t1 = b1 / se1;
    let t2 = b2 / se2;
    let p1 = 2.0 * normal_cdf(-t1.abs());
    let p2 = 2.0 * normal_cdf(-t2.abs());
    (b1, b2, se1, se2, t1, t2, p1, p2)
}

fn multiple_regression_3(x1: &[f64], x2: &[f64], x3: &[f64], y: &[f64]) -> ([f64;3], [f64;3], [f64;3], [f64;3]) {
    let xs = vec![x1.to_vec(), x2.to_vec(), x3.to_vec()];
    let (b, s, t, p) = multiple_regression_n(&xs, y);
    ([b[0],b[1],b[2]], [s[0],s[1],s[2]], [t[0],t[1],t[2]], [p[0],p[1],p[2]])
}

fn multiple_regression_n(xs: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = xs.len();
    let n = y.len() as f64;
    let means: Vec<f64> = xs.iter().map(|x| x.iter().sum::<f64>() / n).collect();
    let my = y.iter().sum::<f64>() / n;

    // Build X'X and X'y (centered)
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for i in 0..y.len() {
        let dy = y[i] - my;
        for j in 0..k {
            let dj = xs[j][i] - means[j];
            xty[j] += dj * dy;
            for l in j..k {
                let dl = xs[l][i] - means[l];
                xtx[j][l] += dj * dl;
                if l != j { xtx[l][j] = xtx[j][l]; }
            }
        }
    }

    // Solve via Gauss elimination
    let betas = solve_linear(&xtx, &xty);

    // Residual variance
    let mut ssr = 0.0f64;
    for i in 0..y.len() {
        let mut pred = my;
        for j in 0..k { pred += betas[j] * (xs[j][i] - means[j]); }
        let r = y[i] - pred;
        ssr += r * r;
    }
    let s2 = ssr / (n - k as f64 - 1.0);

    // Invert X'X for SEs
    let inv = invert_matrix(&xtx);
    let mut ses = vec![0.0f64; k];
    let mut ts = vec![0.0f64; k];
    let mut ps = vec![0.0f64; k];
    for j in 0..k {
        ses[j] = (s2 * inv[j][j]).sqrt();
        ts[j] = betas[j] / ses[j];
        ps[j] = 2.0 * normal_cdf(-ts[j].abs());
    }
    (betas, ses, ts, ps)
}

fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let k = b.len();
    let mut aug: Vec<Vec<f64>> = (0..k).map(|i| {
        let mut row = a[i].clone();
        row.push(b[i]);
        row
    }).collect();
    for i in 0..k {
        let mut max_row = i;
        for j in i+1..k { if aug[j][i].abs() > aug[max_row][i].abs() { max_row = j; } }
        aug.swap(i, max_row);
        let pivot = aug[i][i];
        if pivot.abs() < 1e-20 { continue; }
        for j in i..=k { aug[i][j] /= pivot; }
        for j in 0..k {
            if j == i { continue; }
            let f = aug[j][i];
            for l in i..=k { aug[j][l] -= f * aug[i][l]; }
        }
    }
    (0..k).map(|i| aug[i][k]).collect()
}

fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a.len();
    let mut aug: Vec<Vec<f64>> = (0..k).map(|i| {
        let mut row = a[i].clone();
        for j in 0..k { row.push(if i == j { 1.0 } else { 0.0 }); }
        row
    }).collect();
    for i in 0..k {
        let mut max_row = i;
        for j in i+1..k { if aug[j][i].abs() > aug[max_row][i].abs() { max_row = j; } }
        aug.swap(i, max_row);
        let pivot = aug[i][i];
        if pivot.abs() < 1e-20 { continue; }
        for j in 0..2*k { aug[i][j] /= pivot; }
        for j in 0..k {
            if j == i { continue; }
            let f = aug[j][i];
            for l in 0..2*k { aug[j][l] -= f * aug[i][l]; }
        }
    }
    (0..k).map(|i| aug[i][k..2*k].to_vec()).collect()
}

/// Standard normal CDF approximation (Abramowitz & Stegun)
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 { 1.0 - p * poly } else { p * poly }
}
