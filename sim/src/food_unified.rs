mod pirates;

use pirates::{GameData, Pirate, load_historical_matches, HistMatch};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::io::Write;

const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 7;
const SIM_ITERS: u32 = 10_000;

fn roll(rng: &mut impl Rng, n: u32) -> u32 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) }
}

fn course_counts(pirate: &Pirate, courses: &[usize]) -> (u32, u32) {
    let mut nf = 0u32;
    let mut na = 0u32;
    for &c in courses {
        let is_f = pirate.favorite_courses.contains(&c);
        let is_a = pirate.allergy_courses.contains(&c);
        if is_a { na += 1; } else if is_f { nf += 1; }
    }
    (nf, na)
}

fn allergy_damage(pirate: &Pirate, na: u32, rng: &mut impl Rng) -> u32 {
    let wo = if pirate.weight >= MAX_WEIGHT { 0 }
             else { ((MAX_WEIGHT - pirate.weight) / 2).min(MAX_EFFECT) };
    let mut dmg = 0u32;
    for _ in 0..na { dmg += roll(rng, wo); }
    dmg
}

#[derive(Clone, Copy)]
struct Params {
    base: u32,
    n_rolls: u32,
    fav_mode: u8,    // 0=bulk, 1=multiplicative
    fav_param: u32,  // FAV_DIV for bulk, FAV_PCT for multiplicative
    pos_mode: u8,    // 0=none, 1=mul_after, 2=mul_before, 3=add_after, 4=add_before
    pos_step: u32,
    tiebreak: u8,    // 0=later, 1=earlier, 2=random
    divisor: u32,    // 0=none
}

fn compute_upper(raw_upper: u32, nf: u32, pos: usize, p: &Params) -> u32 {
    match (p.fav_mode, p.pos_mode) {
        // Bulk fav
        (0, 0) => { // no pos
            let red = raw_upper / p.fav_param;
            raw_upper.saturating_sub(nf * red).max(1)
        }
        (0, 1) => { // bulk fav, mul pos after
            let red = raw_upper / p.fav_param;
            let u = raw_upper.saturating_sub(nf * red).max(1);
            (u * (100 - pos as u32 * p.pos_step) / 100).max(1)
        }
        (0, 2) => { // bulk fav, mul pos before
            let u = (raw_upper * (100 - pos as u32 * p.pos_step) / 100).max(1);
            let red = u / p.fav_param;
            u.saturating_sub(nf * red).max(1)
        }
        (0, 3) => { // bulk fav, add pos after
            let red = raw_upper / p.fav_param;
            let u = raw_upper.saturating_sub(nf * red).max(1);
            u.saturating_sub(pos as u32 * p.pos_step).max(1)
        }
        (0, 4) => { // bulk fav, add pos before
            let u = raw_upper.saturating_sub(pos as u32 * p.pos_step).max(1);
            let red = u / p.fav_param;
            u.saturating_sub(nf * red).max(1)
        }
        // Multiplicative fav
        (1, 0) => { // no pos
            let mut u = raw_upper as f64;
            for _ in 0..nf { u *= p.fav_param as f64 / 100.0; }
            (u.floor() as u32).max(1)
        }
        (1, 1) => { // mul fav, mul pos after
            let mut u = raw_upper as f64;
            for _ in 0..nf { u *= p.fav_param as f64 / 100.0; }
            let u = (u.floor() as u32).max(1);
            (u * (100 - pos as u32 * p.pos_step) / 100).max(1)
        }
        (1, 2) => { // mul fav, mul pos before
            let u = (raw_upper * (100 - pos as u32 * p.pos_step) / 100).max(1);
            let mut uf = u as f64;
            for _ in 0..nf { uf *= p.fav_param as f64 / 100.0; }
            (uf.floor() as u32).max(1)
        }
        (1, 3) => { // mul fav, add pos after
            let mut u = raw_upper as f64;
            for _ in 0..nf { u *= p.fav_param as f64 / 100.0; }
            let u = (u.floor() as u32).max(1);
            u.saturating_sub(pos as u32 * p.pos_step).max(1)
        }
        (1, 4) => { // mul fav, add pos before
            let u = raw_upper.saturating_sub(pos as u32 * p.pos_step).max(1);
            let mut uf = u as f64;
            for _ in 0..nf { uf *= p.fav_param as f64 / 100.0; }
            (uf.floor() as u32).max(1)
        }
        _ => raw_upper.max(1),
    }
}

fn sim_arena(pirates: &[&Pirate], courses: &[usize], p: &Params, iters: u32, seed: u64) -> [f64; 4] {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Precompute per-pirate fav/allergy counts
    let counts: [(u32, u32); 4] = std::array::from_fn(|i| course_counts(pirates[i], courses));

    let mut wins = [0u32; 4];
    for _ in 0..iters {
        let mut times = [0u32; 4];
        for pos in 0..4 {
            let dmg = allergy_damage(pirates[pos], counts[pos].1, &mut rng);
            let eff_str = pirates[pos].strength.saturating_sub(dmg);
            let raw_upper = if p.base > eff_str { p.base - eff_str } else { 1 };
            let upper = compute_upper(raw_upper, counts[pos].0, pos, p);

            let mut time = 0u32;
            for _ in 0..p.n_rolls { time += roll(&mut rng, upper); }
            if p.divisor > 0 { time /= p.divisor; }
            times[pos] = time;
        }

        let min_time = *times.iter().min().unwrap();
        let winner = match p.tiebreak {
            0 => { // later
                let mut w = 0;
                for i in 0..4 { if times[i] <= min_time { w = i; } }
                w
            }
            1 => { // earlier
                times.iter().position(|&t| t == min_time).unwrap()
            }
            _ => { // random
                let tied: Vec<usize> = (0..4).filter(|&i| times[i] == min_time).collect();
                tied[rng.gen_range(0..tied.len())]
            }
        };
        wins[winner] += 1;
    }
    std::array::from_fn(|i| wins[i] as f64 / iters as f64)
}

fn eval_ll(data: &GameData, matches: &[Vec<HistMatch>], p: &Params) -> f64 {
    let flat: Vec<&HistMatch> = matches.iter().flat_map(|d| d.iter()).collect();
    let sum_ll: f64 = flat.par_iter().enumerate().map(|(idx, m)| {
        let pirates: Vec<&Pirate> = m.pirate_indices.iter().map(|&i| &data.pirates[i]).collect();
        let probs = sim_arena(
            &[pirates[0], pirates[1], pirates[2], pirates[3]],
            &m.course_indices, p, SIM_ITERS,
            idx as u64 * 7 + 999,
        );
        probs[m.winner_pos].max(1e-10).ln()
    }).sum();
    sum_ll / flat.len() as f64
}

fn main() {
    let pj = std::fs::read_to_string("../pirates.json").unwrap();
    let data = GameData::load(&pj);
    let hj = std::fs::read_to_string("../historical_matches.json").unwrap();
    let hist = load_historical_matches(&data, &hj);

    let train = &hist[..4835.min(hist.len())];
    let test = if hist.len() > 4835 { &hist[4835..] } else { &[] };
    let n_train: usize = train.iter().map(|d| d.len()).sum();
    let n_test: usize = test.iter().map(|d| d.len()).sum();
    println!("Train: {} days ({} arenas), Test: {} days ({} arenas)", train.len(), n_train, test.len(), n_test);
    println!("Sim iters: {}\n", SIM_ITERS);

    let header = format!("{:<65} {:>10} {:>10}", "Model", "Train LL", "Test LL");
    println!("{}", header);
    println!("{}", "-".repeat(87));
    std::io::stdout().flush().unwrap();

    let mut configs: Vec<(&str, Params)> = Vec::new();

    // Baselines
    configs.push(("M1: Bulk+Quant14+LaterTB (b=112,fd=15,r=4)", Params {
        base: 112, n_rolls: 4, fav_mode: 0, fav_param: 15,
        pos_mode: 0, pos_step: 0, tiebreak: 0, divisor: 14,
    }));
    configs.push(("M2: MulFav+PosMul+RandTB (b=109,f93,r=3,pp=7)", Params {
        base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
        pos_mode: 1, pos_step: 7, tiebreak: 2, divisor: 0,
    }));

    // V1: PosMul + Later tiebreak (BOTH mechanisms)
    for &(b, pp) in &[(109,5),(109,6),(109,7),(110,5),(110,6),(110,7),(111,5),(111,6)] {
        let s = Box::leak(format!("V1: MulFav+PosMul+LaterTB (b={},f93,r=3,pp={})", b, pp).into_boxed_str());
        configs.push((s, Params {
            base: b, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 1, pos_step: pp, tiebreak: 0, divisor: 0,
        }));
    }

    // V2: PosMul + Quantization + Later tiebreak
    for &(pp, d) in &[(4,7),(5,7),(4,10),(5,10),(3,14),(4,14)] {
        let s = Box::leak(format!("V2: MulFav+PosMul+Quant+LaterTB (b=109,f93,r=3,pp={},d={})", pp, d).into_boxed_str());
        configs.push((s, Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 1, pos_step: pp, tiebreak: 0, divisor: d,
        }));
    }

    // V3: Additive pos, both tiebreaks
    for &step in &[2,3,4] {
        let s = Box::leak(format!("V3: MulFav+PosAdd+LaterTB (b=109,f93,r=3,step={})", step).into_boxed_str());
        configs.push((s, Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 3, pos_step: step, tiebreak: 0, divisor: 0,
        }));
        let s2 = Box::leak(format!("V3: MulFav+PosAdd+RandTB (b=109,f93,r=3,step={})", step).into_boxed_str());
        configs.push((s2, Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 3, pos_step: step, tiebreak: 2, divisor: 0,
        }));
    }

    // V4: Pos BEFORE favs (mul)
    for &pp in &[6,7,8] {
        let s = Box::leak(format!("V4: MulFav+PosBefore+LaterTB (b=109,f93,r=3,pp={})", pp).into_boxed_str());
        configs.push((s, Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 2, pos_step: pp, tiebreak: 0, divisor: 0,
        }));
        let s2 = Box::leak(format!("V4: MulFav+PosBefore+RandTB (b=109,f93,r=3,pp={})", pp).into_boxed_str());
        configs.push((s2, Params {
            base: 109, n_rolls: 3, fav_mode: 1, fav_param: 93,
            pos_mode: 2, pos_step: pp, tiebreak: 2, divisor: 0,
        }));
    }

    // V5: Bulk fav + PosMul + Later TB (no quant)
    for &(b, pp) in &[(111,5),(111,6),(111,7),(112,5),(112,6),(112,7),(113,5),(113,6)] {
        let s = Box::leak(format!("V5: BulkFav+PosMul+LaterTB (b={},fd=15,r=4,pp={})", b, pp).into_boxed_str());
        configs.push((s, Params {
            base: b, n_rolls: 4, fav_mode: 0, fav_param: 15,
            pos_mode: 1, pos_step: pp, tiebreak: 0, divisor: 0,
        }));
    }

    // V6: Bulk fav + PosMul + Quant + Later TB
    for &(pp, d) in &[(3,14),(4,14),(5,14),(3,10),(4,10)] {
        let s = Box::leak(format!("V6: BulkFav+PosMul+Quant+LaterTB (b=112,fd=15,r=4,pp={},d={})", pp, d).into_boxed_str());
        configs.push((s, Params {
            base: 112, n_rolls: 4, fav_mode: 0, fav_param: 15,
            pos_mode: 1, pos_step: pp, tiebreak: 0, divisor: d,
        }));
    }

    // V7: Additive pos before bulk fav + quant
    for &step in &[2,3,4] {
        let s = Box::leak(format!("V7: BulkFav+AddPosBefore+LaterTB (b=112,fd=15,r=4,step={},d=14)", step).into_boxed_str());
        configs.push((s, Params {
            base: 112, n_rolls: 4, fav_mode: 0, fav_param: 15,
            pos_mode: 4, pos_step: step, tiebreak: 0, divisor: 14,
        }));
    }

    println!("Evaluating {} models...\n", configs.len());
    std::io::stdout().flush().unwrap();

    for (name, params) in &configs {
        let train_ll = eval_ll(&data, train, params);
        let test_ll = if !test.is_empty() { eval_ll(&data, test, params) } else { 0.0 };
        println!("{:<65} {:>10.5} {:>10.5}", name, train_ll, test_ll);
        std::io::stdout().flush().unwrap();
    }

    println!("\nBaseline (uniform): {:.5}", (0.25f64).ln());
}
