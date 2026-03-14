mod pirates;

use pirates::{GameData, Pirate};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// Best model params
const BASE: u32 = 112;
const FAV_DIV: u32 = 15;   // bulk: upper -= nf * floor(upper / 15)
const N_ROLLS: u32 = 4;
const DIVISOR: u32 = 14;
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 7;
const MAX_PAYOUT: u32 = 60;
const SIM_ITERATIONS: u32 = 50_000;

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
fn eating_time(pirate: &Pirate, course_indices: &[usize], rng: &mut impl Rng) -> u32 {
    let (nf, na) = course_counts(pirate, course_indices);

    // Allergy damage: reduce effective strength
    let wo = if pirate.weight >= MAX_WEIGHT { 0 } else { ((MAX_WEIGHT - pirate.weight) / 2).min(MAX_EFFECT) };
    let mut strength = pirate.strength;
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

/// Compute win probabilities via MC simulation. Lowest time wins; ties go to later position.
fn arena_win_probs(
    pirates: &[&Pirate],
    course_indices: &[usize],
    iterations: u32,
    seed: u64,
) -> HashMap<String, f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut wins: HashMap<String, u32> = HashMap::new();
    for p in pirates {
        wins.insert(p.name.clone(), 0);
    }

    for _ in 0..iterations {
        let times: Vec<u32> = pirates.iter()
            .map(|p| eating_time(p, course_indices, &mut rng))
            .collect();

        // Lowest time wins. Ties: later position wins.
        let min_time = *times.iter().min().unwrap();
        let mut winner_pos = 0;
        for (pos, &t) in times.iter().enumerate() {
            if t == min_time {
                winner_pos = pos; // last one with min time wins tie
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
    payout_current: u32,
}

fn make_bets(
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
            let mut payout_current = 1u32;
            let mut pirate_names = Vec::with_capacity(arena_indices.len());

            for (j, &ai) in arena_indices.iter().enumerate() {
                let pirate = &pirate_options[j][combo_indices[j]];
                win_prob *= arena_probs[ai].get(&pirate.name).copied().unwrap_or(0.0);
                payout = (payout * pirate.odds).min(MAX_PAYOUT);
                payout_current = (payout_current * pirate.current_odds.unwrap_or(pirate.odds)).min(MAX_PAYOUT);
                pirate_names.push(pirate.name.clone());
            }

            possible_bets.push(Bet {
                pirate_names,
                win_probability: win_prob,
                payout,
                payout_current,
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

fn bet_won(bet: &Bet, arenas: &[HistArena]) -> bool {
    let winners: std::collections::HashSet<&str> = arenas.iter()
        .map(|a| a.winner.as_str())
        .collect();
    bet.pirate_names.iter().all(|name| winners.contains(name.as_str()))
}

// Result for a single day: opening-odds payout, current-odds payout, expected values
struct DayResult {
    payout_opening: u32,
    payout_current: u32,
    ev_opening: f64,
    ev_current: f64,
    has_current_odds: bool,
}

fn print_stats(label: &str, results: &[&DayResult], use_current: bool) {
    if results.is_empty() { return; }
    let n = results.len() as f64;
    let total_payout: f64 = results.iter().map(|r| {
        if use_current { r.payout_current as f64 } else { r.payout_opening as f64 }
    }).sum();
    let total_ev: f64 = results.iter().map(|r| {
        if use_current { r.ev_current } else { r.ev_opening }
    }).sum();
    let avg_payout = total_payout / n;
    let avg_ev = total_ev / n;
    let roi = (avg_payout - 10.0) / 10.0 * 100.0;

    let mut wins = 0u32;
    let mut losses = 0u32;
    for r in results {
        let p = if use_current { r.payout_current } else { r.payout_opening };
        if p > 10 { wins += 1; } else if p < 10 { losses += 1; }
    }

    println!("  {:<35} {:>5} days | Avg payout {:>6.2} | ROI {:>+6.1}% | EV {:>6.2} | Win% {:>5.1}",
        label, results.len(), avg_payout, roi, avg_ev, wins as f64 / n * 100.0);
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

    let results: Vec<DayResult> = historical.par_iter().enumerate().map(|(day_idx, day_arenas)| {
        let arena_probs: Vec<HashMap<String, f64>> = day_arenas.iter().enumerate().map(|(arena_idx, arena)| {
            let arena_pirates: Vec<&Pirate> = arena.pirates.iter()
                .map(|hp| game_data.pirate_by_name(&hp.name)
                    .unwrap_or_else(|| panic!("Unknown pirate: {}", hp.name)))
                .collect();
            let course_indices: Vec<usize> = arena.foods.iter()
                .filter_map(|food| course_map.get(food.as_str()).copied())
                .collect();
            arena_win_probs(&arena_pirates, &course_indices, SIM_ITERATIONS,
                day_idx as u64 * 17 + arena_idx as u64 * 31 + 42)
        }).collect();

        // Make bets using opening odds (standard)
        let bets = make_bets(&arena_probs, day_arenas);

        let has_current = day_arenas[0].pirates[0].current_odds.is_some();

        let mut payout_opening = 0u32;
        let mut payout_current = 0u32;
        let mut ev_opening = 0.0f64;
        let mut ev_current = 0.0f64;
        for b in &bets {
            let won = bet_won(b, day_arenas);
            if won {
                payout_opening += b.payout;
                payout_current += b.payout_current;
            }
            ev_opening += b.win_probability * b.payout as f64;
            ev_current += b.win_probability * b.payout_current as f64;
        }

        DayResult { payout_opening, payout_current, ev_opening, ev_current, has_current_odds: has_current }
    }).collect();

    let all_results: Vec<&DayResult> = results.iter().collect();
    let has_current: Vec<&DayResult> = results.iter().filter(|r| r.has_current_odds).collect();
    let no_current: Vec<&DayResult> = results.iter().filter(|r| !r.has_current_odds).collect();

    println!("=== BACKTEST RESULTS ===\n");

    println!("Opening odds:");
    print_stats("All data", &all_results, false);
    print_stats("Days with current odds", &has_current, false);
    print_stats("Days without current odds", &no_current, false);

    println!("\nCurrent odds (where available):");
    print_stats("Days with current odds", &has_current, true);

    println!("\nDirect comparison (same {} days):", has_current.len());
    print_stats("Opening odds", &has_current, false);
    print_stats("Current odds", &has_current, true);

    // Detailed: how much do current odds differ from opening?
    let mut higher = 0u32;
    let mut lower = 0u32;
    let mut same = 0u32;
    let mut total_open_ev = 0.0f64;
    let mut total_curr_ev = 0.0f64;
    for r in &has_current {
        total_open_ev += r.ev_opening;
        total_curr_ev += r.ev_current;
        if r.payout_current > r.payout_opening { higher += 1; }
        else if r.payout_current < r.payout_opening { lower += 1; }
        else { same += 1; }
    }
    println!("\n  Days where current payout > opening: {}", higher);
    println!("  Days where current payout < opening: {}", lower);
    println!("  Days where current payout = opening: {}", same);
    println!("  Avg EV opening: {:.3}", total_open_ev / has_current.len() as f64);
    println!("  Avg EV current: {:.3}", total_curr_ev / has_current.len() as f64);
}
