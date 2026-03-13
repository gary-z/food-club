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

#[derive(Deserialize)]
struct HistPirate {
    name: String,
    odds: u32,
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

fn get_payout(bet: &Bet, arenas: &[HistArena]) -> u32 {
    let winners: std::collections::HashSet<&str> = arenas.iter()
        .map(|a| a.winner.as_str())
        .collect();
    let all_correct = bet.pirate_names.iter().all(|name| winners.contains(name.as_str()));
    if all_correct { bet.payout } else { 0 }
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
    println!("Max payout cap: {MAX_PAYOUT}");
    println!();

    let results: Vec<(u32, f64)> = historical.par_iter().enumerate().map(|(day_idx, day_arenas)| {
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

        let bets = make_bets(&arena_probs, day_arenas);
        let total_payout: u32 = bets.iter().map(|b| get_payout(b, day_arenas)).sum();
        let expected_payout: f64 = bets.iter().map(|b| b.win_probability * b.payout as f64).sum();

        (total_payout, expected_payout)
    }).collect();

    let total_days = results.len() as f64;
    let total_actual: f64 = results.iter().map(|(p, _)| *p as f64).sum();
    let total_expected: f64 = results.iter().map(|(_, e)| *e).sum();

    let avg_payout = total_actual / total_days;
    let avg_expected = total_expected / total_days;
    let avg_net_gain = avg_payout - 10.0;

    println!("=== BACKTEST RESULTS ({} days) ===", results.len());
    println!("Average payout per day:   {:.2}", avg_payout);
    println!("Average cost per day:     10.00");
    println!("Average net gain per day: {:.2}", avg_net_gain);
    println!("Average ROI:              {:.1}%", avg_net_gain / 10.0 * 100.0);
    println!("Expected payout per day:  {:.2}", avg_expected);
    println!("Expected net gain:        {:.2}", avg_expected - 10.0);

    let mut wins = 0;
    let mut losses = 0;
    let mut breakeven = 0;
    for (payout, _) in &results {
        match (*payout as i32 - 10).signum() {
            1 => wins += 1,
            -1 => losses += 1,
            _ => breakeven += 1,
        }
    }
    println!("\nWinning days:   {} ({:.1}%)", wins, wins as f64 / total_days * 100.0);
    println!("Losing days:    {} ({:.1}%)", losses, losses as f64 / total_days * 100.0);
    println!("Breakeven days: {} ({:.1}%)", breakeven, breakeven as f64 / total_days * 100.0);
}
