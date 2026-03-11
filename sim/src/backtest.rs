mod pirates;

use pirates::{GameData, Pirate};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// --- PosMul model (best known) ---
const MAX_WEIGHT: u32 = 221;
const MAX_EFFECT: u32 = 10;
const BASE: u32 = 110;
const FAV_PCT: f64 = 0.93;
const ZV_BONUS: u32 = 4;
const POS_PCT: f64 = 7.0;
const MAX_PAYOUT: u32 = 60;
const SIM_ITERATIONS: u32 = 50_000;

fn weight_offset(pirate_weight: u32) -> u32 {
    if pirate_weight >= MAX_WEIGHT {
        return 0;
    }
    ((MAX_WEIGHT - pirate_weight) / 2).min(MAX_EFFECT)
}

fn roll(rng: &mut impl Rng, n: u32) -> i64 {
    if n == 0 { 0 } else { rng.gen_range(1..=n) as i64 }
}

fn posmul_score(pirate: &Pirate, course_indices: &[usize], pos: u32, rng: &mut impl Rng) -> i64 {
    let n_allergy = course_indices.iter()
        .filter(|&&c| pirate.allergy_courses.contains(&c)).count() as u32;
    let n_fav = course_indices.iter()
        .filter(|&&c| pirate.favorite_courses.contains(&c) && !pirate.allergy_courses.contains(&c))
        .count() as u32;

    let wo = weight_offset(pirate.weight);
    let mut life = pirate.strength as i64;
    for _ in 0..n_allergy {
        life -= roll(rng, wo);
    }
    if n_fav == 0 && n_allergy == 0 {
        life += roll(rng, ZV_BONUS);
    }
    let upper_base = ((BASE as i64 - life).max(1) as f64
        * FAV_PCT.powi(n_fav as i32))
        .max(1.0);
    let pos_mul = (100.0 - pos as f64 * POS_PCT) / 100.0;
    let upper = (upper_base * pos_mul).max(1.0) as u32;
    -roll(rng, upper) - roll(rng, upper) - roll(rng, upper)
}

/// Compute win probabilities for 4 pirates in a specific arena with specific foods.
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
        let scores: Vec<(&Pirate, i64)> = pirates.iter().enumerate()
            .map(|(pos, p)| (*p, posmul_score(p, course_indices, pos as u32, &mut rng)))
            .collect();
        let max_score = scores.iter().map(|(_, s)| *s).max().unwrap();
        let tied: Vec<&&Pirate> = scores.iter()
            .filter(|(_, s)| *s == max_score)
            .map(|(p, _)| p)
            .collect();
        let winner = tied[rng.gen_range(0..tied.len())];
        *wins.get_mut(&winner.name).unwrap() += 1;
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
    arena_name: String,
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
}

/// A bet: list of (arena_index, pirate_name), combined payout, combined win probability.
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

    // Enumerate all subsets of arenas (1 to n), and all pirate picks per arena
    for mask in 1u32..(1 << n) {
        let arena_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        // Build pirate options per selected arena
        let pirate_options: Vec<&[HistPirate]> = arena_indices.iter()
            .map(|&i| arenas[i].pirates.as_slice())
            .collect();

        // Enumerate all combinations using iterative Cartesian product
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

            // Advance combo indices
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

    // Sort by expected value descending, take top 10
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
    println!("Model: PosMul (max_w={MAX_WEIGHT}, max_e={MAX_EFFECT}, base={BASE}, fav%={}, zv={ZV_BONUS}, pos%={POS_PCT})",
        (FAV_PCT * 100.0) as u32);
    println!("Sim iterations per arena: {SIM_ITERATIONS}");
    println!("Max payout cap: {MAX_PAYOUT}");
    println!();

    // Process each day in parallel
    let results: Vec<(u32, f64)> = historical.par_iter().enumerate().map(|(day_idx, day_arenas)| {
        // For each arena, compute H19 win probabilities
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

    // Each day costs 10 bets of 1 unit each
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

    // Distribution of daily outcomes
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
