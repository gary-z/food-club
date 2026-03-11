mod pirates;

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct HistPirate {
    name: String,
    #[allow(dead_code)]
    odds: u32,
}

#[derive(Deserialize)]
struct HistArena {
    #[allow(dead_code)]
    arena_name: String,
    #[allow(dead_code)]
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
}

fn main() {
    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse");

    let mut appearances: HashMap<String, u32> = HashMap::new();
    let mut wins: HashMap<String, u32> = HashMap::new();
    let mut total_races = 0u32;

    for day in &days {
        for arena in day {
            total_races += 1;
            for hp in &arena.pirates {
                *appearances.entry(hp.name.clone()).or_default() += 1;
                if arena.winner == hp.name {
                    *wins.entry(hp.name.clone()).or_default() += 1;
                }
            }
        }
    }

    let mut pirates: Vec<(String, u32, u32)> = appearances
        .iter()
        .map(|(name, &app)| (name.clone(), app, *wins.get(name).unwrap_or(&0)))
        .collect();
    pirates.sort_by(|a, b| b.1.cmp(&a.1));

    let total_app: u32 = pirates.iter().map(|(_, a, _)| a).sum();
    let total_wins: u32 = pirates.iter().map(|(_, _, w)| w).sum();

    println!("Total races (arena-days): {}", total_races);
    println!("Total pirate appearances: {}", total_app);
    println!("Total wins: {}", total_wins);
    println!("Average win rate: {:.4} (expected: 0.2500)\n", total_wins as f64 / total_app as f64);

    println!("{:<28} {:>8} {:>8} {:>10}", "Pirate", "Appear", "Wins", "WinRate");
    println!("{}", "-".repeat(58));
    for (name, app, w) in &pirates {
        println!("{:<28} {:>8} {:>8} {:>10.4}", name, app, w, *w as f64 / *app as f64);
    }
}
