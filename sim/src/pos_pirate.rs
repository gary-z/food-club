mod pirates;

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct HistPirate {
    name: String,
    odds: u32,
}

#[derive(Deserialize)]
struct HistArena {
    #[allow(dead_code)]
    arena_name: String,
    #[allow(dead_code)]
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    #[allow(dead_code)]
    winner: String,
}

fn main() {
    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse");

    // pirate_name -> [count_at_pos0, count_at_pos1, count_at_pos2, count_at_pos3]
    let mut pos_counts: HashMap<String, [u32; 4]> = HashMap::new();
    let mut total_apps: HashMap<String, u32> = HashMap::new();

    for day in &days {
        for arena in day {
            for (pos, hp) in arena.pirates.iter().enumerate() {
                let entry = pos_counts.entry(hp.name.clone()).or_insert([0; 4]);
                entry[pos] += 1;
                *total_apps.entry(hp.name.clone()).or_default() += 1;
            }
        }
    }

    // Sort by total appearances descending (all should be equal at 4835)
    let mut pirates: Vec<(String, [u32; 4], u32)> = pos_counts
        .into_iter()
        .map(|(name, counts)| {
            let total = *total_apps.get(&name).unwrap();
            (name, counts, total)
        })
        .collect();
    pirates.sort_by(|a, b| a.0.cmp(&b.0));

    println!("{:<28} {:>6} {:>8} {:>8} {:>8} {:>8}", "Pirate", "total", "pos0%", "pos1%", "pos2%", "pos3%");
    println!("{}", "-".repeat(72));
    for (name, counts, total) in &pirates {
        println!("{:<28} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            name, total,
            counts[0] as f64 / *total as f64 * 100.0,
            counts[1] as f64 / *total as f64 * 100.0,
            counts[2] as f64 / *total as f64 * 100.0,
            counts[3] as f64 / *total as f64 * 100.0,
        );
    }

    // Also show: for each pirate, what's their average odds when at each position?
    println!("\n\nAverage odds by position for each pirate:");
    let mut odds_by_pos: HashMap<String, [Vec<u32>; 4]> = HashMap::new();
    for day in &days {
        for arena in day {
            for (pos, hp) in arena.pirates.iter().enumerate() {
                let entry = odds_by_pos.entry(hp.name.clone())
                    .or_insert_with(|| [Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
                entry[pos].push(hp.odds);
            }
        }
    }

    let mut odds_pirates: Vec<(String, [f64; 4])> = odds_by_pos
        .into_iter()
        .map(|(name, vecs)| {
            let avgs: [f64; 4] = [
                vecs[0].iter().map(|&x| x as f64).sum::<f64>() / vecs[0].len().max(1) as f64,
                vecs[1].iter().map(|&x| x as f64).sum::<f64>() / vecs[1].len().max(1) as f64,
                vecs[2].iter().map(|&x| x as f64).sum::<f64>() / vecs[2].len().max(1) as f64,
                vecs[3].iter().map(|&x| x as f64).sum::<f64>() / vecs[3].len().max(1) as f64,
            ];
            (name, avgs)
        })
        .collect();
    odds_pirates.sort_by(|a, b| a.0.cmp(&b.0));

    println!("{:<28} {:>8} {:>8} {:>8} {:>8}", "Pirate", "pos0", "pos1", "pos2", "pos3");
    println!("{}", "-".repeat(64));
    for (name, avgs) in &odds_pirates {
        println!("{:<28} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            name, avgs[0], avgs[1], avgs[2], avgs[3]);
    }
}
