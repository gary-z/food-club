mod pirates;

use pirates::GameData;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

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
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
}

#[derive(Default)]
struct Record {
    appearances: u32,
    wins: u32,
}

fn main() {
    let pirates_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = GameData::load(&pirates_json);

    let hist_json =
        std::fs::read_to_string("../historical_matches.json").expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse historical_matches.json");

    let course_idx = data.course_name_to_index();

    // pirate_name -> (n_fav, n_allergy) -> Record
    let mut stats: HashMap<String, BTreeMap<(u32, u32), Record>> = HashMap::new();

    for day in &days {
        for arena in day {
            // Resolve food names to course indices
            let courses: Vec<usize> = arena
                .foods
                .iter()
                .filter_map(|food| course_idx.get(food.as_str()).copied())
                .collect();

            if courses.len() != 10 {
                // skip arenas with unresolvable foods
                continue;
            }

            for hp in &arena.pirates {
                let pirate = match data.pirate_by_name(&hp.name) {
                    Some(p) => p,
                    None => continue,
                };

                // Count favorites (excluding overlaps with allergies) and allergies
                let n_allergy = courses
                    .iter()
                    .filter(|&&c| pirate.allergy_courses.contains(&c))
                    .count() as u32;
                let n_fav = courses
                    .iter()
                    .filter(|&&c| {
                        pirate.favorite_courses.contains(&c) && !pirate.allergy_courses.contains(&c)
                    })
                    .count() as u32;

                let key = (n_fav, n_allergy);
                let rec = stats
                    .entry(hp.name.clone())
                    .or_default()
                    .entry(key)
                    .or_default();
                rec.appearances += 1;
                if arena.winner == hp.name {
                    rec.wins += 1;
                }
            }
        }
    }

    // Sort pirates by weight (heaviest first)
    let mut pirate_list: Vec<&pirates::Pirate> = data.pirates.iter().collect();
    pirate_list.sort_by(|a, b| b.weight.cmp(&a.weight));

    for pirate in &pirate_list {
        let buckets = match stats.get(&pirate.name) {
            Some(b) => b,
            None => continue,
        };

        println!(
            "\n{} (weight={}, strength={})",
            pirate.name, pirate.weight, pirate.strength
        );
        println!(
            "  {:>5} {:>5}  {:>6} {:>5} {:>8}",
            "n_fav", "n_all", "appear", "wins", "win_rate"
        );
        println!("  {}", "-".repeat(38));

        for (&(n_fav, n_allergy), rec) in buckets {
            let rate = if rec.appearances > 0 {
                rec.wins as f64 / rec.appearances as f64
            } else {
                0.0
            };
            println!(
                "  {:>5} {:>5}  {:>6} {:>5} {:>8.4}",
                n_fav, n_allergy, rec.appearances, rec.wins, rate
            );
        }
    }
}
