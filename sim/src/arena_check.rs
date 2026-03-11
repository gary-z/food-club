mod pirates;

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
    arena_name: String,
    foods: Vec<String>,
    pirates: Vec<HistPirate>,
    winner: String,
}

fn chi_squared_p_approx(chi2: f64, df: u32) -> f64 {
    // Wilson-Hilferty approximation for chi-squared CDF
    if df == 0 { return 1.0; }
    let k = df as f64;
    let z = (chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k));
    let z = z / (2.0 / (9.0 * k)).sqrt();
    // Standard normal CDF approximation
    let p = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    p // this is P(X < chi2), so p-value = 1 - p
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn main() {
    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse");

    // Collect all arena names in order they appear
    let mut arena_names: Vec<String> = Vec::new();
    if let Some(first_day) = days.first() {
        for arena in first_day {
            arena_names.push(arena.arena_name.clone());
        }
    }
    let n_arenas = arena_names.len();
    println!("Arenas: {:?}\n", arena_names);
    println!("Total days: {}\n", days.len());

    // === CHECK 1: Pirate win rates by arena ===
    println!("=== PIRATE WIN RATES BY ARENA ===");
    // pirate_name -> arena_idx -> (appearances, wins)
    let mut pirate_arena: HashMap<String, Vec<(u32, u32)>> = HashMap::new();

    for day in &days {
        for (ai, arena) in day.iter().enumerate() {
            for hp in &arena.pirates {
                let entry = pirate_arena.entry(hp.name.clone()).or_insert_with(|| vec![(0, 0); n_arenas]);
                if ai < n_arenas {
                    entry[ai].0 += 1;
                    if arena.winner == hp.name {
                        entry[ai].1 += 1;
                    }
                }
            }
        }
    }

    // For each pirate, chi-squared test of win rate independence across arenas
    let mut pirate_results: Vec<(String, f64, f64, u32)> = Vec::new();
    for (name, arena_stats) in &pirate_arena {
        let total_app: u32 = arena_stats.iter().map(|(a, _)| a).sum();
        let total_wins: u32 = arena_stats.iter().map(|(_, w)| w).sum();
        if total_app < 100 { continue; }
        let overall_rate = total_wins as f64 / total_app as f64;

        let mut chi2 = 0.0;
        let mut df = 0u32;
        for (app, wins) in arena_stats {
            if *app < 5 { continue; }
            let expected_wins = *app as f64 * overall_rate;
            let expected_losses = *app as f64 * (1.0 - overall_rate);
            if expected_wins < 1.0 || expected_losses < 1.0 { continue; }
            chi2 += (*wins as f64 - expected_wins).powi(2) / expected_wins;
            chi2 += ((*app - *wins) as f64 - expected_losses).powi(2) / expected_losses;
            df += 1;
        }
        if df > 1 {
            df -= 1; // df = (categories - 1)
            let p_val = 1.0 - chi_squared_p_approx(chi2, df);
            pirate_results.push((name.clone(), chi2, p_val, df));
        }
    }

    pirate_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("{:<28} {:>8} {:>4} {:>10}", "Pirate", "Chi2", "df", "p-value");
    println!("{}", "-".repeat(54));
    for (name, chi2, p, df) in &pirate_results {
        let sig = if *p < 0.05 { " *" } else { "" };
        println!("{:<28} {:>8.2} {:>4} {:>10.4}{}", name, chi2, df, p, sig);
    }

    // Print per-arena win rates for top 5 most "significant" pirates
    println!("\nPer-arena win rates for pirates with lowest p-values:");
    for (name, _, _, _) in pirate_results.iter().take(5) {
        let stats = &pirate_arena[name];
        print!("  {:<24}", name);
        for (ai, aname) in arena_names.iter().enumerate() {
            let (app, wins) = stats[ai];
            if app > 0 {
                print!("  {}:{:.3}({})", &aname[..aname.len().min(8)], wins as f64 / app as f64, app);
            }
        }
        println!();
    }

    // === CHECK 2: Food appearance rates by arena ===
    println!("\n\n=== FOOD APPEARANCE RATES BY ARENA ===");
    // food_name -> arena_idx -> count
    let mut food_arena: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut arena_totals = vec![0u32; n_arenas];

    for day in &days {
        for (ai, arena) in day.iter().enumerate() {
            if ai >= n_arenas { continue; }
            arena_totals[ai] += 1;
            for food in &arena.foods {
                let entry = food_arena.entry(food.clone()).or_insert_with(|| vec![0; n_arenas]);
                entry[ai] += 1;
            }
        }
    }

    let total_days = days.len() as f64;
    let mut food_results: Vec<(String, f64, f64, u32)> = Vec::new();
    for (food, counts) in &food_arena {
        let total: u32 = counts.iter().sum();
        if total < 20 { continue; }
        let overall_rate = total as f64 / (total_days * n_arenas as f64);

        let mut chi2 = 0.0;
        let mut df = 0u32;
        for (ai, &count) in counts.iter().enumerate() {
            let expected = arena_totals[ai] as f64 * 10.0 * overall_rate;
            // Each arena has 10 food slots per day, but foods are distinct per arena
            // Simpler: expected = total / n_arenas
            let expected = total as f64 / n_arenas as f64;
            if expected < 1.0 { continue; }
            chi2 += (count as f64 - expected).powi(2) / expected;
            df += 1;
        }
        if df > 1 {
            df -= 1;
            let p_val = 1.0 - chi_squared_p_approx(chi2, df);
            food_results.push((food.clone(), chi2, p_val, df));
        }
    }

    food_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("{:<30} {:>8} {:>4} {:>10}", "Food", "Chi2", "df", "p-value");
    println!("{}", "-".repeat(56));
    for (name, chi2, p, df) in &food_results {
        let sig = if *p < 0.05 { " *" } else { "" };
        println!("{:<30} {:>8.2} {:>4} {:>10.4}{}", name, chi2, df, p, sig);
    }

    let sig_foods = food_results.iter().filter(|(_, _, p, _)| *p < 0.05).count();
    let sig_pirates = pirate_results.iter().filter(|(_, _, p, _)| *p < 0.05).count();
    println!("\n=== SUMMARY ===");
    println!("Pirates with p < 0.05: {} / {}", sig_pirates, pirate_results.len());
    println!("Foods with p < 0.05:   {} / {}", sig_foods, food_results.len());
    println!("Expected by chance at 5%: ~{:.0} pirates, ~{:.0} foods",
        pirate_results.len() as f64 * 0.05, food_results.len() as f64 * 0.05);
}
