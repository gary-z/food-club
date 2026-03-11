mod pirates;

use pirates::GameData;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

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

fn chi2_pvalue(chi2: f64, df: u32) -> f64 {
    if df == 0 { return 1.0; }
    let k = df as f64;
    let z = (chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k));
    let z = z / (2.0 / (9.0 * k)).sqrt();
    let p = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    1.0 - p
}

fn erf(x: f64) -> f64 {
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
    let pirates_json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let game_data = GameData::load(&pirates_json);
    let course_idx = game_data.course_name_to_index();

    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse");

    // ============================================================
    // CHECK 1: Win rate by position within arena (slot 0-3)
    // ============================================================
    println!("=== CHECK 1: WIN RATE BY POSITION WITHIN ARENA ===\n");

    // pirate_name -> position -> (appearances, wins)
    let mut pos_stats: HashMap<String, Vec<(u32, u32)>> = HashMap::new();

    for day in &days {
        for arena in day {
            for (pos, hp) in arena.pirates.iter().enumerate() {
                let entry = pos_stats.entry(hp.name.clone()).or_insert_with(|| vec![(0, 0); 4]);
                entry[pos].0 += 1;
                if arena.winner == hp.name {
                    entry[pos].1 += 1;
                }
            }
        }
    }

    // Aggregate across all pirates
    let mut total_by_pos = vec![(0u32, 0u32); 4];
    for stats in pos_stats.values() {
        for (pos, (app, wins)) in stats.iter().enumerate() {
            total_by_pos[pos].0 += app;
            total_by_pos[pos].1 += wins;
        }
    }
    println!("  Aggregate across all pirates:");
    for (pos, (app, wins)) in total_by_pos.iter().enumerate() {
        println!("    Position {}: {}/{} = {:.4}", pos, wins, app, *wins as f64 / *app as f64);
    }

    // Chi-squared per pirate
    let mut pirate_pos_results: Vec<(String, f64, f64)> = Vec::new();
    for (name, stats) in &pos_stats {
        let total_app: u32 = stats.iter().map(|(a, _)| a).sum();
        let total_wins: u32 = stats.iter().map(|(_, w)| w).sum();
        if total_app < 100 { continue; }
        let overall_rate = total_wins as f64 / total_app as f64;
        let mut chi2 = 0.0;
        let mut df = 0u32;
        for (app, wins) in stats {
            if *app < 5 { continue; }
            let exp_w = *app as f64 * overall_rate;
            let exp_l = *app as f64 * (1.0 - overall_rate);
            if exp_w < 1.0 || exp_l < 1.0 { continue; }
            chi2 += (*wins as f64 - exp_w).powi(2) / exp_w;
            chi2 += ((*app - *wins) as f64 - exp_l).powi(2) / exp_l;
            df += 1;
        }
        if df > 1 {
            df -= 1;
            let p = chi2_pvalue(chi2, df);
            pirate_pos_results.push((name.clone(), chi2, p));
        }
    }
    pirate_pos_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("\n  Per-pirate chi-squared (position independence):");
    println!("  {:<28} {:>8} {:>10}", "Pirate", "Chi2", "p-value");
    println!("  {}", "-".repeat(50));
    for (name, chi2, p) in &pirate_pos_results {
        let sig = if *p < 0.05 { " *" } else { "" };
        println!("  {:<28} {:>8.2} {:>10.4}{}", name, chi2, p, sig);
    }
    let n_sig = pirate_pos_results.iter().filter(|(_, _, p)| *p < 0.05).count();
    println!("  Significant at p<0.05: {} / {} (expected ~{})",
        n_sig, pirate_pos_results.len(), (pirate_pos_results.len() as f64 * 0.05) as usize);

    // ============================================================
    // CHECK 2: Food ordering doesn't matter
    // ============================================================
    println!("\n\n=== CHECK 2: FOOD ORDERING (FAV/ALLERGY POSITION IN COURSE LIST) ===\n");

    // For each pirate, group matches by (n_fav, n_allergy), then split by whether
    // the first fav/allergy appears in the first half (positions 0-4) vs second half (5-9).
    // pirate -> (n_fav, n_allergy) -> first_half(app, wins), second_half(app, wins)

    // Simpler approach: for fixed (n_fav, n_allergy), does the *average position* of favs/allergies
    // correlate with win rate? Split into "fav-early" vs "fav-late".
    // Actually simplest: for each pirate with n_allergy >= 1, check if having the first allergy
    // in positions 0-4 vs 5-9 changes win rate.

    // Let's check: for each pirate, when they have exactly 1 allergy,
    // does win rate differ based on which position (0-9) that allergy is in?
    // Group into "early" (0-4) vs "late" (5-9).

    let mut ordering_results: Vec<(String, u32, u32, f64, u32, u32, f64)> = Vec::new();

    for pirate_data in &game_data.pirates {
        let mut early = (0u32, 0u32); // (app, wins) when first allergy is in positions 0-4
        let mut late = (0u32, 0u32);  // (app, wins) when first allergy is in positions 5-9

        for day in &days {
            for arena in day {
                let hp = match arena.pirates.iter().find(|p| p.name == pirate_data.name) {
                    Some(p) => p,
                    None => continue,
                };

                let courses: Vec<usize> = arena.foods.iter()
                    .filter_map(|f| course_idx.get(f.as_str()).copied())
                    .collect();
                if courses.len() != 10 { continue; }

                // Count allergies
                let allergy_positions: Vec<usize> = courses.iter().enumerate()
                    .filter(|(_, &c)| pirate_data.allergy_courses.contains(&c))
                    .map(|(i, _)| i)
                    .collect();

                if allergy_positions.len() != 1 { continue; } // exactly 1 allergy

                let won = arena.winner == hp.name;
                if allergy_positions[0] < 5 {
                    early.0 += 1;
                    if won { early.1 += 1; }
                } else {
                    late.0 += 1;
                    if won { late.1 += 1; }
                }
            }
        }

        if early.0 >= 20 && late.0 >= 20 {
            let early_rate = early.1 as f64 / early.0 as f64;
            let late_rate = late.1 as f64 / late.0 as f64;
            ordering_results.push((
                pirate_data.name.clone(),
                early.0, early.1, early_rate,
                late.0, late.1, late_rate,
            ));
        }
    }

    println!("  When pirate has exactly 1 allergy, split by allergy position (early=0-4, late=5-9):");
    println!("  {:<28} {:>12} {:>12} {:>8}", "Pirate", "Early(n,wr)", "Late(n,wr)", "diff");
    println!("  {}", "-".repeat(65));
    for (name, ea, ew, er, la, lw, lr) in &ordering_results {
        println!("  {:<28} {:>4}:{:.3}   {:>4}:{:.3}   {:>+.3}",
            name, ea, er, la, lr, er - lr);
    }

    // Same check for favorites
    let mut fav_ordering: Vec<(String, u32, u32, f64, u32, u32, f64)> = Vec::new();

    for pirate_data in &game_data.pirates {
        let mut early = (0u32, 0u32);
        let mut late = (0u32, 0u32);

        for day in &days {
            for arena in day {
                let hp = match arena.pirates.iter().find(|p| p.name == pirate_data.name) {
                    Some(p) => p,
                    None => continue,
                };

                let courses: Vec<usize> = arena.foods.iter()
                    .filter_map(|f| course_idx.get(f.as_str()).copied())
                    .collect();
                if courses.len() != 10 { continue; }

                let fav_positions: Vec<usize> = courses.iter().enumerate()
                    .filter(|(_, &c)| pirate_data.favorite_courses.contains(&c)
                        && !pirate_data.allergy_courses.contains(&c))
                    .map(|(i, _)| i)
                    .collect();

                if fav_positions.len() != 1 { continue; }

                let won = arena.winner == hp.name;
                if fav_positions[0] < 5 {
                    early.0 += 1;
                    if won { early.1 += 1; }
                } else {
                    late.0 += 1;
                    if won { late.1 += 1; }
                }
            }
        }

        if early.0 >= 20 && late.0 >= 20 {
            let early_rate = early.1 as f64 / early.0 as f64;
            let late_rate = late.1 as f64 / late.0 as f64;
            fav_ordering.push((
                pirate_data.name.clone(),
                early.0, early.1, early_rate,
                late.0, late.1, late_rate,
            ));
        }
    }

    println!("\n  When pirate has exactly 1 favorite, split by favorite position:");
    println!("  {:<28} {:>12} {:>12} {:>8}", "Pirate", "Early(n,wr)", "Late(n,wr)", "diff");
    println!("  {}", "-".repeat(65));
    for (name, ea, ew, er, la, lw, lr) in &fav_ordering {
        println!("  {:<28} {:>4}:{:.3}   {:>4}:{:.3}   {:>+.3}",
            name, ea, er, la, lr, er - lr);
    }

    // ============================================================
    // CHECK 3: Pairwise matchup outliers
    // ============================================================
    println!("\n\n=== CHECK 3: PAIRWISE MATCHUP OUTLIERS ===\n");

    // For every pair of pirates that appeared in the same arena,
    // track how often each beat the other (head-to-head).
    // Compare to expected rate based on overall win rates.

    // pair (a, b) where a < b alphabetically -> (times_together, a_wins_when_together, b_wins_when_together)
    let mut pairs: HashMap<(String, String), (u32, u32, u32)> = HashMap::new();

    // Also need overall win rates
    let mut overall_wins: HashMap<String, u32> = HashMap::new();
    let mut overall_apps: HashMap<String, u32> = HashMap::new();

    for day in &days {
        for arena in day {
            let winner = &arena.winner;
            let names: Vec<&str> = arena.pirates.iter().map(|p| p.name.as_str()).collect();
            for name in &names {
                *overall_apps.entry(name.to_string()).or_default() += 1;
                if *name == winner.as_str() {
                    *overall_wins.entry(name.to_string()).or_default() += 1;
                }
            }

            // All pairs
            for i in 0..names.len() {
                for j in (i+1)..names.len() {
                    let (a, b) = if names[i] < names[j] {
                        (names[i].to_string(), names[j].to_string())
                    } else {
                        (names[j].to_string(), names[i].to_string())
                    };
                    let entry = pairs.entry((a.clone(), b.clone())).or_insert((0, 0, 0));
                    entry.0 += 1;
                    if winner == &a { entry.1 += 1; }
                    if winner == &b { entry.2 += 1; }
                }
            }
        }
    }

    // For each pair, compute expected head-to-head win rates from overall rates
    // Expected: P(A wins | A and B in same arena) ≈ overall_wr(A) (approximately,
    // since there are 4 pirates total)
    // Better: compare A_wins / (A_wins + B_wins) to overall_wr(A) / (overall_wr(A) + overall_wr(B))

    let mut outliers: Vec<(String, String, u32, f64, f64, f64)> = Vec::new();

    for ((a, b), (together, a_wins, b_wins)) in &pairs {
        if *together < 100 { continue; }
        let h2h_total = a_wins + b_wins;
        if h2h_total < 20 { continue; }

        let wr_a = *overall_wins.get(a).unwrap_or(&0) as f64 / *overall_apps.get(a).unwrap_or(&1) as f64;
        let wr_b = *overall_wins.get(b).unwrap_or(&0) as f64 / *overall_apps.get(b).unwrap_or(&1) as f64;

        let expected_a_ratio = wr_a / (wr_a + wr_b);
        let actual_a_ratio = *a_wins as f64 / h2h_total as f64;
        let diff = (actual_a_ratio - expected_a_ratio).abs();

        outliers.push((a.clone(), b.clone(), *together, actual_a_ratio, expected_a_ratio, diff));
    }

    outliers.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());

    println!("  Top 20 pairwise matchup deviations (A wins / (A+B wins) vs expected from overall rates):");
    println!("  {:<25} {:<25} {:>5} {:>8} {:>8} {:>8}", "Pirate A", "Pirate B", "n", "actual", "expect", "diff");
    println!("  {}", "-".repeat(85));
    for (a, b, n, actual, expected, diff) in outliers.iter().take(20) {
        println!("  {:<25} {:<25} {:>5} {:>8.3} {:>8.3} {:>8.3}",
            a, b, n, actual, expected, diff);
    }

    // Summary stats
    let mean_diff: f64 = outliers.iter().map(|(_, _, _, _, _, d)| d).sum::<f64>() / outliers.len() as f64;
    let n_large = outliers.iter().filter(|(_, _, _, _, _, d)| *d > 0.05).count();
    println!("\n  Total pairs: {}", outliers.len());
    println!("  Mean absolute deviation: {:.4}", mean_diff);
    println!("  Pairs with |diff| > 0.05: {}", n_large);
}
