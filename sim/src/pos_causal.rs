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
    winner: String,
}

fn main() {
    let hist_json = std::fs::read_to_string("../historical_matches.json")
        .expect("historical_matches.json not found");
    let days: Vec<Vec<HistArena>> =
        serde_json::from_str(&hist_json).expect("Failed to parse");

    // ============================================================
    // TEST 1: Controlling for odds, does position predict winning?
    // ============================================================
    // Key: (odds, position) -> (appearances, wins)
    let mut odds_pos: HashMap<(u32, usize), (u32, u32)> = HashMap::new();
    // Also: (odds) -> (appearances, wins) for baseline
    let mut odds_only: HashMap<u32, (u32, u32)> = HashMap::new();

    for day in &days {
        for arena in day {
            for (pos, hp) in arena.pirates.iter().enumerate() {
                let won = arena.winner == hp.name;
                let e = odds_pos.entry((hp.odds, pos)).or_insert((0, 0));
                e.0 += 1;
                if won { e.1 += 1; }
                let e2 = odds_only.entry(hp.odds).or_insert((0, 0));
                e2.0 += 1;
                if won { e2.1 += 1; }
            }
        }
    }

    println!("=== TEST 1: WIN RATE BY POSITION, CONTROLLING FOR ODDS ===\n");
    println!("For each odds value, win rate at each position:");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}", "odds", "overall", "pos0", "pos1", "pos2", "pos3");
    println!("{}", "-".repeat(60));

    let mut odds_vals: Vec<u32> = odds_only.keys().copied().collect();
    odds_vals.sort();

    for &odds in &odds_vals {
        let (total_app, total_wins) = odds_only[&odds];
        if total_app < 100 { continue; }
        let overall_wr = total_wins as f64 / total_app as f64;
        let mut pos_strs = Vec::new();
        for pos in 0..4 {
            if let Some(&(app, wins)) = odds_pos.get(&(odds, pos)) {
                if app >= 10 {
                    pos_strs.push(format!("{:.3}({})", wins as f64 / app as f64, app));
                } else {
                    pos_strs.push(format!("   -   "));
                }
            } else {
                pos_strs.push(format!("   -   "));
            }
        }
        println!("{:>5} {:>8.3}({}) {} {} {} {}",
            odds, overall_wr, total_app,
            pos_strs[0], pos_strs[1], pos_strs[2], pos_strs[3]);
    }

    // ============================================================
    // TEST 2: Aggregate - position win rate weighted by odds group
    // ============================================================
    println!("\n\n=== TEST 2: AGGREGATE POSITION EFFECT (ODDS-CONTROLLED) ===\n");
    // For each odds value with enough data, compute (actual_pos_wr - overall_wr) per position
    // Then average across odds values, weighted by sample size
    let mut pos_excess = [0.0f64; 4];
    let mut pos_weight = [0.0f64; 4];

    for &odds in &odds_vals {
        let (total_app, total_wins) = odds_only[&odds];
        if total_app < 200 { continue; }
        let overall_wr = total_wins as f64 / total_app as f64;
        for pos in 0..4 {
            if let Some(&(app, wins)) = odds_pos.get(&(odds, pos)) {
                if app >= 20 {
                    let wr = wins as f64 / app as f64;
                    pos_excess[pos] += (wr - overall_wr) * app as f64;
                    pos_weight[pos] += app as f64;
                }
            }
        }
    }

    println!("Position effect after controlling for odds:");
    for pos in 0..4 {
        if pos_weight[pos] > 0.0 {
            println!("  Position {}: excess win rate = {:+.4} (weighted across {} appearances)",
                pos, pos_excess[pos] / pos_weight[pos], pos_weight[pos] as u64);
        }
    }

    // ============================================================
    // TEST 3: Are pirates sorted by odds within each arena?
    // ============================================================
    println!("\n\n=== TEST 3: ORDERING CHECK - ARE PIRATES SORTED BY ODDS? ===\n");

    let mut sorted_asc = 0u32;
    let mut sorted_desc = 0u32;
    let mut total_arenas = 0u32;
    let mut inversions_total = 0u64;
    let mut pairs_total = 0u64;

    // Check if odds are monotonically related to position
    // Count inversions (pos i < pos j but odds[i] < odds[j] means "ascending tendency")
    for day in &days {
        for arena in day {
            total_arenas += 1;
            let odds: Vec<u32> = arena.pirates.iter().map(|p| p.odds).collect();
            let is_asc = odds.windows(2).all(|w| w[0] <= w[1]);
            let is_desc = odds.windows(2).all(|w| w[0] >= w[1]);
            if is_asc { sorted_asc += 1; }
            if is_desc { sorted_desc += 1; }

            // Count inversions (how often later position has lower odds)
            for i in 0..odds.len() {
                for j in (i+1)..odds.len() {
                    pairs_total += 1;
                    if odds[j] > odds[i] {
                        inversions_total += 1; // later position has HIGHER odds (weaker)
                    }
                }
            }
        }
    }

    println!("  Total arenas: {}", total_arenas);
    println!("  Sorted ascending (weak->strong): {} ({:.1}%)", sorted_asc, sorted_asc as f64 / total_arenas as f64 * 100.0);
    println!("  Sorted descending (strong->weak): {} ({:.1}%)", sorted_desc, sorted_desc as f64 / total_arenas as f64 * 100.0);
    println!("  Fraction of pairs where later pos has HIGHER odds: {:.4} (0.5 = random)",
        inversions_total as f64 / pairs_total as f64);

    // ============================================================
    // TEST 4: Correlation between position and odds
    // ============================================================
    println!("\n\n=== TEST 4: AVERAGE ODDS BY POSITION ===\n");
    let mut pos_odds_sum = [0u64; 4];
    let mut pos_odds_count = [0u64; 4];
    for day in &days {
        for arena in day {
            for (pos, hp) in arena.pirates.iter().enumerate() {
                pos_odds_sum[pos] += hp.odds as u64;
                pos_odds_count[pos] += 1;
            }
        }
    }
    for pos in 0..4 {
        println!("  Position {}: avg odds = {:.3}", pos,
            pos_odds_sum[pos] as f64 / pos_odds_count[pos] as f64);
    }

    // ============================================================
    // TEST 5: Within matched-odds groups, is position assignment uniform?
    // ============================================================
    println!("\n\n=== TEST 5: POSITION DISTRIBUTION WITHIN EACH ODDS VALUE ===\n");
    println!("{:>5} {:>8} {:>8} {:>8} {:>8} {:>8}", "odds", "total", "pos0%", "pos1%", "pos2%", "pos3%");
    println!("{}", "-".repeat(50));
    for &odds in &odds_vals {
        let (total_app, _) = odds_only[&odds];
        if total_app < 200 { continue; }
        let mut pos_counts = [0u32; 4];
        for pos in 0..4 {
            if let Some(&(app, _)) = odds_pos.get(&(odds, pos)) {
                pos_counts[pos] = app;
            }
        }
        println!("{:>5} {:>8} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            odds, total_app,
            pos_counts[0] as f64 / total_app as f64 * 100.0,
            pos_counts[1] as f64 / total_app as f64 * 100.0,
            pos_counts[2] as f64 / total_app as f64 * 100.0,
            pos_counts[3] as f64 / total_app as f64 * 100.0);
    }

    // ============================================================
    // TEST 6: Does position predict winning WITHIN a single arena?
    // ============================================================
    println!("\n\n=== TEST 6: WINNER'S POSITION DISTRIBUTION ===\n");
    let mut winner_pos_count = [0u32; 4];
    let mut total_contests = 0u32;
    for day in &days {
        for arena in day {
            total_contests += 1;
            for (pos, hp) in arena.pirates.iter().enumerate() {
                if arena.winner == hp.name {
                    winner_pos_count[pos] += 1;
                }
            }
        }
    }
    println!("  Total contests: {}", total_contests);
    for pos in 0..4 {
        println!("  Winner at position {}: {} ({:.2}%)", pos,
            winner_pos_count[pos],
            winner_pos_count[pos] as f64 / total_contests as f64 * 100.0);
    }
}
