use serde::Deserialize;

#[derive(Deserialize)]
struct HistPirate {
    #[allow(dead_code)]
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

    let num_days = days.len();
    let mut total_arenas = 0u64;

    // Per-position stats
    let mut sum_odds = [0.0f64; 4];
    let mut count = [0u64; 4];
    let mut lowest_at_pos = [0u64; 4]; // how often position i has the lowest odds

    // Sorting check
    let mut always_ascending = true;
    let mut always_descending = true;
    let mut ascending_count = 0u64;
    let mut descending_count = 0u64;

    for day in &days {
        for arena in day {
            if arena.pirates.len() != 4 {
                continue;
            }
            total_arenas += 1;

            let odds: Vec<u32> = arena.pirates.iter().map(|p| p.odds).collect();

            // Accumulate per-position odds
            for (pos, &o) in odds.iter().enumerate() {
                sum_odds[pos] += o as f64;
                count[pos] += 1;
            }

            // Which position has the lowest odds?
            let min_odds = *odds.iter().min().unwrap();
            for (pos, &o) in odds.iter().enumerate() {
                if o == min_odds {
                    lowest_at_pos[pos] += 1;
                    break; // count only first position in case of tie
                }
            }

            // Check ascending/descending
            let is_asc = odds.windows(2).all(|w| w[0] <= w[1]);
            let is_desc = odds.windows(2).all(|w| w[0] >= w[1]);
            if is_asc {
                ascending_count += 1;
            }
            if is_desc {
                descending_count += 1;
            }
            if !is_asc {
                always_ascending = false;
            }
            if !is_desc {
                always_descending = false;
            }
        }
    }

    println!("=== POSITION-ODDS ANALYSIS ===\n");
    println!("Total days: {}", num_days);
    println!("Total arenas: {}\n", total_arenas);

    println!("--- Average odds by position ---");
    for pos in 0..4 {
        if count[pos] > 0 {
            println!(
                "  Position {}: avg odds = {:.2}  (n={})",
                pos,
                sum_odds[pos] / count[pos] as f64,
                count[pos]
            );
        }
    }

    println!("\n--- Fraction of times each position has the lowest odds (strongest pirate) ---");
    for pos in 0..4 {
        println!(
            "  Position {}: {}/{} = {:.4}",
            pos,
            lowest_at_pos[pos],
            total_arenas,
            lowest_at_pos[pos] as f64 / total_arenas as f64
        );
    }

    println!("\n--- Are odds sorted within each arena? ---");
    println!("  Always ascending (non-decreasing):  {}", always_ascending);
    println!("  Always descending (non-increasing): {}", always_descending);
    println!(
        "  Ascending count:  {}/{} = {:.4}",
        ascending_count,
        total_arenas,
        ascending_count as f64 / total_arenas as f64
    );
    println!(
        "  Descending count: {}/{} = {:.4}",
        descending_count,
        total_arenas,
        descending_count as f64 / total_arenas as f64
    );

    // Also show the distribution of odds at each position as a histogram-like summary
    // by showing min, max, median
    let mut odds_by_pos: Vec<Vec<u32>> = vec![Vec::new(); 4];
    for day in &days {
        for arena in day {
            if arena.pirates.len() != 4 {
                continue;
            }
            for (pos, p) in arena.pirates.iter().enumerate() {
                odds_by_pos[pos].push(p.odds);
            }
        }
    }

    println!("\n--- Odds distribution by position (min, p25, median, p75, max) ---");
    for pos in 0..4 {
        let v = &mut odds_by_pos[pos];
        v.sort();
        let n = v.len();
        if n == 0 {
            continue;
        }
        let min = v[0];
        let max = v[n - 1];
        let p25 = v[n / 4];
        let median = v[n / 2];
        let p75 = v[3 * n / 4];
        println!(
            "  Position {}: min={}, p25={}, median={}, p75={}, max={}",
            pos, min, p25, median, p75, max
        );
    }
}
