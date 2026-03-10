mod pirates;

use pirates::GameData;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const ITERATIONS: u64 = 2_000_000;
const STRENGTH_BASE: f64 = 112.5;
const FAV_STRENGTH_BONUS: f64 = 2.7;
const ALLERGY_STRENGTH_PENALTY: f64 = 3.0;

fn pirate_score(pirate: &pirates::Pirate, courses: &[usize], rng: &mut impl Rng) -> f64 {
    // Match Python exactly: count allergies first, then favorites that are NOT also allergies
    let num_allergy = courses.iter().filter(|&&c| pirate.allergy_courses.contains(&c)).count();
    let num_fav = courses
        .iter()
        .filter(|&&c| pirate.favorite_courses.contains(&c) && !pirate.allergy_courses.contains(&c))
        .count();
    let effective_strength =
        pirate.strength as f64 + num_fav as f64 * FAV_STRENGTH_BONUS - num_allergy as f64 * ALLERGY_STRENGTH_PENALTY;
    // Lower total time = faster eater = wins; return negative so max() picks winner
    let time: f64 = (0..3)
        .map(|_| (STRENGTH_BASE - effective_strength) * rng.gen::<f64>())
        .sum();
    -time
}

fn simulate_chunk(data: &GameData, days: u64, seed: u64) -> HashMap<String, u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut wins: HashMap<String, u64> = HashMap::new();
    let n = data.pirates.len();
    let nc = data.num_courses();
    let mut pirate_order: Vec<usize> = (0..n).collect();

    for _ in 0..days {
        pirate_order.shuffle(&mut rng);
        let course_indices: Vec<usize> = sample(&mut rng, nc, 10).into_vec();

        for group in pirate_order.chunks(4) {
            let winner_idx = group
                .iter()
                .map(|&pi| {
                    let score = pirate_score(&data.pirates[pi], &course_indices, &mut rng);
                    (pi, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(pi, _)| pi)
                .unwrap();
            *wins.entry(data.pirates[winner_idx].name.clone()).or_insert(0) += 1;
        }
    }
    wins
}

fn log_ratio_avg(sim_rates: &HashMap<String, f64>, data: &GameData) -> f64 {
    let total: f64 = data
        .pirates
        .iter()
        .map(|p| {
            let sim = sim_rates.get(&p.name).copied().unwrap_or(1e-9).max(1e-9);
            (sim / p.win_rate).ln().abs()
        })
        .sum();
    total / data.pirates.len() as f64
}

fn main() {
    let json = std::fs::read_to_string("../pirates.json").expect("pirates.json not found");
    let data = Arc::new(GameData::load(&json));

    let n_threads = rayon::current_num_threads() as u64;
    let chunk = ITERATIONS / n_threads;
    let total = chunk * n_threads;

    let all_wins: Vec<HashMap<String, u64>> = (0..n_threads)
        .into_par_iter()
        .map(|i| simulate_chunk(&data, chunk, i * 1337 + 42))
        .collect();

    let mut total_wins: HashMap<String, u64> = HashMap::new();
    for chunk_wins in all_wins {
        for (name, count) in chunk_wins {
            *total_wins.entry(name).or_insert(0) += count;
        }
    }

    let sim_rates: HashMap<String, f64> = total_wins
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / total as f64))
        .collect();

    let mut pirates_sorted = data.pirates.clone();
    pirates_sorted.sort_by(|a, b| a.win_rate.partial_cmp(&b.win_rate).unwrap());

    for p in &pirates_sorted {
        let sim = sim_rates.get(&p.name).copied().unwrap_or(0.0);
        println!(
            "{:.2}\t{:.2}\t{}\t{}\t{}",
            p.win_rate,
            sim,
            p.favorite_courses.len(),
            p.allergy_courses.len(),
            p.name
        );
    }

    println!("Log ratio avg {:.3}", log_ratio_avg(&sim_rates, &data));
}
