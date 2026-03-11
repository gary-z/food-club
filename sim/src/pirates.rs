use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Deserialize)]
struct RawPirate {
    pub name: String,
    pub weight: u32,
    pub strength: u32,
    pub win_rate: f64,
    pub favorites: Vec<String>,
    pub allergies: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawData {
    pirates: Vec<RawPirate>,
    courses: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Pirate {
    pub name: String,
    pub weight: u32,
    pub strength: u32,
    pub win_rate: f64,
    pub favorite_courses: HashSet<usize>,
    pub allergy_courses: HashSet<usize>,
}

#[derive(Debug, Clone)]
pub struct GameData {
    pub pirates: Vec<Pirate>,
    pub courses: Vec<String>, // course index -> course name
}

impl GameData {
    pub fn load(json: &str) -> Self {
        let raw: RawData = serde_json::from_str(json).expect("Failed to parse pirates.json");

        // Build course list and index
        let courses: Vec<String> = raw.courses.keys().cloned().collect();
        let course_index: HashMap<&str, usize> = courses
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Build category -> course indices map
        let mut category_courses: HashMap<&str, Vec<usize>> = HashMap::new();
        for (course_name, cats) in &raw.courses {
            let idx = course_index[course_name.as_str()];
            for cat in cats {
                category_courses.entry(cat.as_str()).or_default().push(idx);
            }
        }

        let pirates = raw
            .pirates
            .into_iter()
            .map(|p| {
                let favorite_courses = p
                    .favorites
                    .iter()
                    .flat_map(|cat| {
                        category_courses
                            .get(cat.as_str())
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect();
                let allergy_courses = p
                    .allergies
                    .iter()
                    .flat_map(|cat| {
                        category_courses
                            .get(cat.as_str())
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect();
                Pirate {
                    name: p.name,
                    weight: p.weight,
                    strength: p.strength,
                    win_rate: p.win_rate,
                    favorite_courses,
                    allergy_courses,
                }
            })
            .collect();

        GameData { pirates, courses }
    }

    pub fn num_courses(&self) -> usize {
        self.courses.len()
    }

    pub fn course_name_to_index(&self) -> HashMap<&str, usize> {
        self.courses
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect()
    }

    pub fn pirate_by_name(&self, name: &str) -> Option<&Pirate> {
        self.pirates.iter().find(|p| p.name == name)
    }

    pub fn pirate_index(&self, name: &str) -> usize {
        self.pirates.iter().position(|p| p.name == name)
            .unwrap_or_else(|| panic!("Unknown pirate: {}", name))
    }

    pub fn course_index(&self, name: &str) -> Option<usize> {
        self.courses.iter().position(|c| c == name)
    }
}

/// A single arena match from historical data.
#[derive(Debug, Clone)]
pub struct HistMatch {
    pub pirate_indices: [usize; 4],
    pub course_indices: Vec<usize>,
    pub winner_pos: usize, // 0-3 index into pirate_indices
}

/// Load historical matches and map to GameData indices.
pub fn load_historical_matches(data: &GameData, json: &str) -> Vec<Vec<HistMatch>> {
    #[derive(Deserialize)]
    struct HP { name: String, #[allow(dead_code)] odds: u32 }
    #[derive(Deserialize)]
    struct HA { #[allow(dead_code)] arena_name: String, foods: Vec<String>, pirates: Vec<HP>, winner: String }

    let days: Vec<Vec<HA>> = serde_json::from_str(json).expect("Failed to parse historical matches");

    days.into_iter().map(|day| {
        day.into_iter().map(|arena| {
            let pirate_indices: [usize; 4] = [
                data.pirate_index(&arena.pirates[0].name),
                data.pirate_index(&arena.pirates[1].name),
                data.pirate_index(&arena.pirates[2].name),
                data.pirate_index(&arena.pirates[3].name),
            ];
            let course_indices: Vec<usize> = arena.foods.iter()
                .filter_map(|f| data.course_index(f))
                .collect();
            let winner_pos = arena.pirates.iter()
                .position(|p| p.name == arena.winner)
                .expect("Winner not in arena");
            HistMatch { pirate_indices, course_indices, winner_pos }
        }).collect()
    }).collect()
}
