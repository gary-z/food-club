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
}
