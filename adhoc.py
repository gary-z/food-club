from food_club_match import Arena, parse_historical_data, HistoricalArena
from collections import Counter


if __name__ == "__main__":
    with open("historical_matches.json", "r", encoding="utf-8") as f:
        json_str = f.read()
    historical_data = parse_historical_data(json_str)
    pair = ["Gooblah the Grarrl", "Scurvy Dan the Blade"]
    matches_with_goob_and_blade = [
        match
        for day in historical_data
        for match in day
        if sum(pirate.name in pair for pirate in match.pirates) == len(pair)
    ]

    count_by_indices = Counter()
    wins_by_indices = Counter()

    for match in matches_with_goob_and_blade:
        indices = tuple(
            sorted(i for i, pirate in enumerate(match.pirates) if pirate.name in pair)
        )
        count_by_indices[indices] += 1
    print(count_by_indices)
