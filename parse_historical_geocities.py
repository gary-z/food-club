from pirates import pirates, courses, arenas
import json

food_id_to_name = {
    1: "Hotfish",
    2: "Broccoli",
    3: "Wriggling Grub",
    4: "Joint Of Ham",
    5: "Rainbow Negg",
    6: "Streaky Bacon",
    7: "Ultimate Burger",
    8: "Bacon Muffin",
    9: "Hot Cakes",
    10: "Spicy Wings",
    11: "Apple Onion Rings",
    12: "Sushi",
    13: "Negg Stew",
    14: "Ice Chocolate Cake",
    15: "Strochal",
    16: "Mallowicious Bar",
    17: "Fungi Pizza",
    18: "Broccoli and Cheese Pizza",
    19: "Bubbling Blueberry Pizza",
    20: "Grapity Slush",
    21: "Rainborific Slush",
    22: "Tangy Tropic Slush",
    23: "Blueberry Tomato Blend",
    24: "Lemon Blitz",
    25: "Fresh Seaweed Pie",
    26: "Flaming Burnumup",
    27: "Hot Tyrannian Pepper",
    28: "Eye Candy",
    29: "Cheese and Tomato Sub",
    30: "Asparagus Pie",
    31: "Wild Chocomato",
    32: "Cinnamon Swirl",
    33: "Anchovies",
    34: "Flaming Fire Faerie Pizza",
    35: "Orange Negg",
    36: "Fish Negg",
    37: "Super Lemon Grape Slush",
    38: "Rasmelon",
    39: "Mustard Ice Cream",
    40: "Worm and Leech Pizza",
}


def load_from_geo_cities_json(json_str):
    raw = json.loads(json_str)
    arena_datas = []

    for food_ids, pirate_ids, winner_index, opening_odds, arena_name in zip(
        raw["foods"], raw["pirates"], raw["winners"], raw["openingOdds"], arenas
    ):
        foods = [food_id_to_name[food_id] for food_id in food_ids]
        for f in foods:
            assert f in courses

        pirate_odds = [
            {"name": pirates[pirate_id - 1].name, "odds": opening_odd}
            for pirate_id, opening_odd in zip(pirate_ids, opening_odds[1:])
        ]
        assert len(pirate_odds) == 4

        arena_datas.append(
            {
                "pirates": pirate_odds,
                "winner": pirate_odds[winner_index - 1]["name"],
                "foods": foods,
                "arena_name": arena_name,
            }
        )
    assert len(arena_datas) == 5
    return arena_datas


geo_city_days = []
for i in range(3000, 9000):
    file_name = "historical/%s.json" % i
    try:
        with open(file_name, "r") as file:
            contents = file.read()
    except FileNotFoundError:
        continue

    raw = json.loads(contents)

    if "foods" not in raw:
        continue

    geo_city_days.append(load_from_geo_cities_json(contents))

with open("historical_geo_cities.json", "w") as file:
    json.dump(geo_city_days, file, indent=4)
