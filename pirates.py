from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class PirateInternal:
    name: str
    weight: int
    strength: int
    win_rate: float
    favorites: Tuple[str]
    allergies: Tuple[str]


pirates_internal = [
    PirateInternal(
        name="Scurvy Dan the Blade",
        weight=166,
        strength=87,
        win_rate=0.9033212010919017,
        favorites=("Salty foods", "Meats"),
        allergies=("Candy",),
    ),
    PirateInternal(
        name="Young Sproggie",
        weight=112,
        strength=73,
        win_rate=0.24235224235224234,
        favorites=("Meats", "Neggs"),
        allergies=("Gross foods",),
    ),
    PirateInternal(
        name="Orvinn the First Mate",
        weight=221,
        strength=52,
        win_rate=0.12158176943699732,
        favorites=("Candy", "Slushies", "Pizza"),
        allergies=("Fruits",),
    ),
    PirateInternal(
        name="Lucky McKyriggan",
        weight=182,
        strength=82,
        win_rate=0.3834325396825397,
        favorites=("Gross foods",),
        allergies=("Pizza",),
    ),
    PirateInternal(
        name="Sir Edmund Ogletree",
        weight=177,
        strength=79,
        win_rate=0.369903389552972,
        favorites=("Dairy",),
        allergies=("Breads",),
    ),
    PirateInternal(
        name="Peg Leg Percival",
        weight=202,
        strength=73,
        win_rate=0.16566810645116345,
        favorites=("Spicy foods",),
        allergies=("Smoothies",),
    ),
    PirateInternal(
        name="Bonnie Pip Culliford",
        weight=116,
        strength=76,
        win_rate=0.3071395094516482,
        favorites=("Candy", "Smoothies"),
        allergies=("Spicy foods",),
    ),
    PirateInternal(
        name="Puffo the Waister",
        weight=180,
        strength=68,
        win_rate=0.15041935927402722,
        favorites=("Candy", "Smoothies", "Slushies"),
        allergies=("Meats",),
    ),
    PirateInternal(
        name="Stuff-A-Roo",
        weight=211,
        strength=59,
        win_rate=0.06288109756097561,
        favorites=("Pizza",),
        allergies=("Neggs",),
    ),
    PirateInternal(
        name="Squire Venable",
        weight=213,
        strength=61,
        win_rate=0.06735553004209721,
        favorites=("Breads",),
        allergies=("Fruits",),
    ),
    PirateInternal(
        name="Captain Crossblades",
        weight=185,
        strength=66,
        win_rate=0.12686868686868688,
        favorites=("Slushies", "Pizza"),
        allergies=("Salty foods",),
    ),
    PirateInternal(
        name="Ol' Stripey",
        weight=189,
        strength=74,
        win_rate=0.26680799515445186,
        favorites=("Meats", "Slushies"),
        allergies=("Breads",),
    ),
    PirateInternal(
        name="Ned the Skipper",
        weight=169,
        strength=79,
        win_rate=0.3122647427854454,
        favorites=("Meats",),
        allergies=("Dairy",),
    ),
    PirateInternal(
        name="Fairfax the Deckhand",
        weight=151,
        strength=71,
        win_rate=0.23152781866352665,
        favorites=("Vegetables", "Fruits"),
        allergies=("Salty foods",),
    ),
    PirateInternal(
        name="Gooblah the Grarrl",
        weight=199,
        strength=93,
        win_rate=1.835593220338983,
        favorites=("Meats",),
        allergies=("Slushies",),
    ),
    PirateInternal(
        name="Franchisco Corvallio",
        weight=165,
        strength=81,
        win_rate=0.5463955637707948,
        favorites=("Spicy foods", "Meats"),
        allergies=("Candy",),
    ),
    PirateInternal(
        name="Federismo Corvallio",
        weight=166,
        strength=81,
        win_rate=0.5587851686230669,
        favorites=("Gross foods", "Pizza"),
        allergies=("Smoothies",),
    ),
    PirateInternal(
        name="Admiral Blackbeard",
        weight=171,
        strength=76,
        win_rate=0.20823104693140795,
        favorites=("Vegetables", "Fruits"),
        allergies=("Dairy",),
    ),
    PirateInternal(
        name="Buck Cutlass",
        weight=189,
        strength=89,
        win_rate=0.8086900129701686,
        favorites=("Candy",),
        allergies=("Vegetables",),
    ),
    PirateInternal(
        name="The Tailhook Kid",
        weight=207,
        strength=81,
        win_rate=0.46309898565932145,
        favorites=("Vegetables",),
        allergies=("Neggs",),
    ),
]

courses_to_categories = {
    "Hotfish": ["Salty foods", "Meats"],
    "Wriggling Grub": ["Gross foods"],
    "Joint Of Ham": ["Meats"],
    "Rainbow Negg": ["Neggs"],
    "Streaky Bacon": ["Meats"],
    "Ultimate Burger": ["Meats"],
    "Bacon Muffin": ["Meats", "Breads"],
    "Hot Cakes": ["Breads"],
    "Spicy Wings": ["Spicy foods", "Meats"],
    "Apple Onion Rings": ["Fruits", "Gross foods"],
    "Sushi": ["Salty foods", "Meats"],
    "Negg Stew": ["Neggs"],
    "Ice Chocolate Cake": ["Candy"],
    "Strochal": ["Candy"],
    "Mallowicious Bar": ["Candy"],
    "Fungi Pizza": ["Gross foods", "Pizza"],
    "Broccoli and Cheese Pizza": ["Vegetables", "Dairy", "Pizza"],
    "Bubbling Blueberry Pizza": ["Fruits", "Pizza"],
    "Grapity Slush": ["Slushies"],
    "Rainborific Slush": ["Slushies"],
    "Tangy Tropic Slush": ["Slushies"],
    "Blueberry Tomato Blend": ["Fruits", "Dairy", "Smoothies"],
    "Lemon Blitz": ["Fruits", "Dairy", "Smoothies"],
    "Fresh Seaweed Pie": ["Salty foods", "Gross foods"],
    "Flaming Burnumup": ["Spicy foods", "Vegetables"],
    "Hot Tyrannian Pepper": ["Spicy foods", "Vegetables"],
    "Eye Candy": ["Candy", "Gross foods"],
    "Cheese and Tomato Sub": ["Fruits", "Breads", "Dairy"],
    "Asparagus Pie": ["Vegetables"],
    "Wild Chocomato": ["Dairy", "Smoothies"],
    "Cinnamon Swirl": ["Candy", "Breads"],
    "Anchovies": ["Salty foods", "Meats"],
    "Flaming Fire Faerie Pizza": ["Spicy foods", "Vegetables", "Pizza"],
    "Orange Negg": ["Neggs"],
    "Fish Negg": ["Neggs"],
    "Super Lemon Grape Slush": ["Slushies"],
    "Rasmelon": ["Smoothies"],
    "Mustard Ice Cream": ["Dairy", "Gross foods"],
    "Worm and Leech Pizza": ["Gross foods", "Pizza"],
    "Broccoli": ["Vegetables"],
}


@dataclass(frozen=True)
class Pirate:
    name: str
    weight: int
    strength: int
    win_rate: float
    favorite_courses: FrozenSet[str]
    allergy_courses: FrozenSet[str]


def get_courses_matching_categories(categories):
    return frozenset(
        course
        for course, course_categories in courses_to_categories.items()
        if set(categories) & set(course_categories)
    )


pirates = tuple(
    Pirate(
        name=pirate.name,
        weight=pirate.weight,
        strength=pirate.strength,
        win_rate=pirate.win_rate,
        favorite_courses=get_courses_matching_categories(pirate.favorites),
        allergy_courses=get_courses_matching_categories(pirate.allergies),
    )
    for pirate in pirates_internal
)

courses = tuple(courses_to_categories)

arenas = ("Shipwreck", "Lagoon", "Treasure Island", "Hidden Cove", "Harpoon Harry's")

__all__ = ["pirates", "arenas", "courses"]
