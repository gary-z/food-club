import json
import os
from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class Pirate:
    name: str
    weight: int
    strength: int
    win_rate: float
    favorite_courses: FrozenSet[str]
    allergy_courses: FrozenSet[str]


def _load():
    path = os.path.join(os.path.dirname(__file__), "pirates.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    courses_to_categories = data["courses"]

    def courses_matching(categories):
        return frozenset(
            course
            for course, cats in courses_to_categories.items()
            if set(categories) & set(cats)
        )

    pirates = tuple(
        Pirate(
            name=p["name"],
            weight=p["weight"],
            strength=p["strength"],
            win_rate=p["win_rate"],
            favorite_courses=courses_matching(p["favorites"]),
            allergy_courses=courses_matching(p["allergies"]),
        )
        for p in data["pirates"]
    )

    courses = tuple(courses_to_categories)
    arenas = tuple(data["arenas"])
    return pirates, courses, arenas


PIRATES, COURSES, ARENAS = _load()

__all__ = ["PIRATES", "ARENAS", "COURSES", "Pirate"]
