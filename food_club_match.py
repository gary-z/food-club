from dataclasses import dataclass
from typing import Tuple, List
import json


@dataclass(frozen=True)
class PirateOdds:
    name: str
    odds: int


@dataclass(frozen=True)
class Arena:
    arena_name: str
    foods: List[str]
    pirates: List[PirateOdds]


@dataclass(frozen=True)
class HistoricalArena(Arena):
    winner: str


def parse_historical_data(json_str):
    return [parse_day(day) for day in json.loads(json_str)]


def parse_day(arenas):
    return [parse_arena(arena) for arena in arenas]


def parse_arena(arena):
    return HistoricalArena(
        arena_name=arena["arena_name"],
        foods=arena["foods"],
        pirates=[parse_pirate(pirate) for pirate in arena["pirates"]],
        winner=arena["winner"],
    )


def parse_pirate(pirate):
    return PirateOdds(name=pirate["name"], odds=pirate["odds"])


__all__ = ["PirateOdds", "Arena"]
