from dataclasses import dataclass
import math
import random
from pirates import pirates, Pirate, courses
from collections import defaultdict, namedtuple
from typing import List

DAYS = 150000


@dataclass(frozen=True)
class SimulationParams:
    favorite_upper: int
    allergy_upper: int
    normal_upper: int


def tweak_params(params: SimulationParams):
    return SimulationParams(
        favorite_upper=params.favorite_upper + random.randint(-20, 20),
        allergy_upper=params.allergy_upper + random.randint(-20, 20),
        normal_upper=params.normal_upper + random.randint(-20, 20),
    )


def get_simulation_score(params: SimulationParams):
    pirates_mine = list(pirates)
    wins = defaultdict(int)
    for _ in range(DAYS):
        random.shuffle(pirates_mine)
        groups = [pirates_mine[i : i + 4] for i in range(0, len(pirates_mine), 4)]
        for group in groups:
            winner = get_group_winner(
                params,
                group,
            )
            wins[winner.name] += 1

    actual_win_rates = [pirate.win_rate for pirate in pirates]
    simulated_win_rates = [wins[pirate.name] / DAYS for pirate in pirates]

    return average_log_ratio_difference(simulated_win_rates, actual_win_rates)


def get_group_winner(params: SimulationParams, group: List[Pirate]):
    cs = random.sample(courses, 10)
    winner = max(group, key=lambda pirate: get_pirate_score(params, pirate, cs))
    return winner


def get_pirate_score(params: SimulationParams, pirate: Pirate, cs: List[str]):
    score = 0
    for course in cs:
        is_fav = course in pirate.favorite_courses
        is_allergy = course in pirate.allergy_courses

        if is_fav and not is_allergy:
            score += (params.favorite_upper - pirate.strength) * random.random()
        elif is_allergy and not is_fav:
            score += (params.allergy_upper - pirate.strength) * random.random()
        else:
            score += (params.normal_upper - pirate.strength) * random.random()
    return -score


def average_log_ratio_difference(simulated, historical):
    if len(simulated) != len(historical):
        raise ValueError("Simulated and historical lists must have the same length.")

    total_log_diff = 0
    count = len(simulated)

    for sim, hist in zip(simulated, historical):
        if hist == 0:
            raise ValueError("Historical win rates must not contain zeros.")
        # Compute the absolute log ratio difference
        total_log_diff += abs(math.log(sim / hist))

    # Return the average
    return total_log_diff / count


def hill_climbing(params: SimulationParams):
    best_score = get_simulation_score(params)
    for _ in range(1000):
        candidate_params = tweak_params(params)
        candidate_score = get_simulation_score(params)
        print("Candidate: %.3f\t%s" % (candidate_score, str(candidate_params)))
        if candidate_score < best_score:
            best_score = candidate_score
            params = candidate_params
            print("New best: %.3f\t%s" % (candidate_score, str(candidate_params)))


hill_climbing(SimulationParams(favorite_upper=135, allergy_upper=170, normal_upper=155))
