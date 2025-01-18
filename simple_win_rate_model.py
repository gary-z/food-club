from dataclasses import dataclass
import math
import random
from pirates import pirates, Pirate, courses
from collections import defaultdict, namedtuple
from typing import List
from concurrent.futures import ProcessPoolExecutor
import os

DAYS = 1000000


@dataclass(frozen=True)
class SimulationParams:
    favorite_upper: int
    allergy_upper: int
    normal_upper: int


def _simulate_chunk(params, days):
    wins = defaultdict(int)
    pirates_mine_local = list(pirates)
    for _ in range(days):
        random.shuffle(pirates_mine_local)
        groups = [
            pirates_mine_local[i : i + 4] for i in range(0, len(pirates_mine_local), 4)
        ]
        for group in groups:
            winner = get_group_winner(params, group)
            wins[winner.name] += 1
    return wins


def get_simulation_win_rates(params: SimulationParams):
    n_procs = n_procs = os.cpu_count()
    iterations = DAYS
    iterations -= iterations % n_procs
    chunk_size = iterations // n_procs
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        # Launch parallel tasks, each simulating a portion of DAYS
        results = list(
            executor.map(_simulate_chunk, [params] * n_procs, [chunk_size] * n_procs)
        )
    # Aggregate the results
    total_wins = defaultdict(int)
    for r in results:
        for pirate_name, count in r.items():
            total_wins[pirate_name] += count
    return {p.name: total_wins[p.name] / iterations for p in pirates}


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
            score += (
                params.favorite_upper - pirate.strength - pirate.weight**0.5
            ) * random.random()
        elif is_allergy:
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


if __name__ == "__main__":
    params = SimulationParams(favorite_upper=145, allergy_upper=170, normal_upper=155)
    simulated_win_rates = get_simulation_win_rates(params)

    for pirate in sorted(pirates, key=lambda p: p.win_rate):
        print(
            "%.2f\t%.2f\t%d\t%d\t%s"
            % (
                pirate.win_rate,
                simulated_win_rates[pirate.name],
                len(pirate.favorite_courses),
                len(pirate.allergy_courses),
                pirate.name,
            )
        )

    print(
        "Log ratio avg %.3f"
        % average_log_ratio_difference(
            [p.win_rate for p in pirates],
            [simulated_win_rates[p.name] for p in pirates],
        ),
    )
