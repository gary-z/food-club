from dataclasses import dataclass
import math
import random
from pirates import PIRATES, Pirate, COURSES
from collections import defaultdict, namedtuple, Counter
from typing import List
from concurrent.futures import ProcessPoolExecutor
import os

DAYS = 2000000


@dataclass(frozen=True)
class SimulationParams:
    favorite_upper: int
    allergy_upper: int
    normal_upper: int


def _simulate_chunk(params, days):
    wins = defaultdict(int)
    pirates_mine_local = list(PIRATES)
    for _ in range(days):
        random.shuffle(pirates_mine_local)
        groups = [
            pirates_mine_local[i : i + 4] for i in range(0, len(pirates_mine_local), 4)
        ]
        for group in groups:
            winner = get_group_winner(params, group, random.sample(COURSES, k=10))
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
    return {p.name: total_wins[p.name] / iterations for p in PIRATES}


def get_group_winner(params: SimulationParams, group: List[Pirate], courses: List[str]):
    winner = max(group, key=lambda pirate: get_pirate_score(params, pirate, courses))
    return winner


def get_pirate_score(params: SimulationParams, pirate: Pirate, cs: List[str]):
    score = 0
    for course in cs:
        is_fav = course in pirate.favorite_courses
        is_allergy = course in pirate.allergy_courses

        time_to_finish_course = (
            params.normal_upper - pirate.strength - 5
        ) * random.random()

        if is_allergy:
            time_to_finish_course += 15
        elif is_fav:
            time_to_finish_course -= 10

        score += time_to_finish_course

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


def get_default_params():
    return SimulationParams(favorite_upper=140, allergy_upper=170, normal_upper=155)


def get_arena_win_probabilities(
    arena_pirate_names: List[str], arena_courses: List[str], num_iterations=50000
):
    pirates_by_name = {pirate.name: pirate for pirate in PIRATES}
    arena_pirates = [pirates_by_name[pirate_name] for pirate_name in arena_pirate_names]
    win_counts = Counter(
        get_group_winner(get_default_params(), arena_pirates, arena_courses).name
        for _ in range(num_iterations)
    )
    return {
        pirate_name: count / num_iterations for pirate_name, count in win_counts.items()
    }


if __name__ == "__main__":
    params = get_default_params()
    simulated_win_rates = get_simulation_win_rates(params)

    for pirate in sorted(PIRATES, key=lambda p: p.win_rate):
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
            [p.win_rate for p in PIRATES],
            [simulated_win_rates[p.name] for p in PIRATES],
        ),
    )

__all__ = "get_arena_win_probabilities"
