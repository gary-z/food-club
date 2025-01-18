import math
import random
from pirates import pirates, Pirate, courses
from collections import defaultdict, namedtuple
from typing import List

DAYS = 30000

wins = defaultdict(int)
FOOD_MULTIPLIER = 2


def get_group_winner(group: List[Pirate]):
    cs = random.sample(courses, 10)
    winner = max(group, key=lambda pirate: get_pirate_score(pirate, cs))
    return winner


deviation = 20


def get_pirate_score(pirate: Pirate, cs: List[str]):
    score = 0
    for course in cs:
        is_fav = course in pirate.favorite_courses
        is_allergy = course in pirate.allergy_courses

        if is_fav and not is_allergy:
            score += (155 - pirate.strength - pirate.weight / 10) * random.random()
        else:
            score += (155 - pirate.strength) * random.random()
    return -score


pirates_mine = list(pirates)
for _ in range(DAYS):
    random.shuffle(pirates_mine)
    groups = [pirates_mine[i : i + 4] for i in range(0, len(pirates_mine), 4)]
    for group in groups:
        winner = get_group_winner(group)
        wins[winner.name] += 1


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


actual_win_rates = [pirate.win_rate for pirate in pirates]
simulated_win_rates = [wins[pirate.name] / DAYS for pirate in pirates]


for pirate in sorted(pirates, key=lambda p: p.win_rate):
    print(
        "%.2f\t%.2f\t%d\t%d\t%s"
        % (
            pirate.win_rate,
            wins[pirate.name] / DAYS,
            len(pirate.favorite_courses),
            len(pirate.allergy_courses),
            pirate.name,
        )
    )

print(
    "Log ratio difference %.3f"
    % average_log_ratio_difference(simulated_win_rates, actual_win_rates)
)
