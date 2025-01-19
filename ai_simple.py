from dataclasses import dataclass
from typing import Tuple, List
from food_club_match import Arena, parse_historical_data, HistoricalArena
from simple_win_rate_model import get_arena_win_probabilities
import itertools


@dataclass(frozen=True)
class Bet:
    pirate_names: List[str]
    win_probability: float
    payout: int


def make_bets(arenas: List[Arena], max_odds=60):
    arena_win_probabilities = {
        arena.arena_name: get_arena_win_probabilities(
            [pirate.name for pirate in arena.pirates], arena.foods, 10000
        )
        for arena in arenas
    }

    # (expected_winning, dict[arena_name -> pirate])
    possible_bets = []
    for num_arenas_to_bet_on in range(1, 5):  # avoid 5 arena bets
        for arenas_to_be_on in itertools.combinations(arenas, num_arenas_to_bet_on):
            for pirates_to_bet_on in itertools.product(
                *(arena.pirates for arena in arenas_to_be_on)
            ):
                win_probability = 1
                payout = 1
                for arena, pirate_odds in zip(arenas_to_be_on, pirates_to_bet_on):
                    win_probability *= arena_win_probabilities[arena.arena_name][
                        pirate_odds.name
                    ]
                    payout = min(pirate_odds.odds * payout, max_odds)
                possible_bets.append(
                    Bet(
                        pirate_names=[pirate.name for pirate in pirates_to_bet_on],
                        win_probability=win_probability,
                        payout=payout,
                    )
                )

    possible_bets.sort(key=lambda bet: -bet.win_probability * bet.payout)
    return possible_bets[:10]


def get_payout(bet: Bet, arenas: List[HistoricalArena]):
    winners = set(arena.winner for arena in arenas)
    num_correct = sum(pirate_name in winners for pirate_name in bet.pirate_names)
    if num_correct == len(bet.pirate_names):
        return bet.payout
    return 0


if __name__ == "__main__":
    with open("historical_matches.json", "r", encoding="utf-8") as f:
        json_str = f.read()
    historical_data = parse_historical_data(json_str)
    net_gains = 0
    for i, day_arenas in enumerate(historical_data):
        bets = make_bets(day_arenas)
        total_payout = sum(get_payout(bet, day_arenas) for bet in bets)
        delta = total_payout - 10
        net_gains += delta
        print("%d\t%.1f" % (delta, net_gains / (i + 1)))
