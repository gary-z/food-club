#!/usr/bin/env python3
"""
Deeper TER analysis:
1. What does the TER strategy actually bet on? (How many 2:1 anchors?)
2. Compare with our anchor strategy from finding #30
3. Track win days correctly
4. Break down where TER's profit comes from
"""

import json
import numpy as np
from collections import Counter, defaultdict
from math import exp, floor

# Reuse the core functions from backtest_ter.py
exec(open('backtest_ter.py').read().split("###############################################################################\n# BACKTEST")[0])

MAX_BET = 10000
N_BETS = 10

print(f"=== TER STRATEGY DEEP DIVE ===\n")

# Analyze TER bet composition
bet_stats = {
    'n_arenas_per_bet': [],  # how many arenas covered per bet
    'has_2': [],  # does bet contain a 2:1 pirate
    'min_odds_in_bet': [],
    'bet_odds': [],
}

# Track profit by bet type
profit_by_type = {
    'with_2': {'profit': 0, 'spent': 0, 'n_bets': 0},
    'without_2': {'profit': 0, 'spent': 0, 'n_bets': 0},
}

# Our anchor strategy
anchor_profit = 0
anchor_spent = 0
anchor_wins = 0
anchor_days_with_win = 0

# TER tracking
ter_profit = 0
ter_spent = 0
ter_wins = 0
ter_days_with_win = 0

for day_idx, day_data in enumerate(data):
    if day_idx % 1000 == 0:
        print(f"  Day {day_idx}/{len(data)}...")

    try:
        arenas = resolve_arena(day_data)
    except ValueError:
        continue

    arena_odds_arr = []
    logit_probs_arr = []
    winners = []

    for arena in arenas:
        arena_odds_arr.append(arena['odds'])
        logit_probs_arr.append(compute_logit_probs(arena['pirates'], arena['foods']))
        winners.append(arena['winner_pos'])

    # TER logit bets
    bets = generate_max_ter_bets(arena_odds_arr, logit_probs_arr, MAX_BET, N_BETS)

    day_ter_won = 0
    day_ter_spent = 0
    for spec, bet_amount, odds, prob, er in bets:
        n_arenas = sum(1 for s in spec if s > 0)
        has_2 = any(arena_odds_arr[ai][spec[ai]-1] == 2 for ai in range(5) if spec[ai] > 0)
        min_odd = min(arena_odds_arr[ai][spec[ai]-1] for ai in range(5) if spec[ai] > 0)

        bet_stats['n_arenas_per_bet'].append(n_arenas)
        bet_stats['has_2'].append(has_2)
        bet_stats['min_odds_in_bet'].append(min_odd)
        bet_stats['bet_odds'].append(odds)

        # Check win
        won = True
        for arena_idx, pirate_choice in enumerate(spec):
            if pirate_choice > 0 and (pirate_choice - 1) != winners[arena_idx]:
                won = False
                break

        payout = min(1_000_000, odds * bet_amount) if won else 0
        profit = payout - bet_amount

        day_ter_spent += bet_amount
        day_ter_won += payout

        key = 'with_2' if has_2 else 'without_2'
        profit_by_type[key]['profit'] += profit
        profit_by_type[key]['spent'] += bet_amount
        profit_by_type[key]['n_bets'] += 1

    ter_profit += day_ter_won - day_ter_spent
    ter_spent += day_ter_spent
    if day_ter_won > 0:
        ter_days_with_win += 1

    # Our anchor strategy: every bet must include at least one 2:1 pirate
    # with logit p >= 0.50, padded with best-odds pirates to maximize EV
    # (simplified: use the same TER approach but filter to bets containing a good 2:1)
    good_2s = []
    for ai in range(5):
        for pi in range(4):
            if arena_odds_arr[ai][pi] == 2 and logit_probs_arr[ai][pi] >= 0.50:
                good_2s.append((ai, pi))

    if good_2s:
        # Generate all combos, filter to those containing at least one good 2:1
        all_combos = generate_max_ter_bets(arena_odds_arr, logit_probs_arr, MAX_BET, 3125)
        anchor_bets = []
        for spec, bet_amount, odds, prob, er in all_combos:
            contains_good_2 = False
            for ai, pi in good_2s:
                if spec[ai] == pi + 1:
                    contains_good_2 = True
                    break
            if contains_good_2:
                anchor_bets.append((spec, bet_amount, odds, prob, er))
            if len(anchor_bets) >= N_BETS:
                break

        day_anchor_won = 0
        day_anchor_spent = 0
        for spec, bet_amount, odds, prob, er in anchor_bets:
            day_anchor_spent += bet_amount
            won = True
            for arena_idx, pirate_choice in enumerate(spec):
                if pirate_choice > 0 and (pirate_choice - 1) != winners[arena_idx]:
                    won = False
                    break
            if won:
                payout = min(1_000_000, odds * bet_amount)
                day_anchor_won += payout

        anchor_profit += day_anchor_won - day_anchor_spent
        anchor_spent += day_anchor_spent
        if day_anchor_won > 0:
            anchor_days_with_win += 1

print("\n" + "=" * 70)
print("TER BET COMPOSITION")
print("=" * 70)

n_bets_arr = np.array(bet_stats['n_arenas_per_bet'])
has_2_arr = np.array(bet_stats['has_2'])
odds_arr = np.array(bet_stats['bet_odds'])

print(f"Total bets analyzed: {len(n_bets_arr)}")
print(f"\nArenas per bet distribution:")
for n in range(1, 6):
    pct = (n_bets_arr == n).mean() * 100
    print(f"  {n} arena(s): {pct:.1f}%")

print(f"\nBets containing a 2:1 pirate: {has_2_arr.mean()*100:.1f}%")
print(f"Bets WITHOUT any 2:1 pirate: {(~has_2_arr).mean()*100:.1f}%")

print(f"\nBet odds distribution:")
for label, lo, hi in [("2-4", 2, 5), ("4-8", 4, 9), ("8-16", 8, 17),
                        ("16-32", 16, 33), ("32-64", 32, 65), ("64+", 64, 100000)]:
    pct = ((odds_arr >= lo) & (odds_arr < hi)).mean() * 100
    if pct > 0.1:
        print(f"  {label}: {pct:.1f}%")

print(f"\n" + "=" * 70)
print("PROFIT BY BET TYPE")
print("=" * 70)

for key, r in profit_by_type.items():
    roi = r['profit'] / r['spent'] * 100 if r['spent'] > 0 else 0
    print(f"{key:15s}: profit={r['profit']:>12.0f}, spent={r['spent']:>12.0f}, "
          f"ROI={roi:+.1f}%, n_bets={r['n_bets']}")

print(f"\n" + "=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)

ter_roi = ter_profit / ter_spent * 100 if ter_spent > 0 else 0
anchor_roi = anchor_profit / anchor_spent * 100 if anchor_spent > 0 else 0

print(f"{'Strategy':25s} {'Profit':>12s} {'Spent':>12s} {'ROI%':>8s} {'WinDays':>8s}")
print(f"{'TER Logit':25s} {ter_profit:12.0f} {ter_spent:12.0f} {ter_roi:8.1f} {ter_days_with_win:8d}")
print(f"{'Anchor (2:1 p>=0.50)':25s} {anchor_profit:12.0f} {anchor_spent:12.0f} {anchor_roi:8.1f} {anchor_days_with_win:8d}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
