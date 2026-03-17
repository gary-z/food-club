#!/usr/bin/env python3
"""
Backtest the NeoFoodClub TER (Total Expected Return) bet-making strategy
and compare against our strategies.

NeoFoodClub has two probability models:
1. Legacy: odds-based (invert opening odds, clamp, normalize)
2. Logit: multinomial logit with pirate-specific intercepts + fav/allergy/position coefficients

The Max TER strategy:
- Enumerate all 3^5 - 1 = 242 possible bet combinations (can skip arenas)
  Actually it's (pirates+skip)^arenas: each arena picks pirate 1-4 or skip(0).
  Total combos: 5^5 = 3125 including all-skip.
- For each combo: odds = product of selected pirate odds, prob = product of selected pirate probs
- Compute expected return: prob * min(odds * bet_amount, 1_000_000) / bet_amount
  (capped at 1M payout)
- Sort by expected return, take top 10
- Bet amount per bet: min(max_bet, floor(1_000_000 / odds))

For our backtest with historical data, we don't have a max_bet parameter.
We'll use a fixed bet amount (like 10000 NP) and also test with the cap.
"""

import json
import numpy as np
from collections import defaultdict
from math import exp, log, floor

with open('historical_matches.json') as f:
    data = json.load(f)
with open('pirates.json') as f:
    pdata = json.load(f)

pirate_info = {p['name']: p for p in pdata['pirates']}
courses = pdata['courses']
pirate_names_list = [p['name'] for p in pdata['pirates']]

# Build pirate name -> ID mapping (1-indexed as in reference.js)
# Need to figure out the ID mapping from the logit coefficients
# The IDs 1-20 correspond to the pirates in some order
# Let's check: sp0 has intercepts. The pirate with ID 15 has intercept 0 (reference/baseline).
# From the strengths and the logit intercepts, we can figure out the mapping.

# Actually, let's look at the food tables. qw[pirate_id][food_id] = number of fav categories matched.
# We need the pirate ID mapping. Let me extract it from the minified code.

# From the NeoFoodClub source, pirates are indexed 1-20 in a specific order.
# Let me look for the pirate name array.

# Based on standard NeoFoodClub ordering (from their codebase):
PIRATE_NAMES = {
    1: "Scurvy Dan the Blade",
    2: "Young Sproggie",
    3: "Orvinn the First Mate",
    4: "Lucky McKyriggan",
    5: "Sir Edmund Ogletree",
    6: "Peg Leg Percival",
    7: "Bonnie Pip Culliford",
    8: "Puffo the Waister",
    9: "Stuff-A-Roo",
    10: "Squire Venable",
    11: "Captain Crossblades",
    12: "Ol' Stripey",
    13: "Ned the Skipper",
    14: "Fairfax the Deckhand",
    15: "Gooblah the Grarrl",
    16: "Franchisco Corvallio",
    17: "Federismo Corvallio",
    18: "Admiral Blackbeard",
    19: "Buck Cutlass",
    20: "The Tailhook Kid",
}
PIRATE_ID = {v: k for k, v in PIRATE_NAMES.items()}

# Food name -> ID mapping (1-indexed, standard NeoFoodClub order)
FOOD_NAMES = {
    1: "Hotfish", 2: "Wriggling Grub", 3: "Joint Of Ham", 4: "Rainbow Negg",
    5: "Streaky Bacon", 6: "Ultimate Burger", 7: "Bacon Muffin", 8: "Hot Cakes",
    9: "Spicy Wings", 10: "Apple Onion Rings", 11: "Sushi", 12: "Negg Stew",
    13: "Ice Chocolate Cake", 14: "Strochal", 15: "Mallowicious Bar",
    16: "Fungi Pizza", 17: "Broccoli and Cheese Pizza", 18: "Bubbling Blueberry Pizza",
    19: "Grapity Slush", 20: "Rainborific Slush", 21: "Tangy Tropic Slush",
    22: "Blueberry Tomato Blend", 23: "Lemon Blitz", 24: "Fresh Seaweed Pie",
    25: "Flaming Burnumup", 26: "Hot Tyrannian Pepper", 27: "Eye Candy",
    28: "Cheese and Tomato Sub", 29: "Asparagus Pie", 30: "Wild Chocomato",
    31: "Cinnamon Swirl", 32: "Anchovies", 33: "Flaming Fire Faerie Pizza",
    34: "Orange Negg", 35: "Fish Negg", 36: "Super Lemon Grape Slush",
    37: "Rasmelon", 38: "Mustard Ice Cream", 39: "Worm and Leech Pizza",
    40: "Broccoli",
}
FOOD_ID = {v: k for k, v in FOOD_NAMES.items()}

# Logit model coefficients from reference.js
sp0 = {1:-.5794546137649016,2:-2.3340176069570022,3:-3.474037823978662,4:-1.4578830613659912,5:-1.8102069310612203,6:-2.422310039511212,7:-2.3184881291648933,8:-2.928866300910098,9:-3.9102802356052813,10:-3.5320077355301014,11:-3.1314070773414184,12:-2.3663937896941962,13:-1.715608068650775,14:-2.5531507387746055,15:0,16:-1.262467332140883,17:-1.103465227110361,18:-2.2579227035112517,19:-.5650341699607364,20:-1.558154401612824}
bp0 = {1:.15129605892144243,2:.2530527649979796,3:.23974280606406095,4:.17517164487162887,5:.26530335015021544,6:.28883341735028595,7:.23618502676327702,8:.27632288797217713,9:.33911291319200493,10:.20645506433963406,11:.16258334490695758,12:.2267205822583252,13:.2373537543131154,14:.24942106458862776,15:.2667678760178308,16:.1831021036769767,17:.15741968373448073,18:.18353049785395714,19:.253471447136691,20:.27045548959604065}
pp0 = {1:.4696065161547852,2:.32395212072128166,3:.2890471502222602,4:.5181910373170984,5:.3804292931289034,6:.39126649935517344,7:.31072511120701124,8:.3044291939500264,9:.23833833294864382,10:.3437123089907127,11:.40922317817650533,12:.4675485779040732,13:.46902371461304243,14:.3603534368675031,15:.4832855296602312,16:.43249902125801176,17:.4720557851980187,18:.463493123784612,19:.420171654938422,20:.3705750455448484}
Mp0 = {1:.04376402324072572,2:.02538118023096806,3:.2384139905094721,4:.2735844593834547,5:.1793367367423826,6:.14811293125663813,7:.40193527419447694,8:.06210568149558645,9:.23653557508585984,10:.537623959110779,11:.5730376051750228,12:.3351573750340522,13:.37369492849128266,14:.20858138294274442,15:.16606944424047773,16:.08813480508408332,17:.07565821843203682,18:.43401118190707755,19:.21672179670665537,20:.0875079874018897}
lp0 = {1:.3524928498440687,2:.3843393293695538,3:.6484090757422685,4:.5600714124680725,5:.4562077578067562,6:.3215799883521333,7:.5903302898393084,8:.2600818572454927,9:.6099452657297066,10:.8491220677783762,11:.595170109365996,12:.5926666694068489,13:.5537704402652582,14:.4772478538294615,15:.40225534376314526,16:.37971456716173024,17:.240666666912557,18:.6649015928873812,19:.5324813774827821,20:.5108294958415198}
zp0 = {1:.556254968120439,2:.6153941832210503,3:.8483294355086128,4:.8610947241531649,5:.7620600438458967,6:.6380752472938512,7:.8766237076115079,8:.6572134020019397,9:1.0251541436609148,10:1.0175029272045701,11:1.0785825159241023,12:.9848969531866039,13:.9509999512072826,14:.7193771178192622,15:.5562556504529854,16:.737359077852164,17:.5458781702452122,18:.9633552191404998,19:.7259614979158966,20:.7715448357294399}

# Build fav/allergy count tables per pirate per food
# qw[pirate_id][food_id] = number of fav categories
# mw[pirate_id][food_id] = number of allergy categories
# We build these from our pirates.json data

def build_food_tables():
    """Build fav and allergy count tables matching NeoFoodClub's qw/mw."""
    qw = {}
    mw = {}
    for pid, pname in PIRATE_NAMES.items():
        pi = pirate_info[pname]
        fav_cats = set(pi['favorites'])
        allg_cats = set(pi['allergies'])
        qw[pid] = {}
        mw[pid] = {}
        for fid, fname in FOOD_NAMES.items():
            food_cats = set(courses.get(fname, []))
            # Fav count = number of matching fav categories
            qw[pid][fid] = len(fav_cats & food_cats)
            # Allergy count = number of matching allergy categories
            mw[pid][fid] = len(allg_cats & food_cats)
    return qw, mw

qw, mw = build_food_tables()


def compute_logit_probs(arena_pirates, arena_foods):
    """
    Compute logit model probabilities for 4 pirates in an arena.
    arena_pirates: list of 4 pirate IDs (1-indexed)
    arena_foods: list of 10 food IDs (1-indexed)
    Returns: list of 4 probabilities
    """
    # Compute fav and allergy counts per pirate
    fav_counts = [0] * 4
    allg_counts = [0] * 4
    for i in range(4):
        pid = arena_pirates[i]
        for fid in arena_foods:
            fav_counts[i] += qw[pid][fid]
            allg_counts[i] += mw[pid][fid]

    # Compute logit scores
    scores = [0.0] * 4
    for i in range(4):
        pid = arena_pirates[i]
        s = sp0[pid]  # intercept
        s += bp0[pid] * fav_counts[i]  # fav effect
        s += pp0[pid] * (-allg_counts[i])  # allergy effect (note: negative)
        # Wait - let me recheck. In the code: s+=pp0[c]*p where p=o[i][a] which is -allg_count
        # So pp0 * (-allg_count) = -pp0 * allg_count. Allergies reduce the score.
        # Actually o[i][a] = -mw sum, so p is negative. s += pp0[pid] * p means
        # s += pp0[pid] * (-allg_count). Since pp0 is positive, this reduces s.
        # That's correct.

        # Position effects (position 1=idx1, 2=idx2, 3=idx3)
        if i == 1:
            s += Mp0[pid]
        elif i == 2:
            s += lp0[pid]
        elif i == 3:
            s += zp0[pid]

        scores[i] = exp(s)

    total = sum(scores)
    probs = [s / total for s in scores]
    return probs


def compute_legacy_probs(arena_odds):
    """
    Compute legacy (odds-based) probabilities.
    arena_odds: list of 4 opening odds values
    """
    # From dp0 in reference.js:
    # For each pirate: compute min and max prob from odds
    # odds=13 → [0, 1/13], odds=2 → [1/3, 1], else → [1/(1+odds), 1/odds]
    # Then normalize using the "std" estimation procedure

    mins = [0.0] * 4
    maxs = [0.0] * 4
    for i in range(4):
        odds = arena_odds[i]
        if odds == 13:
            mins[i] = 0
            maxs[i] = 1/13
        elif odds == 2:
            mins[i] = 1/3
            maxs[i] = 1
        else:
            mins[i] = 1 / (1 + odds)
            maxs[i] = 1 / odds

    # Compute std probabilities
    std = [0.0] * 4
    min_sum = sum(mins)
    max_sum = sum(maxs)

    for i in range(4):
        odds = arena_odds[i]
        if odds == 13:
            std[i] = 0.05
        else:
            std[i] = (mins[i] + maxs[i]) / 2

    # Adjustment loop (from dp0)
    a = 2
    while a <= 13:
        s = sum(std)
        if s == 1:
            break

        count = 0
        deficit = 0
        min_margin = 1
        for i in range(4):
            if arena_odds[i] <= a:
                count += 1
                deficit += std[i] - mins[i]
                min_margin = min(min_margin, maxs[i] - mins[i])

        if s - deficit > 1 or count == 0 or deficit + 1 - s > min_margin * count:
            a += 1
            continue

        adjustment = (1 - s + deficit) / count
        for i in range(4):
            if arena_odds[i] <= a:
                std[i] = mins[i] + (deficit + 1 - s) / count
        break

    # Normalize
    total = sum(std)
    probs = [s / total for s in std]
    return probs


def generate_max_ter_bets(arena_odds, arena_probs, max_bet, n_bets=10):
    """
    Generate Max TER bet set.
    arena_odds: 5x4 array of opening odds (0-indexed arenas, 0-indexed pirates)
    arena_probs: 5x4 array of win probabilities
    max_bet: maximum bet amount per bet
    n_bets: number of bets to place (default 10)
    Returns: list of (bet_spec, bet_amount, odds, prob, er) tuples
    """
    # Enumerate all 5^5 = 3125 combinations
    # Each arena: 0=skip, 1-4=pirate index
    combos = []

    for a0 in range(5):
        for a1 in range(5):
            for a2 in range(5):
                for a3 in range(5):
                    for a4 in range(5):
                        spec = (a0, a1, a2, a3, a4)
                        if all(s == 0 for s in spec):
                            continue  # skip all-empty

                        odds = 1
                        prob = 1.0
                        for arena_idx, pirate_choice in enumerate(spec):
                            if pirate_choice > 0:
                                odds *= arena_odds[arena_idx][pirate_choice - 1]
                                prob *= arena_probs[arena_idx][pirate_choice - 1]

                        bet_amount = min(max_bet, max(1, floor(1_000_000 / odds)))
                        payout = min(1_000_000, odds * bet_amount)

                        if max_bet < 50:
                            # Very small bets: use simple ratio
                            er = odds * prob
                        else:
                            er = prob * payout / bet_amount

                        combos.append((spec, bet_amount, odds, prob, er))

    # Sort by expected return ratio, take top n_bets
    combos.sort(key=lambda x: x[4], reverse=True)
    return combos[:n_bets]


def resolve_arena(day_data):
    """Convert a day's data into arena_pirates, arena_odds, arena_foods, winners."""
    arenas = []
    for arena in day_data:
        pirates = []
        odds = []
        foods = []
        winner_pos = -1

        for i, p in enumerate(arena['pirates']):
            pid = PIRATE_ID.get(p['name'])
            if pid is None:
                raise ValueError(f"Unknown pirate: {p['name']}")
            pirates.append(pid)
            odds.append(p['odds'])
            if p['name'] == arena['winner']:
                winner_pos = i

        for food_name in arena['foods']:
            fid = FOOD_ID.get(food_name)
            if fid is None:
                raise ValueError(f"Unknown food: {food_name}")
            foods.append(fid)

        arenas.append({
            'pirates': pirates,
            'odds': odds,
            'foods': foods,
            'winner_pos': winner_pos,
        })
    return arenas


###############################################################################
# BACKTEST
###############################################################################

MAX_BET = 10000  # Fixed bet amount for comparison
N_BETS = 10

print(f"=== BACKTEST: NeoFoodClub Max TER Strategy ===")
print(f"Max bet: {MAX_BET}, Bets per day: {N_BETS}")
print(f"Days: {len(data)}\n")

# Track results for each strategy
results = {
    'ter_logit': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
    'ter_legacy': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
}

# Also implement our strategies for comparison
# Strategy: bet on all odds=2 pirates (simple singles)
our_results = {
    'all_2s': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
    'filtered_2s_50': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
}

for day_idx, day_data in enumerate(data):
    if day_idx % 500 == 0:
        print(f"  Processing day {day_idx}/{len(data)}...")

    try:
        arenas = resolve_arena(day_data)
    except ValueError as e:
        continue

    # Compute probabilities for each arena
    arena_odds_arr = []
    logit_probs_arr = []
    legacy_probs_arr = []
    winners = []

    for arena in arenas:
        arena_odds_arr.append(arena['odds'])
        logit_probs_arr.append(compute_logit_probs(arena['pirates'], arena['foods']))
        legacy_probs_arr.append(compute_legacy_probs(arena['odds']))
        winners.append(arena['winner_pos'])

    # Generate Max TER bets (logit)
    logit_bets = generate_max_ter_bets(arena_odds_arr, logit_probs_arr, MAX_BET, N_BETS)

    # Generate Max TER bets (legacy)
    legacy_bets = generate_max_ter_bets(arena_odds_arr, legacy_probs_arr, MAX_BET, N_BETS)

    # Evaluate bets
    for strategy_name, bets in [('ter_logit', logit_bets), ('ter_legacy', legacy_bets)]:
        day_spent = 0
        day_won = 0
        for spec, bet_amount, odds, prob, er in bets:
            day_spent += bet_amount
            # Check if bet wins
            won = True
            for arena_idx, pirate_choice in enumerate(spec):
                if pirate_choice > 0 and (pirate_choice - 1) != winners[arena_idx]:
                    won = False
                    break
            if won:
                payout = min(1_000_000, odds * bet_amount)
                day_won += payout

        results[strategy_name]['spent'] += day_spent
        results[strategy_name]['profit'] += day_won - day_spent
        results[strategy_name]['days'] += 1
        if day_won > 0:
            results[strategy_name]['wins'] += 1

    # Our strategies
    # All 2:1 singles
    day_2s_spent = 0
    day_2s_won = 0
    for ai, arena in enumerate(arenas):
        for pi in range(4):
            if arena['odds'][pi] == 2:
                day_2s_spent += MAX_BET
                if pi == winners[ai]:
                    day_2s_won += 2 * MAX_BET

    our_results['all_2s']['spent'] += day_2s_spent
    our_results['all_2s']['profit'] += day_2s_won - day_2s_spent
    our_results['all_2s']['days'] += 1

    # Filtered 2:1 (logit p >= 0.50)
    day_f2_spent = 0
    day_f2_won = 0
    for ai, arena in enumerate(arenas):
        for pi in range(4):
            if arena['odds'][pi] == 2 and logit_probs_arr[ai][pi] >= 0.50:
                day_f2_spent += MAX_BET
                if pi == winners[ai]:
                    day_f2_won += 2 * MAX_BET

    our_results['filtered_2s_50']['spent'] += day_f2_spent
    our_results['filtered_2s_50']['profit'] += day_f2_won - day_f2_spent
    our_results['filtered_2s_50']['days'] += 1


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n{'Strategy':25s} {'Profit':>12s} {'Spent':>12s} {'ROI%':>8s} {'Win Days':>10s} {'Win%':>7s}")
for name, r in {**results, **our_results}.items():
    roi = r['profit'] / r['spent'] * 100 if r['spent'] > 0 else 0
    win_pct = r['wins'] / r['days'] * 100 if r['days'] > 0 else 0
    print(f"{name:25s} {r['profit']:12.0f} {r['spent']:12.0f} {roi:8.1f} {r['wins']:10d} {win_pct:7.1f}")

# Per-era comparison
print("\n" + "=" * 70)
print("BY ERA")
print("=" * 70)

for era_name, era_start, era_end in [("Legacy (0-4449)", 0, 4450), ("Modern (4450+)", 4450, len(data))]:
    print(f"\n--- {era_name} ---")

    era_results = {
        'ter_logit': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
        'ter_legacy': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
        'all_2s': {'profit': 0, 'spent': 0, 'wins': 0, 'days': 0},
    }

    for day_idx in range(era_start, min(era_end, len(data))):
        day_data = data[day_idx]
        try:
            arenas = resolve_arena(day_data)
        except ValueError:
            continue

        arena_odds_arr = []
        logit_probs_arr = []
        legacy_probs_arr = []
        winners = []

        for arena in arenas:
            arena_odds_arr.append(arena['odds'])
            logit_probs_arr.append(compute_logit_probs(arena['pirates'], arena['foods']))
            legacy_probs_arr.append(compute_legacy_probs(arena['odds']))
            winners.append(arena['winner_pos'])

        for strategy_name, bets_fn, probs in [
            ('ter_logit', generate_max_ter_bets, logit_probs_arr),
            ('ter_legacy', generate_max_ter_bets, legacy_probs_arr),
        ]:
            bets = bets_fn(arena_odds_arr, probs, MAX_BET, N_BETS)
            day_spent = 0
            day_won = 0
            for spec, bet_amount, odds, prob, er in bets:
                day_spent += bet_amount
                won = True
                for arena_idx, pirate_choice in enumerate(spec):
                    if pirate_choice > 0 and (pirate_choice - 1) != winners[arena_idx]:
                        won = False
                        break
                if won:
                    payout = min(1_000_000, odds * bet_amount)
                    day_won += payout

            era_results[strategy_name]['spent'] += day_spent
            era_results[strategy_name]['profit'] += day_won - day_spent
            era_results[strategy_name]['days'] += 1
            if day_won > 0:
                era_results[strategy_name]['wins'] += 1

        # All 2s
        day_2s_spent = 0
        day_2s_won = 0
        for ai, arena in enumerate(arenas):
            for pi in range(4):
                if arena['odds'][pi] == 2:
                    day_2s_spent += MAX_BET
                    if pi == winners[ai]:
                        day_2s_won += 2 * MAX_BET
        era_results['all_2s']['spent'] += day_2s_spent
        era_results['all_2s']['profit'] += day_2s_won - day_2s_spent
        era_results['all_2s']['days'] += 1

    print(f"{'Strategy':25s} {'Profit':>12s} {'Spent':>12s} {'ROI%':>8s} {'Win%':>7s}")
    for name, r in era_results.items():
        roi = r['profit'] / r['spent'] * 100 if r['spent'] > 0 else 0
        win_pct = r['wins'] / r['days'] * 100 if r['days'] > 0 else 0
        print(f"{name:25s} {r['profit']:12.0f} {r['spent']:12.0f} {roi:8.1f} {win_pct:7.1f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
