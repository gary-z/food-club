import json
from collections import defaultdict

with open("historical_matches.json") as f:
    days = json.load(f)

with open("pirates.json") as f:
    pdata = json.load(f)

pirate_strength = {}
pirate_weight = {}
for p in pdata["pirates"]:
    pirate_strength[p["name"]] = p["strength"]
    pirate_weight[p["name"]] = p["weight"]

# For each pirate in each arena, compute a "composite" that captures more
# than just odds: (pirate_name, odds, n_fav, n_allergy)
# Then check if position still predicts winning after controlling for all of these

# First, we need pirate course data
course_names = pdata["courses"]
course_idx = {name: i for i, name in enumerate(course_names)}
pirate_favs = {}
pirate_allergies = {}
for p in pdata["pirates"]:
    pirate_favs[p["name"]] = set(p["favorites"])
    pirate_allergies[p["name"]] = set(p["allergies"])

# ============================================================
# TEST: Control for (pirate, odds, n_fav, n_allergy, n_opponents_with_lower_odds)
# ============================================================
# (pirate, odds, n_fav, n_allergy) -> pos -> [app, wins]
full_key_pos = defaultdict(lambda: defaultdict(lambda: [0, 0]))
full_key = defaultdict(lambda: [0, 0])

for day in days:
    for arena in day:
        foods = arena["foods"]
        courses = [course_idx.get(f, -1) for f in foods]
        courses = [c for c in courses if c >= 0]

        for pos, p in enumerate(arena["pirates"]):
            name = p["name"]
            odds = p["odds"]
            won = name == arena["winner"]

            # Count favs and allergies using food names directly
            favs = pirate_favs.get(name, set())
            allergies = pirate_allergies.get(name, set())
            n_fav = sum(1 for f in foods if f in favs and f not in allergies)
            n_allergy = sum(1 for f in foods if f in allergies)

            key = (name, odds, n_fav, n_allergy)
            full_key_pos[key][pos][0] += 1
            if won: full_key_pos[key][pos][1] += 1
            full_key[key][0] += 1
            if won: full_key[key][1] += 1

pos_excess = [0.0] * 4
pos_weight = [0.0] * 4

for key, (total_app, total_wins) in full_key.items():
    if total_app < 20:
        continue
    overall_wr = total_wins / total_app
    for pos in range(4):
        app, wins = full_key_pos[key].get(pos, (0, 0))
        if isinstance(app, int) and app >= 3:
            wr = wins / app
            pos_excess[pos] += (wr - overall_wr) * app
            pos_weight[pos] += app

print("=== POSITION EFFECT, CONTROLLING FOR (PIRATE, ODDS, N_FAV, N_ALLERGY) ===\n")
for pos in range(4):
    if pos_weight[pos] > 0:
        print(f"  Position {pos}: excess win rate = {pos_excess[pos]/pos_weight[pos]:+.5f} (n={int(pos_weight[pos])})")

# ============================================================
# TEST 2: What if position IS the sort order of some internal score?
# Check: within an arena, does the winner tend to be at a specific position
# MORE than expected from their odds alone?
# ============================================================
print("\n\n=== WITHIN-ARENA ANALYSIS ===\n")

# For each arena, compute expected win prob from odds: P(i) ∝ 1/odds(i)
# Then check if position adds information beyond odds
pos_lift = defaultdict(list)  # pos -> list of (expected_from_odds, actually_won)

for day in days:
    for arena in day:
        pirates = arena["pirates"]
        inv_odds = [1.0 / p["odds"] for p in pirates]
        total_inv = sum(inv_odds)
        probs = [x / total_inv for x in inv_odds]

        for pos, p in enumerate(pirates):
            won = 1 if p["name"] == arena["winner"] else 0
            pos_lift[pos].append((probs[pos], won))

print("Position vs expected-from-odds:")
print(f"  {'pos':>4} {'avg_expected':>14} {'avg_actual':>12} {'lift':>10}")
for pos in range(4):
    data = pos_lift[pos]
    avg_exp = sum(e for e, w in data) / len(data)
    avg_act = sum(w for e, w in data) / len(data)
    print(f"  {pos:>4} {avg_exp:>14.4f} {avg_act:>12.4f} {avg_act - avg_exp:>+10.4f}")

# ============================================================
# TEST 3: Bin by expected probability from odds, check position effect
# ============================================================
print("\n\n=== POSITION EFFECT WITHIN EXPECTED-PROBABILITY BINS ===\n")
# Bin expected probs into deciles
all_data = []  # (pos, expected_prob, won)
for day in days:
    for arena in day:
        pirates = arena["pirates"]
        inv_odds = [1.0 / p["odds"] for p in pirates]
        total_inv = sum(inv_odds)
        probs = [x / total_inv for x in inv_odds]
        for pos, p in enumerate(pirates):
            won = 1 if p["name"] == arena["winner"] else 0
            all_data.append((pos, probs[pos], won))

# Sort by expected prob and bin
all_data.sort(key=lambda x: x[1])
bin_size = len(all_data) // 10

print(f"  {'bin':>4} {'prob_range':>20} {'pos0_wr':>10} {'pos1_wr':>10} {'pos2_wr':>10} {'pos3_wr':>10}")
print(f"  {'-'*65}")

for b in range(10):
    start = b * bin_size
    end = start + bin_size if b < 9 else len(all_data)
    subset = all_data[start:end]
    prob_lo = subset[0][1]
    prob_hi = subset[-1][1]

    pos_data = defaultdict(lambda: [0, 0])
    for pos, prob, won in subset:
        pos_data[pos][0] += 1
        pos_data[pos][1] += won

    strs = []
    for p in range(4):
        app, wins = pos_data[p]
        if app >= 10:
            strs.append(f"{wins/app:.3f}({app:>4})")
        else:
            strs.append(f"{'  -':>10}")
    print(f"  {b:>4} {prob_lo:.3f}-{prob_hi:.3f}  {strs[0]} {strs[1]} {strs[2]} {strs[3]}")
