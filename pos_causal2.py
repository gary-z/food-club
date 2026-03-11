import json
from collections import defaultdict

with open("historical_matches.json") as f:
    days = json.load(f)

with open("pirates.json") as f:
    pdata = json.load(f)

# Build pirate lookup for strength
pirate_strength = {}
for p in pdata["pirates"]:
    pirate_strength[p["name"]] = p["strength"]

# ============================================================
# TEST A: Within odds=2, average strength at each position
# ============================================================
print("=== TEST A: AVERAGE STRENGTH BY POSITION WITHIN EACH ODDS VALUE ===\n")
# (odds, pos) -> list of strengths
odds_pos_strength = defaultdict(list)

for day in days:
    for arena in day:
        for pos, p in enumerate(arena["pirates"]):
            if p["name"] in pirate_strength:
                odds_pos_strength[(p["odds"], pos)].append(pirate_strength[p["name"]])

print(f"{'odds':>5} {'pos0_str':>10} {'pos1_str':>10} {'pos2_str':>10} {'pos3_str':>10}")
print("-" * 50)
for odds in sorted(set(k[0] for k in odds_pos_strength)):
    vals = []
    any_data = False
    for pos in range(4):
        s = odds_pos_strength.get((odds, pos), [])
        if len(s) >= 20:
            vals.append(f"{sum(s)/len(s):>10.1f}")
            any_data = True
        else:
            vals.append(f"{'  -':>10}")
    if any_data:
        print(f"{odds:>5} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

# ============================================================
# TEST B: Control for BOTH odds AND pirate identity
# ============================================================
print("\n\n=== TEST B: POSITION EFFECT CONTROLLING FOR PIRATE IDENTITY ===\n")
# For each pirate, compute their win rate at each position
# Then compare: does a pirate win more at position 3 than position 0?
# This controls for pirate identity (strength, weight, etc.)

pirate_pos = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # pirate -> pos -> [app, wins]

for day in days:
    for arena in day:
        for pos, p in enumerate(arena["pirates"]):
            pirate_pos[p["name"]][pos][0] += 1
            if p["name"] == arena["winner"]:
                pirate_pos[p["name"]][pos][1] += 1

print(f"{'Pirate':<28} {'overall':>8} {'pos0':>8} {'pos1':>8} {'pos2':>8} {'pos3':>8}")
print("-" * 72)

pos_excess_pirate = [0.0] * 4
pos_weight_pirate = [0.0] * 4

pirates_sorted = sorted(pirate_pos.keys())
for name in pirates_sorted:
    total_app = sum(pirate_pos[name][p][0] for p in range(4))
    total_wins = sum(pirate_pos[name][p][1] for p in range(4))
    overall_wr = total_wins / total_app if total_app > 0 else 0

    pos_wrs = []
    for pos in range(4):
        app, wins = pirate_pos[name][pos]
        if app >= 50:
            wr = wins / app
            pos_wrs.append(f"{wr:.3f}")
            pos_excess_pirate[pos] += (wr - overall_wr) * app
            pos_weight_pirate[pos] += app
        else:
            pos_wrs.append("  -  ")

    print(f"{name:<28} {overall_wr:>8.3f} {pos_wrs[0]:>8} {pos_wrs[1]:>8} {pos_wrs[2]:>8} {pos_wrs[3]:>8}")

print(f"\nAggregate position effect (controlled for pirate identity):")
for pos in range(4):
    if pos_weight_pirate[pos] > 0:
        print(f"  Position {pos}: excess win rate = {pos_excess_pirate[pos]/pos_weight_pirate[pos]:+.5f}")

# ============================================================
# TEST C: Control for pirate identity AND odds simultaneously
# ============================================================
print("\n\n=== TEST C: POSITION EFFECT CONTROLLING FOR PIRATE + ODDS ===\n")
# (pirate, odds, pos) -> [app, wins]
pko = defaultdict(lambda: [0, 0])
# (pirate, odds) -> [app, wins]
pk = defaultdict(lambda: [0, 0])

for day in days:
    for arena in day:
        for pos, p in enumerate(arena["pirates"]):
            won = p["name"] == arena["winner"]
            pko[(p["name"], p["odds"], pos)][0] += 1
            if won: pko[(p["name"], p["odds"], pos)][1] += 1
            pk[(p["name"], p["odds"])][0] += 1
            if won: pk[(p["name"], p["odds"])][1] += 1

pos_excess_full = [0.0] * 4
pos_weight_full = [0.0] * 4

for (name, odds), (total_app, total_wins) in pk.items():
    if total_app < 40:
        continue
    overall_wr = total_wins / total_app
    for pos in range(4):
        app, wins = pko.get((name, odds, pos), (0, 0))
        if app >= 5:
            wr = wins / app
            pos_excess_full[pos] += (wr - overall_wr) * app
            pos_weight_full[pos] += app

print("Aggregate position effect (controlled for pirate identity AND odds):")
for pos in range(4):
    if pos_weight_full[pos] > 0:
        print(f"  Position {pos}: excess win rate = {pos_excess_full[pos]/pos_weight_full[pos]:+.5f}")

# ============================================================
# TEST D: Does the ordering reflect a sort by something other than odds?
# ============================================================
print("\n\n=== TEST D: WHAT IS THE ORDERING? ===\n")
# Check: within each arena, are pirates sorted by strength?
sorted_by_str_desc = 0
sorted_by_str_asc = 0
total = 0
str_inversions = 0
str_pairs = 0

for day in days:
    for arena in day:
        total += 1
        strengths = [pirate_strength.get(p["name"], 0) for p in arena["pirates"]]
        if all(strengths[i] <= strengths[i+1] for i in range(3)):
            sorted_by_str_asc += 1
        if all(strengths[i] >= strengths[i+1] for i in range(3)):
            sorted_by_str_desc += 1
        for i in range(len(strengths)):
            for j in range(i+1, len(strengths)):
                str_pairs += 1
                if strengths[j] > strengths[i]:
                    str_inversions += 1

print(f"  Sorted by strength ascending: {sorted_by_str_asc} ({sorted_by_str_asc/total*100:.1f}%)")
print(f"  Sorted by strength descending: {sorted_by_str_desc} ({sorted_by_str_desc/total*100:.1f}%)")
print(f"  Fraction of pairs where later pos has HIGHER strength: {str_inversions/str_pairs:.4f} (0.5 = random)")

# Compare with odds sorting
print(f"\n  (For reference, odds inversion rate was 0.3390)")

# ============================================================
# TEST E: Correlation matrix (position, odds, strength, win)
# ============================================================
print("\n\n=== TEST E: WITHIN ODDS=2, WIN RATE BY POSITION AND STRENGTH ===\n")
# Among odds=2 pirates, split by strength quartile and position
odds2_data = []  # (pos, strength, won)
for day in days:
    for arena in day:
        for pos, p in enumerate(arena["pirates"]):
            if p["odds"] == 2:
                s = pirate_strength.get(p["name"], 0)
                won = p["name"] == arena["winner"]
                odds2_data.append((pos, s, won))

# Split strength into bins
strengths_2 = sorted(set(d[1] for d in odds2_data))
mid = strengths_2[len(strengths_2)//2]
print(f"  Median strength for odds=2: {mid}")
print(f"  {'':>12} {'pos0':>10} {'pos1':>10} {'pos2':>10} {'pos3':>10}")
for label, filt in [("str<=med", lambda s: s <= mid), ("str>med", lambda s: s > mid)]:
    vals = []
    for pos in range(4):
        matches = [(s, w) for (p, s, w) in odds2_data if p == pos and filt(s)]
        if len(matches) >= 20:
            wr = sum(w for s, w in matches) / len(matches)
            vals.append(f"{wr:.3f}({len(matches):>4})")
        else:
            vals.append("     -     ")
    print(f"  {label:>12} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")
