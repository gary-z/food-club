"""
Analyze win rates for str 81/82 pirates when they have 0 allergies in a match.
Question: Is win rate directly proportional to number of favorites?
"""
import json
from collections import defaultdict

# Load data
with open("pirates.json") as f:
    data = json.load(f)

with open("historical_matches.json") as f:
    historical = json.load(f)

# Build course -> categories mapping
course_cats = {}
for course_name, cats in data["courses"].items():
    course_cats[course_name] = set(cats)

# Build pirate lookup
pirate_map = {p["name"]: p for p in data["pirates"]}

# Target pirates (str 81 and 82)
targets = [
    "Franchisco Corvallio",   # str=81, w=165, favs=[Spicy,Meats], allergy=[Candy]
    "Federismo Corvallio",    # str=81, w=166, favs=[Gross,Pizza], allergy=[Smoothies]
    "The Tailhook Kid",       # str=81, w=207, favs=[Vegetables], allergy=[Neggs]
    "Lucky McKyriggan",       # str=82, w=182, favs=[Gross foods], allergy=[Pizza]
]

def count_favs_allergies(pirate_name, foods):
    """Count effective favorites and allergies for a pirate given the 10 foods."""
    p = pirate_map[pirate_name]
    fav_cats = set(p["favorites"])
    allergy_cats = set(p["allergies"])

    nf = 0
    na = 0
    for food in foods:
        cats = course_cats.get(food, set())
        is_fav = bool(cats & fav_cats)
        is_allergy = bool(cats & allergy_cats)
        if is_fav and is_allergy:
            na += 1  # allergy wins over favorite
        elif is_allergy:
            na += 1
        elif is_fav:
            nf += 1
    return nf, na


# Collect: per pirate, per n_fav (when n_allergy=0): wins and total
# stats[pirate_name][n_fav] = [wins, total]
stats = {name: defaultdict(lambda: [0, 0]) for name in targets}

# Also collect overall stats for any n_allergy
stats_all = {name: defaultdict(lambda: [0, 0]) for name in targets}

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        pirates_in_arena = [p["name"] for p in arena["pirates"]]

        for pname in pirates_in_arena:
            if pname not in stats:
                continue
            nf, na = count_favs_allergies(pname, foods)

            won = 1 if pname == winner else 0

            stats_all[pname][(nf, na)][0] += won
            stats_all[pname][(nf, na)][1] += 1

            if na == 0:
                stats[pname][nf][0] += won
                stats[pname][nf][1] += 1

print("=" * 80)
print("WIN RATES BY N_FAV WHEN N_ALLERGY = 0")
print("(Weight only matters for allergies, so with 0 allergies, weight is irrelevant)")
print("=" * 80)

for name in targets:
    p = pirate_map[name]
    print(f"\n{name} (str={p['strength']}, w={p['weight']}, "
          f"fav_cats={p['favorites']}, allergy_cats={p['allergies']})")
    print(f"  {'n_fav':>5}  {'wins':>6}  {'total':>6}  {'win_rate':>8}  {'ratio_to_0fav':>13}")

    sorted_favs = sorted(stats[name].keys())
    base_wr = None
    for nf in sorted_favs:
        wins, total = stats[name][nf]
        wr = wins / total if total > 0 else 0
        if nf == 0:
            base_wr = wr
        ratio = f"{wr/base_wr:.3f}" if base_wr and base_wr > 0 else "---"
        print(f"  {nf:>5}  {wins:>6}  {total:>6}  {wr:>8.4f}  {ratio:>13}")

# Now let's do a direct comparison: same n_fav, 0 allergies, across pirates
print("\n" + "=" * 80)
print("CROSS-PIRATE COMPARISON: SAME N_FAV, 0 ALLERGIES")
print("If weight doesn't matter (no allergies), pirates with same strength")
print("should have same base win rate. Differences = effect of n_fav.")
print("=" * 80)

# Collect all n_fav values across all targets
all_nfavs = set()
for name in targets:
    all_nfavs.update(stats[name].keys())

for nf in sorted(all_nfavs):
    print(f"\nn_fav = {nf}:")
    for name in targets:
        p = pirate_map[name]
        if nf in stats[name]:
            wins, total = stats[name][nf]
            wr = wins / total if total > 0 else 0
            print(f"  {name:30s} str={p['strength']} w={p['weight']}  "
                  f"wins={wins:>4}/{total:<5}  WR={wr:.4f}")

# Show total fav counts distribution (how often each n_fav occurs)
print("\n" + "=" * 80)
print("FAVORITE COUNT DISTRIBUTION (when 0 allergies)")
print("=" * 80)
for name in targets:
    p = pirate_map[name]
    total_matches = sum(v[1] for v in stats[name].values())
    print(f"\n{name} (fav_cats={p['favorites']}):")
    for nf in sorted(stats[name].keys()):
        wins, total = stats[name][nf]
        pct = total / total_matches * 100 if total_matches > 0 else 0
        print(f"  n_fav={nf}: {total:>5} matches ({pct:5.1f}%)")

# Key test: for str=81 pirates at n_fav=0, n_allergy=0, win rates should be identical
# regardless of weight (since weight only affects allergies)
print("\n" + "=" * 80)
print("KEY TEST: str=81 pirates, 0 favs, 0 allergies")
print("If weight ONLY affects allergies, these should have IDENTICAL win rates")
print("(they differ only in weight: 165, 166, 207)")
print("=" * 80)

str81_pirates = ["Franchisco Corvallio", "Federismo Corvallio", "The Tailhook Kid"]
for name in str81_pirates:
    p = pirate_map[name]
    if 0 in stats[name]:
        wins, total = stats[name][0]
        wr = wins / total if total > 0 else 0
        print(f"  {name:30s} w={p['weight']:>3}  wins={wins:>4}/{total:<5}  WR={wr:.4f}")
    else:
        print(f"  {name:30s} w={p['weight']:>3}  NO DATA for 0 fav, 0 allergy")

print("\n" + "=" * 80)
print("PROPORTIONALITY TEST")
print("If win rate is proportional to n_fav, then WR(n) / WR(0) should be linear in n")
print("=" * 80)

for name in targets:
    p = pirate_map[name]
    sorted_favs = sorted(stats[name].keys())
    if 0 not in stats[name] or stats[name][0][1] < 50:
        print(f"\n{name}: insufficient data at n_fav=0")
        continue

    base_wins, base_total = stats[name][0]
    base_wr = base_wins / base_total

    print(f"\n{name} (str={p['strength']}):")
    print(f"  {'n_fav':>5}  {'WR':>8}  {'WR/WR(0)':>10}  {'linear(1+k*n)':>14}  {'note':>20}")

    # Find best-fit linear coefficient
    data_points = []
    for nf in sorted_favs:
        wins, total = stats[name][nf]
        if total >= 30:
            wr = wins / total
            data_points.append((nf, wr / base_wr))

    # Simple linear regression through (0, 1): ratio = 1 + k*n
    if len(data_points) > 1:
        sum_n_r = sum(n * r for n, r in data_points if n > 0)
        sum_n2 = sum(n * n for n, r in data_points if n > 0)
        k = (sum_n_r - sum(n for n, _ in data_points if n > 0)) / sum_n2 if sum_n2 > 0 else 0

        for nf in sorted_favs:
            wins, total = stats[name][nf]
            if total < 30:
                continue
            wr = wins / total
            ratio = wr / base_wr
            linear_pred = 1 + k * nf
            note = f"k={k:.4f}" if nf == 1 else ""
            print(f"  {nf:>5}  {wr:>8.4f}  {ratio:>10.4f}  {linear_pred:>14.4f}  {note:>20}")
