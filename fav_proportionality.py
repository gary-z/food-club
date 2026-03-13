"""
Pooled analysis: str=81 pirates, 0 allergies.
Test: is win rate directly proportional to number of favorites?
Also verify weight has no effect by comparing within same n_fav.
"""
import json
from collections import defaultdict

with open("pirates.json") as f:
    data = json.load(f)
with open("historical_matches.json") as f:
    historical = json.load(f)

course_cats = {name: set(cats) for name, cats in data["courses"].items()}
pirate_map = {p["name"]: p for p in data["pirates"]}

str81 = ["Franchisco Corvallio", "Federismo Corvallio", "The Tailhook Kid"]
str82 = ["Lucky McKyriggan"]
all_targets = str81 + str82

def count_fav_allergy(pname, foods):
    p = pirate_map[pname]
    fav_cats = set(p["favorites"])
    allergy_cats = set(p["allergies"])
    nf = na = 0
    for food in foods:
        cats = course_cats.get(food, set())
        is_f = bool(cats & fav_cats)
        is_a = bool(cats & allergy_cats)
        if is_f and is_a:
            na += 1
        elif is_a:
            na += 1
        elif is_f:
            nf += 1
    return nf, na

# Pooled stats for str=81: stats_81[n_fav] = [wins, total]
stats_81 = defaultdict(lambda: [0, 0])
# Per-pirate (for weight comparison)
per_pirate = {n: defaultdict(lambda: [0, 0]) for n in all_targets}
# str=82
stats_82 = defaultdict(lambda: [0, 0])

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        pirate_names = [p["name"] for p in arena["pirates"]]

        for pname in pirate_names:
            if pname not in pirate_map or pname not in all_targets:
                continue
            nf, na = count_fav_allergy(pname, foods)
            if na != 0:
                continue
            won = 1 if pname == winner else 0
            per_pirate[pname][nf][0] += won
            per_pirate[pname][nf][1] += 1
            if pname in str81:
                stats_81[nf][0] += won
                stats_81[nf][1] += 1
            else:
                stats_82[nf][0] += won
                stats_82[nf][1] += 1

print("=" * 70)
print("POOLED STR=81 PIRATES (0 allergies) - Franchisco, Federismo, Tailhook")
print("=" * 70)
print(f"{'n_fav':>5} {'wins':>6} {'total':>6} {'WR':>8} {'WR/WR(0)':>10} {'WR/WR(1)':>10}")

wr_at = {}
for nf in sorted(stats_81.keys()):
    w, t = stats_81[nf]
    wr = w / t if t > 0 else 0
    wr_at[nf] = wr
    r0 = f"{wr/wr_at[0]:.3f}" if 0 in wr_at and wr_at[0] > 0 else "---"
    r1 = f"{wr/wr_at.get(1,1):.3f}" if 1 in wr_at and wr_at.get(1,0) > 0 else "---"
    print(f"{nf:>5} {w:>6} {t:>6} {wr:>8.4f} {r0:>10} {r1:>10}")

print(f"\nSTR=82 (Lucky McKyriggan, 0 allergies):")
print(f"{'n_fav':>5} {'wins':>6} {'total':>6} {'WR':>8} {'WR/WR(0)':>10}")
wr82 = {}
for nf in sorted(stats_82.keys()):
    w, t = stats_82[nf]
    wr = w / t if t > 0 else 0
    wr82[nf] = wr
    r0 = f"{wr/wr82[0]:.3f}" if 0 in wr82 and wr82[0] > 0 else "---"
    print(f"{nf:>5} {w:>6} {t:>6} {wr:>8.4f} {r0:>10}")

# Direct proportionality test: WR(n) = c * n would mean WR(0)=0.
# More likely the question is: WR(n) - WR(0) proportional to n?
# i.e., each fav adds a constant boost?
print("\n" + "=" * 70)
print("PROPORTIONALITY TEST (pooled str=81)")
print("Direct proportional: WR(n)/n should be constant")
print("Additive: WR(n)-WR(0) should be linear in n")
print("Multiplicative: WR(n)/WR(0) should follow a^n pattern")
print("=" * 70)

print(f"\n{'n_fav':>5} {'WR':>8} {'WR/n':>8} {'WR-WR(0)':>10} {'delta/n':>8} {'WR/WR(0)':>10} {'per-fav mult':>12}")
base = wr_at.get(0, 0)
for nf in sorted(wr_at.keys()):
    wr = wr_at[nf]
    wr_over_n = f"{wr/nf:.4f}" if nf > 0 else "---"
    delta = wr - base
    delta_over_n = f"{delta/nf:.4f}" if nf > 0 else "---"
    ratio = wr / base if base > 0 else 0
    per_fav = f"{ratio**(1/nf):.4f}" if nf > 0 and ratio > 0 else "---"
    print(f"{nf:>5} {wr:>8.4f} {wr_over_n:>8} {delta:>10.4f} {delta_over_n:>8} {ratio:>10.4f} {per_fav:>12}")

# Weight comparison within n_fav buckets (the cleanest test)
print("\n" + "=" * 70)
print("WEIGHT COMPARISON: same str=81, same n_fav, different weights")
print("If weight doesn't matter with 0 allergies, WRs should be ~equal")
print("=" * 70)

for nf in range(6):
    rows = []
    for pname in str81:
        p = pirate_map[pname]
        if nf in per_pirate[pname]:
            w, t = per_pirate[pname][nf]
            if t >= 30:
                wr = w / t
                rows.append((pname, p["weight"], w, t, wr))
    if len(rows) >= 2:
        print(f"\nn_fav={nf}:")
        for pname, weight, w, t, wr in rows:
            print(f"  {pname:30s} w={weight:>3}  {w:>4}/{t:<5}  WR={wr:.4f}")
        wrs = [r[4] for r in rows]
        spread = max(wrs) - min(wrs)
        print(f"  -> Spread: {spread:.4f} ({spread*100:.1f}pp)")
