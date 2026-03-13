"""
Weight isolation test: allergy=0, controlling for n_fav.
1) str=81/82 pirates at fav=1 and fav=2 (focused)
2) ALL pirates at fav=1 and fav=2 - does weight predict residual after strength?
3) Head-to-head: when two str=81 pirates share the same arena, same n_allergy=0,
   compare their outcomes directly (controls for opponents perfectly).
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
targets = set(str81 + str82)

def count_fav_allergy(pname, foods):
    p = pirate_map[pname]
    fav_cats, allergy_cats = set(p["favorites"]), set(p["allergies"])
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

# ---- PART 1: Focused str=81/82, allergy=0, fav=1 and fav=2 ----
# per_pirate[(name, nf)] = [wins, total]
per_pirate = defaultdict(lambda: [0, 0])

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        for p in arena["pirates"]:
            pname = p["name"]
            if pname not in targets:
                continue
            nf, na = count_fav_allergy(pname, foods)
            if na != 0:
                continue
            won = 1 if pname == winner else 0
            per_pirate[(pname, nf)][0] += won
            per_pirate[(pname, nf)][1] += 1

print("=" * 75)
print("PART 1: str=81/82 pirates, allergy=0, fav=1 and fav=2")
print("If weight ONLY affects allergies, these should depend only on str+fav")
print("=" * 75)

for nf in [0, 1, 2, 3, 4]:
    print(f"\nn_fav={nf}, n_allergy=0:")
    rows = []
    for pname in str81 + str82:
        p = pirate_map[pname]
        key = (pname, nf)
        if key in per_pirate and per_pirate[key][1] >= 20:
            w, t = per_pirate[key]
            wr = w / t
            rows.append((pname, p["strength"], p["weight"], w, t, wr))
    for pname, s, wt, w, t, wr in rows:
        print(f"  {pname:30s} str={s} w={wt:>3}  {w:>4}/{t:<5} WR={wr:.4f}")
    if len(rows) >= 2:
        # Show str=81 only spread
        s81_rows = [r for r in rows if r[1] == 81]
        if len(s81_rows) >= 2:
            wrs = [r[5] for r in s81_rows]
            print(f"  str=81 spread: {(max(wrs)-min(wrs))*100:.1f}pp "
                  f"(weights: {', '.join(str(r[2]) for r in s81_rows)})")

# ---- PART 2: ALL pirates, allergy=0, fav=1 and fav=2 ----
# Check if weight predicts win rate beyond strength
all_stats = defaultdict(lambda: [0, 0])

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        for p in arena["pirates"]:
            pname = p["name"]
            nf, na = count_fav_allergy(pname, foods)
            if na != 0:
                continue
            won = 1 if pname == winner else 0
            all_stats[(pname, nf)][0] += won
            all_stats[(pname, nf)][1] += 1

print("\n\n" + "=" * 75)
print("PART 2: ALL pirates, allergy=0 - does weight predict WR beyond strength?")
print("=" * 75)

for nf in [1, 2]:
    print(f"\nn_fav={nf}, n_allergy=0 (sorted by strength):")
    print(f"  {'Pirate':30s} {'str':>4} {'w':>4} {'wins':>5} {'total':>5} {'WR':>7} {'w_offset':>8}")
    rows = []
    for p in data["pirates"]:
        key = (p["name"], nf)
        if key in all_stats and all_stats[key][1] >= 50:
            w, t = all_stats[key]
            wr = w / t
            wo = min((221 - p["weight"]) // 2, 10)
            rows.append((p["name"], p["strength"], p["weight"], w, t, wr, wo))
    rows.sort(key=lambda r: r[1])
    for pname, s, wt, w, t, wr, wo in rows:
        print(f"  {pname:30s} {s:>4} {wt:>4} {w:>5} {t:>5} {wr:>7.4f} {wo:>8}")

    # Compute correlation of weight with WR residual (after linear strength fit)
    if len(rows) >= 5:
        import numpy as np
        strengths = np.array([r[1] for r in rows], dtype=float)
        weights = np.array([r[2] for r in rows], dtype=float)
        wrs = np.array([r[5] for r in rows], dtype=float)

        # Linear fit: WR ~ a + b*strength
        A = np.column_stack([np.ones_like(strengths), strengths])
        coeffs = np.linalg.lstsq(A, wrs, rcond=None)[0]
        predicted = A @ coeffs
        residuals = wrs - predicted

        # Correlation of residuals with weight
        corr = np.corrcoef(weights, residuals)[0, 1]
        print(f"\n  Strength->WR fit: WR = {coeffs[0]:.4f} + {coeffs[1]:.4f}*str")
        print(f"  Correlation of weight with WR residual: r={corr:.3f}")
        print(f"  (If weight matters, expect significant positive or negative r)")

# ---- PART 3: Head-to-head within same arena ----
print("\n\n" + "=" * 75)
print("PART 3: HEAD-TO-HEAD - str=81 pirates in the SAME arena, both allergy=0")
print("(Perfect opponent control)")
print("=" * 75)

# For each pair of str=81 pirates, find arenas where both appear with allergy=0
from itertools import combinations

pair_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
# pair_stats[(p1,p2)][(nf1,nf2)] = [p1_wins_among_pair, total]
# Actually simpler: pair_stats[(p1,p2)] = [p1_wins, p2_wins, neither_wins, total]
pair_overall = defaultdict(lambda: [0, 0, 0, 0])

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        pirate_names = [p["name"] for p in arena["pirates"]]

        # Find str=81 pirates in this arena
        present = []
        for pname in pirate_names:
            if pname in str81:
                nf, na = count_fav_allergy(pname, foods)
                if na == 0:
                    present.append((pname, nf))

        if len(present) < 2:
            continue

        for i in range(len(present)):
            for j in range(i+1, len(present)):
                p1, nf1 = present[i]
                p2, nf2 = present[j]
                # Canonical order
                key = tuple(sorted([p1, p2]))
                if winner == p1:
                    pair_overall[key][0] += 1
                elif winner == p2:
                    pair_overall[key][1] += 1
                else:
                    pair_overall[key][2] += 1
                pair_overall[key][3] += 1

                # Also track by (nf1, nf2) or (nf_diff)
                nf_diff = nf1 - nf2 if p1 == key[0] else nf2 - nf1
                pair_stats[key][nf_diff][0] += (1 if winner == key[0] else 0)
                pair_stats[key][nf_diff][1] += 1

print("\nOverall head-to-head (both allergy=0 in same arena):")
for key in sorted(pair_overall.keys()):
    p1, p2 = key
    w1, w2, wother, total = pair_overall[key]
    w1_pct = w1/total*100 if total > 0 else 0
    w2_pct = w2/total*100 if total > 0 else 0
    wt1 = pirate_map[p1]["weight"]
    wt2 = pirate_map[p2]["weight"]
    print(f"\n  {p1} (w={wt1}) vs {p2} (w={wt2})")
    print(f"    N={total}, {p1} wins: {w1} ({w1_pct:.1f}%), "
          f"{p2} wins: {w2} ({w2_pct:.1f}%), other: {wother}")

    # Breakdown by fav advantage
    print(f"    By fav advantage (positive = first pirate has more favs):")
    for nf_diff in sorted(pair_stats[key].keys()):
        wins_p1, tot = pair_stats[key][nf_diff]
        wins_p2 = sum(1 for _ in [])  # need to track separately
        p1_wr = wins_p1 / tot if tot > 0 else 0
        if tot >= 5:
            print(f"      fav_diff={nf_diff:+d}: N={tot}, {key[0].split()[0]} wins {wins_p1} ({p1_wr:.1%})")

# ---- PART 4: Same-fav head-to-head (the purest test) ----
print("\n\n" + "=" * 75)
print("PART 4: SAME FAV COUNT head-to-head - str=81 in same arena,")
print("both allergy=0 AND same n_fav (isolates ONLY weight)")
print("=" * 75)

same_fav_h2h = defaultdict(lambda: [0, 0, 0])  # [p1_wins, p2_wins, other]

for day in historical:
    for arena in day:
        foods = arena["foods"]
        winner = arena["winner"]
        pirate_names = [p["name"] for p in arena["pirates"]]

        present = []
        for pname in pirate_names:
            if pname in str81:
                nf, na = count_fav_allergy(pname, foods)
                if na == 0:
                    present.append((pname, nf))

        if len(present) < 2:
            continue

        for i in range(len(present)):
            for j in range(i+1, len(present)):
                p1, nf1 = present[i]
                p2, nf2 = present[j]
                if nf1 != nf2:
                    continue
                key = tuple(sorted([p1, p2]))
                if winner == key[0]:
                    same_fav_h2h[key][0] += 1
                elif winner == key[1]:
                    same_fav_h2h[key][1] += 1
                else:
                    same_fav_h2h[key][2] += 1

for key in sorted(same_fav_h2h.keys()):
    p1, p2 = key
    w1, w2, wother = same_fav_h2h[key]
    total = w1 + w2 + wother
    wt1 = pirate_map[p1]["weight"]
    wt2 = pirate_map[p2]["weight"]
    print(f"\n  {p1} (w={wt1}) vs {p2} (w={wt2}), same fav count, both allergy=0:")
    print(f"    N={total}, {p1.split()[0]} wins: {w1} ({w1/(w1+w2)*100:.1f}% of pair wins), "
          f"{p2.split()[0]} wins: {w2} ({w2/(w1+w2)*100:.1f}% of pair wins), other: {wother}")
    if total > 0:
        heavier = p1 if wt1 > wt2 else p2
        lighter = p2 if wt1 > wt2 else p1
        hw = w1 if wt1 > wt2 else w2
        lw = w2 if wt1 > wt2 else w1
        print(f"    Heavier pirate wins: {hw}/{hw+lw} = {hw/(hw+lw)*100:.1f}% (expect ~50% if weight irrelevant)")
