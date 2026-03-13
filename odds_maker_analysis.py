"""
Analyze the odds maker's behavior to determine:
1. Heuristic vs Monte Carlo?
2. Aware of position advantages?
3. Aware of favorites/allergies (in context of arena foods)?
4. Aware that allergy overrides favorite on overlap foods?
"""

import json
import os
from collections import defaultdict
from pirates import PIRATES, COURSES

# Build lookup
PIRATE_BY_NAME = {p.name: p for p in PIRATES}

# Load course category mapping for overlap detection
with open(os.path.join(os.path.dirname(__file__), "pirates.json")) as f:
    RAW = json.load(f)
COURSE_CATS = RAW["courses"]


def load_matches():
    with open(os.path.join(os.path.dirname(__file__), "historical_matches.json")) as f:
        return json.load(f)


def course_counts(pirate, foods):
    """Count favorites (excl overlap) and allergies for a pirate given arena foods."""
    nf = 0
    na = 0
    for food in foods:
        is_fav = food in pirate.favorite_courses
        is_allergy = food in pirate.allergy_courses
        if is_allergy:
            na += 1
        elif is_fav:
            nf += 1
    return nf, na


def course_counts_naive(pirate, foods):
    """Count favorites and allergies WITHOUT overlap logic (fav counts even if also allergy)."""
    nf = 0
    na = 0
    for food in foods:
        is_fav = food in pirate.favorite_courses
        is_allergy = food in pirate.allergy_courses
        if is_fav:
            nf += 1
        if is_allergy:
            na += 1
    return nf, na


def main():
    data = load_matches()
    print(f"Loaded {len(data)} days of data\n")

    # Collect per-entry records: (pirate, odds, position, nf, na, n_overlap, strength, weight, won)
    records = []
    for day in data:
        for arena in day:
            foods = arena["foods"]
            pirates = arena["pirates"]
            winner = arena["winner"]
            for pos, p in enumerate(pirates):
                pirate = PIRATE_BY_NAME[p["name"]]
                nf, na = course_counts(pirate, foods)
                nf_naive, na_naive = course_counts_naive(pirate, foods)
                n_overlap = nf_naive - nf  # foods that are both fav and allergy
                won = 1 if p["name"] == winner else 0
                records.append({
                    "name": p["name"],
                    "odds": p["odds"],
                    "pos": pos,
                    "nf": nf,
                    "na": na,
                    "nf_naive": nf_naive,
                    "n_overlap": n_overlap,
                    "strength": pirate.strength,
                    "weight": pirate.weight,
                    "won": won,
                })

    print(f"Total records: {len(records)}")
    print()

    # =========================================================================
    # 1. ODDS DISTRIBUTION — are they quantized in a heuristic-like way?
    # =========================================================================
    print("=" * 70)
    print("1. ODDS DISTRIBUTION")
    print("=" * 70)
    odds_counts = defaultdict(int)
    odds_wins = defaultdict(int)
    for r in records:
        odds_counts[r["odds"]] += 1
        odds_wins[r["odds"]] += r["won"]

    print(f"{'Odds':>4} {'Count':>8} {'Wins':>6} {'WinRate':>8} {'Implied':>8} {'Actual/Implied':>14}")
    for odds in sorted(odds_counts):
        cnt = odds_counts[odds]
        wins = odds_wins[odds]
        wr = wins / cnt
        implied = 1.0 / odds
        print(f"{odds:4d} {cnt:8d} {wins:6d} {wr:8.4f} {implied:8.4f} {wr/implied:14.4f}")

    # =========================================================================
    # 2. DO ODDS CORRELATE WITH STRENGTH?
    # =========================================================================
    print()
    print("=" * 70)
    print("2. ODDS vs STRENGTH")
    print("=" * 70)
    # For each pirate, what's their average odds?
    pirate_odds = defaultdict(list)
    for r in records:
        pirate_odds[r["name"]].append(r["odds"])

    print(f"{'Pirate':<30} {'Str':>3} {'Wt':>3} {'AvgOdds':>7} {'MinOdds':>7} {'MaxOdds':>7} {'WinRate':>7}")
    pirate_list = sorted(PIRATE_BY_NAME.items(), key=lambda x: x[1].strength, reverse=True)
    for name, p in pirate_list:
        odds_list = pirate_odds[name]
        avg_o = sum(odds_list) / len(odds_list)
        wr = sum(1 for r in records if r["name"] == name and r["won"]) / len(odds_list)
        print(f"{name:<30} {p.strength:3d} {p.weight:3d} {avg_o:7.2f} {min(odds_list):7d} {max(odds_list):7d} {wr:7.4f}")

    # =========================================================================
    # 3. DO ODDS VARY WITH FOOD CONTEXT (FAVORITES/ALLERGIES)?
    # =========================================================================
    print()
    print("=" * 70)
    print("3. ODDS vs FAVORITES/ALLERGIES (food-aware?)")
    print("=" * 70)

    # Group by (pirate_strength_bin, nf, na) to see if odds change with food context
    # First: overall, does odds correlate with nf and na?
    nf_odds = defaultdict(list)
    na_odds = defaultdict(list)
    for r in records:
        nf_odds[r["nf"]].append(r["odds"])
        na_odds[r["na"]].append(r["odds"])

    print("\nAverage odds by number of favorites (allergy-override applied):")
    print(f"{'nFav':>4} {'Count':>8} {'AvgOdds':>8} {'StdOdds':>8}")
    for nf in sorted(nf_odds):
        vals = nf_odds[nf]
        avg = sum(vals) / len(vals)
        std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"{nf:4d} {len(vals):8d} {avg:8.3f} {std:8.3f}")

    print("\nAverage odds by number of allergies:")
    print(f"{'nAll':>4} {'Count':>8} {'AvgOdds':>8} {'StdOdds':>8}")
    for na in sorted(na_odds):
        vals = na_odds[na]
        avg = sum(vals) / len(vals)
        std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"{na:4d} {len(vals):8d} {avg:8.3f} {std:8.3f}")

    # =========================================================================
    # 3b. CONTROL FOR STRENGTH: same pirate, different food contexts
    # =========================================================================
    print()
    print("=" * 70)
    print("3b. SAME PIRATE, DIFFERENT FOOD CONTEXT (controls for strength)")
    print("=" * 70)
    print("  If odds maker knows about foods, same pirate should get different")
    print("  odds depending on how many favs/allergies are in the arena.\n")

    # Per-pirate: average odds by (nf, na)
    pirate_food_odds = defaultdict(lambda: defaultdict(list))
    pirate_food_wins = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["nf"], r["na"])
        pirate_food_odds[r["name"]][key].append(r["odds"])
        pirate_food_wins[r["name"]][key].append(r["won"])

    # Show a few example pirates with high variance
    print(f"{'Pirate':<25} {'(nf,na)':>8} {'Count':>6} {'AvgOdds':>8} {'WinRate':>8}")
    for name, p in sorted(PIRATE_BY_NAME.items(), key=lambda x: x[1].strength, reverse=True)[:6]:
        contexts = pirate_food_odds[name]
        for key in sorted(contexts):
            nf, na = key
            vals = contexts[key]
            wins = pirate_food_wins[name][key]
            avg = sum(vals) / len(vals)
            wr = sum(wins) / len(wins)
            if len(vals) >= 20:
                print(f"{name:<25} ({nf},{na}){' ':>3} {len(vals):6d} {avg:8.2f} {wr:8.4f}")
        print()

    # =========================================================================
    # 4. DO ODDS VARY WITH POSITION?
    # =========================================================================
    print()
    print("=" * 70)
    print("4. ODDS vs POSITION")
    print("=" * 70)

    pos_odds = defaultdict(list)
    pos_wins = defaultdict(list)
    for r in records:
        pos_odds[r["pos"]].append(r["odds"])
        pos_wins[r["pos"]].append(r["won"])

    print(f"{'Pos':>3} {'Count':>8} {'AvgOdds':>8} {'WinRate':>8}")
    for pos in range(4):
        vals = pos_odds[pos]
        wins = pos_wins[pos]
        avg = sum(vals) / len(vals)
        wr = sum(wins) / len(wins)
        print(f"{pos:3d} {len(vals):8d} {avg:8.3f} {wr:8.4f}")

    # =========================================================================
    # 4b. SAME PIRATE, SAME ODDS, DIFFERENT POSITION — does position still matter?
    # =========================================================================
    print()
    print("=" * 70)
    print("4b. SAME PIRATE + SAME ODDS at different positions")
    print("=" * 70)
    print("  If odds maker accounts for position, same pirate at same odds")
    print("  should win at the SAME rate regardless of position.")
    print("  If NOT, position 3 should still win more.\n")

    # Group by (pirate, odds) -> per position win rates
    po_pos = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [wins, count]
    for r in records:
        key = (r["name"], r["odds"])
        po_pos[key][r["pos"]][0] += r["won"]
        po_pos[key][r["pos"]][1] += 1

    # Aggregate across all (pirate, odds) combos
    agg_pos = defaultdict(lambda: [0, 0])
    for key, positions in po_pos.items():
        for pos, (w, c) in positions.items():
            agg_pos[pos][0] += w
            agg_pos[pos][1] += c

    print("Controlling for (pirate, odds):")
    print(f"{'Pos':>3} {'Wins':>8} {'Count':>8} {'WinRate':>8}")
    for pos in range(4):
        w, c = agg_pos[pos]
        print(f"{pos:3d} {w:8d} {c:8d} {w/c:8.4f}")

    # Further control: (pirate, odds, nf, na)
    print("\nControlling for (pirate, odds, nf, na):")
    agg_pos2 = defaultdict(lambda: [0, 0])
    for r in records:
        agg_pos2[r["pos"]][0] += r["won"]
        agg_pos2[r["pos"]][1] += 1

    # Actually do proper control: within each (pirate, odds, nf, na) group,
    # compute per-position win rates, then average the differences
    group_key = lambda r: (r["name"], r["odds"], r["nf"], r["na"])
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        groups[group_key(r)][r["pos"]][0] += r["won"]
        groups[group_key(r)][r["pos"]][1] += 1

    pos_excess = defaultdict(list)
    for gk, positions in groups.items():
        total_w = sum(v[0] for v in positions.values())
        total_c = sum(v[1] for v in positions.values())
        if total_c < 40:
            continue
        base_wr = total_w / total_c
        for pos in range(4):
            if positions[pos][1] >= 5:
                wr = positions[pos][0] / positions[pos][1]
                pos_excess[pos].append(wr - base_wr)

    print(f"{'Pos':>3} {'Groups':>6} {'AvgExcessWR':>12}")
    for pos in range(4):
        vals = pos_excess[pos]
        if vals:
            print(f"{pos:3d} {len(vals):6d} {sum(vals)/len(vals):12.4f}")

    # =========================================================================
    # 5. OVERLAP TEST: does odds maker know allergy overrides favorite?
    # =========================================================================
    print()
    print("=" * 70)
    print("5. OVERLAP TEST: allergy overrides favorite?")
    print("=" * 70)
    print("  Compare odds when pirate has overlap foods vs pure allergy vs pure fav\n")

    # Group records by overlap status
    overlap_data = defaultdict(lambda: {"odds": [], "wins": [], "count": 0})
    for r in records:
        if r["n_overlap"] > 0:
            category = f"overlap={r['n_overlap']}"
        else:
            category = f"nf={r['nf']},na={r['na']}"
        overlap_data[category]["odds"].append(r["odds"])
        overlap_data[category]["wins"].append(r["won"])
        overlap_data[category]["count"] += 1

    # More targeted: compare same pirate with overlap vs without
    # For each pirate, compare arenas where they have overlaps vs not
    print("Per pirate: odds and win rates with overlap vs without")
    print(f"{'Pirate':<25} {'Type':>12} {'Count':>6} {'AvgOdds':>8} {'WinRate':>8}")
    for name, p in sorted(PIRATE_BY_NAME.items(), key=lambda x: x[1].strength, reverse=True):
        # Check if this pirate CAN have overlaps
        overlap_courses = p.favorite_courses & p.allergy_courses
        if not overlap_courses:
            continue

        with_overlap = [r for r in records if r["name"] == name and r["n_overlap"] > 0]
        no_overlap = [r for r in records if r["name"] == name and r["n_overlap"] == 0]

        if len(with_overlap) >= 20 and len(no_overlap) >= 20:
            avg_o_ov = sum(r["odds"] for r in with_overlap) / len(with_overlap)
            wr_ov = sum(r["won"] for r in with_overlap) / len(with_overlap)
            avg_o_no = sum(r["odds"] for r in no_overlap) / len(no_overlap)
            wr_no = sum(r["won"] for r in no_overlap) / len(no_overlap)
            print(f"{name:<25} {'w/ overlap':>12} {len(with_overlap):6d} {avg_o_ov:8.2f} {wr_ov:8.4f}")
            print(f"{'':<25} {'no overlap':>12} {len(no_overlap):6d} {avg_o_no:8.2f} {wr_no:8.4f}")
            print()

    # =========================================================================
    # 5b. CRITICAL TEST: Same pirate, same nf_naive, same na_naive,
    #     WITH vs WITHOUT overlap — does odds maker differentiate?
    # =========================================================================
    print()
    print("=" * 70)
    print("5b. OVERLAP: controlling for naive fav/allergy count")
    print("=" * 70)
    print("  If odds maker uses allergy-overrides-fav, then having an overlap")
    print("  food (which is BOTH fav and allergy) should result in HIGHER odds")
    print("  (weaker) compared to a pure-fav food, even if naive counts match.\n")
    print("  Alternatively, if odds maker counts favs/allergies naively,")
    print("  overlap won't change the odds.\n")

    # For pirates with overlap potential, compare:
    # - Records where n_overlap > 0 vs n_overlap == 0
    # - Controlling for (pirate, nf_naive, na_naive) or (pirate, nf+n_overlap, na)
    overlap_test = defaultdict(lambda: {"ov": [], "no_ov": []})
    for r in records:
        pirate = PIRATE_BY_NAME[r["name"]]
        if not (pirate.favorite_courses & pirate.allergy_courses):
            continue
        # Group by (pirate, nf_naive, na_naive)
        key = (r["name"], r["nf_naive"], r["na"] + r["n_overlap"])
        # Wait - na_naive = na (allergies always count), nf_naive = nf + n_overlap
        # Let me regroup by (pirate, nf_naive, na_naive) to match raw counts
        key = (r["name"], r["nf_naive"], r["na"] + r["n_overlap"])
        if r["n_overlap"] > 0:
            overlap_test[key]["ov"].append(r)
        else:
            overlap_test[key]["no_ov"].append(r)

    # Actually let me just use proper naive counts
    # nf_naive = # foods in favorite_courses (regardless of allergy)
    # na_naive = # foods in allergy_courses (regardless of favorite)
    # When there's overlap: nf_naive includes the overlap food, na_naive includes it too
    # When no overlap: nf_naive = nf, na_naive = na

    # Better approach: for each pirate that has overlap potential,
    # group by the TOTAL number of foods that match any preference
    overlap_test2 = defaultdict(lambda: {"ov": [], "no_ov": []})
    for r in records:
        pirate = PIRATE_BY_NAME[r["name"]]
        if not (pirate.favorite_courses & pirate.allergy_courses):
            continue
        key = (r["name"], r["nf"] + r["n_overlap"], r["na"])
        # nf + n_overlap = how many foods WOULD be favorites if no override
        # na = how many pure allergies
        if r["n_overlap"] > 0:
            overlap_test2[key]["ov"].append(r)
        else:
            overlap_test2[key]["no_ov"].append(r)

    # Aggregate: average odds difference when overlap present vs not
    ov_odds_diffs = []
    ov_wr_diffs = []
    print(f"{'Pirate':<25} {'nf_would':>8} {'na':>3} {'Type':>8} {'N':>5} {'AvgOdds':>8} {'WR':>6}")
    for key in sorted(overlap_test2):
        ov = overlap_test2[key]["ov"]
        no_ov = overlap_test2[key]["no_ov"]
        if len(ov) >= 15 and len(no_ov) >= 15:
            name, nf_would, na = key
            avg_ov = sum(r["odds"] for r in ov) / len(ov)
            avg_no = sum(r["odds"] for r in no_ov) / len(no_ov)
            wr_ov = sum(r["won"] for r in ov) / len(ov)
            wr_no = sum(r["won"] for r in no_ov) / len(no_ov)
            ov_odds_diffs.append(avg_ov - avg_no)
            ov_wr_diffs.append(wr_ov - wr_no)
            print(f"{name:<25} {nf_would:8d} {na:3d} {'overlap':>8} {len(ov):5d} {avg_ov:8.2f} {wr_ov:6.3f}")
            print(f"{'':<25} {'':>8} {'':>3} {'pure':>8} {len(no_ov):5d} {avg_no:8.2f} {wr_no:6.3f}")
            print()

    if ov_odds_diffs:
        print(f"Average odds increase when overlap present: {sum(ov_odds_diffs)/len(ov_odds_diffs):+.3f}")
        print(f"Average WR decrease when overlap present:   {sum(ov_wr_diffs)/len(ov_wr_diffs):+.4f}")

    # =========================================================================
    # 6. HEURISTIC vs MONTE CARLO: analyze odds granularity
    # =========================================================================
    print()
    print("=" * 70)
    print("6. HEURISTIC vs MONTE CARLO DETECTION")
    print("=" * 70)

    # If Monte Carlo: we'd expect the odds to be derived from simulated win probs
    # mapped to integer odds. The mapping should be smooth and the residuals small.
    # If heuristic: odds might follow a simple formula like:
    #   odds = f(strength, position, nf, na)
    # and we should be able to reconstruct the formula.

    # Test: can we predict odds from (strength, nf, na, pos)?
    # Try a simple linear model
    print("\n6a. Linear model: odds ~ strength + nf + na + pos")
    import numpy as np

    X = np.array([[r["strength"], r["nf"], r["na"], r["pos"]] for r in records], dtype=float)
    y = np.array([r["odds"] for r in records], dtype=float)

    # Add intercept
    X_int = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    y_pred = X_int @ beta
    residuals = y - y_pred
    rmse = (residuals ** 2).mean() ** 0.5
    r_sq = 1 - (residuals ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print(f"  Intercept:  {beta[0]:8.3f}")
    print(f"  Strength:   {beta[1]:8.4f}")
    print(f"  nFav:       {beta[2]:8.4f}")
    print(f"  nAllergy:   {beta[3]:8.4f}")
    print(f"  Position:   {beta[4]:8.4f}")
    print(f"  R²:         {r_sq:8.4f}")
    print(f"  RMSE:       {rmse:8.4f}")

    # Test: for a given pirate at a given (nf, na), what's the distribution of odds?
    # If heuristic, odds should be DETERMINISTIC for a given (pirate, nf, na, pos)
    # If Monte Carlo, there might be variation from simulation noise
    print("\n6b. Odds determinism: same (pirate, nf, na) -> same odds?")
    combo_odds = defaultdict(list)
    for r in records:
        key = (r["name"], r["nf"], r["na"])
        combo_odds[key].append(r["odds"])

    n_deterministic = 0
    n_total = 0
    n_varies = 0
    varies_examples = []
    for key, vals in combo_odds.items():
        if len(vals) >= 20:
            n_total += 1
            unique = set(vals)
            if len(unique) == 1:
                n_deterministic += 1
            else:
                n_varies += 1
                if len(varies_examples) < 5:
                    from collections import Counter
                    c = Counter(vals)
                    varies_examples.append((key, len(vals), c.most_common()))

    print(f"  Combos with >=20 samples: {n_total}")
    print(f"  Always same odds: {n_deterministic} ({100*n_deterministic/max(1,n_total):.1f}%)")
    print(f"  Odds vary: {n_varies} ({100*n_varies/max(1,n_total):.1f}%)")

    if varies_examples:
        print("\n  Examples of varying odds:")
        for key, n, dist in varies_examples:
            print(f"    {key[0]:<25} nf={key[1]} na={key[2]}: N={n}, dist={dist}")

    # Now check with position included
    print("\n6c. Same (pirate, nf, na, pos) -> same odds?")
    combo_odds_pos = defaultdict(list)
    for r in records:
        key = (r["name"], r["nf"], r["na"], r["pos"])
        combo_odds_pos[key].append(r["odds"])

    n_det = 0
    n_tot = 0
    n_var = 0
    var_examples = []
    for key, vals in combo_odds_pos.items():
        if len(vals) >= 10:
            n_tot += 1
            unique = set(vals)
            if len(unique) == 1:
                n_det += 1
            else:
                n_var += 1
                if len(var_examples) < 5:
                    from collections import Counter
                    c = Counter(vals)
                    var_examples.append((key, len(vals), c.most_common()))

    print(f"  Combos with >=10 samples: {n_tot}")
    print(f"  Always same odds: {n_det} ({100*n_det/max(1,n_tot):.1f}%)")
    print(f"  Odds vary: {n_var} ({100*n_var/max(1,n_tot):.1f}%)")

    if var_examples:
        print("\n  Examples of varying odds (with position):")
        for key, n, dist in var_examples:
            print(f"    {key[0]:<25} nf={key[1]} na={key[2]} pos={key[3]}: N={n}, dist={dist}")

    # =========================================================================
    # 6d. Check if odds account for OPPONENT strength
    # =========================================================================
    print()
    print("=" * 70)
    print("6d. DO ODDS DEPEND ON OPPONENTS?")
    print("=" * 70)
    print("  If Monte Carlo, odds should change based on who the opponents are.")
    print("  If heuristic based only on own stats, opponents shouldn't matter.\n")

    # For a fixed (pirate, nf, na), check if odds vary with opponent strength
    pirate_context = defaultdict(list)
    for day in data:
        for arena in day:
            foods = arena["foods"]
            pirates = arena["pirates"]
            winner = arena["winner"]
            for pos, p in enumerate(pirates):
                pirate = PIRATE_BY_NAME[p["name"]]
                nf, na = course_counts(pirate, foods)
                # Get opponent strengths
                opp_strengths = []
                for j, op in enumerate(pirates):
                    if j != pos:
                        opp = PIRATE_BY_NAME[op["name"]]
                        opp_nf, opp_na = course_counts(opp, foods)
                        opp_strengths.append((opp.strength, opp_nf, opp_na))
                avg_opp_str = sum(s for s, _, _ in opp_strengths) / 3
                pirate_context[(p["name"], nf, na)].append({
                    "odds": p["odds"],
                    "avg_opp_str": avg_opp_str,
                    "opp_strengths": opp_strengths,
                })

    # For each group, see if odds correlate with opponent strength
    print(f"{'Pirate':<25} {'nf':>2} {'na':>2} {'N':>5} | {'Corr(odds,opp_str)':>18}")
    correlations = []
    for key in sorted(pirate_context, key=lambda k: -PIRATE_BY_NAME[k[0]].strength):
        entries = pirate_context[key]
        if len(entries) < 50:
            continue
        odds_arr = np.array([e["odds"] for e in entries])
        opp_arr = np.array([e["avg_opp_str"] for e in entries])
        if odds_arr.std() < 0.01 or opp_arr.std() < 0.01:
            continue
        corr = np.corrcoef(odds_arr, opp_arr)[0, 1]
        correlations.append(corr)
        name, nf, na = key
        if abs(corr) > 0.05 or len(correlations) <= 10:
            print(f"{name:<25} {nf:2d} {na:2d} {len(entries):5d} | {corr:18.4f}")

    if correlations:
        print(f"\nMean correlation: {np.mean(correlations):.4f}")
        print(f"Median correlation: {np.median(correlations):.4f}")
        print(f"Std correlation: {np.std(correlations):.4f}")

    # =========================================================================
    # 7. ODDS FORMULA RECONSTRUCTION
    # =========================================================================
    print()
    print("=" * 70)
    print("7. ODDS FORMULA RECONSTRUCTION")
    print("=" * 70)
    print("  Try to find the exact mapping: (pirate, nf, na, pos) -> odds\n")

    # For each (pirate, nf, na), what's the modal (most common) odds value?
    print(f"{'Pirate':<25} {'Str':>3} {'nf':>2} {'na':>2} {'ModeOdds':>8} {'Pct':>5} {'N':>5}")
    from collections import Counter
    for name, p in sorted(PIRATE_BY_NAME.items(), key=lambda x: -x[1].strength)[:5]:
        for (nf, na) in sorted(set((r["nf"], r["na"]) for r in records if r["name"] == name)):
            vals = [r["odds"] for r in records if r["name"] == name and r["nf"] == nf and r["na"] == na]
            if len(vals) < 10:
                continue
            c = Counter(vals)
            mode, mode_cnt = c.most_common(1)[0]
            pct = mode_cnt / len(vals) * 100
            print(f"{name:<25} {p.strength:3d} {nf:2d} {na:2d} {mode:8d} {pct:5.1f}% {len(vals):5d}")
        print()

    # =========================================================================
    # 8. DOES THE SUM OF 1/ODDS = 1? (Overround analysis)
    # =========================================================================
    print()
    print("=" * 70)
    print("8. OVERROUND ANALYSIS (sum of implied probabilities per arena)")
    print("=" * 70)

    overrounds = []
    for day in data:
        for arena in day:
            s = sum(1.0 / p["odds"] for p in arena["pirates"])
            overrounds.append(s)

    overrounds = np.array(overrounds)
    print(f"  Mean overround: {overrounds.mean():.4f}")
    print(f"  Std overround:  {overrounds.std():.4f}")
    print(f"  Min overround:  {overrounds.min():.4f}")
    print(f"  Max overround:  {overrounds.max():.4f}")

    # Distribution of overround
    from collections import Counter
    or_rounded = Counter(round(x, 2) for x in overrounds)
    print(f"\n  Overround distribution (top 10):")
    for val, cnt in or_rounded.most_common(10):
        print(f"    {val:.2f}: {cnt} ({100*cnt/len(overrounds):.1f}%)")

    # =========================================================================
    # 9. ODDS BY (PIRATE, POSITION): does position shift odds?
    # =========================================================================
    print()
    print("=" * 70)
    print("9. SAME PIRATE AT DIFFERENT POSITIONS: odds shift?")
    print("=" * 70)

    pirate_pos_odds = defaultdict(lambda: defaultdict(list))
    for r in records:
        pirate_pos_odds[r["name"]][r["pos"]].append(r["odds"])

    print(f"{'Pirate':<25} {'Str':>3} {'Pos0':>6} {'Pos1':>6} {'Pos2':>6} {'Pos3':>6} {'P3-P0':>6}")
    for name, p in sorted(PIRATE_BY_NAME.items(), key=lambda x: -x[1].strength):
        avgs = {}
        for pos in range(4):
            vals = pirate_pos_odds[name][pos]
            if vals:
                avgs[pos] = sum(vals) / len(vals)
        if all(pos in avgs for pos in range(4)):
            print(f"{name:<25} {p.strength:3d} {avgs[0]:6.2f} {avgs[1]:6.2f} {avgs[2]:6.2f} {avgs[3]:6.2f} {avgs[3]-avgs[0]:+6.2f}")


if __name__ == "__main__":
    main()
