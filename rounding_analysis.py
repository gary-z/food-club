"""
Investigate whether the game rounds strength/weight to nearest 5, 10, etc.
If rounding occurs, pirates in the same bucket should have suspiciously
similar win rates after controlling for favorites/allergies.
"""

import json
import os
import math
from collections import defaultdict
from itertools import combinations
from pirates import PIRATES

PIRATE_BY_NAME = {p.name: p for p in PIRATES}


def load_matches():
    with open(os.path.join(os.path.dirname(__file__), "historical_matches.json")) as f:
        return json.load(f)


def course_counts(pirate, foods):
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


def round_floor(val, step):
    return (val // step) * step


def round_ceil(val, step):
    return math.ceil(val / step) * step


def round_nearest(val, step):
    return round(val / step) * step


def main():
    data = load_matches()
    print(f"Loaded {len(data)} days\n")

    # Build per-pirate per-(nf,na) win rate records
    pirate_food_records = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [wins, count]
    # Also build per-pirate per-(nf,na) per-position records for odds
    pirate_food_odds = defaultdict(lambda: defaultdict(list))

    for day in data:
        for arena in day:
            foods = arena["foods"]
            winner = arena["winner"]
            for pos, p in enumerate(arena["pirates"]):
                pirate = PIRATE_BY_NAME[p["name"]]
                nf, na = course_counts(pirate, foods)
                won = 1 if p["name"] == winner else 0
                pirate_food_records[p["name"]][(nf, na)][0] += won
                pirate_food_records[p["name"]][(nf, na)][1] += 1
                pirate_food_odds[p["name"]][(nf, na)].append(p["odds"])

    # =========================================================================
    # 1. Show raw pirate stats sorted by strength
    # =========================================================================
    print("=" * 80)
    print("1. RAW PIRATE STATS")
    print("=" * 80)
    pirates_sorted = sorted(PIRATES, key=lambda p: p.strength)
    print(f"{'Name':<28} {'Str':>3} {'Wt':>3} {'WR':>6} {'nFavCats':>8} {'nAllCats':>8}")
    for p in pirates_sorted:
        with open(os.path.join(os.path.dirname(__file__), "pirates.json")) as f:
            raw = json.load(f)
        pdata = [x for x in raw["pirates"] if x["name"] == p.name][0]
        print(f"{p.name:<28} {p.strength:3d} {p.weight:3d} {p.win_rate:6.3f} "
              f"{len(pdata['favorites']):8d} {len(pdata['allergies']):8d}")

    # =========================================================================
    # 2. Test strength rounding: for each rounding scheme, group pirates
    #    and check within-group win rate variance
    # =========================================================================
    print()
    print("=" * 80)
    print("2. STRENGTH ROUNDING ANALYSIS")
    print("=" * 80)

    # For each (nf, na) context, compute win rate for each pirate
    # Then check if pirates with same rounded strength have closer WRs
    # than expected from their raw strength difference

    # First, get controlled win rates at common food contexts
    # Use (nf=1, na=1) as it has good sample sizes for most pirates
    common_contexts = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]

    for ctx in common_contexts:
        nf, na = ctx
        print(f"\n--- Context nf={nf}, na={na} ---")
        ctx_data = []
        for p in pirates_sorted:
            rec = pirate_food_records[p.name].get((nf, na))
            odds_list = pirate_food_odds[p.name].get((nf, na), [])
            if rec and rec[1] >= 30:
                wr = rec[0] / rec[1]
                avg_odds = sum(odds_list) / len(odds_list) if odds_list else 0
                ctx_data.append((p.name, p.strength, p.weight, wr, avg_odds, rec[1]))

        if len(ctx_data) < 5:
            print("  (too few pirates with sufficient data)")
            continue

        print(f"  {'Name':<28} {'Str':>3} {'Wt':>3} {'WR':>7} {'AvgOdds':>7} {'N':>5}")
        for name, s, w, wr, ao, n in ctx_data:
            print(f"  {name:<28} {s:3d} {w:3d} {wr:7.4f} {ao:7.2f} {n:5d}")

    # =========================================================================
    # 3. Systematic rounding test
    # =========================================================================
    print()
    print("=" * 80)
    print("3. SYSTEMATIC ROUNDING TEST")
    print("=" * 80)
    print("  For each rounding scheme, group pirates by rounded strength.")
    print("  Compare within-group WR variance to what raw strength predicts.")
    print("  If rounding is real, within-group WR differences should be ~0.\n")

    # Build a global controlled WR for each pirate that averages across food contexts
    # Weight by sample size
    pirate_controlled_wr = {}
    for p in PIRATES:
        total_w = 0
        total_n = 0
        for (nf, na), (wins, count) in pirate_food_records[p.name].items():
            if count >= 30:
                total_w += wins
                total_n += count
        if total_n > 0:
            pirate_controlled_wr[p.name] = total_w / total_n

    schemes = [
        ("floor5", lambda v: round_floor(v, 5)),
        ("ceil5", lambda v: round_ceil(v, 5)),
        ("nearest5", lambda v: round_nearest(v, 5)),
        ("floor10", lambda v: round_floor(v, 10)),
        ("ceil10", lambda v: round_ceil(v, 10)),
        ("nearest10", lambda v: round_nearest(v, 10)),
        ("floor3", lambda v: round_floor(v, 3)),
        ("nearest3", lambda v: round_nearest(v, 3)),
        ("floor4", lambda v: round_floor(v, 4)),
        ("nearest4", lambda v: round_nearest(v, 4)),
        ("floor7", lambda v: round_floor(v, 7)),
        ("nearest7", lambda v: round_nearest(v, 7)),
        ("floor15", lambda v: round_floor(v, 15)),
        ("nearest15", lambda v: round_nearest(v, 15)),
        ("floor20", lambda v: round_floor(v, 20)),
        ("nearest20", lambda v: round_nearest(v, 20)),
    ]

    print(f"{'Scheme':<12} {'Groups':>6} {'GroupsWithMultiple':>18} {'Pirate pairs in same bucket':>28}")
    print(f"{'':12} {'':>6} {'':>18} {'and their WR + odds comparison':>28}")
    print()

    for scheme_name, fn in schemes:
        groups = defaultdict(list)
        for p in PIRATES:
            rounded = fn(p.strength)
            groups[rounded].append(p)

        multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
        if not multi_groups:
            print(f"{scheme_name:<12} {len(groups):6d} {0:18d}  (no collisions)")
            continue

        print(f"\n{scheme_name:<12} {len(groups):6d} {len(multi_groups):18d}")
        for rounded_val, members in sorted(multi_groups.items()):
            names = [(p.name, p.strength, p.weight) for p in members]
            print(f"  Bucket {rounded_val}: {', '.join(f'{n} (s={s},w={w})' for n,s,w in names)}")

    # =========================================================================
    # 4. PAIR-WISE ANALYSIS: pirates with same rounded strength
    #    Do they have the same odds at the same (nf, na)?
    # =========================================================================
    print()
    print("=" * 80)
    print("4. PAIRWISE ODDS COMPARISON FOR CLOSE-STRENGTH PIRATES")
    print("=" * 80)
    print("  If strength is rounded, pirates in the same bucket should get")
    print("  identical odds distributions at the same (nf, na, pos).\n")

    # Find pairs of pirates with strength difference <= various thresholds
    # Compare their odds distributions at same (nf, na)
    for max_diff in [0, 1, 2, 3, 5]:
        print(f"\n--- Pirates with strength diff <= {max_diff} ---")
        pairs = []
        for i, p1 in enumerate(PIRATES):
            for p2 in list(PIRATES)[i+1:]:
                if abs(p1.strength - p2.strength) <= max_diff:
                    pairs.append((p1, p2))

        if not pairs:
            print("  (no pairs)")
            continue

        for p1, p2 in pairs:
            print(f"\n  {p1.name} (s={p1.strength},w={p1.weight}) vs "
                  f"{p2.name} (s={p2.strength},w={p2.weight})")
            print(f"  {'(nf,na)':<10} {'WR1':>7} {'WR2':>7} {'Odds1':>7} {'Odds2':>7} {'N1':>5} {'N2':>5} {'WR_diff':>8} {'Odds_diff':>9}")

            shared_contexts = set(pirate_food_records[p1.name].keys()) & set(pirate_food_records[p2.name].keys())
            for ctx in sorted(shared_contexts):
                r1 = pirate_food_records[p1.name][ctx]
                r2 = pirate_food_records[p2.name][ctx]
                o1 = pirate_food_odds[p1.name].get(ctx, [])
                o2 = pirate_food_odds[p2.name].get(ctx, [])
                if r1[1] >= 50 and r2[1] >= 50:
                    wr1 = r1[0] / r1[1]
                    wr2 = r2[0] / r2[1]
                    ao1 = sum(o1) / len(o1) if o1 else 0
                    ao2 = sum(o2) / len(o2) if o2 else 0
                    print(f"  ({ctx[0]},{ctx[1]}){'':<5} {wr1:7.4f} {wr2:7.4f} {ao1:7.2f} {ao2:7.2f} "
                          f"{r1[1]:5d} {r2[1]:5d} {wr1-wr2:+8.4f} {ao1-ao2:+9.2f}")

    # =========================================================================
    # 5. WEIGHT ROUNDING ANALYSIS
    # =========================================================================
    print()
    print("=" * 80)
    print("5. WEIGHT ROUNDING ANALYSIS")
    print("=" * 80)
    print("  Weight affects allergy penalty: weight_offset = (max_w - weight) / 2")
    print("  If weight is rounded, pirates with similar weights should have")
    print("  identical allergy responses.\n")

    # For each pirate, compute allergy-specific win rate ratios
    # Compare pirates with close weights but different strengths
    print(f"{'Name':<28} {'Str':>3} {'Wt':>3} {'WO':>4} {'WR_na0':>7} {'WR_na1':>7} {'WR_na2':>7} {'ratio_1/0':>9} {'ratio_2/0':>9}")
    for p in sorted(PIRATES, key=lambda p: p.weight):
        # weight_offset = min((221 - weight) / 2, 10)
        wo = min((221 - p.weight) // 2, 10)
        r0 = pirate_food_records[p.name].get((1, 0), [0, 0])
        r1 = pirate_food_records[p.name].get((1, 1), [0, 0])
        r2 = pirate_food_records[p.name].get((1, 2), [0, 0])
        wr0 = r0[0] / r0[1] if r0[1] >= 30 else None
        wr1 = r1[0] / r1[1] if r1[1] >= 30 else None
        wr2 = r2[0] / r2[1] if r2[1] >= 30 else None
        ratio1 = f"{wr1/wr0:9.4f}" if wr0 and wr1 and wr0 > 0 else "     N/A"
        ratio2 = f"{wr2/wr0:9.4f}" if wr0 and wr2 and wr0 > 0 else "     N/A"
        wr0_s = f"{wr0:7.4f}" if wr0 else "    N/A"
        wr1_s = f"{wr1:7.4f}" if wr1 else "    N/A"
        wr2_s = f"{wr2:7.4f}" if wr2 else "    N/A"
        print(f"{p.name:<28} {p.strength:3d} {p.weight:3d} {wo:4d} {wr0_s} {wr1_s} {wr2_s} {ratio1} {ratio2}")

    # =========================================================================
    # 6. DIRECT TEST: Do the odds differentiate pirates with same
    #    rounded strength? (The killer test)
    # =========================================================================
    print()
    print("=" * 80)
    print("6. KILLER TEST: odds differentiation within rounding buckets")
    print("=" * 80)
    print("  If the game rounds strength to nearest 5, then pirates with")
    print("  strengths 79 and 81 (both round to 80) should get the SAME odds")
    print("  in the same food context. If odds differ, rounding doesn't happen.\n")

    # Key pairs to test:
    # Franchisco (81, w=165) vs Federismo (81, w=166) - SAME strength, nearly same weight
    # Franchisco (81) vs Sir Edmund (79) - diff 2, would be same in round-to-5
    # Ned (79) vs Sir Edmund (79) - SAME strength
    # Young Sproggie (73) vs Peg Leg (73) - SAME strength, very different weight
    # Bonnie Pip (76) vs Admiral Blackbeard (76) - SAME strength, very different weight

    test_pairs = [
        ("Franchisco Corvallio", "Federismo Corvallio", "same str=81, similar wt"),
        ("Sir Edmund Ogletree", "Ned the Skipper", "same str=79, diff wt (177 vs 169)"),
        ("Young Sproggie", "Peg Leg Percival", "same str=73, VERY diff wt (112 vs 202)"),
        ("Bonnie Pip Culliford", "Admiral Blackbeard", "same str=76, diff wt (116 vs 171)"),
        ("Franchisco Corvallio", "Sir Edmund Ogletree", "str 81 vs 79 (round-5 bucket=80)"),
        ("Franchisco Corvallio", "The Tailhook Kid", "both str=81, diff wt (165 vs 207)"),
        ("Lucky McKyriggan", "Franchisco Corvallio", "str 82 vs 81 (round-5 bucket=80)"),
        ("Scurvy Dan the Blade", "Buck Cutlass", "str 87 vs 89 (round-5 bucket=90)"),
    ]

    for name1, name2, desc in test_pairs:
        p1 = PIRATE_BY_NAME[name1]
        p2 = PIRATE_BY_NAME[name2]
        print(f"\n  {name1} (s={p1.strength},w={p1.weight}) vs "
              f"{name2} (s={p2.strength},w={p2.weight})")
        print(f"  [{desc}]")
        print(f"  {'(nf,na)':<10} {'Odds1':>7} {'Odds2':>7} {'WR1':>7} {'WR2':>7} {'N1':>5} {'N2':>5}")

        shared = set(pirate_food_records[name1].keys()) & set(pirate_food_records[name2].keys())
        for ctx in sorted(shared):
            r1 = pirate_food_records[name1][ctx]
            r2 = pirate_food_records[name2][ctx]
            o1 = pirate_food_odds[name1].get(ctx, [])
            o2 = pirate_food_odds[name2].get(ctx, [])
            if r1[1] >= 50 and r2[1] >= 50:
                wr1 = r1[0] / r1[1]
                wr2 = r2[0] / r2[1]
                ao1 = sum(o1) / len(o1) if o1 else 0
                ao2 = sum(o2) / len(o2) if o2 else 0
                marker = ""
                if abs(ao1 - ao2) < 0.15:
                    marker = " <-- SAME ODDS"
                print(f"  ({ctx[0]},{ctx[1]}){'':<5} {ao1:7.2f} {ao2:7.2f} {wr1:7.4f} {wr2:7.4f} "
                      f"{r1[1]:5d} {r2[1]:5d}{marker}")

    # =========================================================================
    # 7. ODDS DISTRIBUTION COMPARISON for same-strength pairs
    # =========================================================================
    print()
    print("=" * 80)
    print("7. FULL ODDS DISTRIBUTION for same-strength pirates")
    print("=" * 80)
    print("  If strength is used raw (not rounded), same-strength pirates with")
    print("  different weights should get DIFFERENT odds due to weight effect.\n")

    from collections import Counter

    same_str_pairs = [
        ("Franchisco Corvallio", "Federismo Corvallio"),  # 81, w=165 vs 166
        ("Franchisco Corvallio", "The Tailhook Kid"),     # 81, w=165 vs 207
        ("Sir Edmund Ogletree", "Ned the Skipper"),       # 79, w=177 vs 169
        ("Young Sproggie", "Peg Leg Percival"),           # 73, w=112 vs 202
        ("Bonnie Pip Culliford", "Admiral Blackbeard"),   # 76, w=116 vs 171
    ]

    for name1, name2 in same_str_pairs:
        p1 = PIRATE_BY_NAME[name1]
        p2 = PIRATE_BY_NAME[name2]
        print(f"\n  {name1} (s={p1.strength},w={p1.weight}) vs "
              f"{name2} (s={p2.strength},w={p2.weight})")

        # Pick a well-populated context
        for ctx in [(1, 1), (2, 1), (1, 0), (0, 1)]:
            o1 = pirate_food_odds[name1].get(ctx, [])
            o2 = pirate_food_odds[name2].get(ctx, [])
            if len(o1) >= 50 and len(o2) >= 50:
                c1 = Counter(o1)
                c2 = Counter(o2)
                print(f"  Context ({ctx[0]},{ctx[1]}): N1={len(o1)}, N2={len(o2)}")
                print(f"    {'Odds':>4} {'P1%':>6} {'P2%':>6} {'Diff':>6}")
                for odds in range(2, 14):
                    pct1 = 100 * c1.get(odds, 0) / len(o1)
                    pct2 = 100 * c2.get(odds, 0) / len(o2)
                    if pct1 > 0.5 or pct2 > 0.5:
                        print(f"    {odds:4d} {pct1:6.1f} {pct2:6.1f} {pct1-pct2:+6.1f}")
                break

    # =========================================================================
    # 8. QUANTITATIVE TEST: strength sensitivity
    # =========================================================================
    print()
    print("=" * 80)
    print("8. STRENGTH SENSITIVITY: avg odds per strength point")
    print("=" * 80)
    print("  If rounding to 5, we'd see flat regions in the odds-vs-strength curve")
    print("  with jumps at boundaries. If raw, it should be smooth.\n")

    # For each pirate at context (1,1), plot strength vs avg odds
    ctx = (1, 1)
    print(f"  Context: nf=1, na=1")
    print(f"  {'Str':>3} {'Name':<28} {'AvgOdds':>7} {'WR':>7} {'N':>5}")
    for p in sorted(PIRATES, key=lambda p: p.strength):
        odds_list = pirate_food_odds[p.name].get(ctx, [])
        rec = pirate_food_records[p.name].get(ctx, [0, 0])
        if len(odds_list) >= 30 and rec[1] >= 30:
            avg_o = sum(odds_list) / len(odds_list)
            wr = rec[0] / rec[1]
            print(f"  {p.strength:3d} {p.name:<28} {avg_o:7.2f} {wr:7.4f} {rec[1]:5d}")

    # Also do (0,0) - pure strength, no food effects
    ctx = (0, 0)
    print(f"\n  Context: nf=0, na=0 (pure strength, no food effects)")
    print(f"  {'Str':>3} {'Wt':>3} {'Name':<28} {'AvgOdds':>7} {'WR':>7} {'N':>5}")
    for p in sorted(PIRATES, key=lambda p: p.strength):
        odds_list = pirate_food_odds[p.name].get(ctx, [])
        rec = pirate_food_records[p.name].get(ctx, [0, 0])
        if len(odds_list) >= 30 and rec[1] >= 30:
            avg_o = sum(odds_list) / len(odds_list)
            wr = rec[0] / rec[1]
            print(f"  {p.strength:3d} {p.weight:3d} {p.name:<28} {avg_o:7.2f} {wr:7.4f} {rec[1]:5d}")

    # =========================================================================
    # 9. WEIGHT SENSITIVITY: same strength, different weight
    # =========================================================================
    print()
    print("=" * 80)
    print("9. WEIGHT EFFECT: same-strength pirates, different allergy responses")
    print("=" * 80)
    print("  Weight only matters via allergy penalty. Compare pirates with")
    print("  same strength at (nf=0, na=2) vs (nf=0, na=0) to isolate weight.\n")

    print(f"  {'Name':<28} {'Str':>3} {'Wt':>3} {'WO':>3} {'WR(0,0)':>8} {'WR(0,2)':>8} {'Ratio':>6} {'Odds(0,0)':>9} {'Odds(0,2)':>9}")
    for p in sorted(PIRATES, key=lambda p: (p.strength, p.weight)):
        wo = min((221 - p.weight) // 2, 10)
        r00 = pirate_food_records[p.name].get((0, 0), [0, 0])
        r02 = pirate_food_records[p.name].get((0, 2), [0, 0])
        o00 = pirate_food_odds[p.name].get((0, 0), [])
        o02 = pirate_food_odds[p.name].get((0, 2), [])
        if r00[1] >= 30 and r02[1] >= 30:
            wr00 = r00[0] / r00[1]
            wr02 = r02[0] / r02[1]
            ao00 = sum(o00) / len(o00)
            ao02 = sum(o02) / len(o02)
            ratio = wr02 / wr00 if wr00 > 0 else 0
            print(f"  {p.name:<28} {p.strength:3d} {p.weight:3d} {wo:3d} {wr00:8.4f} {wr02:8.4f} "
                  f"{ratio:6.3f} {ao00:9.2f} {ao02:9.2f}")


if __name__ == "__main__":
    main()
