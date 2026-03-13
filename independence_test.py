"""
Independence test: for each pair of pirates in the same arena, compute the
head-to-head win rate (P(A wins | A or B won)).  Split by strength of the
other two pirates.  If scores are independent, h2h rate should be stable
regardless of opponent strength.
"""
import json
from collections import defaultdict
from itertools import combinations

with open("pirates.json") as f:
    data = json.load(f)
with open("historical_matches.json") as f:
    historical = json.load(f)

pirate_map = {p["name"]: p for p in data["pirates"]}

# For each ordered pair (p1, p2) where p1 < p2 alphabetically,
# collect: other_strength -> [p1_wins, p2_wins]
# We'll bin other_strength into terciles later.
PairKey = tuple  # (p1_name, p2_name) canonical order

pair_data = defaultdict(list)  # key -> list of (other_str_sum, winner_name)

for day in historical:
    for arena in day:
        pirate_names = [p["name"] for p in arena["pirates"]]
        winner = arena["winner"]
        strengths = {p["name"]: pirate_map[p["name"]]["strength"] for p in arena["pirates"]}

        for i in range(4):
            for j in range(i+1, 4):
                p1, p2 = pirate_names[i], pirate_names[j]
                # Only count if one of them won
                if winner != p1 and winner != p2:
                    continue
                key = tuple(sorted([p1, p2]))
                others = [pirate_names[k] for k in range(4) if k != i and k != j]
                other_str = sum(strengths[o] for o in others)
                pair_data[key].append((other_str, winner))

# For each pair with enough data, split into low/high other_strength
# and compare h2h rates
print("=" * 85)
print("HEAD-TO-HEAD INDEPENDENCE TEST")
print("If scores are independent, h2h rate should NOT change with opponent strength")
print("=" * 85)

results = []

for key in sorted(pair_data.keys()):
    records = pair_data[key]
    if len(records) < 60:
        continue

    p1, p2 = key
    s1 = pirate_map[p1]["strength"]
    s2 = pirate_map[p2]["strength"]

    # Sort by other_strength and split into terciles
    records.sort(key=lambda r: r[0])
    n = len(records)
    tercile_size = n // 3

    terciles = [
        records[:tercile_size],
        records[tercile_size:2*tercile_size],
        records[2*tercile_size:],
    ]
    labels = ["weak opps", "mid opps ", "strong opps"]

    h2h_rates = []
    row_data = []
    for label, group in zip(labels, terciles):
        p1_wins = sum(1 for _, w in group if w == p1)
        p2_wins = sum(1 for _, w in group if w == p2)
        total = p1_wins + p2_wins
        avg_other = sum(s for s, _ in group) / len(group)
        h2h = p1_wins / total if total > 0 else 0.5
        h2h_rates.append(h2h)
        row_data.append((label, p1_wins, p2_wins, total, h2h, avg_other))

    spread = max(h2h_rates) - min(h2h_rates)
    overall_p1 = sum(1 for _, w in records if w == p1)
    overall_total = len(records)
    overall_h2h = overall_p1 / overall_total

    results.append((spread, key, s1, s2, overall_h2h, overall_total, row_data))

# Sort by spread descending to highlight biggest deviations
results.sort(key=lambda r: -r[0])

print(f"\n{'Pair':50s} {'str':>7} {'N':>5} {'h2h':>6}  "
      f"{'weak':>6} {'mid':>6} {'strong':>6} {'spread':>6}")
print("-" * 105)

for spread, key, s1, s2, overall_h2h, overall_total, row_data in results:
    p1, p2 = key
    short1 = p1.split()[0]
    short2 = p2.split()[0]
    pair_label = f"{short1} vs {short2}"
    h2h_vals = [rd[3] and rd[4] for rd in row_data]
    print(f"{pair_label:50s} {s1:>3}/{s2:<3} {overall_total:>5} {overall_h2h:>6.3f}  "
          f"{row_data[0][4]:>6.3f} {row_data[1][4]:>6.3f} {row_data[2][4]:>6.3f} {spread:>6.3f}")

# Summary statistics
spreads = [r[0] for r in results]
print(f"\n{len(results)} pairs with N>=60")
print(f"Mean spread: {sum(spreads)/len(spreads):.3f}")
print(f"Median spread: {sorted(spreads)[len(spreads)//2]:.3f}")
print(f"Max spread: {max(spreads):.3f}")
print(f"Pairs with spread > 0.15: {sum(1 for s in spreads if s > 0.15)}")
print(f"Pairs with spread > 0.10: {sum(1 for s in spreads if s > 0.10)}")

# Show detailed breakdown for top 10 most variable pairs
print("\n" + "=" * 85)
print("TOP 10 MOST VARIABLE PAIRS (detailed)")
print("=" * 85)

for spread, key, s1, s2, overall_h2h, overall_total, row_data in results[:10]:
    p1, p2 = key
    print(f"\n{p1} (str={s1}) vs {p2} (str={s2})")
    print(f"  Overall: N={overall_total}, {p1.split()[0]} h2h = {overall_h2h:.3f}")
    for label, p1w, p2w, tot, h2h, avg_other in row_data:
        print(f"  {label}: avg_other_str={avg_other:.0f}, "
              f"{p1.split()[0]}={p1w} {p2.split()[0]}={p2w} "
              f"h2h={h2h:.3f} (N={tot})")
    print(f"  Spread: {spread:.3f}")

# Statistical test: under independence, h2h rate per tercile should be
# binomial with same p.  Chi-squared test across terciles.
print("\n" + "=" * 85)
print("CHI-SQUARED TEST: how many pairs show significant (p<0.05) variation?")
print("=" * 85)

from scipy.stats import chi2_contingency

sig_count = 0
total_tested = 0

for spread, key, s1, s2, overall_h2h, overall_total, row_data in results:
    # Build 2x3 contingency table: [p1_wins, p2_wins] x [weak, mid, strong]
    table = []
    for label, p1w, p2w, tot, h2h, avg_other in row_data:
        table.append([p1w, p2w])

    # Skip if any cell is 0
    if any(cell == 0 for row in table for cell in row):
        continue

    total_tested += 1
    chi2, p_val, dof, expected = chi2_contingency(table)
    if p_val < 0.05:
        sig_count += 1

print(f"Tested: {total_tested} pairs")
print(f"Significant at p<0.05: {sig_count} ({sig_count/total_tested*100:.1f}%)")
print(f"Expected by chance: {total_tested*0.05:.1f} ({5.0:.1f}%)")
