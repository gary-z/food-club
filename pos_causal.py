import json
from collections import defaultdict

with open("historical_matches.json") as f:
    days = json.load(f)

# (odds, position) -> [appearances, wins]
odds_pos = defaultdict(lambda: [0, 0])
# odds -> [appearances, wins]
odds_only = defaultdict(lambda: [0, 0])

# winner position counts
winner_pos = [0, 0, 0, 0]
total_arenas = 0

# ordering check
inversions = 0
pairs_total = 0
sorted_asc = 0
sorted_desc = 0

# position -> sum of odds, count
pos_odds_sum = [0, 0, 0, 0]
pos_odds_count = [0, 0, 0, 0]

for day in days:
    for arena in day:
        total_arenas += 1
        pirates = arena["pirates"]
        winner = arena["winner"]
        odds_list = [p["odds"] for p in pirates]

        for pos, p in enumerate(pirates):
            odds = p["odds"]
            won = p["name"] == winner
            odds_pos[(odds, pos)][0] += 1
            if won:
                odds_pos[(odds, pos)][1] += 1
            odds_only[odds][0] += 1
            if won:
                odds_only[odds][1] += 1

            pos_odds_sum[pos] += odds
            pos_odds_count[pos] += 1

            if won:
                winner_pos[pos] += 1

        # Check ordering
        is_asc = all(odds_list[i] <= odds_list[i+1] for i in range(len(odds_list)-1))
        is_desc = all(odds_list[i] >= odds_list[i+1] for i in range(len(odds_list)-1))
        if is_asc: sorted_asc += 1
        if is_desc: sorted_desc += 1

        for i in range(len(odds_list)):
            for j in range(i+1, len(odds_list)):
                pairs_total += 1
                if odds_list[j] > odds_list[i]:
                    inversions += 1

# ============================================================
# TEST 1: Win rate by position controlling for odds
# ============================================================
print("=== TEST 1: WIN RATE BY POSITION, CONTROLLING FOR ODDS ===\n")
print(f"{'odds':>5} {'overall':>10} {'pos0':>12} {'pos1':>12} {'pos2':>12} {'pos3':>12}")
print("-" * 70)

odds_vals = sorted(odds_only.keys())
for odds in odds_vals:
    total_app, total_wins = odds_only[odds]
    if total_app < 100:
        continue
    overall_wr = total_wins / total_app
    pos_strs = []
    for pos in range(4):
        app, wins = odds_pos.get((odds, pos), (0, 0))
        if app >= 10:
            pos_strs.append(f"{wins/app:.3f}({app:>4})")
        else:
            pos_strs.append(f"{'  -':>11}")
    print(f"{odds:>5} {overall_wr:>8.3f}({total_app:>5}) {pos_strs[0]} {pos_strs[1]} {pos_strs[2]} {pos_strs[3]}")

# ============================================================
# TEST 2: Aggregate position effect (odds-controlled)
# ============================================================
print("\n\n=== TEST 2: AGGREGATE POSITION EFFECT (ODDS-CONTROLLED) ===\n")
pos_excess = [0.0] * 4
pos_weight = [0.0] * 4

for odds in odds_vals:
    total_app, total_wins = odds_only[odds]
    if total_app < 200:
        continue
    overall_wr = total_wins / total_app
    for pos in range(4):
        app, wins = odds_pos.get((odds, pos), (0, 0))
        if app >= 20:
            wr = wins / app
            pos_excess[pos] += (wr - overall_wr) * app
            pos_weight[pos] += app

print("Position effect after controlling for odds:")
for pos in range(4):
    if pos_weight[pos] > 0:
        print(f"  Position {pos}: excess win rate = {pos_excess[pos]/pos_weight[pos]:+.4f} (weighted across {int(pos_weight[pos])} appearances)")

# ============================================================
# TEST 3: Ordering check
# ============================================================
print(f"\n\n=== TEST 3: ORDERING CHECK ===\n")
print(f"  Total arenas: {total_arenas}")
print(f"  Sorted ascending (weak->strong): {sorted_asc} ({sorted_asc/total_arenas*100:.1f}%)")
print(f"  Sorted descending (strong->weak): {sorted_desc} ({sorted_desc/total_arenas*100:.1f}%)")
print(f"  Fraction of pairs where later pos has HIGHER odds: {inversions/pairs_total:.4f} (0.5 = random)")

# ============================================================
# TEST 4: Average odds by position
# ============================================================
print(f"\n\n=== TEST 4: AVERAGE ODDS BY POSITION ===\n")
for pos in range(4):
    print(f"  Position {pos}: avg odds = {pos_odds_sum[pos]/pos_odds_count[pos]:.3f}")

# ============================================================
# TEST 5: Position distribution within each odds value
# ============================================================
print(f"\n\n=== TEST 5: POSITION DISTRIBUTION WITHIN EACH ODDS VALUE ===\n")
print(f"{'odds':>5} {'total':>8} {'pos0%':>8} {'pos1%':>8} {'pos2%':>8} {'pos3%':>8}")
print("-" * 50)
for odds in odds_vals:
    total_app, _ = odds_only[odds]
    if total_app < 200:
        continue
    pos_counts = [odds_pos.get((odds, p), (0, 0))[0] for p in range(4)]
    print(f"{odds:>5} {total_app:>8} {pos_counts[0]/total_app*100:>8.1f} {pos_counts[1]/total_app*100:>8.1f} {pos_counts[2]/total_app*100:>8.1f} {pos_counts[3]/total_app*100:>8.1f}")

# ============================================================
# TEST 6: Winner's position distribution
# ============================================================
print(f"\n\n=== TEST 6: WINNER'S POSITION DISTRIBUTION ===\n")
print(f"  Total contests: {total_arenas}")
for pos in range(4):
    print(f"  Winner at position {pos}: {winner_pos[pos]} ({winner_pos[pos]/total_arenas*100:.2f}%)")

# ============================================================
# TEST 7: Within each arena, rank pirates by odds, check if rank predicts position
# ============================================================
print(f"\n\n=== TEST 7: ODDS RANK VS POSITION ===\n")
# For each arena, rank pirates 0-3 by odds (0=lowest=strongest)
# Then check: what fraction of the time does odds-rank match position?
rank_at_pos = defaultdict(lambda: [0]*4)  # rank -> position counts
for day in days:
    for arena in day:
        pirates = arena["pirates"]
        odds_with_pos = [(p["odds"], pos) for pos, p in enumerate(pirates)]
        # Sort by odds to get rank (lowest odds = rank 0 = strongest)
        sorted_by_odds = sorted(odds_with_pos, key=lambda x: (x[0], x[1]))
        for rank, (odds, pos) in enumerate(sorted_by_odds):
            rank_at_pos[rank][pos] += 1

print("  Where does each odds-rank end up positionally?")
print(f"  {'rank':>6} {'pos0%':>8} {'pos1%':>8} {'pos2%':>8} {'pos3%':>8}")
print(f"  {'-'*42}")
for rank in range(4):
    total = sum(rank_at_pos[rank])
    print(f"  rank {rank} {rank_at_pos[rank][0]/total*100:>8.1f} {rank_at_pos[rank][1]/total*100:>8.1f} {rank_at_pos[rank][2]/total*100:>8.1f} {rank_at_pos[rank][3]/total*100:>8.1f}")
