"""Analyse Orvinn's historical win rate to understand why models underpredict him."""
from food_club_match import parse_historical_data
from pirates import PIRATES

PIRATE_INFO = {p.name: p for p in PIRATES}

with open("historical_matches.json", "r", encoding="utf-8") as f:
    historical_data = parse_historical_data(f.read())

orvinn_arenas = [
    arena
    for day in historical_data
    for arena in day
    if any(p.name == "Orvinn the First Mate" for p in arena.pirates)
]

orvinn = PIRATE_INFO["Orvinn the First Mate"]
total = len(orvinn_arenas)
wins = sum(a.winner == "Orvinn the First Mate" for a in orvinn_arenas)
print(f"Orvinn appearances: {total}, wins: {wins}, rate: {wins/total:.3f}\n")

# --- Win rate by number of favorite courses in arena ---
print("Win rate by # of Orvinn's favorite courses in arena:")
from collections import defaultdict
by_nfav = defaultdict(lambda: [0, 0])  # [wins, total]
for arena in orvinn_arenas:
    n_fav = sum(f in orvinn.favorite_courses for f in arena.foods)
    by_nfav[n_fav][1] += 1
    if arena.winner == "Orvinn the First Mate":
        by_nfav[n_fav][0] += 1
for nf in sorted(by_nfav):
    w, t = by_nfav[nf]
    print(f"  n_fav={nf}: {w}/{t} = {w/t:.3f}")

# --- Win rate by number of allergy courses in arena ---
print("\nWin rate by # of Orvinn's allergy courses in arena:")
by_nall = defaultdict(lambda: [0, 0])
for arena in orvinn_arenas:
    n_all = sum(f in orvinn.allergy_courses for f in arena.foods)
    by_nall[n_all][1] += 1
    if arena.winner == "Orvinn the First Mate":
        by_nall[n_all][0] += 1
for na in sorted(by_nall):
    w, t = by_nall[na]
    print(f"  n_allergy={na}: {w}/{t} = {w/t:.3f}")

# --- Win rate by sum of opponent strengths ---
print("\nWin rate by total opponent strength (terciles):")
opponent_strengths = []
for arena in orvinn_arenas:
    opp_strength = sum(
        PIRATE_INFO[p.name].strength
        for p in arena.pirates
        if p.name != "Orvinn the First Mate"
    )
    opponent_strengths.append((opp_strength, arena.winner == "Orvinn the First Mate"))

opponent_strengths.sort()
tercile = len(opponent_strengths) // 3
for i, label in enumerate(["weak opponents", "medium opponents", "strong opponents"]):
    chunk = opponent_strengths[i*tercile:(i+1)*tercile]
    w = sum(won for _, won in chunk)
    t = len(chunk)
    strengths = [s for s, _ in chunk]
    print(f"  {label} (avg opp str {sum(strengths)/t:.1f}): {w}/{t} = {w/t:.3f}")

# --- Compare Orvinn vs each other pirate head-to-head ---
print("\nOrvinn head-to-head win rates vs each co-arena pirate:")
h2h = defaultdict(lambda: [0, 0])
for arena in orvinn_arenas:
    opp_names = [p.name for p in arena.pirates if p.name != "Orvinn the First Mate"]
    for opp in opp_names:
        h2h[opp][1] += 1
        if arena.winner == "Orvinn the First Mate":
            h2h[opp][0] += 1

for name, (w, t) in sorted(h2h.items(), key=lambda x: PIRATE_INFO[x[0]].strength):
    rate = w / t
    str_ = PIRATE_INFO[name].strength
    print(f"  vs {name:30s} (str={str_}): {w}/{t} = {rate:.3f}")

# --- Orvinn's odds distribution ---
print("\nOrvinn's odds distribution (proxy for the game's win probability estimate):")
from collections import Counter
odds_counts = Counter(p.odds for a in orvinn_arenas for p in a.pirates if p.name == "Orvinn the First Mate")
odds_wins = defaultdict(lambda: [0, 0])
for arena in orvinn_arenas:
    for p in arena.pirates:
        if p.name == "Orvinn the First Mate":
            odds_wins[p.odds][1] += 1
            if arena.winner == "Orvinn the First Mate":
                odds_wins[p.odds][0] += 1
for odds in sorted(odds_wins):
    w, t = odds_wins[odds]
    print(f"  odds={odds}: {w}/{t} = {w/t:.3f}  (implied prob {1/odds:.3f})")
