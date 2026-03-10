"""
For each historical arena Orvinn appeared in, compute:
  - actual outcome (did he win?)
  - H11 expected win probability for that exact arena + food composition

Then compare actual vs expected broken down by opponent strength tier.
This tells us whether Orvinn beats/underperforms the model more against
strong opponents or weak opponents.

H11: upper = (base - strength) * fav_pct^n_fav * allergy_pct^n_allergy
     3 rolls, same bound. Winner = lowest total (most negative score).
"""
import random
from collections import defaultdict
from food_club_match import parse_historical_data
from pirates import PIRATES

BASE = 110
FAV_PCT = 0.92
ALLERGY_PCT = 1.15
SIMS_PER_ARENA = 2000

PIRATE_INFO = {p.name: p for p in PIRATES}

with open("historical_matches.json", "r", encoding="utf-8") as f:
    historical_data = parse_historical_data(f.read())

orvinn_arenas = [
    arena
    for day in historical_data
    for arena in day
    if any(p.name == "Orvinn the First Mate" for p in arena.pirates)
]


def h11_upper(pirate, courses):
    n_fav = sum(
        c in pirate.favorite_courses and c not in pirate.allergy_courses
        for c in courses
    )
    n_allergy = sum(c in pirate.allergy_courses for c in courses)
    upper = (BASE - pirate.strength) * (FAV_PCT ** n_fav) * (ALLERGY_PCT ** n_allergy)
    return max(1, int(upper))


def simulate_win_prob(arena_pirates, courses, sims):
    """Monte Carlo win probability for each pirate in this arena under H11."""
    uppers = [h11_upper(p, courses) for p in arena_pirates]
    wins = [0] * len(arena_pirates)
    for _ in range(sims):
        scores = [-(random.randint(1, u) + random.randint(1, u) + random.randint(1, u))
                  for u in uppers]
        best = max(scores)
        # handle ties uniformly
        tied = [i for i, s in enumerate(scores) if s == best]
        wins[random.choice(tied)] += 1
    return [w / sims for w in wins]


# For each Orvinn arena: compute actual result + H11 expected prob
results = []
for arena in orvinn_arenas:
    pirates_in_arena = [PIRATE_INFO[p.name] for p in arena.pirates]
    orvinn_idx = next(i for i, p in enumerate(pirates_in_arena)
                      if p.name == "Orvinn the First Mate")
    opponents = [p for p in pirates_in_arena if p.name != "Orvinn the First Mate"]

    actual_win = 1 if arena.winner == "Orvinn the First Mate" else 0
    probs = simulate_win_prob(pirates_in_arena, arena.foods, SIMS_PER_ARENA)
    expected_prob = probs[orvinn_idx]

    max_opp_strength = max(p.strength for p in opponents)
    avg_opp_strength = sum(p.strength for p in opponents) / len(opponents)
    strongest_opp = max(opponents, key=lambda p: p.strength).name

    results.append({
        "actual": actual_win,
        "expected": expected_prob,
        "residual": actual_win - expected_prob,  # positive = Orvinn beats model
        "max_opp_str": max_opp_strength,
        "avg_opp_str": avg_opp_strength,
        "strongest_opp": strongest_opp,
    })

print(f"Total arenas: {len(results)}")
print(f"Overall actual:   {sum(r['actual'] for r in results) / len(results):.4f}")
print(f"Overall expected: {sum(r['expected'] for r in results) / len(results):.4f}")
print(f"Overall residual: {sum(r['residual'] for r in results) / len(results):.4f}")

# --- Break down by max opponent strength tier ---
def analyze_by_tier(results, key, label, n_tiers=4):
    sorted_r = sorted(results, key=lambda r: r[key])
    tier_size = len(sorted_r) // n_tiers
    print(f"\n{label}:")
    print(f"  {'Tier':<22} {'n':>5}  {'avg_opp':>7}  {'actual':>7}  {'expected':>8}  {'residual':>9}  {'ratio':>6}")
    for i in range(n_tiers):
        chunk = sorted_r[i * tier_size : (i + 1) * tier_size]
        avg_opp = sum(r[key] for r in chunk) / len(chunk)
        actual = sum(r['actual'] for r in chunk) / len(chunk)
        expected = sum(r['expected'] for r in chunk) / len(chunk)
        residual = actual - expected
        ratio = actual / expected if expected > 0 else float('inf')
        tier_label = ["weakest", "weak-mid", "strong-mid", "strongest"][i]
        print(f"  {tier_label:<22} {len(chunk):>5}  {avg_opp:>7.1f}  {actual:>7.4f}  {expected:>8.4f}  {residual:>+9.4f}  {ratio:>6.3f}")

analyze_by_tier(results, "max_opp_str", "By strongest opponent in arena (max opp strength)")
analyze_by_tier(results, "avg_opp_str", "By average opponent strength in arena")

# --- Per specific strong opponent ---
print("\nResidual when sharing arena with specific strong pirates:")
by_opponent = defaultdict(list)
for arena, r in zip(orvinn_arenas, results):
    for p in arena.pirates:
        if p.name != "Orvinn the First Mate":
            by_opponent[p.name].append(r)

rows = []
for name, rs in by_opponent.items():
    actual = sum(r['actual'] for r in rs) / len(rs)
    expected = sum(r['expected'] for r in rs) / len(rs)
    residual = actual - expected
    str_ = PIRATE_INFO[name].strength
    rows.append((str_, name, len(rs), actual, expected, residual))

rows.sort()
print(f"  {'Opponent':<30} {'str':>4}  {'n':>5}  {'actual':>7}  {'expected':>8}  {'residual':>9}")
for str_, name, n, actual, expected, residual in rows:
    print(f"  {name:<30} {str_:>4}  {n:>5}  {actual:>7.4f}  {expected:>8.4f}  {residual:>+9.4f}")
