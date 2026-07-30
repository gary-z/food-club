#!/usr/bin/env python3
"""What resolution does the odds maker's probability have?

The published odds histogram is lumpy in a way a smooth probability model
cannot produce: relative to floor(1/p_nn) there is a large excess at odds 12
and 11 and a large deficit at 8, 9 and 6.  That is the signature of the
probability being quantised before 1/p is taken.

For a quantisation step g and rounding rule, each published odds value v maps
to a contiguous *tile* of probability, namely the union of the quantised values
m*g with clamp(floor(1/(m*g)),2,13) == v, widened by the rounding rule.  Two
things follow that can be checked without any model of the game:

  1. a step that is too coarse leaves some odds value with an empty tile, and
     that odds value could then never be published;
  2. within an arena the four probabilities sum to 1, so the four tiles must
     admit a vector summing to 1.

Test 2 is sharp: the tiles for a 1% step are narrow and irregular, so a wrong
step size is exposed by arenas whose tiles cannot reach 1.
"""
import os
from collections import Counter

import numpy as np

import fc_data
from fc_odds import publish, tiles

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "odds_grid_test.txt")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


d = fc_data.load_arenas()
odds = d["odds"].astype(np.int64)
n = odds.shape[0]
win = np.zeros((n, 4), dtype=bool)
win[np.arange(n), d["winner"]] = True

say("=" * 78)
say("Resolution of the odds maker's probability")
say("=" * 78)

# --------------------------------------------------- 1. tiles and feasibility
cands = [("continuous floor(1/p)", None, "round"),
         ("step 5%, round", 0.05, "round"),
         ("step 2%, round", 0.02, "round"),
         ("step 1%, round", 0.01, "round"),
         ("step 1%, floor", 0.01, "floor"),
         ("step 0.5%, round", 0.005, "round"),
         ("step 0.2%, round", 0.002, "round"),
         ("step 0.1%, round", 0.001, "round")]

say("\n--- feasibility of sum(p)=1 given the published odds (model free) ---")
say(f"{'rule':<24}{'empty tiles':>28}{'arenas infeasible':>20}")
results = {}
for name, step, mode in cands:
    T = tiles(step, mode)
    empty = [v for v in range(2, 14) if T[v] is None]
    if empty:
        say(f"{name:<24}{('odds ' + str(empty)):>28}{'n/a':>20}")
        results[name] = None
        continue
    lo = np.array([T[v][0] for v in range(2, 14)])
    hi = np.array([T[v][1] for v in range(2, 14)])
    slo = lo[odds - 2].sum(axis=1)
    shi = hi[odds - 2].sum(axis=1)
    bad = (slo > 1.0 + 1e-12) | (shi < 1.0 - 1e-12)
    results[name] = bad
    say(f"{name:<24}{'-':>28}{f'{bad.sum()} / {n}':>20}")

say("\ntile boundaries for the surviving 1% rules (probability ranges):")
for name, step, mode in cands:
    if step != 0.01:
        continue
    T = tiles(step, mode)
    say(f"  {name}")
    say("    " + "  ".join(f"{v}:[{T[v][0]:.3f},{T[v][1]:.3f})" for v in range(13, 1, -1)))

# the arenas that fail the strict 1%-floor rule
T = tiles(0.01, "floor")
lo = np.array([T[v][0] for v in range(2, 14)])
hi = np.array([T[v][1] for v in range(2, 14)])
slo = lo[odds - 2].sum(axis=1)
bad = slo > 1.0 + 1e-12
if bad.sum():
    say(f"\narenas incompatible with the 1%-floor rule ({bad.sum()}):")
    for a in np.where(bad)[0][:10]:
        say(f"  day {d['day'][a]} arena {d['arena'][a]}: odds={tuple(odds[a])} "
            f"tile lower bounds sum to {slo[a]:.3f} > 1")
say("(the 1%-round rule has to hold for every arena, so rounding, not truncation)")

# -------------------------------------------- 2. does the test have power?
say("\n--- does test 1 have power? simulate finer/other resolutions ---")
say("draw p from p_nn, quantise it with the stated rule, publish the odds, then")
say("ask how often the resulting line is incompatible with a 1%-round line")
p_nn = np.load(os.path.join(ROOT, "nn_probs.npz"))["p"]
T1 = tiles(0.01, "round")
lo1 = np.array([T1[v][0] for v in range(2, 14)])
hi1 = np.array([T1[v][1] for v in range(2, 14)])


def frac_incompatible_with_1pct(o):
    slo = lo1[o - 2].sum(axis=1)
    shi = hi1[o - 2].sum(axis=1)
    return ((slo > 1.0 + 1e-12) | (shi < 1.0 - 1e-12)).mean()


rng = np.random.default_rng(0)
say(f"{'generating rule':<26}{'incompatible with 1%-round':>30}")
for nm, st, md in [("continuous (exact p)", None, "round"),
                   ("step 0.5%, round", 0.005, "round"),
                   ("step 0.2%, round", 0.002, "round"),
                   ("step 1%, round", 0.01, "round"),
                   ("step 1%, floor", 0.01, "floor"),
                   ("step 2%, round", 0.02, "round")]:
    o = publish(p_nn, st, md)
    say(f"{nm:<26}{100*frac_incompatible_with_1pct(o):>29.2f}%")
for N in (100, 111, 200, 500):
    k = rng.multinomial(N, p_nn)
    with np.errstate(divide="ignore"):
        o = np.where(k == 0, 13, np.clip(np.floor(N / np.maximum(k, 1)), 2, 13)).astype(np.int64)
    say(f"{('monte carlo N=' + str(N)):<26}{100*frac_incompatible_with_1pct(o):>29.2f}%")
say(f"{'OBSERVED':<26}{100*frac_incompatible_with_1pct(odds):>29.2f}%")

# ---------------------------------------------------- 3. the odds histogram
say("\n--- odds histogram: observed vs each publishing rule applied to p_nn ---")
obs = np.array([(odds == v).sum() for v in range(2, 14)])
say(f"{'odds':>5}{'observed':>10}" + "".join(
    f"{nm:>12}" for nm in ["exact", "1% round", "1% floor", "0.5%", "MC 111"]))
cols = []
for st, md in [(None, "round"), (0.01, "round"), (0.01, "floor"), (0.005, "round")]:
    o = publish(p_nn, st, md)
    cols.append(np.array([(o == v).sum() for v in range(2, 14)]))
k = rng.multinomial(111, p_nn)
with np.errstate(divide="ignore"):
    omc = np.where(k == 0, 13, np.clip(np.floor(111 / np.maximum(k, 1)), 2, 13)).astype(np.int64)
cols.append(np.array([(omc == v).sum() for v in range(2, 14)]))
for i, v in enumerate(range(2, 14)):
    say(f"{v:>5}{obs[i]:>10}" + "".join(f"{c[i]:>12}" for c in cols))
say(f"{'chi2':>5}{'':>10}" + "".join(
    f"{(((obs - c) ** 2) / np.maximum(c, 1)).sum():>12.0f}" for c in cols))
say("(chi2 against 12 bins, so anything above ~30 is a decisive misfit; all rules")
say(" share the same p_nn, so the differences isolate the publishing rule)")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
