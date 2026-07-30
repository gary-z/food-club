#!/usr/bin/env python3
"""Structural falsification tests of:
   "the odds maker gives no edge when opening odds are anything but 2:1"

Null hypothesis H0 (finding 26/27): opening odds N = max(2, min(13, floor(1/p)))
where p is the TRUE win probability.  Consequences that must hold for N >= 3:
   (a) p <= 1/N            -> EV = p*N <= 1  (no +EV single bet)
   (b) p >  1/(N+1) for N in 3..12 (13 is clamped from below)
   (c) sum_i 1/N_i >= 1 in any arena with no 2:1 pirate (no Dutch book)

Every test below is a chance for the data to violate one of these.
Multiplicity is controlled with a parametric bootstrap under the *boundary*
null p_i = 1/N_i, which is the most generous version of H0 (EV exactly 1.0),
so a rejection is a rejection of even the friendliest reading of the claim.

Writes edge_structural_results.txt
"""
import json
import numpy as np
from collections import defaultdict
from scipy import stats

RNG = np.random.default_rng(20260730)
OUT = []


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ load data
with open("pirates.json") as f:
    raw = json.load(f)
course_idx = {n: i for i, n in enumerate(raw["courses"])}
cat_courses = defaultdict(set)
for cname, cats in raw["courses"].items():
    for cat in cats:
        cat_courses[cat].add(course_idx[cname])
pinfo = {}
for d in raw["pirates"]:
    fav = set()
    for c in d["favorites"]:
        fav |= cat_courses.get(c, set())
    alg = set()
    for c in d["allergies"]:
        alg |= cat_courses.get(c, set())
    pinfo[d["name"]] = (d["strength"], d["weight"], fav, alg)

with open("historical_matches.json") as f:
    hist = json.load(f)

rows = []  # one row per (arena, pirate)
arena_id = 0
for di, dayarenas in enumerate(hist):
    for a in dayarenas:
        food_ids = [course_idx[f] for f in a["foods"] if f in course_idx]
        odds = [p["odds"] for p in a["pirates"]]
        S = sum(1.0 / o for o in odds)
        S_no2 = sum(1.0 / o for o in odds if o >= 3)
        n2 = sum(1 for o in odds if o == 2)
        for pos, p in enumerate(a["pirates"]):
            st, wt, fav, alg = pinfo[p["name"]]
            nf = na = 0
            for c in food_ids:
                if c in alg:
                    na += 1
                elif c in fav:
                    nf += 1
            rows.append((di, arena_id, pos, p["name"], p["odds"],
                         p.get("current_odds") or 0, int(p["name"] == a["winner"]),
                         int(a.get("legacy", False)), nf, na, S, S_no2, n2, st, wt))
        arena_id += 1

names = np.array([r[3] for r in rows])
day = np.array([r[0] for r in rows])
aid = np.array([r[1] for r in rows])
pos = np.array([r[2] for r in rows])
odds = np.array([r[4] for r in rows])
cur = np.array([r[5] for r in rows])
won = np.array([r[6] for r in rows])
legacy = np.array([r[7] for r in rows], dtype=bool)
nf = np.array([r[8] for r in rows])
na = np.array([r[9] for r in rows])
Sarena = np.array([r[10] for r in rows])
Sno2 = np.array([r[11] for r in rows])
n2arena = np.array([r[12] for r in rows])
strength = np.array([r[13] for r in rows])

emit("=" * 78)
emit("STRUCTURAL FALSIFICATION TESTS — 'no edge at opening odds != 2'")
emit("=" * 78)
emit(f"{len(rows)} pirate-slots, {aid.max()+1} arenas, {day.max()+1} days "
     f"({(~legacy).sum()//4} modern arenas)")
emit()


# --------------------------------------------------------------- helper stats
def cell_stats(mask):
    """Under boundary null p=1/N: expected wins, z on wins, realized ROI."""
    n = int(mask.sum())
    if n == 0:
        return None
    w = int(won[mask].sum())
    p0 = 1.0 / odds[mask]
    exp = p0.sum()
    var = (p0 * (1 - p0)).sum()
    z = (w - exp) / np.sqrt(var) if var > 0 else 0.0
    profit = (won[mask] * odds[mask]).sum() - n
    roi = profit / n
    # profit z (variance under boundary null)
    pv = (p0 * (1 - p0) * odds[mask] ** 2).sum()
    zp = ((won[mask] * odds[mask]).sum() - (p0 * odds[mask]).sum()) / np.sqrt(pv)
    return dict(n=n, wins=w, exp=exp, z=z, roi=roi, profit=profit, zp=zp,
                wr=w / n, mask=mask)


def boundary_bootstrap_max(masks, n_sim=20000):
    """FWER-controlled max-z under the boundary null p_i = 1/N_i.

    Simulated independently per bet; within-arena outcomes are negatively
    correlated in reality, so independent simulation over-states the null
    variance -> the resulting p-value is conservative.
    """
    idx = [np.flatnonzero(m) for m in masks]
    allidx = np.unique(np.concatenate(idx)) if idx else np.array([], int)
    p0 = 1.0 / odds[allidx]
    posmap = {v: i for i, v in enumerate(allidx)}
    cols = [np.array([posmap[v] for v in ix]) for ix in idx]
    exps = [p0[c].sum() for c in cols]
    sds = [np.sqrt((p0[c] * (1 - p0[c])).sum()) for c in cols]
    maxz = np.empty(n_sim)
    for s in range(n_sim):
        sim = (RNG.random(len(allidx)) < p0)
        maxz[s] = max((sim[c].sum() - e) / sd for c, e, sd in zip(cols, exps, sds))
    return maxz


# =============================================================== TEST 1
emit("-" * 78)
emit("TEST 1. Per-odds-level bound test:  is WR > 1/N ?  (impossible under H0)")
emit("-" * 78)
emit(f"{'odds':>4} {'bets':>7} {'wins':>6} {'WR':>8} {'1/N':>8} {'ROI':>8} "
     f"{'z(WR>1/N)':>10} {'p1side':>9}")
bin_masks, bin_labels = [], []
for N in range(3, 14):
    m = odds == N
    c = cell_stats(m)
    p1 = stats.norm.sf(c["z"])
    emit(f"{N:>4} {c['n']:>7} {c['wins']:>6} {c['wr']:>8.4f} {1/N:>8.4f} "
         f"{c['roi']*100:>7.2f}% {c['z']:>10.2f} {p1:>9.4f}")
    bin_masks.append(m)
    bin_labels.append(f"odds={N}")
c2 = cell_stats(odds == 2)
emit(f"{2:>4} {c2['n']:>7} {c2['wins']:>6} {c2['wr']:>8.4f} {0.5:>8.4f} "
     f"{c2['roi']*100:>7.2f}% {c2['z']:>10.2f}   (clamped bin — positive control)")

maxz = boundary_bootstrap_max(bin_masks, 20000)
obs = max(cell_stats(m)["z"] for m in bin_masks)
emit(f"\nmax z over the 11 bins = {obs:.2f};  FWER p = {(maxz >= obs).mean():.4f} "
     f"(boundary-null bootstrap, 20k sims)")
emit()

# =============================================================== TEST 2
emit("-" * 78)
emit("TEST 2. Dutch book: sum_i 1/N_i over an arena (H0 requires >= 1 with no 2:1)")
emit("-" * 78)
first = np.flatnonzero(np.r_[True, np.diff(aid) != 0])
Sa, n2a, lega = Sarena[first], n2arena[first], legacy[first]
for lbl, sel in [("all arenas", np.ones(len(Sa), bool)),
                 ("arenas with no 2:1", n2a == 0),
                 ("no 2:1, modern", (n2a == 0) & ~lega)]:
    s = Sa[sel]
    if len(s) == 0:
        continue
    emit(f"  {lbl:<22} n={len(s):>6}  min={s.min():.4f}  "
         f"below 1.0: {(s < 1.0).sum():>6} ({(s<1.0).mean():.2%})")
emit("  (arenas containing a 2:1 legitimately fall below 1: the 2 bin is clamped")
emit("   from above, p can be up to 1.0, so 1/2 is not an upper bound on p.)")
# lower-bound consistency: sum of bin lower bounds must be <= 1
lower = []
for s0, e0 in zip(first, np.r_[first[1:], len(rows)]):
    o = odds[s0:e0]
    lb = sum(1.0 / 3 if x == 2 else (0.0 if x == 13 else 1.0 / (x + 1)) for x in o)
    lower.append(lb)
lower = np.array(lower)
emit(f"  sum of bin LOWER bounds > 1 (would falsify floor rule): "
     f"{(lower > 1 + 1e-12).sum()} / {len(lower)} arenas   max={lower.max():.4f}")
emit()

# =============================================================== TEST 3
emit("-" * 78)
emit("TEST 3. Cell scan: every (pirate x odds) cell with n>=150, WR vs 1/N")
emit("-" * 78)
masks, labels = [], []
for nm in sorted(set(names)):
    for N in range(3, 14):
        m = (names == nm) & (odds == N)
        if m.sum() >= 150:
            masks.append(m)
            labels.append(f"{nm} @ {N}:1")
res = [cell_stats(m) for m in masks]
order = np.argsort([-r["z"] for r in res])
emit(f"  {len(masks)} cells tested. Top 10 by z:")
emit(f"  {'cell':<40} {'n':>6} {'WR':>7} {'1/N':>7} {'ROI':>8} {'z':>6}")
for i in order[:10]:
    r = res[i]
    emit(f"  {labels[i]:<40} {r['n']:>6} {r['wr']:>7.4f} "
         f"{1/odds[masks[i]][0]:>7.4f} {r['roi']*100:>7.1f}% {r['z']:>6.2f}")
maxz = boundary_bootstrap_max(masks, 5000)
obs = max(r["z"] for r in res)
emit(f"  max z = {obs:.2f};  FWER p = {(maxz >= obs).mean():.4f} (5k sims)")
emit()

# =============================================================== TEST 3b
emit("-" * 78)
emit("TEST 3b. Cell scan at odds>=3: (pirate), (pirate x na), (pirate x nf)")
emit("-" * 78)
masks, labels = [], []
o3 = odds >= 3
for nm in sorted(set(names)):
    pm = (names == nm) & o3
    masks.append(pm); labels.append(f"{nm} [all odds>=3]")
    for lo, hi, tag in [(0, 1, "na=0"), (1, 2, "na=1"), (2, 3, "na=2"), (3, 99, "na>=3")]:
        masks.append(pm & (na >= lo) & (na < hi)); labels.append(f"{nm} {tag}")
    for lo, hi, tag in [(0, 1, "nf=0"), (1, 2, "nf=1"), (2, 3, "nf=2"), (3, 99, "nf>=3")]:
        masks.append(pm & (nf >= lo) & (nf < hi)); labels.append(f"{nm} {tag}")
keep = [i for i, m in enumerate(masks) if m.sum() >= 150]
masks = [masks[i] for i in keep]; labels = [labels[i] for i in keep]
res = [cell_stats(m) for m in masks]
order = np.argsort([-r["z"] for r in res])
emit(f"  {len(masks)} cells tested. Top 8 by z:")
emit(f"  {'cell':<40} {'n':>6} {'WR':>7} {'mean 1/N':>9} {'ROI':>8} {'z':>6}")
for i in order[:8]:
    r = res[i]
    emit(f"  {labels[i]:<40} {r['n']:>6} {r['wr']:>7.4f} "
         f"{np.mean(1/odds[masks[i]]):>9.4f} {r['roi']*100:>7.1f}% {r['z']:>6.2f}")
maxz = boundary_bootstrap_max(masks, 5000)
obs = max(r["z"] for r in res)
emit(f"  max z = {obs:.2f};  FWER p = {(maxz >= obs).mean():.4f} (5k sims)")
emit()

# =============================================================== TEST 4
emit("-" * 78)
emit("TEST 4. Cell scan: (odds x context) cells — position, nf, na, regime, overround")
emit("-" * 78)
masks, labels = [], []
for N in range(3, 14):
    on = odds == N
    for k in range(4):
        masks.append(on & (pos == k)); labels.append(f"odds={N} pos={k}")
    for k in range(0, 5):
        masks.append(on & (nf == k)); labels.append(f"odds={N} nf={k}")
        masks.append(on & (na == k)); labels.append(f"odds={N} na={k}")
    masks.append(on & legacy); labels.append(f"odds={N} legacy")
    masks.append(on & ~legacy); labels.append(f"odds={N} modern")
    for lo, hi in [(0, 1.0), (1.0, 1.05), (1.05, 1.15), (1.15, 9)]:
        masks.append(on & (Sarena >= lo) & (Sarena < hi))
        labels.append(f"odds={N} overround[{lo},{hi})")
keep = [i for i, m in enumerate(masks) if m.sum() >= 150]
masks = [masks[i] for i in keep]; labels = [labels[i] for i in keep]
res = [cell_stats(m) for m in masks]
order = np.argsort([-r["z"] for r in res])
emit(f"  {len(masks)} cells tested. Top 10 by z:")
emit(f"  {'cell':<40} {'n':>6} {'WR':>7} {'1/N':>7} {'ROI':>8} {'z':>6}")
for i in order[:10]:
    r = res[i]
    emit(f"  {labels[i]:<40} {r['n']:>6} {r['wr']:>7.4f} "
         f"{1/odds[masks[i]][0]:>7.4f} {r['roi']*100:>7.1f}% {r['z']:>6.2f}")
maxz = boundary_bootstrap_max(masks, 5000)
obs = max(r["z"] for r in res)
emit(f"  max z = {obs:.2f};  FWER p = {(maxz >= obs).mean():.4f} (5k sims)")
emit()

# =============================================================== TEST 5
emit("-" * 78)
emit("TEST 5. Pre-registered targeted tests (single tests, no multiplicity)")
emit("-" * 78)
targets = [
    ("odds 9-12 pooled (coarse discretization)", (odds >= 9) & (odds <= 12)),
    ("odds >= 6 pooled", (odds >= 6) & (odds <= 12)),
    ("all odds >= 3 pooled", odds >= 3),
    ("all odds >= 3, modern only", (odds >= 3) & ~legacy),
    ("Orvinn, odds>=3, na>=1, MODERN (finding 34)",
     (names == "Orvinn the First Mate") & (odds >= 3) & (na >= 1) & ~legacy),
    ("Orvinn, odds>=3, na>=1, legacy (control)",
     (names == "Orvinn the First Mate") & (odds >= 3) & (na >= 1) & legacy),
    ("Orvinn, odds>=3, MODERN (all na)",
     (names == "Orvinn the First Mate") & (odds >= 3) & ~legacy),
    ("Gooblah, odds>=3, na>=2 (finding 4b)",
     (names == "Gooblah the Grarrl") & (odds >= 3) & (na >= 2)),
    ("odds>=3 in low-overround arenas (S<1.02)", (odds >= 3) & (Sarena < 1.02)),
    ("odds=13 with nf>=3 (fav-boosted longshots)", (odds == 13) & (nf >= 3)),
]
emit(f"  {'test':<46} {'n':>6} {'WR':>7} {'1/N̄':>7} {'ROI':>8} {'z':>6} {'p':>7}")
for lbl, m in targets:
    c = cell_stats(m)
    if c is None or c["n"] == 0:
        emit(f"  {lbl:<46} (no data)")
        continue
    emit(f"  {lbl:<46} {c['n']:>6} {c['wr']:>7.4f} {np.mean(1/odds[m]):>7.4f} "
         f"{c['roi']*100:>7.1f}% {c['z']:>6.2f} {stats.norm.sf(c['z']):>7.4f}")
emit()

# =============================================================== TEST 6
emit("-" * 78)
emit("TEST 6. Split-half replication: pick the best cell on half A, test it on half B")
emit("-" * 78)
half = (day % 2 == 0)
allm, alll = [], []
for nm in sorted(set(names)):
    for N in range(3, 14):
        m = (names == nm) & (odds == N)
        if m.sum() >= 150:
            allm.append(m); alll.append(f"{nm} @ {N}:1")
for split_name, A, B in [("even days -> odd days", half, ~half),
                         ("odd days -> even days", ~half, half)]:
    zs = []
    for m in allm:
        c = cell_stats(m & A)
        zs.append(c["z"] if c and c["n"] >= 60 else -99)
    top = int(np.argmax(zs))
    cb = cell_stats(allm[top] & B)
    emit(f"  {split_name}: best in-sample cell = {alll[top]} (z={zs[top]:.2f})")
    emit(f"     out-of-sample: n={cb['n']} WR={cb['wr']:.4f} vs 1/N={1/odds[allm[top]][0]:.4f} "
         f"ROI={cb['roi']*100:+.1f}% z={cb['z']:+.2f}")
    # top 5 aggregated
    top5 = list(np.argsort(zs)[-5:])
    magg = np.zeros(len(rows), bool)
    for t in top5:
        magg |= allm[t] & B
    ca = cell_stats(magg)
    emit(f"     top-5 cells aggregated out-of-sample: n={ca['n']} ROI={ca['roi']*100:+.1f}% "
         f"z={ca['z']:+.2f}")
emit()

# =============================================================== TEST 7
emit("-" * 78)
emit("TEST 7. Time stability of per-bin ROI (is the odds maker drifting?)")
emit("-" * 78)
nd = day.max() + 1
edges = np.linspace(0, nd, 6).astype(int)
emit(f"  {'odds':>4} " + " ".join(f"{f'd{edges[i]}-{edges[i+1]}':>13}" for i in range(5)))
for N in [3, 5, 8, 9, 10, 11, 12, 13, 2]:
    cells = []
    for i in range(5):
        m = (odds == N) & (day >= edges[i]) & (day < edges[i + 1])
        c = cell_stats(m)
        cells.append(f"{c['roi']*100:>+7.1f}%({c['n']:>4})" if c else " " * 13)
    emit(f"  {N:>4} " + " ".join(cells))
emit()

# =============================================================== TEST 8
emit("-" * 78)
emit("TEST 8. Odds drift: opening odds >= 3 whose CURRENT odds rose")
emit("-" * 78)
emit("  H0 implies p > 1/(N+1) for N in 3..12, so a payout of N+1 or more is")
emit("  strictly +EV — an edge on pirates whose OPENING odds are not 2:1.")
has_cur = cur > 0
emit(f"  {'selection':<42} {'bets':>6} {'WR':>7} {'payout':>7} {'ROI':>8} {'z':>7}")
for lbl, m in [
    ("open>=3, current >= open+1", has_cur & (odds >= 3) & (odds <= 12) & (cur >= odds + 1)),
    ("open>=3, current >= open+2", has_cur & (odds >= 3) & (odds <= 12) & (cur >= odds + 2)),
    ("open>=3, current == open (control)", has_cur & (odds >= 3) & (odds <= 12) & (cur == odds)),
    ("open>=3, current < open (control)", has_cur & (odds >= 3) & (odds <= 12) & (cur < odds)),
    ("open=13, current==13 (no lower bound)", has_cur & (odds == 13) & (cur == 13)),
]:
    n = int(m.sum())
    if n == 0:
        continue
    pay = cur[m]
    profit = (won[m] * pay).sum() - n
    roi = profit / n
    # z against the strict structural bound p = 1/(N+1)
    p0 = 1.0 / (odds[m] + 1.0)
    mu = (p0 * pay).sum() - n
    sd = np.sqrt((p0 * (1 - p0) * pay ** 2).sum())
    emit(f"  {lbl:<42} {n:>6} {won[m].mean():>7.4f} {pay.mean():>7.2f} "
         f"{roi*100:>7.1f}% {(profit-0)/np.sqrt((won[m].mean()*(1-won[m].mean())*pay**2).sum()):>7.2f}")
emit("  (z is for ROI>0 using the empirical win rate for variance.)")
emit()

# =============================================================== TEST 9
emit("-" * 78)
emit("TEST 9. Power: how large an edge can we still NOT rule out?")
emit("-" * 78)
emit("  One-sided 95% upper confidence limit on true ROI (normal approx on the")
emit("  realised per-bet profit; a real edge above this would have been seen).")
emit(f"  {'selection':<32} {'bets':>7} {'ROI':>8} {'95% UCL on ROI':>16}")
sels = [(f"odds={N}", odds == N) for N in range(3, 14)]
sels += [("odds 3-12 pooled", (odds >= 3) & (odds <= 12)),
         ("odds 9-12 pooled", (odds >= 9) & (odds <= 12)),
         ("odds>=3 modern", (odds >= 3) & ~legacy),
         ("odds=2 (control)", odds == 2)]
for lbl, m in sels:
    prof = won[m] * odds[m] - 1.0
    ucl = prof.mean() + 1.645 * prof.std(ddof=1) / np.sqrt(len(prof))
    emit(f"  {lbl:<32} {len(prof):>7} {prof.mean()*100:>7.1f}% {ucl*100:>15.1f}%")
emit()

with open("edge_structural_results.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("wrote edge_structural_results.txt")
