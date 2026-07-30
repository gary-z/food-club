#!/usr/bin/env python3
"""Deep dive on the two candidate falsifications that survived the first pass:

  (a) the NN-selected "EV >= 1.15 at opening odds >= 3" strategy (+14.2%, z=2.75)
  (b) the odds=11 bin, which keeps showing up above its 1/N ceiling

Both are post-hoc selections from a search, so the only honest arbiter is
out-of-sample replication.  Also tests the *necessary condition* for any
NN-based edge: does the NN rank pirates within an odds bin better than chance?

Writes edge_deepdive_results.txt
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(777)
OUT = []


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


d = np.load("edge_oof_market.npz", allow_pickle=False)
db = np.load("edge_oof_base.npz", allow_pickle=False)
di = np.load("edge_oof_ident.npz", allow_pickle=False)
Y = d["Y"]
n_arenas = len(Y)
odds = d["odds"].reshape(-1)
day = np.repeat(d["day"], 4)
legacy = np.repeat(d["legacy"], 4)
na = d["na"].reshape(-1)
nf = d["nf"].reshape(-1)
pid = d["pid"].reshape(-1)
pnames = d["pirate_names"]
posn = np.tile(np.arange(4), n_arenas)
won = np.zeros((n_arenas, 4), dtype=np.int64)
won[np.arange(n_arenas), Y] = 1
won = won.reshape(-1)
p_mkt = d["oof"].reshape(-1)
p_base = db["oof"].reshape(-1)
p_ident = di["oof"].reshape(-1)
ev = p_mkt * odds
nd = day.max() + 1

emit("=" * 78)
emit("DEEP DIVE: are the surviving candidates real, or search artefacts?")
emit("=" * 78)
emit()


def stat_block(mask, label, width=46):
    n = int(mask.sum())
    if n == 0:
        emit(f"  {label:<{width}} (empty)")
        return None
    w = won[mask]
    o = odds[mask]
    p0 = 1.0 / o
    roi = (w * o).sum() / n - 1
    mu = (p0 * o).sum() / n - 1
    sd = np.sqrt((p0 * (1 - p0) * o ** 2).sum()) / n
    z = (roi - mu) / sd
    emit(f"  {label:<{width}} {n:>6} {w.mean():>7.4f} {roi*100:>+7.1f}% {z:>+6.2f}")
    return dict(n=n, roi=roi, z=z)


# =========================================================== 1. decomposition
emit("-" * 78)
emit("1. Where does the 'EV >= 1.15, odds >= 3' profit actually come from?")
emit("-" * 78)
sel = (odds >= 3) & (ev >= 1.15)
emit(f"  {'slice':<46} {'bets':>6} {'WR':>7} {'ROI':>8} {'z':>6}")
stat_block(sel, "ALL (the headline number)")
emit("  by opening odds:")
for N in range(3, 14):
    m = sel & (odds == N)
    if m.sum() >= 20:
        stat_block(m, f"  odds={N}")
emit("  by pirate (cells with >=100 bets):")
for i, nm in enumerate(pnames):
    m = sel & (pid == i)
    if m.sum() >= 100:
        stat_block(m, f"  {nm}")
emit()

# ======================================================== 2. temporal splits
emit("-" * 78)
emit("2. Out-of-sample replication in time")
emit("-" * 78)
emit(f"  {'slice':<46} {'bets':>6} {'WR':>7} {'ROI':>8} {'z':>6}")
emit("  a) headline strategy (EV>=1.15, odds>=3) by fifths of the dataset:")
edges = np.linspace(0, nd, 6).astype(int)
for i in range(5):
    m = sel & (day >= edges[i]) & (day < edges[i + 1])
    stat_block(m, f"  days {edges[i]}-{edges[i+1]}")
emit("  b) the same, first half vs second half:")
stat_block(sel & (day < nd // 2), "  first half")
stat_block(sel & (day >= nd // 2), "  second half")
emit("  c) odds=11 raw bin by fifths:")
for i in range(5):
    m = (odds == 11) & (day >= edges[i]) & (day < edges[i + 1])
    stat_block(m, f"  days {edges[i]}-{edges[i+1]}")
emit("  d) odds=11 + EV>=1.10 by half:")
m11 = (odds == 11) & (ev >= 1.10)
stat_block(m11 & (day < nd // 2), "  first half")
stat_block(m11 & (day >= nd // 2), "  second half")
emit()

# ============================================ 3. honest walk-forward protocol
emit("-" * 78)
emit("3. Strict walk-forward: pick the rule on early data, bet it on later data")
emit("-" * 78)
emit("  Rule space: (min odds in {3,6,9,11}, threshold T in {1.0,1.05,...,1.3}),")
emit("  chosen by highest z on the training window, then applied out-of-sample.")
grid = [(lo, T) for lo in (3, 6, 9, 11) for T in (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)]
emit(f"  {'protocol':<46} {'bets':>6} {'WR':>7} {'ROI':>8} {'z':>6}")
for frac, tag in [(0.5, "train first 50% -> test last 50%"),
                  (0.667, "train first 2/3 -> test last 1/3")]:
    cut = int(nd * frac)
    tr = day < cut
    te = day >= cut
    best, bz = None, -99
    for lo, T in grid:
        m = tr & (odds >= lo) & (ev >= T)
        if m.sum() < 200:
            continue
        o = odds[m]
        p0 = 1.0 / o
        roi = (won[m] * o).sum() / m.sum() - 1
        mu = (p0 * o).sum() / m.sum() - 1
        sd = np.sqrt((p0 * (1 - p0) * o ** 2).sum()) / m.sum()
        z = (roi - mu) / sd
        if z > bz:
            bz, best = z, (lo, T)
    lo, T = best
    emit(f"  {tag}: picked odds>={lo}, T={T} (train z={bz:+.2f})")
    stat_block(te & (odds >= lo) & (ev >= T), f"  out-of-sample result")
emit("  Rolling-origin: retrain the choice every fifth, bet the next fifth:")
tot_n = tot_prof = 0
for i in range(1, 5):
    tr = day < edges[i]
    te = (day >= edges[i]) & (day < edges[i + 1])
    best, bz = None, -99
    for lo, T in grid:
        m = tr & (odds >= lo) & (ev >= T)
        if m.sum() < 200:
            continue
        o = odds[m]
        p0 = 1.0 / o
        z = ((won[m] * o).sum() - (p0 * o).sum()) / np.sqrt((p0 * (1 - p0) * o ** 2).sum())
        if z > bz:
            bz, best = z, (lo, T)
    lo, T = best
    m = te & (odds >= lo) & (ev >= T)
    prof = (won[m] * odds[m]).sum() - m.sum()
    tot_n += m.sum()
    tot_prof += prof
    emit(f"    window {i}: rule odds>={lo} T={T:.2f} -> {int(m.sum()):>5} bets, "
         f"ROI {prof/max(m.sum(),1)*100:+.1f}%")
emit(f"    combined walk-forward: {int(tot_n)} bets, ROI {tot_prof/max(tot_n,1)*100:+.1f}%")
emit()

# ================================== 4. necessary condition: within-bin ranking
emit("-" * 78)
emit("4. Necessary condition — does the NN rank pirates WITHIN an odds bin?")
emit("-" * 78)
emit("  If the odds maker's p is the truth, the bin still spans p in (1/(N+1), 1/N],")
emit("  so within-bin ranking ability is expected and is NOT itself a falsification.")
emit("  It only tells us the NN has resolution to work with.")
emit(f"  {'odds':>4} {'bets':>6} " + " ".join(f"{'Q'+str(q+1):>7}" for q in range(5))
     + f" {'trend z':>8} {'1/N':>7}")
tot_z = []
for N in range(3, 14):
    m = np.flatnonzero(odds == N)
    if len(m) < 500:
        continue
    q = np.argsort(np.argsort(p_mkt[m])) * 5 // len(m)
    wrs = [won[m][q == k].mean() for k in range(5)]
    # Cochran-Armitage trend test
    x = q.astype(float)
    y = won[m].astype(float)
    r = np.corrcoef(x, y)[0, 1]
    z = r * np.sqrt(len(m) - 1)
    tot_z.append(z)
    emit(f"  {N:>4} {len(m):>6} " + " ".join(f"{w:>7.4f}" for w in wrs)
         + f" {z:>8.2f} {1/N:>7.4f}")
emit(f"  Stouffer-combined trend z across bins = {np.sum(tot_z)/np.sqrt(len(tot_z)):.2f}")
emit()
emit("  Top-quintile WR vs the bin ceiling 1/N (this IS the falsification test):")
emit(f"  {'odds':>4} {'bets':>6} {'Q5 WR':>8} {'1/N':>8} {'ROI':>8} {'z':>6}")
for N in range(3, 14):
    m = np.flatnonzero(odds == N)
    if len(m) < 500:
        continue
    q = np.argsort(np.argsort(p_mkt[m])) * 5 // len(m)
    top = m[q == 4]
    mm = np.zeros(len(odds), bool)
    mm[top] = True
    wr = won[top].mean()
    p0 = 1.0 / N
    z = (wr - p0) / np.sqrt(p0 * (1 - p0) / len(top))
    emit(f"  {N:>4} {len(top):>6} {wr:>8.4f} {p0:>8.4f} {(wr*N-1)*100:>7.1f}% {z:>6.2f}")
emit()

# ============================================== 5. consensus / robustness view
emit("-" * 78)
emit("5. Robustness of the headline pocket to the choice of NN")
emit("-" * 78)
emit(f"  {'selection':<46} {'bets':>6} {'WR':>7} {'ROI':>8} {'z':>6}")
for nm, p in [("market", p_mkt), ("base", p_base), ("ident", p_ident)]:
    stat_block((odds >= 3) & (p * odds >= 1.15), f"  {nm}: EV>=1.15")
stat_block((odds >= 3) & (p_mkt * odds >= 1.15) & (p_base * odds >= 1.15)
           & (p_ident * odds >= 1.15), "  all three agree EV>=1.15")
stat_block((odds >= 3) & (((p_mkt * odds >= 1.15).astype(int)
                           + (p_base * odds >= 1.15).astype(int)
                           + (p_ident * odds >= 1.15).astype(int)) == 1),
           "  exactly one model says EV>=1.15")
emit()

# ====================================================== 6. how many chances?
emit("-" * 78)
emit("6. How many independent chances did the search have?")
emit("-" * 78)
emit("  Counting the tests run across all three scripts: 11 odds bins, 160")
emit("  (pirate x odds) cells, 210 (odds x context) cells, 340 (pirate x food)")
emit("  cells, 11 EV thresholds, 57 (odds,T) cells, 3 NN variants.")
emit("  A z of 2.75 on one of ~800 correlated cells is unremarkable; the")
emit("  boundary-null bootstrap already returned FWER p=0.03 for the threshold")
emit("  family alone, and p=0.07 for the (odds,T) family. Neither survives")
emit("  the full search, which is why replication above is the deciding test.")
emit()

with open("edge_deepdive_results.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("wrote edge_deepdive_results.txt")
