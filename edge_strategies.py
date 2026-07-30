#!/usr/bin/env python3
"""Falsification attempt #2: can a neural net find +EV bets at opening odds >= 3?

Uses the out-of-fold (OOF) probabilities from edge_train_nn.py.  Every
probability used to *select* a bet was produced by a model that never saw that
arena, so realized ROI on a selected set is an unbiased estimate of that set's
true edge (selection on a noisy estimate biases the estimate, not the outcome).

H0: opening odds N = max(2, min(13, floor(1/p))) with p the true probability,
    hence p <= 1/N and EV = p*N <= 1 for every N >= 3.

Writes edge_strategy_results.txt
"""
import itertools
import numpy as np
from scipy import stats

RNG = np.random.default_rng(31337)
OUT = []
VARIANTS = ["base", "ident", "market"]


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ load OOF
data = {}
for v in VARIANTS:
    try:
        data[v] = np.load(f"edge_oof_{v}.npz", allow_pickle=False)
    except FileNotFoundError:
        print(f"missing edge_oof_{v}.npz — skipping")
d0 = data[VARIANTS[0]] if VARIANTS[0] in data else next(iter(data.values()))
Y = d0["Y"]
day_a = d0["day"]
legacy_a = d0["legacy"]
odds_a = d0["odds"]          # [arena, 4]
cur_a = d0["cur"]
na_a = d0["na"]
pnames = d0["pirate_names"]
pid_a = d0["pid"]
n_arenas = len(Y)

# flatten to per-bet arrays
def flat(x):
    return x.reshape(-1)


odds = flat(odds_a)
cur = flat(cur_a)
na = flat(na_a)
pid = flat(pid_a)
day = np.repeat(day_a, 4)
legacy = np.repeat(legacy_a, 4)
arena = np.repeat(np.arange(n_arenas), 4)
won = np.zeros((n_arenas, 4), dtype=np.int64)
won[np.arange(n_arenas), Y] = 1
won = flat(won)
P = {v: flat(data[v]["oof"]) for v in data}
S_arena = (1.0 / odds_a).sum(axis=1)
S = np.repeat(S_arena, 4)
n2 = np.repeat((odds_a == 2).sum(axis=1), 4)

# odds maker's own probability estimate (normalised implied)
imp_a = (1.0 / odds_a)
mkt = flat(imp_a / imp_a.sum(axis=1, keepdims=True))

emit("=" * 78)
emit("NN-BASED FALSIFICATION TESTS — 'no edge at opening odds != 2'")
emit("=" * 78)
emit(f"{n_arenas} arenas / {len(odds)} pirate-slots; OOF variants: {list(data)}")
emit()


# ------------------------------------------------------------------- helpers
def day_bootstrap_roi(mask, payout=None, n_boot=4000):
    """Day-clustered bootstrap CI on ROI (bets on the same day share a draw)."""
    if payout is None:
        payout = odds
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return (np.nan, np.nan, np.nan)
    d = day[idx]
    prof = won[idx] * payout[idx] - 1.0
    ud, inv = np.unique(d, return_inverse=True)
    sums = np.bincount(inv, weights=prof)
    cnts = np.bincount(inv)
    k = len(ud)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.integers(0, k, k)
        boot[b] = sums[pick].sum() / max(cnts[pick].sum(), 1)
    return prof.sum() / len(idx), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def boundary_z(mask, payout=None):
    """z-score of realized profit against the boundary null p_i = 1/N_i (EV=1)."""
    if payout is None:
        payout = odds
    m = np.flatnonzero(mask)
    if len(m) == 0:
        return np.nan
    p0 = 1.0 / odds[m]
    pay = payout[m]
    obs = (won[m] * pay).sum() - len(m)
    mu = (p0 * pay).sum() - len(m)
    sd = np.sqrt((p0 * (1 - p0) * pay ** 2).sum())
    return (obs - mu) / sd


def boundary_boot_maxz(masks, n_sim=20000):
    idx = [np.flatnonzero(m) for m in masks]
    keep = [i for i in idx if len(i) > 0]
    if not keep:
        return np.zeros(n_sim)
    allidx = np.unique(np.concatenate(keep))
    p0 = 1.0 / odds[allidx]
    pos = {v: i for i, v in enumerate(allidx)}
    cols = [np.array([pos[v] for v in ix]) if len(ix) else np.array([], int) for ix in idx]
    pay = [odds[ix] for ix in idx]
    mus = [(1.0 / odds[ix] * odds[ix]).sum() - len(ix) if len(ix) else 0.0 for ix in idx]
    sds = [np.sqrt(((1.0 / odds[ix]) * (1 - 1.0 / odds[ix]) * odds[ix] ** 2).sum())
           if len(ix) else 1.0 for ix in idx]
    out = np.full(n_sim, -99.0)
    for s in range(n_sim):
        sim = RNG.random(len(allidx)) < p0
        best = -99.0
        for c, py, mu, sd, ix in zip(cols, pay, mus, sds, idx):
            if len(ix) == 0:
                continue
            obs = (sim[c] * py).sum() - len(ix)
            z = (obs - mu) / sd
            if z > best:
                best = z
        out[s] = best
    return out


# ====================================================== 1. model vs odds maker
emit("-" * 78)
emit("1. Does the NN actually know more than the odds maker?")
emit("-" * 78)
emit("   Per-arena log-likelihood of the realised winner (higher = better).")
mkt_a = imp_a / imp_a.sum(axis=1, keepdims=True)
ll_mkt = np.log(mkt_a[np.arange(n_arenas), Y])
emit(f"   {'model':<28} {'LL all':>9} {'LL legacy':>10} {'LL modern':>10} "
     f"{'vs market':>10} {'95% CI':>18}")
emit(f"   {'odds maker (norm. 1/N)':<28} {ll_mkt.mean():>9.5f} "
     f"{ll_mkt[legacy_a].mean():>10.5f} {ll_mkt[~legacy_a].mean():>10.5f}")
for v in data:
    ll = np.log(data[v]["oof"][np.arange(n_arenas), Y])
    diff = ll - ll_mkt
    ud, inv = np.unique(day_a, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    k = len(ud)
    boot = np.array([sums[p].sum() / cnts[p].sum()
                     for p in (RNG.integers(0, k, k) for _ in range(2000))])
    emit(f"   {'NN ' + v:<28} {ll.mean():>9.5f} {ll[legacy_a].mean():>10.5f} "
         f"{ll[~legacy_a].mean():>10.5f} {diff.mean():>+10.5f} "
         f"[{np.percentile(boot,2.5):+.5f},{np.percentile(boot,97.5):+.5f}]")
emit("   (Reference: hand-rolled Model 4 scores -1.06314 on modern data.)")
emit()

BEST = "market" if "market" in data else list(data)[0]
p_nn = P[BEST]
ev_nn = p_nn * odds
emit(f"   Using '{BEST}' OOF probabilities for the strategy tests below.")
emit()

# =============================================== 2. calibration above the bound
emit("-" * 78)
emit("2. Calibration of the NN where it claims an edge (odds >= 3)")
emit("-" * 78)
emit("   If H0 holds, realised WR should track min(NN p, 1/N) — the NN's claimed")
emit("   excess above the bin ceiling should evaporate.")
emit(f"   {'NN EV bucket':<16} {'bets':>7} {'NN p':>8} {'1/N':>8} {'real WR':>8} "
     f"{'ROI':>8} {'z(EV>1)':>8}")
m3 = odds >= 3
edges = [0.0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 9.9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = m3 & (ev_nn >= lo) & (ev_nn < hi)
    if m.sum() < 30:
        continue
    emit(f"   [{lo:.2f},{hi:.2f}){'':<5} {m.sum():>7} {p_nn[m].mean():>8.4f} "
         f"{(1/odds[m]).mean():>8.4f} {won[m].mean():>8.4f} "
         f"{((won[m]*odds[m]).sum()/m.sum()-1)*100:>7.1f}% {boundary_z(m):>8.2f}")
emit()

# ===================================================== 3. the headline strategy
emit("-" * 78)
emit("3. STRATEGY: bet 1 unit on every pirate with opening odds >= 3 and NN EV >= T")
emit("-" * 78)
emit(f"   {'T':>5} {'bets':>7} {'WR':>7} {'mean N':>7} {'NN EV':>7} {'ROI':>8} "
     f"{'95% CI (day-boot)':>22} {'z vs EV=1':>10}")
Ts = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00]
masks, zs = [], []
for T in Ts:
    m = m3 & (ev_nn >= T)
    masks.append(m)
    if m.sum() == 0:
        emit(f"   {T:>5.2f}       0")
        zs.append(-99)
        continue
    roi, lo, hi = day_bootstrap_roi(m)
    z = boundary_z(m)
    zs.append(z)
    emit(f"   {T:>5.2f} {m.sum():>7} {won[m].mean():>7.4f} {odds[m].mean():>7.2f} "
         f"{ev_nn[m].mean():>7.3f} {roi*100:>7.1f}% "
         f"[{lo*100:>+6.1f}%,{hi*100:>+6.1f}%] {z:>10.2f}")
maxz = boundary_boot_maxz(masks, 10000)
obs = max(zs)
emit(f"   max z over the {len(Ts)} thresholds = {obs:.2f}; "
     f"FWER p = {(maxz >= obs).mean():.4f} (boundary-null bootstrap)")
emit()

emit("   Same sweep for the other NN variants (does the choice of model matter?):")
emit(f"   {'variant':<8} {'T':>5} {'bets':>7} {'WR':>7} {'ROI':>8} {'z':>7}")
for v in data:
    for T in [1.0, 1.1, 1.2, 1.3]:
        m = m3 & (P[v] * odds >= T)
        if m.sum() < 20:
            continue
        emit(f"   {v:<8} {T:>5.2f} {m.sum():>7} {won[m].mean():>7.4f} "
             f"{((won[m]*odds[m]).sum()/m.sum()-1)*100:>7.1f}% {boundary_z(m):>7.2f}")
emit()

emit("   Same sweep, restricted to odds >= 3 in the MODERN regime:")
emit(f"   {'T':>5} {'bets':>7} {'WR':>7} {'ROI':>8} {'z':>7}")
for T in [1.0, 1.1, 1.2, 1.3, 1.5]:
    m = m3 & (ev_nn >= T) & ~legacy
    if m.sum() < 20:
        continue
    emit(f"   {T:>5.2f} {m.sum():>7} {won[m].mean():>7.4f} "
         f"{((won[m]*odds[m]).sum()/m.sum()-1)*100:>7.1f}% {boundary_z(m):>7.2f}")
emit()

# =================================================== 4. consensus of 3 models
if len(data) >= 2:
    emit("-" * 78)
    emit("4. 'Very confident': every NN variant independently says EV >= T")
    emit("-" * 78)
    emit(f"   {'T':>5} {'bets':>7} {'WR':>7} {'mean N':>7} {'ROI':>8} "
         f"{'95% CI (day-boot)':>22} {'z':>7}")
    cmasks, czs = [], []
    for T in [1.0, 1.05, 1.10, 1.20, 1.30, 1.50]:
        m = m3.copy()
        for v in data:
            m &= (P[v] * odds >= T)
        cmasks.append(m)
        if m.sum() < 20:
            czs.append(-99)
            emit(f"   {T:>5.2f} {m.sum():>7}  (too few)")
            continue
        roi, lo, hi = day_bootstrap_roi(m)
        z = boundary_z(m)
        czs.append(z)
        emit(f"   {T:>5.2f} {m.sum():>7} {won[m].mean():>7.4f} {odds[m].mean():>7.2f} "
             f"{roi*100:>7.1f}% [{lo*100:>+6.1f}%,{hi*100:>+6.1f}%] {z:>7.2f}")
    maxz = boundary_boot_maxz(cmasks, 10000)
    emit(f"   max z = {max(czs):.2f}; FWER p = {(maxz >= max(czs)).mean():.4f}")
    emit()

# ================================================ 5. per-odds-level EV sweep
emit("-" * 78)
emit("5. Per-odds-level EV sweep (honest OOF version of finding 31)")
emit("-" * 78)
emit(f"   {'odds':>4} {'best T':>7} {'bets':>6} {'WR':>7} {'1/N':>7} {'ROI':>8} {'z':>6}")
allm, allz = [], []
for N in range(3, 14):
    best = None
    for T in [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
        m = (odds == N) & (ev_nn >= T)
        allm.append(m)
        if m.sum() < 50:
            allz.append(-99)
            continue
        z = boundary_z(m)
        allz.append(z)
        if best is None or z > best[0]:
            best = (z, T, m)
    if best is None:
        emit(f"   {N:>4}   (no cell with >=50 bets)")
        continue
    z, T, m = best
    emit(f"   {N:>4} {T:>7.2f} {m.sum():>6} {won[m].mean():>7.4f} {1/N:>7.4f} "
         f"{((won[m]*odds[m]).sum()/m.sum()-1)*100:>7.1f}% {z:>6.2f}")
maxz = boundary_boot_maxz(allm, 5000)
emit(f"   max z over all {len([z for z in allz if z>-99])} (odds,T) cells = {max(allz):.2f}; "
     f"FWER p = {(maxz >= max(allz)).mean():.4f}")
emit()

# ==================================== 6. single pre-registered weighted statistic
emit("-" * 78)
emit("6. Edge-weighted aggregate statistic (one pre-registered test, no sweep)")
emit("-" * 78)
w = np.maximum(0.0, ev_nn - 1.0) * m3
sel = w > 0
stat = (w[sel] * (won[sel] * odds[sel] - 1.0)).sum()
p0 = 1.0 / odds[sel]
mu = (w[sel] * (p0 * odds[sel] - 1.0)).sum()
sd = np.sqrt((w[sel] ** 2 * p0 * (1 - p0) * odds[sel] ** 2).sum())
emit(f"   sum of w_i*(N_i*X_i - 1) with w_i = max(0, NN_EV_i - 1), odds>=3")
emit(f"   n={sel.sum()} bets, statistic={stat:.1f}, boundary-null mean={mu:.1f}, "
     f"sd={sd:.1f}, z={(stat-mu)/sd:+.2f}, p={stats.norm.sf((stat-mu)/sd):.4f}")
emit()

# ============================================= 7. tight no-2:1 arenas (sharp H0)
emit("-" * 78)
emit("7. Sharpest region: arenas with NO 2:1 pirate, sorted by slack S-1")
emit("-" * 78)
emit("   With no 2:1, sum_i p_i = 1 and p_i <= 1/N_i, so every pirate obeys")
emit("   1/N_i - (S-1) <= p_i <= 1/N_i:  H0 pins EV into [1 - N(S-1), 1].")
no2 = n2 == 0
emit(f"   {'slack S-1':<14} {'bets':>6} {'WR':>7} {'mean 1/N':>9} {'ROI':>8} "
     f"{'H0 floor':>9} {'z':>7}")
for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 9)]:
    m = no2 & (S - 1 >= lo) & (S - 1 < hi)
    if m.sum() < 20:
        continue
    floor = (1 - odds[m] * (S[m] - 1)).mean() - 1
    emit(f"   [{lo:.2f},{hi:.2f}){'':<3} {m.sum():>6} {won[m].mean():>7.4f} "
         f"{(1/odds[m]).mean():>9.4f} {((won[m]*odds[m]).sum()/m.sum()-1)*100:>7.1f}% "
         f"{floor*100:>8.1f}% {boundary_z(m):>7.2f}")
m = no2
roi, lo_, hi_ = day_bootstrap_roi(m)
emit(f"   ALL no-2:1 arenas: {m.sum()} bets, ROI={roi*100:+.1f}% "
     f"[{lo_*100:+.1f}%,{hi_*100:+.1f}%], z={boundary_z(m):+.2f}")
emit()

# ================================================= 8. parlays of non-2:1 only
emit("-" * 78)
emit("8. NN-picked parlays that contain NO 2:1 pirate (top-10 EV per day)")
emit("-" * 78)
emit("   Under H0 every leg has EV<=1 so any parlay has EV<=1; payout capped at 60")
oofB = np.asarray(data[BEST]["oof"])   # hoisted: NpzFile re-decompresses per access
day_first = {}
for i, d in enumerate(day_a):
    day_first.setdefault(int(d), []).append(i)
tot_bets = tot_profit = 0.0
day_profit = {}
for d, arenas in day_first.items():
    cands = []
    for ar in arenas:
        opts = [None]
        for k in range(4):
            if odds_a[ar][k] >= 3:
                opts.append(k)
        cands.append((ar, opts))
    bets = []
    for combo in itertools.product(*[o for _, o in cands]):
        if all(c is None for c in combo):
            continue
        p, payout, legs = 1.0, 1, []
        for (ar, _), k in zip(cands, combo):
            if k is None:
                continue
            p *= oofB[ar][k]
            payout = min(payout * int(odds_a[ar][k]), 60)
            legs.append((ar, k))
        bets.append((p * payout, p, payout, legs))
    bets.sort(key=lambda x: -x[0])
    dp = 0.0
    for ev, p, payout, legs in bets[:10]:
        tot_bets += 1
        winnings = payout if all(Y[ar] == k for ar, k in legs) else 0
        dp += winnings - 1
    tot_profit += dp
    day_profit[d] = dp
emit(f"   {int(tot_bets)} bets, profit {tot_profit:+.0f}, ROI {tot_profit/tot_bets*100:+.2f}%")
dp = np.array(list(day_profit.values()))
se = dp.std(ddof=1) / np.sqrt(len(dp)) * len(dp) / tot_bets
emit(f"   day-level SE of ROI = {se*100:.2f}%  ->  z = {tot_profit/tot_bets/se:+.2f}")
emit()

# ================================================ 9. current-odds drift edge
emit("-" * 78)
emit("9. Non-2:1 pirates whose CURRENT odds drifted above the opening bound")
emit("-" * 78)
emit("   H0 itself implies p > 1/(N+1), so paying out at N+1 or better is +EV.")
hascur = cur > 0
emit(f"   {'selection':<44} {'bets':>6} {'WR':>7} {'payout':>7} {'ROI':>8} {'95% CI':>20}")
for lbl, m in [
    ("open 3-12, current >= open+1", hascur & (odds >= 3) & (odds <= 12) & (cur >= odds + 1)),
    ("open 3-12, current >= open+2", hascur & (odds >= 3) & (odds <= 12) & (cur >= odds + 2)),
    ("  ... and NN EV(current) >= 1.2",
     hascur & (odds >= 3) & (odds <= 12) & (cur >= odds + 1) & (p_nn * cur >= 1.2)),
    ("open 3-12, current == open (control)", hascur & (odds >= 3) & (odds <= 12) & (cur == odds)),
    ("open 3-12, current < open (control)", hascur & (odds >= 3) & (odds <= 12) & (cur < odds)),
]:
    if m.sum() < 10:
        continue
    roi, lo_, hi_ = day_bootstrap_roi(m, payout=cur.astype(float))
    emit(f"   {lbl:<44} {m.sum():>6} {won[m].mean():>7.4f} {cur[m].mean():>7.2f} "
         f"{roi*100:>7.1f}% [{lo_*100:>+6.1f}%,{hi_*100:>+6.1f}%]")
emit()

with open("edge_strategy_results.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("wrote edge_strategy_results.txt")
