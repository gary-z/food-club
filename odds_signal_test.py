#!/usr/bin/env python3
"""Does the opening line carry signal the NN does not already have?

Three conditional-logit models over the four slots of an arena, all scored by
out-of-fold log-likelihood on the same 5-fold split used by nn_truth.py:

    NN only    z_i = a * log p_nn_i
    line only  z_i = gamma[odds_i]                       (12 free values)
    both       z_i = a * log p_nn_i + gamma[odds_i]

"both" nests the other two, so the improvement of "both" over "NN only" is
exactly the extra signal in the published line, and it is measured per arena so
it can be tested pairwise.  A clamp variant (project p_nn into the probability
tile the odds imply, then renormalise) is included because that is what
sim/src/food_unified.rs does with the odds.

Section 2 repeats the test with current_odds (end-of-day, bet-volume driven),
which exists only for modern rounds.

Section 3 prices the buckets: EV = odds * win rate per odds value, which is
where the whole-percent rounding of finding 41 shows up.
"""
import os

import numpy as np

import fc_data
import fc_odds

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "odds_signal_test.txt")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


d = fc_data.load_arenas()
n = d["feat"].shape[0]
odds = d["odds"].astype(np.int64)
cur = d["cur_odds"].astype(np.int64)
Y = d["winner"].astype(np.int64)
win = np.zeros((n, 4), dtype=bool)
win[np.arange(n), Y] = True
npz = np.load(os.path.join(ROOT, "nn_probs.npz"))
p_nn = npz["p"]
fold = npz["fold"]
log_p = np.log(np.maximum(p_nn, 1e-12))


def fit_logit(feats, train, iters=3000, lr=0.05):
    """feats: list of (n,4) arrays for continuous terms, plus ('cat', codes, k)"""
    cont = [spec[1] for spec in feats if spec[0] == "cont"]
    cats = [(spec[1], spec[2]) for spec in feats if spec[0] == "cat"]
    n_par = len(cont) + sum(k for _, k in cats)
    th = np.zeros(n_par)
    m1 = np.zeros(n_par)
    m2 = np.zeros(n_par)
    idx = np.where(train)[0]

    def scores(th, rows):
        z = np.zeros((len(rows), 4))
        o = 0
        for f in cont:
            z += th[o] * f[rows]
            o += 1
        for c, k in cats:
            z += th[o:o + k][c[rows]]
            o += k
        return z

    for it in range(iters):
        z = scores(th, idx)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        g = win[idx].astype(float) - p            # (m,4)
        grad = np.empty(n_par)
        o = 0
        for f in cont:
            grad[o] = (g * f[idx]).sum() / len(idx)
            o += 1
        for c, k in cats:
            for v in range(k):
                grad[o + v] = g[c[idx] == v].sum() / len(idx)
            o += k
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad ** 2
        th += lr * (m1 / (1 - 0.9 ** (it + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-9)
    z = scores(th, np.arange(n))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return th, e / e.sum(axis=1, keepdims=True)


def crossfit(feats, mask=None, folds=5):
    """out-of-fold per-arena log-likelihood"""
    ll = np.full(n, np.nan)
    use = np.ones(n, dtype=bool) if mask is None else mask
    for k in range(folds):
        te = (fold == k) & use
        tr = (fold != k) & use
        if te.sum() == 0:
            continue
        _, p = fit_logit(feats, tr)
        ll[te] = np.log(np.maximum(p[te][np.arange(te.sum()), Y[te]], 1e-12))
    return ll


def paired(a, b, label_a, label_b, mask):
    dif = (a - b)[mask]
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    say(f"  {label_a} - {label_b}: {dif.mean():+.5f} per arena  "
        f"SE {se:.5f}  t={dif.mean()/se:+.2f}  n={len(dif)}")


say("=" * 78)
say("Does the opening line add signal beyond the NN?")
say("=" * 78)
say(f"arenas {n}; NN out-of-fold LL "
    f"{np.log(np.maximum(p_nn[win], 1e-12)).mean():.5f}")

o_code = odds - 2
feats_nn = [("cont", log_p)]
feats_odds = [("cat", o_code, 12)]
feats_both = [("cont", log_p), ("cat", o_code, 12)]

ll_nn = crossfit(feats_nn)
ll_odds = crossfit(feats_odds)
ll_both = crossfit(feats_both)

# clamp variant: push p_nn into the tile the published odds imply
lo_t, hi_t = fc_odds.tile_arrays(0.01, "round")
lo = lo_t[o_code]
hi = hi_t[o_code]
p_cl = np.clip(p_nn, lo, np.maximum(hi - 1e-9, lo))
for _ in range(50):
    p_cl = p_cl / p_cl.sum(axis=1, keepdims=True)
    p_cl = np.clip(p_cl, lo, np.maximum(hi - 1e-9, lo))
p_cl = p_cl / p_cl.sum(axis=1, keepdims=True)
ll_clamp = np.log(np.maximum(p_cl[win], 1e-12))

ok = ~np.isnan(ll_nn) & ~np.isnan(ll_odds) & ~np.isnan(ll_both)
say()
say("out-of-fold log-likelihood per arena (higher is better):")
say(f"  uniform                     {np.log(0.25):.5f}")
say(f"  published line alone        {ll_odds[ok].mean():.5f}")
say(f"  NN alone                    {ll_nn[ok].mean():.5f}")
say(f"  NN + line                   {ll_both[ok].mean():.5f}")
say(f"  NN clamped into odds tiles  {ll_clamp[ok].mean():.5f}")
say()
say("paired differences:")
paired(ll_both, ll_nn, "NN+line", "NN alone", ok)
paired(ll_both, ll_odds, "NN+line", "line alone", ok)
paired(ll_clamp, ll_nn, "clamped", "NN alone", ok)
say()
say("by regime:")
for lbl, m in (("legacy", d["legacy"] & ok), ("modern", ~d["legacy"] & ok)):
    say(f"  {lbl}: NN {ll_nn[m].mean():.5f}  line {ll_odds[m].mean():.5f}  "
        f"NN+line {ll_both[m].mean():.5f}")
    paired(ll_both, ll_nn, "   NN+line", "NN alone", m)

# ------------------------------------------------------- 2. current odds
say()
say("--- 2. current (end-of-day) odds, modern rounds only ---")
has_cur = (cur > 0).all(axis=1) & ok
say(f"arenas with current_odds: {has_cur.sum()}")
if has_cur.sum() > 500:
    c_code = np.clip(cur, 2, 13) - 2
    ll_nn_m = crossfit(feats_nn, mask=has_cur)
    ll_cur = crossfit([("cat", c_code, 12)], mask=has_cur)
    ll_nn_cur = crossfit([("cont", log_p), ("cat", c_code, 12)], mask=has_cur)
    ll_all3 = crossfit([("cont", log_p), ("cat", o_code, 12), ("cat", c_code, 12)],
                       mask=has_cur)
    ll_open_m = crossfit(feats_both, mask=has_cur)
    say(f"  NN alone            {ll_nn_m[has_cur].mean():.5f}")
    say(f"  current line alone  {ll_cur[has_cur].mean():.5f}")
    say(f"  NN + opening        {ll_open_m[has_cur].mean():.5f}")
    say(f"  NN + current        {ll_nn_cur[has_cur].mean():.5f}")
    say(f"  NN + opening + cur  {ll_all3[has_cur].mean():.5f}")
    paired(ll_nn_cur, ll_nn_m, "NN+current", "NN alone", has_cur)
    paired(ll_all3, ll_open_m, "NN+open+cur", "NN+open", has_cur)

# --------------------------------------------------- 3. what a bucket pays
say()
say("--- 3. what each bucket actually pays ---")
say("EV of a 1-unit bet = odds * win rate.  Under the finding-41 rule the odds")
say("maker's probability is rounded to whole percent BEFORE 1/p is floored, so")
say("the top of each percent tile can pay over 1 even when the odds maker is")
say("exactly right: pct=9 -> odds 11 pays 11*p for p up to 9.5%, i.e. 1.045.")
say(f"{'odds':>5}{'n':>8}{'winRate':>9}{'EV':>8}{'95% CI on EV':>18}"
    f"{'pct tile':>11}{'EV at tile top':>15}")
for v in range(2, 14):
    m = odds == v
    k = int(win[m].sum())
    nn_ = int(m.sum())
    wr = k / nn_
    se = np.sqrt(wr * (1 - wr) / nn_)
    ev = v * wr
    lo_v, hi_v = lo_t[v - 2], hi_t[v - 2]
    say(f"{v:>5}{nn_:>8}{wr:>9.4f}{ev:>8.3f}"
        f"{f'[{v*(wr-1.96*se):.3f},{v*(wr+1.96*se):.3f}]':>18}"
        f"{f'[{100*lo_v:.1f},{100*hi_v:.1f})%':>11}{v*hi_v:>15.3f}")
say()
pool = np.isin(odds, [9, 10, 11])
k = int(win[pool].sum())
tot = int(pool.sum())
ret = (odds[pool] * win[pool]).sum()
roi = ret / tot - 1
var = ((odds[pool] * win[pool] - ret / tot) ** 2).mean()
se = np.sqrt(var / tot)
say(f"pooled odds 9/10/11: {tot} bets, ROI {100*roi:+.2f}% "
    f"SE {100*se:.2f}%  z={roi/se:+.2f}")
pool2 = odds == 2
ret2 = (2 * win[pool2]).sum()
tot2 = int(pool2.sum())
roi2 = ret2 / tot2 - 1
se2 = np.sqrt(((2 * win[pool2] - ret2 / tot2) ** 2).mean() / tot2)
say(f"odds 2 for comparison: {tot2} bets, ROI {100*roi2:+.2f}% "
    f"SE {100*se2:.2f}%  z={roi2/se2:+.2f}")

# ------------------------------------------- 4. where does the extra signal sit?
say()
say("--- 4. which arenas the line helps on ---")
say("The NN sees (strength, weight, nf, na), so it can express any per-pirate")
say("effect except telling apart the twin pairs of finding 46, and it cannot")
say("express a per-era effect such as Orvinn's pre-PHP-fix allergy damage")
say("(findings 4c/34).  If the line's extra signal sits in those places it is a")
say("gap in our features, not superior knowledge on the odds maker's part.")
pir_names = [q["name"] for q in d["pirates"]]
orv = pir_names.index("Orvinn the First Mate")
has_orv = (d["pirate_ix"] == orv).any(axis=1)
twin_ids = []
bykey = {}
for i, q in enumerate(d["pirates"]):
    bykey.setdefault((q["strength"], min((221 - min(q["weight"], 221)) // 2, 7)),
                     []).append(i)
twin_set = {i for v in bykey.values() if len(v) > 1 for i in v}
n_twin = np.isin(d["pirate_ix"], list(twin_set)).sum(axis=1)
has_twin_pair = np.zeros(n, dtype=bool)
for v in bykey.values():
    if len(v) > 1:
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                both = (d["pirate_ix"] == v[a]).any(axis=1) & \
                       (d["pirate_ix"] == v[b]).any(axis=1)
                has_twin_pair |= both
for lbl, m in (("legacy, contains Orvinn", d["legacy"] & has_orv & ok),
               ("legacy, no Orvinn", d["legacy"] & ~has_orv & ok),
               ("modern, contains Orvinn", ~d["legacy"] & has_orv & ok),
               ("modern, no Orvinn", ~d["legacy"] & ~has_orv & ok),
               ("contains a twin pair", has_twin_pair & ok),
               ("no twin pair", ~has_twin_pair & ok)):
    paired(ll_both, ll_nn, f"{lbl:<26} NN+line", "NN", m)

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
