#!/usr/bin/env python3
"""Is the opening odds line a monte-carlo estimate of the real win probability?

Hypothesis under test (H_MC(N)):
    the odds maker runs N trials of the real game, takes p_hat_i = k_i / N,
    and publishes odds_i = clamp(floor(1/p_hat_i), 2, 13).

Four independent lines of evidence, the last two using the cross-fitted NN of
nn_truth.py as the source of truth for the real probability p:

  A  achievability  (model free)  -- which N can even produce the observed
     odds vectors, given sum_i k_i = N
  B  pseudo-likelihood sweep over N of the observed odds given p_nn
  C  win-rate-by-odds calibration: how sharp the observed line is, versus how
     sharp MC-N can be
  D  the odds-implied probability intervals versus reality, which is what
     kills N = infinity (an exact computation of the true probability)

Run nn_truth.py first to produce nn_probs.npz.
"""
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import binom

import fc_data

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "odds_mc_test.txt")

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    m = z * np.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n) / den
    return c - m, c + m


def bounds(o):
    """(lo, hi] implied by published odds o under floor(1/p) clamped to [2,13]"""
    lo = np.where(o <= 12, 1.0 / (o + 1.0), 0.0)
    hi = np.where(o >= 3, 1.0 / o, 1.0)
    return lo, hi


# ------------------------------------------------------------------ data
d = fc_data.load_arenas()
odds = d["odds"].astype(np.int64)
n = odds.shape[0]
win = np.zeros((n, 4), dtype=bool)
win[np.arange(n), d["winner"]] = True
legacy = d["legacy"]

say("=" * 78)
say("Is the opening odds line a monte carlo of the real win probability?")
say("=" * 78)
say(f"arenas {n}  (legacy {legacy.sum()}, modern {(~legacy).sum()})")

# =========================================================== A. achievability
say()
say("--- A. which N can produce the observed odds vectors (model free) ---")
say("k_i is an integer count out of N and sum_i k_i = N, so a published odds")
say("vector is only reachable for N where every odds value has a non-empty k")
say("range and the ranges can sum to exactly N.")


def krange(o, N):
    if o == 2:
        lo, hi = N // 3 + 1, N
    elif o == 13:
        lo, hi = 0, N // 13
    else:
        lo, hi = N // (o + 1) + 1, N // o
    return (lo, hi) if lo <= hi else None


vec_counts = Counter(tuple(sorted(row)) for row in odds)


def achievable_frac(N):
    ok = 0
    for v, c in vec_counts.items():
        tot_lo = tot_hi = 0
        good = True
        for o in v:
            r = krange(o, N)
            if r is None:
                good = False
                break
            tot_lo += r[0]
            tot_hi += r[1]
        if good and tot_lo <= N <= tot_hi:
            ok += c
    return ok / n


Ns_a = list(range(1, 1001)) + [1500, 2000, 5000, 10 ** 4, 10 ** 5, 10 ** 6]
frac = {N: achievable_frac(N) for N in Ns_a}
full = [N for N in Ns_a if frac[N] == 1.0]
say(f"distinct sorted odds vectors: {len(vec_counts)}")
say(f"smallest N that can produce every observed odds vector: {full[0]}")
say(f"N < {full[0]} is impossible for ANY published line. Sample coverage:")
for N in [10, 25, 50, 75, 100, 110, 111, 120, 150, 200, 500, 1000, 10 ** 6]:
    say(f"    N={N:>7}: {100*frac[N]:6.2f}% of arenas reachable")
bad_above = [N for N in range(full[0], 1001) if frac[N] < 1.0]
say(f"N in [{full[0]},1000] still impossible: {len(bad_above)} values, e.g. "
    f"{bad_above[:12]}")
say("Verdict A: every N below 111 is excluded outright, plus many larger ones.")

# ------------------------------------------------------------- NN truth model
npz_path = os.path.join(ROOT, "nn_probs.npz")
if not os.path.exists(npz_path):
    say("\nnn_probs.npz missing -- run nn_truth.py for sections B-D")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(0)

p = np.load(npz_path)["p"]
det = np.clip(np.floor(1.0 / p), 2, 13).astype(np.int64)   # N -> infinity odds
ll_nn = np.log(np.maximum(p[win], 1e-12)).mean()
say()
say(f"NN truth model: out-of-fold LL/arena = {ll_nn:.5f} "
    f"(uniform {np.log(0.25):.5f})")

# ============================================== B. pseudo-likelihood over N
say()
say("--- B. pseudo-likelihood of the published odds given p_nn, by N ---")
say("per slot: P(k_i in the range implied by the published odds), k_i~Bin(N,p_nn)")
flat_p = p.ravel()
flat_o = odds.ravel()


def slot_prob(N):
    lo_k = np.where(flat_o == 2, N // 3, np.where(flat_o == 13, -1, N // (flat_o + 1)))
    hi_k = np.where(flat_o == 2, N, np.where(flat_o == 13, N // 13, N // flat_o))
    pr = binom.cdf(hi_k, N, flat_p) - binom.cdf(lo_k, N, flat_p)
    return np.maximum(pr, 1e-300)


Ns_b = [111, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000,
        10000, 20000, 50000, 100000, 1000000]
say(f"{'N':>9} {'pseudo-LL/slot':>15} {'P(exact match)':>15}")
best = (-1e18, None)
for N in Ns_b:
    pr = slot_prob(N)
    pll = np.log(pr).mean()
    say(f"{N:>9} {pll:>15.5f} {np.exp(pll):>15.5f}")
    if pll > best[0]:
        best = (pll, N)
match_det = (det == odds).mean()
arena_match_det = (det == odds).all(axis=1).mean()
say(f"{'inf':>9} {'(det.)':>15} {match_det:>15.5f}")
say(f"best N in the sweep: {best[1]}  (pseudo-LL {best[0]:.5f})")
say(f"deterministic floor(1/p_nn) matches {100*match_det:.2f}% of slots, "
    f"{100*arena_match_det:.2f}% of whole arenas")

# ================================================== C. sharpness of the line
say()
say("--- C. win rate by published odds: observed line vs MC-N ---")
say("If the odds carry MC noise, each odds bucket is contaminated by pirates")
say("whose real p is far from 1/odds, and the win rate in the bucket gets")
say("dragged toward the average.  Observed buckets are sharp:")
say(f"{'odds':>5} {'n':>7} {'realWR':>8} {'95% CI':>17} {'implied p range':>18} {'meanP_nn':>9}")
obs_wr = {}
for v in range(2, 14):
    m = odds == v
    k = win[m].sum()
    nn_ = m.sum()
    wr = k / nn_
    lo, hi = wilson(k, nn_)
    b = bounds(np.array([v]))
    obs_wr[v] = (wr, lo, hi, nn_)
    say(f"{v:>5} {nn_:>7} {wr:>8.4f} [{lo:.4f},{hi:.4f}] "
        f"({b[0][0]:.4f},{b[1][0]:.4f}] {p[m].mean():>9.4f}")

rng = np.random.default_rng(12345)


def sim_profile(N, reps=4):
    """mean real p per simulated odds bucket, under H_MC(N) with p_nn as truth"""
    acc = {v: [0.0, 0] for v in range(2, 14)}
    for _ in range(reps):
        k = rng.multinomial(N, p)
        with np.errstate(divide="ignore"):
            o = np.where(k == 0, 13, np.clip(np.floor(N / np.maximum(k, 1)), 2, 13))
        o = o.astype(np.int64)
        for v in range(2, 14):
            m = o == v
            acc[v][0] += p[m].sum()
            acc[v][1] += m.sum()
    return {v: (acc[v][0] / acc[v][1] if acc[v][1] else np.nan, acc[v][1] / reps)
            for v in range(2, 14)}


say()
say("model-predicted win rate per odds bucket under H_MC(N) (p_nn as truth):")
hdr = f"{'odds':>5} {'observed':>9} {'95% CI':>17}"
Ns_c = [111, 200, 500, 1000, 5000, 100000]
for N in Ns_c:
    hdr += f" {('N=' + str(N)):>9}"
say(hdr)
prof = {N: sim_profile(N) for N in Ns_c}
for v in range(2, 14):
    wr, lo, hi, cnt = obs_wr[v]
    row = f"{v:>5} {wr:>9.4f} [{lo:.4f},{hi:.4f}]"
    for N in Ns_c:
        row += f" {prof[N][v][0]:>9.4f}"
    say(row)
say()
say("same table as a compatibility count (odds buckets 3..12 whose observed CI")
say("contains the model's predicted win rate):")
for N in Ns_c:
    okc = 0
    for v in range(3, 13):
        wr, lo, hi, cnt = obs_wr[v]
        if lo <= prof[N][v][0] <= hi:
            okc += 1
    say(f"    N={N:>7}: {okc}/10 buckets compatible")
say("bucket sizes drift with N too; observed vs predicted counts:")
row = f"{'odds':>5} {'observed':>9}" + "".join(f" {('N=' + str(N)):>9}" for N in Ns_c)
say(row)
for v in range(2, 14):
    row = f"{v:>5} {obs_wr[v][3]:>9}"
    for N in Ns_c:
        row += f" {int(prof[N][v][1]):>9}"
    say(row)

# =========================================== D. what kills N = infinity
say()
say("--- D. does the published line bracket the real probability? ---")
say("Under H_MC(inf) (exact computation of the real p) the real p must lie in")
say("the interval the published odds imply.  Test it with outcomes only.")
lo_b, hi_b = bounds(odds)
say(f"{'odds':>5} {'n':>7} {'realWR':>8} {'CI':>17} {'implied':>18} {'verdict':>10}")
viol = 0
for v in range(3, 13):
    m = odds == v
    k = win[m].sum()
    nn_ = m.sum()
    lo, hi = wilson(k, nn_)
    b_lo, b_hi = 1.0 / (v + 1.0), 1.0 / v
    bad = (hi < b_lo) or (lo > b_hi)
    viol += bad
    say(f"{v:>5} {nn_:>7} {k/nn_:>8.4f} [{lo:.4f},{hi:.4f}] "
        f"({b_lo:.4f},{b_hi:.4f}] {'VIOLATION' if bad else 'ok':>10}")
say(f"aggregate violations: {viol}/10 -- the line is well calibrated on average")

say()
say("but aggregates hide sign-flipping errors.  Slice by (pirate, regime):")
say("Orvinn the First Mate, whose allergy penalty the game stopped applying")
say("around round 8616 while the odds maker kept applying it (finding 4c/34):")
pir_names = [q["name"] for q in d["pirates"]]
orv = pir_names.index("Orvinn the First Mate")
sel_p = d["pirate_ix"] == orv
for label, regmask in (("legacy", legacy), ("modern", ~legacy)):
    for na_lo, na_hi in ((0, 0), (1, 1), (2, 6)):
        na = d["feat"][:, :, 3].astype(int)
        m = sel_p & regmask[:, None] & (na >= na_lo) & (na <= na_hi)
        if m.sum() < 50:
            continue
        k = win[m].sum()
        nn_ = m.sum()
        lo, hi = wilson(k, nn_)
        imp = (1.0 / odds[m]).mean()
        pnnm = p[m].mean()
        say(f"  {label:>6} na={na_lo}{'' if na_lo==na_hi else '+'}: n={nn_:>5} "
            f"realWR={k/nn_:.4f} CI[{lo:.4f},{hi:.4f}] "
            f"mean 1/odds={imp:.4f}  mean p_nn={pnnm:.4f}")

say()
say("population version, using p_nn as the real probability:")
inside = (p > lo_b) & (p <= hi_b)
say(f"slots whose p_nn lies inside the interval implied by the published odds: "
    f"{100*inside.mean():.2f}%")
for v in range(2, 14):
    m = odds == v
    say(f"  odds={v:>2}: inside {100*inside[m].mean():5.1f}%  "
        f"mean p_nn={p[m].mean():.4f}  implied ({1.0/(v+1) if v<=12 else 0:.4f},"
        f"{1.0/v if v>=3 else 1.0:.4f}]")

say()
say("For H_MC(inf) the *whole arena* must be bracketed. Fraction of arenas with")
say(f"all four p_nn inside: {100*inside.all(axis=1).mean():.2f}%")

# ================================================ E. predictability ceiling
say()
say("--- E. how predictable can a monte-carlo line be? ---")
say("Under H_MC(N) the line is random, so no deterministic rule can reproduce")
say("more than  mean_i max_v P(odds_i = v | p_i, N)  of the published values.")
say("Compare that against what fixed models actually score on the real line.")
import fc_pmf                                            # noqa: E402
import fc_odds                                           # noqa: E402

det_models = {name: fc_pmf.arena_win_probs(d["feat"], m)[0]
              for name, m in fc_pmf.MODELS.items()}
det_models["p_nn"] = p
say(f"{'model':<8}{'exact':>10}{'@1% floor':>12}")
best_real = 0.0
for name, pm in det_models.items():
    a = (fc_odds.publish(pm) == odds).mean()
    b = (fc_odds.publish(pm, 0.01, "floor") == odds).mean()
    best_real = max(best_real, a, b)
    say(f"{name:<8}{100*a:>9.2f}%{100*b:>11.2f}%")
say(f"best fixed rule on the real line: {100*best_real:.2f}% of slots")
say()
say(f"{'N':>8}{'ceiling':>10}{'truth rule':>12}{'verdict':>26}")
for N in (100, 111, 150, 200, 250, 300, 400, 500, 700, 1000, 2000, 5000):
    pr = []
    for v in range(2, 14):
        if v == 2:
            lo_k, hi_k = N // 3, N
        elif v == 13:
            lo_k, hi_k = -1, N // 13
        else:
            lo_k, hi_k = N // (v + 1), N // v
        pr.append(binom.cdf(hi_k, N, flat_p) - binom.cdf(lo_k, N, flat_p))
    pr = np.stack(pr)
    ceiling = pr.max(axis=0).mean()
    dv = fc_odds.publish(flat_p) - 2
    truth = pr[dv, np.arange(len(flat_p))].mean()
    verdict = "EXCLUDED (beaten by a fixed rule)" if ceiling < best_real else ""
    say(f"{N:>8}{100*ceiling:>9.1f}%{100*truth:>11.1f}%{verdict:>26}")

# =============================================== F. is the line deterministic?
say()
say("--- F. monotonicity: is the line a deterministic function of the state? ---")
say("Pirates with the same strength and the same weight offset are identical")
say("inputs to the game.  Inside one arena, if such a pirate has at least as many")
say("favourites, no more allergies and a later (better) position than its twin,")
say("its win probability cannot be lower, so a deterministic odds maker must not")
say("publish worse odds for it.  This needs no model of the game.")
nf_a = d["feat"][:, :, 2].astype(np.int64)
na_a = d["feat"][:, :, 3].astype(np.int64)
pixs = d["pirate_ix"].astype(np.int64)
twin = {}
for i, q in enumerate(d["pirates"]):
    key = (q["strength"], min((221 - min(q["weight"], 221)) // 2, 7))
    twin.setdefault(key, []).append(i)
twin_of = {i: k for k, v in twin.items() if len(v) > 1 for i in v}
say("  interchangeable groups: " + "; ".join(
    f"str{k[0]}: " + "/".join(d["pirates"][i]["name"].split()[0] for i in v)
    for k, v in twin.items() if len(v) > 1))

tests = []
for a in range(n):
    for i in range(4):
        for j in range(4):
            if i <= j or twin_of.get(pixs[a, i]) is None:
                continue
            if twin_of.get(pixs[a, i]) != twin_of.get(pixs[a, j]):
                continue
            if nf_a[a, i] >= nf_a[a, j] and na_a[a, i] <= na_a[a, j]:
                tests.append((a, i, j))      # i is at the later position
n_t = len(tests)
viol = [(a, i, j) for a, i, j in tests if odds[a, i] > odds[a, j]]
say(f"  comparisons: {n_t}   inversions: {len(viol)} "
    f"({100*len(viol)/max(n_t,1):.2f}%)   a deterministic rule allows 0")
say(f"  inversion sizes: " + str(sorted(int(odds[a, i] - odds[a, j])
                                       for a, i, j in viol)))
rng2 = np.random.default_rng(3)
say(f"{'N':>8}{'':>6}{'expected inversions':>21}{'P(<=obs)':>11}{'P(>=obs)':>11}")
for N in (100, 200, 300, 500, 700, 1000, 5000):
    for pct in (False, True):
        cnt = 0
        reps = 10
        for _ in range(reps):
            k = rng2.multinomial(N, p)
            q = k / N
            if pct:
                q = np.round(q * 100) / 100
            with np.errstate(divide="ignore"):
                o = np.where(q <= 0, 13, fc_odds.clamp_floor(
                    1.0 / np.maximum(q, 1e-12))).astype(np.int64)
            cnt += sum(1 for a, i, j in tests if o[a, i] > o[a, j])
        exp = max(cnt / reps, 1e-9)
        lo_p = binom.cdf(len(viol), n_t, exp / n_t)
        hi_p = 1 - binom.cdf(len(viol) - 1, n_t, exp / n_t)
        say(f"{N:>8}{('+pct' if pct else ''):>6}{exp:>21.1f}{lo_p:>11.2e}{hi_p:>11.2e}")

# ==================================== G. granularity and noise disagree on N
say()
say("--- G. the two measurements of N ---")
say("granularity: chi2 of the 12-bin odds histogram (quantisation makes it lumpy,")
say("             monte-carlo noise smooths the lumps away)")
say("noise:       inversions in test F")
obs_h = np.array([(odds == v).sum() for v in range(2, 14)])


def chi2_of(o, reps=1):
    h = np.array([(o == v).sum() for v in range(2, 14)]) / reps
    return (((obs_h - h) ** 2) / np.maximum(h, 1)).sum()


say(f"{'rule':<26}{'hist chi2':>11}{'inversions':>12}")
for lbl, st, md in (("deterministic, exact p", None, "round"),
                    ("deterministic, 1% round", 0.01, "round"),
                    ("deterministic, 1% floor", 0.01, "floor"),
                    ("deterministic, 0.5%", 0.005, "round")):
    o = fc_odds.publish(p, st, md)
    say(f"{lbl:<26}{chi2_of(o):>11.0f}"
        f"{sum(1 for a, i, j in tests if o[a, i] > o[a, j]):>12}")
for N in (100, 150, 200, 300, 500, 1000, 5000):
    for pct in (False, True):
        reps = 4
        os_ = []
        inv = 0
        for _ in range(reps):
            k = rng2.multinomial(N, p)
            q = k / N
            if pct:
                q = np.round(q * 100) / 100
            with np.errstate(divide="ignore"):
                o = np.where(q <= 0, 13, fc_odds.clamp_floor(
                    1.0 / np.maximum(q, 1e-12))).astype(np.int64)
            os_.append(o)
            inv += sum(1 for a, i, j in tests if o[a, i] > o[a, j])
        lbl = f"MC N={N}" + (" + pct" if pct else "")
        say(f"{lbl:<26}{chi2_of(np.concatenate(os_), reps):>11.0f}{inv/reps:>12.1f}")
say(f"{'OBSERVED':<26}{'--':>11}{len(viol):>12}")
say()
say("The histogram is only reproduced when the probability sits on a 1% grid,")
say("which for a monte carlo means N=100; but at N=100 the line would invert")
say("dominance far more often than it does, and a fixed rule already beats the")
say("N=100 predictability ceiling.  Larger N fixes the noise and breaks the")
say("histogram.  No single N does both.")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
