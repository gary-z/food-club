#!/usr/bin/env python3
"""Reverse engineer the opening odds maker, with calibrated controls.

A fit quality is meaningless on its own, so the same per-state rating rule is
fitted against several *known* lines as well as the real one:

  published        the real opening odds
  p_nn             floor(1/p) of the NN's probability.  The NN is a softmax over
                   per-pirate scores, so this line is exactly representable by
                   the rating rule: it measures what the optimiser can do.
  p_nn @1%         the same, but with p rounded to whole percent first
  p_M4             floor(1/p) of the exact dice-race probability of Model 4.  A
                   race is not a Luce model, so this measures how much of a miss
                   non-separability alone can cause.
  p_M4 @1%
  MC N=...         a monte-carlo line drawn from p_nn with N trials.  These
                   calibrate how noisy a line looks to the fitter, which is what
                   lets the real line's noise level be read off.

Needs nn_probs.npz (nn_truth.py).
"""
import os

import numpy as np

import fc_data
import fc_odds
import fc_pmf

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "odds_reverse.txt")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


d = fc_data.load_arenas()
n = d["feat"].shape[0]
odds = d["odds"].astype(np.int64)
pix = d["pirate_ix"].astype(np.int64)
nf = d["feat"][:, :, 2].astype(np.int64)
na = d["feat"][:, :, 3].astype(np.int64)
pos = np.tile(np.arange(4), (n, 1))

ids, uniq = fc_odds.class_ids([pix, nf, na, pos], n)
n_cls = len(uniq)

rng = np.random.default_rng(7)
test = rng.random(n) < 0.2
train = ~test

p_nn = np.load(os.path.join(ROOT, "nn_probs.npz"))["p"]
p_m4, _, _ = fc_pmf.arena_win_probs(d["feat"], fc_pmf.MODELS["M4"])

say("=" * 78)
say("Reverse engineering the opening odds maker")
say("=" * 78)
say(f"arenas {n}, pirate states (pirate,nf,na,pos) {n_cls}")
say("fitting one Luce weight per pirate state on 80% of arenas, scoring exact")
say("reproduction of the line on the held-out 20%")

# initialisation from p_nn keeps every fit in the same basin
lw = np.log(np.maximum(p_nn, 1e-9))
lw = lw - lw.mean(axis=1, keepdims=True)
w0 = np.exp(np.bincount(ids.ravel(), weights=lw.ravel(), minlength=n_cls) /
            np.maximum(np.bincount(ids.ravel(), minlength=n_cls), 1))

targets = [("published", odds)]
targets.append(("p_nn exact", fc_odds.publish(p_nn)))
targets.append(("p_nn @1% round", fc_odds.publish(p_nn, 0.01, "round")))
targets.append(("p_M4 exact", fc_odds.publish(p_m4)))
targets.append(("p_M4 @1% round", fc_odds.publish(p_m4, 0.01, "round")))
for N in (100, 200, 500, 1000, 5000):
    k = rng.multinomial(N, p_nn)
    with np.errstate(divide="ignore"):
        o = np.where(k == 0, 13, fc_odds.clamp_floor(N / np.maximum(k, 1)))
    targets.append((f"MC N={N} from p_nn", o.astype(np.int64)))

rules = [("continuous", None, "round"), ("1% round", 0.01, "round")]

say()
header = f"{'line fitted':<20}" + "".join(
    f"{('tiles ' + r[0]):>26}" for r in rules)
say(header)
say(f"{'':<20}" + "".join(f"{'train / held-out slot':>26}" for _ in rules))
results = {}
for name, tgt in targets:
    row = f"{name:<20}"
    for rname, step, mode in rules:
        lo_t, hi_t = fc_odds.tile_arrays(step, mode)
        w = fc_odds.fit_separable(ids, n_cls, tgt, lo_t, hi_t, train, w0=w0,
                                  sweeps=6, verbose=False)
        pred = fc_odds.predict_separable(w, ids, step, mode)
        tr = (pred[train] == tgt[train]).mean()
        te = (pred[test] == tgt[test]).mean()
        ae = (pred[test] == tgt[test]).all(axis=1).mean()
        results[(name, rname)] = (tr, te, ae, w)
        row += f"{f'{100*tr:.1f}% / {100*te:.1f}%':>26}"
    say(row)

say()
say("arena-exact on held-out (all four odds reproduced):")
for name, _ in targets:
    say(f"  {name:<20}" + "".join(
        f"{('  ' + rules[i][0] + ': ' + f'{100*results[(name, rules[i][0])][2]:.1f}%'):>22}"
        for i in range(len(rules))))

say()
say("--- how noisy is the published line? ---")
say("Read the published line's fit quality against the monte-carlo controls:")
base = results[("published", "1% round")][1]
say(f"  published:      held-out slot-exact {100*base:.1f}%")
for N in (100, 200, 500, 1000, 5000):
    v = results[(f"MC N={N} from p_nn", "1% round")][1]
    say(f"  MC N={N:<6}   held-out slot-exact {100*v:.1f}%")
say(f"  p_nn exact:     held-out slot-exact "
    f"{100*results[('p_nn exact', '1% round')][1]:.1f}%")
say(f"  p_M4 exact:     held-out slot-exact "
    f"{100*results[('p_M4 exact', '1% round')][1]:.1f}%")

# ------------------------------------------------- what the weights look like
w_pub = results[("published", "1% round")][3]
q_pub = fc_odds.separable_probs(w_pub, ids)
say()
say("--- fitted rating for the published line ---")
say("(the separable fit is only a 65% approximation of the line, so read these")
say(" as the best separable summary of it, not as the odds maker's own table;")
say(" medians are used because classes seen only with clamped odds are unbounded)")
inv = {v: k for k, v in uniq.items()}
cnt = np.bincount(ids.ravel(), minlength=n_cls)
pirates = d["pirates"]


def wv(pi, f, a, po, min_n=30):
    c = uniq.get((pi, f, a, po))
    if c is None or cnt[c] < min_n:
        return None
    return w_pub[c]


say("position: median w(pos)/w(pos 0) over states seen with both")
for po in range(4):
    rs = [wv(pi, f, a, po) / wv(pi, f, a, 0)
          for pi in range(20) for f in range(5) for a in range(4)
          if wv(pi, f, a, po) and wv(pi, f, a, 0)]
    say(f"  pos={po}: n={len(rs):>4} ratio={np.median(rs):.4f}  1/ratio={1/np.median(rs):.4f}")
say("favourite: median w(nf)/w(nf-1)")
for f in range(1, 6):
    rs = [wv(pi, f, a, po) / wv(pi, f - 1, a, po)
          for pi in range(20) for a in range(4) for po in range(4)
          if wv(pi, f, a, po) and wv(pi, f - 1, a, po)]
    if rs:
        say(f"  nf {f-1}->{f}: n={len(rs):>4} ratio={np.median(rs):.4f}")
say("allergy: median w(na)/w(na-1) by pirate weight")
for a in range(1, 4):
    rs_all = []
    for pi in sorted(range(20), key=lambda i: -pirates[i]["weight"]):
        rs = [wv(pi, f, a, po) / wv(pi, f, a - 1, po)
              for f in range(5) for po in range(4)
              if wv(pi, f, a, po) and wv(pi, f, a - 1, po)]
        if len(rs) >= 3:
            rs_all.append((pirates[pi]["name"], pirates[pi]["weight"], np.median(rs)))
    say(f"  na {a-1}->{a}: " + ", ".join(
        f"{nm.split()[0]}(wt{wt}) {r:.3f}" for nm, wt, r in rs_all[:8]))

np.savez_compressed(os.path.join(ROOT, "odds_fit.npz"), w=w_pub, ids=ids,
                    keys=np.array(list(uniq.keys())), q=q_pub, test=test,
                    p_m4=p_m4)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT} and odds_fit.npz")
