#!/usr/bin/env python3
"""Is the odds maker's disagreement with the game about realised allergy damage?

Allergy damage is a die roll.  Our models marginalise over it, which is right for
predicting outcomes.  If instead the odds maker computes its probability from one
already-rolled damage value, then its line is a deterministic function of that
draw and looks noisy to any marginalised model -- but only for arenas that
contain allergic pirates.  Arenas where every pirate has na = 0 have nothing to
roll, so there the two must agree.
"""
import os

import numpy as np

import fc_data
import fc_odds
import fc_pmf

d = fc_data.load_arenas()
odds = d["odds"].astype(np.int64)
na = d["feat"][:, :, 3].astype(np.int64)
n = len(odds)
p_nn = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_probs.npz"))["p"]

n_allergic = (na > 0).sum(axis=1)          # pirates in the arena with any allergy
tot_na = na.sum(axis=1)
print(f"arenas by number of allergic pirates: " +
      ", ".join(f"{k}:{(n_allergic == k).sum()}" for k in range(5)))

models = {name: fc_pmf.arena_win_probs(d["feat"], m)[0]
          for name, m in fc_pmf.MODELS.items()}
models["p_nn"] = p_nn

print("\nslot-exact match with the published odds, by arena allergy content")
print(f"{'model':<8}{'rule':<10}" + "".join(f"{('nA=' + str(k)):>9}" for k in range(5))
      + f"{'all':>9}")
for name, pm in models.items():
    for lbl, step, mode in (("exact", None, "round"), ("@1% floor", 0.01, "floor")):
        pred = fc_odds.publish(pm, step, mode)
        row = f"{name:<8}{lbl:<10}"
        for k in range(5):
            m = n_allergic == k
            row += f"{100*(pred[m] == odds[m]).mean():>8.1f}%"
        row += f"{100*(pred == odds).mean():>8.1f}%"
        print(row)

print("\nsame, restricted to slots of pirates that have no allergy themselves")
own_clean = na == 0
for name, pm in models.items():
    pred = fc_odds.publish(pm, 0.01, "floor")
    row = f"{name:<8}{'@1% floor':<10}"
    for k in range(5):
        m = (n_allergic == k)[:, None] & own_clean
        row += f"{100*(pred[m] == odds[m]).mean():>8.1f}%" if m.sum() else f"{'-':>9}"
    print(row)

print("\narena-exact (all four odds) by allergy content, model M5 @1% floor")
pred = fc_odds.publish(models["M5"], 0.01, "floor")
for k in range(5):
    m = n_allergic == k
    print(f"  nA={k}: n={m.sum():>6} arena-exact={100*(pred[m] == odds[m]).all(axis=1).mean():>5.1f}%"
          f"  slot-exact={100*(pred[m] == odds[m]).mean():>5.1f}%")

print("\nmean |published - predicted| by own allergy count (M5 @1% floor)")
for a in range(0, 5):
    m = na == a
    if m.sum() < 50:
        continue
    print(f"  own na={a}: n={m.sum():>6} mean|err|={np.abs(pred[m] - odds[m]).mean():.3f} "
          f"exact={100*(pred[m] == odds[m]).mean():.1f}%")
