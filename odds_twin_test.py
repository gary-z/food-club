#!/usr/bin/env python3
"""Does the odds maker distinguish pirates the game cannot tell apart?

Pairs of pirates with the same strength and the same weight offset are identical
inputs to the game.  Inside arenas that contain both, with the same fav and
allergy counts, the only thing separating them is position.  Regress the odds
difference on the position difference: a non-zero intercept means the odds maker
holds a per-pirate view that the game's inputs do not contain.

The same paired design is run on outcomes (who actually won) as a control: the
game should show no such preference.
"""
import numpy as np

import fc_data

d = fc_data.load_arenas()
n = d["feat"].shape[0]
odds = d["odds"].astype(np.int64)
nf = d["feat"][:, :, 2].astype(np.int64)
na = d["feat"][:, :, 3].astype(np.int64)
pix = d["pirate_ix"].astype(np.int64)
pirates = d["pirates"]
winner = d["winner"]
legacy = d["legacy"]

groups = {}
for i, p in enumerate(pirates):
    wo = min((221 - min(p["weight"], 221)) // 2, 7)
    groups.setdefault((p["strength"], wo), []).append(i)
groups = {k: v for k, v in groups.items() if len(v) > 1}

slot_of = -np.ones((n, 20), dtype=np.int64)
for a in range(n):
    for s in range(4):
        slot_of[a, pix[a, s]] = s


def ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return b, se


print("paired within-arena comparison of interchangeable pirates")
print("model: odds_A - odds_B = c0 + c1*(pos_A - pos_B), matched nf and na")
print(f"{'pair':<44}{'n':>6}{'intercept':>12}{'t':>8}{'pos coef':>10}")
for key, mem in sorted(groups.items()):
    for x in range(len(mem)):
        for y in range(x + 1, len(mem)):
            A, B = mem[x], mem[y]
            rows_d, rows_p, rows_w = [], [], []
            for a in range(n):
                sa, sb = slot_of[a, A], slot_of[a, B]
                if sa < 0 or sb < 0:
                    continue
                if nf[a, sa] != nf[a, sb] or na[a, sa] != na[a, sb]:
                    continue
                rows_d.append(odds[a, sa] - odds[a, sb])
                rows_p.append(sa - sb)
                if winner[a] in (sa, sb):
                    rows_w.append(1.0 if winner[a] == sa else 0.0)
            if len(rows_d) < 40:
                continue
            X = np.column_stack([np.ones(len(rows_d)), rows_p])
            b, se = ols(X, rows_d)
            lbl = (f"{pirates[A]['name'].split()[0]}(wt{pirates[A]['weight']}) vs "
                   f"{pirates[B]['name'].split()[0]}(wt{pirates[B]['weight']}) str{key[0]}")
            print(f"{lbl:<44}{len(rows_d):>6}{b[0]:>+12.3f}{b[0]/se[0]:>8.2f}"
                  f"{b[1]:>+10.3f}")
            if rows_w:
                wr = np.mean(rows_w)
                sew = np.sqrt(wr * (1 - wr) / len(rows_w))
                print(f"{'    head-to-head win share of the first pirate':<44}"
                      f"{len(rows_w):>6}{wr:>12.3f}{(wr-0.5)/max(sew,1e-9):>8.2f}"
                      f"{'  (game control)':>10}")

print("\nsame test on the strongest signal (Ned vs Ogletree) split by era:")
A = [i for i, p in enumerate(pirates) if p["name"].startswith("Ned")][0]
B = [i for i, p in enumerate(pirates) if p["name"].startswith("Sir Edmund")][0]
for lbl, mask in (("legacy", legacy), ("modern", ~legacy)):
    rows_d, rows_p = [], []
    for a in np.where(mask)[0]:
        sa, sb = slot_of[a, A], slot_of[a, B]
        if sa < 0 or sb < 0 or nf[a, sa] != nf[a, sb] or na[a, sa] != na[a, sb]:
            continue
        rows_d.append(odds[a, sa] - odds[a, sb])
        rows_p.append(sa - sb)
    if len(rows_d) > 20:
        b, se = ols(np.column_stack([np.ones(len(rows_d)), rows_p]), rows_d)
        print(f"  {lbl}: n={len(rows_d)} intercept={b[0]:+.3f} t={b[0]/se[0]:.2f}")

print("\nmean published odds by pirate at nf=1, na=1, all positions pooled")
for key, mem in sorted(groups.items()):
    out = []
    for i in mem:
        vals = [odds[a, slot_of[a, i]] for a in range(n)
                if slot_of[a, i] >= 0 and nf[a, slot_of[a, i]] == 1
                and na[a, slot_of[a, i]] == 1]
        if vals:
            out.append(f"{pirates[i]['name'].split()[0]}(wt{pirates[i]['weight']})"
                       f"={np.mean(vals):.2f} n={len(vals)}")
    print(f"  str{key[0]}: " + "   ".join(out))
