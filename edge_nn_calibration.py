#!/usr/bin/env python3
"""Per-pirate calibration of the retrained NN, using out-of-fold predictions
(every arena scored by a model that never saw it).  Comparable to
nn_winrate_table.txt but without that script's test-set early stopping."""
import numpy as np

rows = []
for v in ["base", "ident", "market"]:
    d = np.load(f"edge_oof_{v}.npz", allow_pickle=False)
    Y, oof = d["Y"], d["oof"]
    n = len(Y)
    ll = np.log(oof[np.arange(n), Y])
    leg = d["legacy"]
    rows.append((v, ll.mean(), ll[leg].mean(), ll[~leg].mean()))

d = np.load("edge_oof_base.npz", allow_pickle=False)
Y, oof, pid, pnames = d["Y"], d["oof"], d["pid"], d["pirate_names"]
n = len(Y)
won = np.zeros((n, 4), dtype=int)
won[np.arange(n), Y] = 1

out = []
out.append("NN out-of-fold calibration (5-fold by day, 3-seed ensemble, honest early stopping)")
out.append("")
out.append(f"{'variant':<10} {'OOF LL all':>11} {'legacy':>10} {'modern':>10}")
for v, a, l, m in rows:
    out.append(f"{v:<10} {a:>11.5f} {l:>10.5f} {m:>10.5f}")
out.append("")
out.append("Reference points: hand-rolled Model 4 modern LL = -1.06314;")
out.append("odds maker (normalised 1/N) = -1.09495 all / -1.08829 modern.")
out.append("nn_winrate_table.txt's -1.05821 is not comparable: that run early-stopped")
out.append("on its own test set, so it is an optimistic estimate.")
out.append("")
out.append("Per-pirate OOF calibration (variant 'base', all 5643 days):")
out.append(f"{'Pirate':<30} {'N':>6} {'Pred%':>7} {'Real%':>7} {'95% CI':>16} {'Status':>10}")
out.append("-" * 82)
res = []
for i, nm in enumerate(pnames):
    m = pid == i
    pred = oof[m].mean()
    real = won[m].mean()
    N = int(m.sum())
    z = 1.96
    den = 1 + z ** 2 / N
    c = (real + z ** 2 / (2 * N)) / den
    mar = z * np.sqrt((real * (1 - real) + z ** 2 / (4 * N)) / N) / den
    lo, hi = c - mar, c + mar
    res.append((real, nm, N, pred, lo, hi))
for real, nm, N, pred, lo, hi in sorted(res, reverse=True):
    st = "** OUT **" if (pred < lo or pred > hi) else "ok"
    out.append(f"{nm:<30} {N:>6} {pred*100:>6.1f}% {real*100:>6.1f}% "
               f"[{lo*100:>5.1f}%,{hi*100:>5.1f}%] {st:>10}")
txt = "\n".join(out)
print(txt)
open("edge_nn_oof_calibration.txt", "w").write(txt + "\n")
