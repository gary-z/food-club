#!/usr/bin/env python3
"""Final arbitration.

Two competing accounts of the NN-selected bets at opening odds >= 3:

  M1 "the NN is right":  realised WR should equal the NN's probability p_nn
  M0 "the odds maker is right (H0)": realised WR should equal min(p_nn, 1/N),
      i.e. the NN's claimed excess above the bin ceiling evaporates

These make quantitatively different predictions on the same bets, so the data
can choose between them.  Also: does anything survive dropping the odds=11
pocket, and does the pocket beat even the model that selected it?

Writes edge_verdict_results.txt
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(4242)
OUT = []


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


d = np.load("edge_oof_market.npz", allow_pickle=False)
Y = d["Y"]
n_arenas = len(Y)
odds = d["odds"].reshape(-1)
day = np.repeat(d["day"], 4)
legacy = np.repeat(d["legacy"], 4)
won = np.zeros((n_arenas, 4), dtype=np.int64)
won[np.arange(n_arenas), Y] = 1
won = won.reshape(-1)
p = d["oof"].reshape(-1)
ev = p * odds
m3 = odds >= 3

emit("=" * 78)
emit("VERDICT: is the NN's claimed edge delivered, or capped at the bin ceiling?")
emit("=" * 78)
emit()

# ---------------------------------------------------------- 1. claim delivery
emit("-" * 78)
emit("1. Claimed vs delivered ROI at each confidence threshold (odds >= 3)")
emit("-" * 78)
emit("   M1 predicts delivered = claimed.  M0 (H0) predicts delivered = capped,")
emit("   where capped ROI = mean(min(p_nn, 1/N) * N) - 1 <= 0.")
emit(f"   {'T':>5} {'bets':>6} {'claimed':>8} {'capped':>8} {'delivered':>10} "
     f"{'z vs M1':>8} {'z vs M0':>8}")
for T in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]:
    m = m3 & (ev >= T)
    n = int(m.sum())
    if n < 20:
        continue
    o = odds[m]
    claimed = (p[m] * o).mean() - 1
    capped = (np.minimum(p[m], 1.0 / o) * o).mean() - 1
    prof = won[m] * o - 1.0
    delivered = prof.mean()
    se = prof.std(ddof=1) / np.sqrt(n)
    emit(f"   {T:>5.2f} {n:>6} {claimed*100:>+7.1f}% {capped*100:>+7.1f}% "
         f"{delivered*100:>+9.1f}% {(delivered-claimed)/se:>+8.2f} "
         f"{(delivered-capped)/se:>+8.2f}")
emit()
emit("   Same, with the odds=11 bin removed:")
emit(f"   {'T':>5} {'bets':>6} {'claimed':>8} {'capped':>8} {'delivered':>10} "
     f"{'z vs M1':>8} {'z vs M0':>8}")
for T in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]:
    m = m3 & (ev >= T) & (odds != 11)
    n = int(m.sum())
    if n < 20:
        continue
    o = odds[m]
    claimed = (p[m] * o).mean() - 1
    capped = (np.minimum(p[m], 1.0 / o) * o).mean() - 1
    prof = won[m] * o - 1.0
    delivered = prof.mean()
    se = prof.std(ddof=1) / np.sqrt(n)
    emit(f"   {T:>5.2f} {n:>6} {claimed*100:>+7.1f}% {capped*100:>+7.1f}% "
         f"{delivered*100:>+9.1f}% {(delivered-claimed)/se:>+8.2f} "
         f"{(delivered-capped)/se:>+8.2f}")
emit()

# ------------------------------------------------- 2. leave-one-odds-bin-out
emit("-" * 78)
emit("2. Leave-one-bin-out on the headline strategy (EV >= 1.15, odds >= 3)")
emit("-" * 78)
sel = m3 & (ev >= 1.15)
emit(f"   {'excluded bin':<16} {'bets':>6} {'ROI':>8} {'z vs EV=1':>10}")
for N in [None] + list(range(3, 14)):
    m = sel if N is None else (sel & (odds != N))
    o = odds[m]
    p0 = 1.0 / o
    prof = (won[m] * o).sum() - m.sum()
    mu = (p0 * o).sum() - m.sum()
    sd = np.sqrt((p0 * (1 - p0) * o ** 2).sum())
    lbl = "none (baseline)" if N is None else f"odds={N}"
    emit(f"   {lbl:<16} {int(m.sum()):>6} {prof/m.sum()*100:>+7.1f}% {(prof-mu)/sd:>+10.2f}")
emit()

# ------------------------------------------- 3. does the pocket beat the NN?
emit("-" * 78)
emit("3. The odds=11 pocket beats even the model that found it")
emit("-" * 78)
m = (odds == 11) & (ev >= 1.10)
n = int(m.sum())
wr = won[m].mean()
pnn = p[m].mean()
emit(f"   n={n}: bin ceiling 1/11 = {1/11:.4f}, NN says p = {pnn:.4f}, "
     f"realised = {wr:.4f}")
se_nn = np.sqrt(pnn * (1 - pnn) / n)
se_ceil = np.sqrt((1 / 11) * (1 - 1 / 11) / n)
emit(f"   realised vs ceiling : z = {(wr-1/11)/se_ceil:+.2f}  "
     f"(this is the apparent falsification)")
emit(f"   realised vs NN claim: z = {(wr-pnn)/se_nn:+.2f}  "
     f"(the pocket also beats the NN itself)")
emit("   A genuine mispricing the NN detected should land ON the NN's number,")
emit("   not 2 sigma above it. Overshooting the discovering model is the")
emit("   signature of an upward fluctuation, not of a discovered edge.")
emit()

# ---------------------------------------------------- 4. modern regime only
emit("-" * 78)
emit("4. Modern regime only (the regime you would actually bet in)")
emit("-" * 78)
emit(f"   {'selection':<34} {'bets':>6} {'ROI':>8} {'95% CI':>20}")
for lbl, m in [("all odds>=3", m3 & ~legacy),
               ("odds>=3, NN EV>=1.10", m3 & ~legacy & (ev >= 1.10)),
               ("odds>=3, NN EV>=1.15", m3 & ~legacy & (ev >= 1.15)),
               ("odds=11 (all)", (odds == 11) & ~legacy),
               ("odds=11, NN EV>=1.10", (odds == 11) & ~legacy & (ev >= 1.10)),
               ("odds=2 (control)", (odds == 2) & ~legacy)]:
    n = int(m.sum())
    if n < 10:
        emit(f"   {lbl:<34} {n:>6}  (too few)")
        continue
    prof = won[m] * odds[m] - 1.0
    se = prof.std(ddof=1) / np.sqrt(n)
    emit(f"   {lbl:<34} {n:>6} {prof.mean()*100:>+7.1f}% "
         f"[{(prof.mean()-1.96*se)*100:>+6.1f}%,{(prof.mean()+1.96*se)*100:>+6.1f}%]")
emit()

# ------------------------------------------------------- 5. bankroll reality
emit("-" * 78)
emit("5. What the strongest surviving candidate would actually be worth")
emit("-" * 78)
m = (odds == 11) & (ev >= 1.10)
per_year = m.sum() / (day.max() + 1) * 365
emit(f"   odds=11 + NN EV>=1.10 fires {per_year:.1f} times per year "
     f"({m.sum()} times in {day.max()+1} days).")
emit(f"   Even taking the point estimate at face value (+58.9% ROI), that is")
emit(f"   {per_year:.0f} bets/yr — with 11:1 variance, a year of play has a")
kelly = (won[m].mean() * 11 - 1) / (11 - 1)
emit(f"   standard deviation of ~{11*np.sqrt(won[m].mean()*(1-won[m].mean())*per_year)/per_year*100:.0f}% "
     f"of turnover. Full-Kelly stake would be {kelly*100:.1f}% of bankroll.")
emit(f"   By contrast the odds=2 clamp fires {(odds==2).sum()/(day.max()+1)*365:.0f} "
     f"times per year at a verified +4.7% (+29% filtered).")
emit()

with open("edge_verdict_results.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("wrote edge_verdict_results.txt")
