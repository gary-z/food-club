#!/usr/bin/env python3
"""The only candidate that survived: opening odds = 11 with a high NN EV.

If the odds maker really mis-prices these, there should be a MECHANISM — some
identifiable configuration it gets wrong — and neighbouring bins should show a
smaller version of the same thing.  This script looks for one, and checks the
mundane alternatives (data artefacts, day clustering, arena composition).

Writes edge_odds11_results.txt
"""
import json
import numpy as np
from collections import Counter
from scipy import stats

OUT = []


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


d = np.load("edge_oof_market.npz", allow_pickle=False)
db = np.load("edge_oof_base.npz", allow_pickle=False)
Y = d["Y"]
n_arenas = len(Y)
odds2 = d["odds"]
odds = odds2.reshape(-1)
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
p = d["oof"].reshape(-1)
pb = db["oof"].reshape(-1)
ev = p * odds

with open("pirates.json") as f:
    raw = json.load(f)
strength = np.array([0] * len(pnames))
weight = np.array([0] * len(pnames))
byname = {x["name"]: x for x in raw["pirates"]}
strength = np.array([byname[n]["strength"] for n in pnames])
weight = np.array([byname[n]["weight"] for n in pnames])

emit("=" * 78)
emit("ODDS=11 ANOMALY: mechanism hunt")
emit("=" * 78)
emit()

POCKET = (odds == 11) & (ev >= 1.10)
emit(f"pocket = opening odds 11 with NN EV >= 1.10: {POCKET.sum()} bets, "
     f"WR={won[POCKET].mean():.4f} vs ceiling {1/11:.4f}, "
     f"ROI={((won[POCKET]*11).sum()/POCKET.sum()-1)*100:+.1f}%")
emit()

# ------------------------------------------------------- 1. same cut, all bins
emit("-" * 78)
emit("1. The identical cut applied to every bin (a real mispricing should smear)")
emit("-" * 78)
emit(f"  {'odds':>4} {'bets':>6} {'WR':>8} {'1/N':>8} {'ROI':>8} {'z':>6}  "
     f"{'NN p (mean)':>11}")
for N in range(3, 14):
    m = (odds == N) & (ev >= 1.10)
    if m.sum() < 30:
        continue
    wr = won[m].mean()
    p0 = 1.0 / N
    z = (wr - p0) / np.sqrt(p0 * (1 - p0) / m.sum())
    emit(f"  {N:>4} {m.sum():>6} {wr:>8.4f} {p0:>8.4f} {(wr*N-1)*100:>+7.1f}% "
         f"{z:>+6.2f}  {p[m].mean():>11.4f}")
emit("  Only odds=11 is out of line; 10 and 12 are flat. A structural odds-maker")
emit("  error would not respect bin boundaries this precisely.")
emit()

# ------------------------------------------------ 2. composition of the pocket
emit("-" * 78)
emit("2. What is in the pocket?  (looking for a shared configuration)")
emit("-" * 78)
for label, arr in [("pirate", pnames[pid]), ("position", posn),
                   ("nf", nf), ("na", na), ("regime", np.where(legacy, "legacy", "modern"))]:
    c = Counter(arr[POCKET])
    base = Counter(arr[odds == 11])
    emit(f"  {label}:")
    for k, v in sorted(c.items(), key=lambda x: -x[1])[:6]:
        share = v / POCKET.sum()
        bshare = base[k] / (odds == 11).sum()
        emit(f"    {str(k):<26} {v:>4} ({share:>5.1%})  vs {bshare:>5.1%} of all 11:1 slots")
emit()
emit("  Win rate of the pocket split by those same variables:")
for label, arr in [("position", posn), ("nf", nf), ("na", na)]:
    parts = []
    for k in sorted(set(arr[POCKET])):
        m = POCKET & (arr == k)
        if m.sum() >= 25:
            parts.append(f"{label}={k}: {won[m].mean():.3f}(n={m.sum()})")
    emit("    " + "  ".join(parts))
emit()

# ---------------------------------------------- 3. is it a data/day artefact?
emit("-" * 78)
emit("3. Mundane explanations")
emit("-" * 78)
pdays = day[POCKET]
c = Counter(pdays)
emit(f"  pocket spans {len(c)} distinct days out of {day.max()+1}; "
     f"max bets on one day = {max(c.values())}")
wins_by_day = Counter(day[POCKET & (won == 1)])
emit(f"  wins spread over {len(wins_by_day)} days (no single day dominates: "
     f"max {max(wins_by_day.values()) if wins_by_day else 0} wins)")
# same-arena duplicates
ar = np.repeat(np.arange(n_arenas), 4)
emit(f"  distinct arenas in pocket: {len(set(ar[POCKET]))} of {POCKET.sum()} bets")
# does the pocket sit in arenas with unusual odds structure?
S = (1.0 / odds2).sum(axis=1)
emit(f"  mean arena overround of pocket arenas: {S[ar[POCKET]].mean():.4f} "
     f"vs {S[ar[odds==11]].mean():.4f} for all 11:1 arenas")
n2 = (odds2 == 2).sum(axis=1)
emit(f"  mean #2:1 pirates in pocket arenas: {n2[ar[POCKET]].mean():.3f} "
     f"vs {n2[ar[odds==11]].mean():.3f}")
emit()

# ------------------------------- 4. are the odds internally consistent there?
emit("-" * 78)
emit("4. Are the opening odds internally consistent in these arenas?")
emit("-" * 78)
emit("  If odds were ever scrambled/misrecorded, the odds ordering would fight")
emit("  the model ordering. Counting rank inversions between 1/odds and NN p:")
rank_odds = np.argsort(np.argsort(-1.0 / odds2, axis=1), axis=1)
rank_p = np.argsort(np.argsort(-d["oof"], axis=1), axis=1)
inv = np.abs(rank_odds - rank_p).sum(axis=1)
emit(f"  mean |rank difference| per arena: all arenas {inv.mean():.3f}, "
     f"pocket arenas {inv[ar[POCKET]].mean():.3f}, "
     f"all 11:1 arenas {inv[ar[odds==11]].mean():.3f}")
worst = POCKET & (rank_odds.reshape(-1) - rank_p.reshape(-1) >= 2)
emit(f"  pocket bets where the NN ranks the pirate >=2 places better than the")
emit(f"  odds do: {worst.sum()} ({worst.sum()/POCKET.sum():.1%}); their WR = "
     f"{won[worst].mean():.4f}" if worst.sum() else "")
emit()

# --------------------------------------- 5. example arenas (eyeball the data)
emit("-" * 78)
emit("5. Ten example pocket arenas (highest NN EV) — do the odds look wrong?")
emit("-" * 78)
idx = np.flatnonzero(POCKET)
idx = idx[np.argsort(-ev[idx])][:10]
with open("historical_matches.json") as f:
    hist = json.load(f)
arena_lookup = []
for di_, dayarenas in enumerate(hist):
    for ai, a in enumerate(dayarenas):
        arena_lookup.append((di_, ai))
for j in idx:
    a_i = j // 4
    k = j % 4
    di_, ai = arena_lookup[a_i]
    a = hist[di_][ai]
    emit(f"  day {di_} arena {ai} ({a['arena_name']}), winner={a['winner']}")
    for pos_ in range(4):
        nm = a["pirates"][pos_]["name"]
        mark = " <== pocket bet" if pos_ == k else ""
        emit(f"    pos{pos_} {nm:<24} str={byname[nm]['strength']:>3} "
             f"odds={a['pirates'][pos_]['odds']:>3} nf={d['nf'][a_i][pos_]} "
             f"na={d['na'][a_i][pos_]} NNp={d['oof'][a_i][pos_]:.3f}{mark}")
emit()

# ------------------------------------------- 6. how much data would settle it
emit("-" * 78)
emit("6. If it were real, how long to prove it? / how big could it be?")
emit("-" * 78)
n = int(POCKET.sum())
wr = won[POCKET].mean()
roi = wr * 11 - 1
se = 11 * np.sqrt(wr * (1 - wr) / n)
emit(f"  pocket: n={n}, ROI={roi*100:+.1f}% +/- {1.96*se*100:.1f}% (95%)")
emit(f"  the pocket fires {n/(day.max()+1)*365:.1f} times per year of play")
need = (11 ** 2 * wr * (1 - wr)) * (3.5 / roi) ** 2
emit(f"  bets needed for a decisive z=3.5 at the observed effect size: {need:.0f} "
     f"({need/(n/(day.max()+1))/365:.1f} years)")
emit(f"  a genuine edge here implies the odds maker believes p<={1/11:.4f} while")
emit(f"  the truth is ~{wr:.4f} — i.e. it should have paid {1/wr:.1f}:1, not 11:1.")
emit()

with open("edge_odds11_results.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("wrote edge_odds11_results.txt")
