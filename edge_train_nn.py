#!/usr/bin/env python3
"""Train the arena-winner NN with K-fold CV, producing honest out-of-fold (OOF)
win probabilities for every (arena, position) in the dataset.

Differences from train_nn.py / nn_winrate_check.py:
  * K-fold by DAY so every arena gets a prediction from a model that never saw it.
  * Early stopping on a validation split carved out of the TRAINING folds
    (the old scripts early-stopped on the test set, which leaks).
  * Seed-ensembled probabilities (average of several runs) for calibration.
  * Several feature variants, including one that sees the opening odds
    (so the NN can only "win" by adding information the odds maker lacks).

Usage: edge_train_nn.py <variant>   where variant in {base, ident, market}
Writes edge_oof_<variant>.npz
"""
import json, sys, os
import numpy as np
from collections import defaultdict

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
N_FOLDS = 5
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
HIDDEN = [128, 64]
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "300"))
PATIENCE = 30

# ---------------------------------------------------------------- data loading
with open("pirates.json") as f:
    raw = json.load(f)

course_names = list(raw["courses"].keys())
course_idx = {n: i for i, n in enumerate(course_names)}
cat_courses = defaultdict(set)
for cname, cats in raw["courses"].items():
    for cat in cats:
        cat_courses[cat].add(course_idx[cname])


class PirateInfo:
    def __init__(self, d):
        self.name = d["name"]
        self.strength = d["strength"]
        self.weight = d["weight"]
        self.fav = set()
        for c in d["favorites"]:
            self.fav |= cat_courses.get(c, set())
        self.alg = set()
        for c in d["allergies"]:
            self.alg |= cat_courses.get(c, set())


pirates_list = [PirateInfo(d) for d in raw["pirates"]]
pindex = {p.name: i for i, p in enumerate(pirates_list)}
N_PIRATES = len(pirates_list)

with open("historical_matches.json") as f:
    hist = json.load(f)

feat_rows = []      # [arena][pos][feature]
winners = []        # winning position per arena
meta_day = []
meta_legacy = []
meta_pid = []       # [arena][pos] pirate index
meta_odds = []      # [arena][pos] opening odds
meta_cur = []       # [arena][pos] current odds (0 if missing)
meta_nf = []
meta_na = []

for day_idx, day in enumerate(hist):
    for arena in day:
        food_ids = [course_idx[f] for f in arena["foods"] if f in course_idx]
        legacy = arena.get("legacy", False)
        winner_name = arena["winner"]

        odds = [p["odds"] for p in arena["pirates"]]
        implied = np.array([1.0 / o for o in odds])
        implied_norm = implied / implied.sum()

        feats, pids, nfs, nas, curs = [], [], [], [], []
        winner_pos = -1
        for pos, pd in enumerate(arena["pirates"]):
            p = pirates_list[pindex[pd["name"]]]
            if pd["name"] == winner_name:
                winner_pos = pos
            nf = na = 0
            for c in food_ids:
                if c in p.alg:
                    na += 1
                elif c in p.fav:
                    nf += 1

            f = [p.strength, p.weight, nf, na]
            if VARIANT in ("ident", "market"):
                onehot = [0.0] * N_PIRATES
                onehot[pindex[pd["name"]]] = 1.0
                f = f + onehot + [1.0 if legacy else 0.0]
            if VARIANT == "market":
                f = f + [implied[pos], implied_norm[pos], float(odds[pos])]
            feats.append(f)
            pids.append(pindex[pd["name"]])
            nfs.append(nf)
            nas.append(na)
            curs.append(pd.get("current_odds") or 0)

        feat_rows.append(feats)
        winners.append(winner_pos)
        meta_day.append(day_idx)
        meta_legacy.append(legacy)
        meta_pid.append(pids)
        meta_odds.append(odds)
        meta_cur.append(curs)
        meta_nf.append(nfs)
        meta_na.append(nas)

X = np.array(feat_rows, dtype=np.float32)
Y = np.array(winners, dtype=np.int64)
day = np.array(meta_day, dtype=np.int64)
legacy = np.array(meta_legacy, dtype=bool)
odds = np.array(meta_odds, dtype=np.int32)
cur = np.array(meta_cur, dtype=np.int32)
pid = np.array(meta_pid, dtype=np.int32)
nf = np.array(meta_nf, dtype=np.int32)
na = np.array(meta_na, dtype=np.int32)

n_arenas, _, n_feat = X.shape
print(f"[{VARIANT}] arenas={n_arenas} features={n_feat} days={day.max()+1}", flush=True)

# ------------------------------------------------------------------- training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


class SiameseScoringNet(nn.Module):
    def __init__(self, n_features, hidden_sizes):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.scorer = nn.Sequential(*layers)
        self.pos_bias = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        b = x.shape[0]
        scores = self.scorer(x.view(b * 4, -1)).view(b, 4)
        return scores + self.pos_bias.unsqueeze(0)


def eval_ll(model, Xt, Yt):
    model.eval()
    with torch.no_grad():
        lp = torch.log_softmax(-model(Xt), dim=1)
        return lp[torch.arange(len(Yt)), Yt].mean().item()


def probs_of(model, Xt):
    model.eval()
    with torch.no_grad():
        return torch.softmax(-model(Xt), dim=1).numpy()


# Fold assignment by day (deterministic hash so it is reproducible).
def day_hash(d):
    h = (d * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 31
    h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 29
    return h


fold_of_day = {d: day_hash(d) % N_FOLDS for d in range(day.max() + 1)}
fold = np.array([fold_of_day[d] for d in day])
# validation days carved out of the training folds (10%)
val_of_day = {d: (day_hash(d) // N_FOLDS) % 10 == 0 for d in range(day.max() + 1)}
is_val_day = np.array([val_of_day[d] for d in day])

oof = np.zeros((n_arenas, 4), dtype=np.float64)

for k in range(N_FOLDS):
    test_mask = fold == k
    trainable = ~test_mask
    val_mask = trainable & is_val_day
    fit_mask = trainable & ~is_val_day

    means = X[fit_mask].reshape(-1, n_feat).mean(axis=0)
    stds = X[fit_mask].reshape(-1, n_feat).std(axis=0)
    stds[stds == 0] = 1.0
    Xn = (X - means) / stds

    Xf = torch.tensor(Xn[fit_mask])
    Yf = torch.tensor(Y[fit_mask])
    Xv = torch.tensor(Xn[val_mask])
    Yv = torch.tensor(Y[val_mask])
    Xk = torch.tensor(Xn[test_mask])
    Yk = torch.tensor(Y[test_mask])

    acc = np.zeros((test_mask.sum(), 4))
    for seed in range(N_SEEDS):
        torch.manual_seed(1000 * k + seed)
        model = SiameseScoringNet(n_feat, HIDDEN)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        loader = DataLoader(TensorDataset(Xf, Yf), batch_size=1024, shuffle=True)

        best_val, wait, best_state = -999, 0, None
        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(-model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vll = eval_ll(model, Xv, Yv)
            sched.step(-vll)
            if vll > best_val:
                best_val, wait = vll, 0
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        acc += probs_of(model, Xk)
        print(f"[{VARIANT}] fold {k} seed {seed}: val={best_val:.5f} "
              f"heldout={eval_ll(model, Xk, Yk):.5f} (epoch {epoch})", flush=True)

    oof[test_mask] = acc / N_SEEDS

oof_ll = np.log(oof[np.arange(n_arenas), Y]).mean()
mod = ~legacy
print(f"[{VARIANT}] OOF LL all={oof_ll:.5f} "
      f"modern={np.log(oof[mod, Y[mod]]).mean():.5f} "
      f"legacy={np.log(oof[~mod, Y[~mod]]).mean():.5f}", flush=True)

np.savez_compressed(
    f"edge_oof_{VARIANT}.npz",
    oof=oof, Y=Y, day=day, legacy=legacy, odds=odds, cur=cur,
    pid=pid, nf=nf, na=na,
    pirate_names=np.array([p.name for p in pirates_list]),
)
print(f"[{VARIANT}] wrote edge_oof_{VARIANT}.npz", flush=True)
