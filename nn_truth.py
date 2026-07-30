#!/usr/bin/env python3
"""Train the reference NN and cache out-of-fold win probabilities.

The NN (same architecture as train_nn.py / nn_winrate_check.py: a siamese
scorer over (strength, weight, nf, na) plus a learned position bias, softmaxed
over the four positions) is the best available estimate of the *real* win
probability, so it is used as the source of truth when testing hypotheses about
the odds maker.

Probabilities are produced by 5-fold cross-fitting: every arena is scored by a
model that never saw it, and early stopping uses a validation slice carved out
of the training folds only (never the scored fold).  That keeps p_nn free of
the in-sample optimism that would otherwise make the odds maker look noisier
than it is.

Writes nn_probs.npz:  p (n,4) float64, fold (n,) int8
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import fc_data

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "nn_probs.npz")


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
        batch_size = x.shape[0]
        flat = x.view(batch_size * 4, -1)
        scores = self.scorer(flat).view(batch_size, 4)
        return scores + self.pos_bias.unsqueeze(0)


def eval_ll(model, X, Y):
    model.eval()
    with torch.no_grad():
        lp = torch.log_softmax(-model(X), dim=1)
        return lp[torch.arange(len(Y)), Y].mean().item()


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(-model(X), dim=1).numpy().astype(np.float64)


def train_one(X_tr, Y_tr, X_va, Y_va, hidden, epochs, patience, seed):
    torch.manual_seed(seed)
    model = SiameseScoringNet(X_tr.shape[2], hidden)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=1024, shuffle=True)
    best_ll, best_state, stale = -999.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(-model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = eval_ll(model, X_va, Y_va)
        sched.step(-va)
        if va > best_ll:
            best_ll, stale = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)

    d = fc_data.load_arenas()
    X = d["feat"]
    Y = d["winner"].astype(np.int64)
    n = len(X)

    # deterministic fold assignment (md5 of the arena index, as in nn_winrate_check.py)
    fold = np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % args.folds
                     for i in range(n)], dtype=np.int8)

    p_out = np.zeros((n, 4), dtype=np.float64)

    for k in range(args.folds):
        te = fold == k
        tr_all = ~te
        # validation slice from the training folds only
        idx_tr = np.where(tr_all)[0]
        rng = np.random.default_rng(1234 + k)
        perm = rng.permutation(len(idx_tr))
        n_va = len(idx_tr) // 10
        va_idx = idx_tr[perm[:n_va]]
        tr_idx = idx_tr[perm[n_va:]]

        # normalisation from the training rows only
        means = X[tr_idx].reshape(-1, X.shape[2]).mean(axis=0)
        stds = X[tr_idx].reshape(-1, X.shape[2]).std(axis=0)
        stds[stds == 0] = 1.0
        Xn = (X - means) / stds

        X_tr = torch.tensor(Xn[tr_idx])
        Y_tr = torch.tensor(Y[tr_idx])
        X_va = torch.tensor(Xn[va_idx])
        Y_va = torch.tensor(Y[va_idx])
        X_te = torch.tensor(Xn[te])
        Y_te = torch.tensor(Y[te])

        best = (-999.0, None)
        for run in range(args.runs):
            model, va_ll = train_one(X_tr, Y_tr, X_va, Y_va, args.hidden,
                                     args.epochs, args.patience, seed=1000 * k + run)
            te_ll = eval_ll(model, X_te, Y_te)
            print(f"fold {k} run {run}: val={va_ll:.5f} heldout={te_ll:.5f}", flush=True)
            if va_ll > best[0]:
                best = (va_ll, model)
        model = best[1]
        p_out[te] = predict(model, X_te)
        print(f"fold {k}: held-out LL = {eval_ll(model, X_te, Y_te):.5f}", flush=True)

    ll = np.log(np.maximum(p_out[np.arange(n), Y], 1e-12)).mean()
    ll_leg = np.log(np.maximum(p_out[d["legacy"], :][np.arange(d["legacy"].sum()),
                                                     Y[d["legacy"]]], 1e-12)).mean()
    mod = ~d["legacy"]
    ll_mod = np.log(np.maximum(p_out[mod, :][np.arange(mod.sum()), Y[mod]], 1e-12)).mean()
    print(f"\nout-of-fold LL: all={ll:.5f}  legacy={ll_leg:.5f}  modern={ll_mod:.5f}")
    print(f"(uniform baseline = {np.log(0.25):.5f})")

    # calibration of the truth model itself
    print("\nreliability of p_nn (out-of-fold):")
    edges = np.array([0, .05, .10, .15, .20, .25, .30, .35, .40, .50, .60, 1.01])
    flat_p = p_out.ravel()
    win = np.zeros((n, 4), dtype=bool)
    win[np.arange(n), Y] = True
    flat_w = win.ravel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (flat_p >= lo) & (flat_p < hi)
        if m.sum() == 0:
            continue
        se = np.sqrt(flat_w[m].mean() * (1 - flat_w[m].mean()) / m.sum())
        print(f"  p in [{lo:.2f},{hi:.2f}): n={m.sum():>6} pred={flat_p[m].mean():.4f} "
              f"actual={flat_w[m].mean():.4f} +-{1.96*se:.4f}")

    np.savez_compressed(OUT, p=p_out, fold=fold)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
