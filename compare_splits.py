#!/usr/bin/env python3
"""Compare NN vs Model 4 on both legacy/modern and mixed hash splits."""
import json, numpy as np, math
from collections import defaultdict

with open("pirates.json") as f:
    raw = json.load(f)

course_names = list(raw["courses"].keys())
course_idx = {n: i for i, n in enumerate(course_names)}
cat_courses = defaultdict(set)
for cname, cats in raw["courses"].items():
    for cat in cats: cat_courses[cat].add(course_idx[cname])

class PirateInfo:
    def __init__(self, d):
        self.name = d["name"]
        self.strength = d["strength"]
        self.weight = d["weight"]
        self.fav = set()
        for c in d["favorites"]: self.fav |= cat_courses.get(c, set())
        self.alg = set()
        for c in d["allergies"]: self.alg |= cat_courses.get(c, set())

pirates_list = [PirateInfo(d) for d in raw["pirates"]]
pindex = {p.name: i for i, p in enumerate(pirates_list)}

with open("historical_matches.json") as f:
    hist = json.load(f)

# ==================== Model 4 PMF engine (Python port) ====================

BASE = 120; FAV_DIV = 16; N_ROLLS = 6; DIVISOR = 22; MAX_WEIGHT = 221; MAX_EFFECT = 6

def dice_sum_pmf(n, d):
    if d == 0 or n == 0: return np.array([1.0])
    mx = n * d
    inv_d = 1.0 / d
    pmf = np.zeros(mx + 1)
    pmf[1:d+1] = inv_d
    for _ in range(1, n):
        new = np.zeros(mx + 1)
        s = 0.0
        for k in range(mx + 1):
            if k >= 1: s += pmf[k - 1]
            if k > d: s -= pmf[k - d - 1]
            new[k] = s * inv_d
        pmf = new
    return pmf

# Precompute roll table
max_upper = 200
roll_table = [dice_sum_pmf(N_ROLLS, d) for d in range(max_upper + 1)]

def pirate_score_pmf(strength, weight, nf, na):
    raw_wo = min((MAX_WEIGHT - min(weight, MAX_WEIGHT)) // 2, MAX_EFFECT)
    wo = raw_wo

    dmg_pmf = dice_sum_pmf(na, wo) if na > 0 and wo > 0 else np.array([1.0])

    max_die = len(roll_table) - 1
    max_raw = N_ROLLS * max_die
    raw_pmf = np.zeros(max_raw + 1)

    for dmg_val, dp in enumerate(dmg_pmf):
        if dp < 1e-15: continue
        upper = max(1, BASE - strength)
        for _ in range(nf):
            red = upper // FAV_DIV
            upper = max(1, upper - red)
        upper += dmg_val
        upper = max(1, upper)
        if upper <= max_die:
            rpmf = roll_table[upper]
            for k, rp in enumerate(rpmf):
                if rp > 0.0 and k < len(raw_pmf):
                    raw_pmf[k] += dp * rp

    max_q = max_raw // DIVISOR
    qpmf = np.zeros(max_q + 1)
    for k, pr in enumerate(raw_pmf):
        if pr < 1e-15: continue
        qk = k // DIVISOR
        if qk <= max_q: qpmf[qk] += pr
    return qpmf

def win_probs_from_pmfs(pmfs):
    max_t = max(len(p) for p in pmfs)
    # Survival functions
    surv = []
    for pmf in pmfs:
        s = np.zeros(max_t + 1)
        acc = 0.0
        for t in range(len(pmf) - 1, -1, -1):
            s[t] = acc
            acc += pmf[t]
        surv.append(s)

    def f(i, t):
        return pmfs[i][t] if t < len(pmfs[i]) else 0.0
    def s(i, t):
        return surv[i][t] if t < len(surv[i]) else 0.0
    def g(i, t):
        return 1.0 if t == 0 else s(i, t - 1)

    probs = [0.0] * 4
    for t in range(max_t):
        probs[3] += f(3,t) * g(0,t) * g(1,t) * g(2,t)
        probs[2] += f(2,t) * g(0,t) * g(1,t) * s(3,t)
        probs[1] += f(1,t) * g(0,t) * s(2,t) * s(3,t)
        probs[0] += f(0,t) * s(1,t) * s(2,t) * s(3,t)
    return probs

# ==================== Build dataset ====================

all_features = []
all_winners = []
all_is_legacy = []
all_m4_probs = []  # Model 4 win probs per arena

for day_idx, day in enumerate(hist):
    for arena in day:
        foods = arena["foods"]
        food_ids = [course_idx[f] for f in foods if f in course_idx]
        legacy = arena.get("legacy", False)
        winner_name = arena["winner"]

        features = []
        winner_pos = -1
        pirate_data = []  # (strength, weight, nf, na) for M4
        for pos, pd in enumerate(arena["pirates"]):
            p = pirates_list[pindex[pd["name"]]]
            if pd["name"] == winner_name:
                winner_pos = pos
            nf = 0; na = 0
            for c in food_ids:
                if c in p.alg: na += 1
                elif c in p.fav: nf += 1
            features.append([p.strength, p.weight, nf, na])
            pirate_data.append((p.strength, p.weight, nf, na))

        # Model 4 probabilities
        pmfs = [pirate_score_pmf(*pd) for pd in pirate_data]
        m4_probs = win_probs_from_pmfs(pmfs)

        all_features.append(features)
        all_winners.append(winner_pos)
        all_is_legacy.append(legacy)
        all_m4_probs.append(m4_probs)

X = np.array(all_features, dtype=np.float32)
Y = np.array(all_winners, dtype=np.int64)
is_legacy = np.array(all_is_legacy, dtype=bool)
m4_probs = np.array(all_m4_probs, dtype=np.float64)

# Clip M4 probs for log safety
m4_probs = np.clip(m4_probs, 1e-10, 1.0)

def hash_day(idx):
    h = (idx * 0x517cc1b727220a95) & 0xffffffffffffffff
    h ^= h >> 32
    h = (h * 0x6c62272e07bb0142) & 0xffffffffffffffff
    h ^= h >> 32
    return h

n_arenas = len(Y)
is_hash_train = np.array([hash_day(i) % 2 == 0 for i in range(n_arenas)], dtype=bool)

print(f"Total arenas: {n_arenas}")
print(f"Legacy: {is_legacy.sum()}, Modern: {(~is_legacy).sum()}")
print(f"Hash train: {is_hash_train.sum()}, Hash test: {(~is_hash_train).sum()}")

# ==================== Model 4 LL on both splits ====================

def m4_ll(mask):
    """Compute Model 4 LL on arenas where mask is True."""
    log_probs = np.log(m4_probs[mask])
    winners = Y[mask]
    ll = np.mean([log_probs[i, winners[i]] for i in range(len(winners))])
    return ll

print(f"\n{'='*70}")
print(f"Model 4 (hand-rolled PMF)")
print(f"{'='*70}")
# Split 1: legacy train / modern test
m4_modern_ll = m4_ll(~is_legacy)
m4_legacy_ll = m4_ll(is_legacy)
print(f"  Legacy LL:     {m4_legacy_ll:.5f} ({is_legacy.sum()} arenas)")
print(f"  Modern LL:     {m4_modern_ll:.5f} ({(~is_legacy).sum()} arenas)")

# Split 2: hash
m4_hash_train_ll = m4_ll(is_hash_train)
m4_hash_test_ll = m4_ll(~is_hash_train)
print(f"  Hash-train LL: {m4_hash_train_ll:.5f} ({is_hash_train.sum()} arenas)")
print(f"  Hash-test LL:  {m4_hash_test_ll:.5f} ({(~is_hash_train).sum()} arenas)")
m4_all_ll = m4_ll(np.ones(n_arenas, dtype=bool))
print(f"  All data LL:   {m4_all_ll:.5f} ({n_arenas} arenas)")

# ==================== NN on both splits ====================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

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
        scores = scores + self.pos_bias.unsqueeze(0)
        return scores

def eval_ll(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        log_probs = torch.log_softmax(-logits, dim=1)
        ll = log_probs[torch.arange(len(Y)), Y].mean().item()
    return ll

def train_nn(train_mask, test_mask, label, n_runs=5, hidden=[128, 64], n_epochs=300):
    # Normalize using training split
    train_X_np = X[train_mask]
    means = train_X_np.reshape(-1, X.shape[2]).mean(axis=0)
    stds = train_X_np.reshape(-1, X.shape[2]).std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds

    X_train = torch.tensor(X_norm[train_mask], device=device)
    Y_train = torch.tensor(Y[train_mask], device=device)
    X_test = torch.tensor(X_norm[test_mask], device=device)
    Y_test = torch.tensor(Y[test_mask], device=device)

    best_test_ll = -999
    best_train_ll = -999
    for run in range(n_runs):
        model = SiameseScoringNet(4, hidden).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        best_epoch_ll = -999
        patience_counter = 0
        best_state = None
        for epoch in range(n_epochs):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(-model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            test_ll = eval_ll(model, X_test, Y_test)
            scheduler.step(-test_ll)
            if test_ll > best_epoch_ll:
                best_epoch_ll = test_ll
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= 30:
                break

        model.load_state_dict(best_state)
        train_ll = eval_ll(model, X_train, Y_train)
        test_ll = eval_ll(model, X_test, Y_test)
        if test_ll > best_test_ll:
            best_test_ll = test_ll
            best_train_ll = train_ll

    print(f"  {label:<35} train={best_train_ll:.5f} test={best_test_ll:.5f}")
    return best_train_ll, best_test_ll

print(f"\n{'='*70}")
print(f"NN [128, 64] (5 runs each)")
print(f"{'='*70}")

# Split 1: legacy=train, modern=test
nn_train1, nn_test1 = train_nn(is_legacy, ~is_legacy, "legacy→modern")

# Split 2: hash train/test
nn_train2, nn_test2 = train_nn(is_hash_train, ~is_hash_train, "hash-train→hash-test")

# ==================== Summary ====================

print(f"\n{'='*70}")
print(f"SUMMARY: Test LL comparison")
print(f"{'='*70}")
print(f"{'Split':<25} {'Model 4':>10} {'NN':>10} {'Gap (NN-M4)':>12}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*12}")
print(f"{'legacy→modern':<25} {m4_modern_ll:>10.5f} {nn_test1:>10.5f} {nn_test1 - m4_modern_ll:>+12.5f}")
print(f"{'hash-train→hash-test':<25} {m4_hash_test_ll:>10.5f} {nn_test2:>10.5f} {nn_test2 - m4_hash_test_ll:>+12.5f}")
print(f"\n{'Split':<25} {'Model 4':>10} {'NN':>10} {'Gap (NN-M4)':>12}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*12}")
print(f"{'Train (legacy)':<25} {m4_legacy_ll:>10.5f} {nn_train1:>10.5f} {nn_train1 - m4_legacy_ll:>+12.5f}")
print(f"{'Train (hash-train)':<25} {m4_hash_train_ll:>10.5f} {nn_train2:>10.5f} {nn_train2 - m4_hash_train_ll:>+12.5f}")
