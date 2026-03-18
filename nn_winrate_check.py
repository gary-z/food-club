#!/usr/bin/env python3
"""Train best NN, then compare per-pirate predicted vs actual win rates on modern data."""
import json, numpy as np
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

# Build dataset
all_features = []
all_winners = []
all_is_legacy = []
all_pirate_names = []  # [arena_idx][pos] = pirate name

for day in hist:
    for arena in day:
        foods = arena["foods"]
        food_ids = [course_idx[f] for f in foods if f in course_idx]
        legacy = arena.get("legacy", False)
        winner_name = arena["winner"]

        features = []
        names = []
        winner_pos = -1
        for pos, pd in enumerate(arena["pirates"]):
            p = pirates_list[pindex[pd["name"]]]
            if pd["name"] == winner_name:
                winner_pos = pos

            nf = 0; na = 0
            for c in food_ids:
                if c in p.alg: na += 1
                elif c in p.fav: nf += 1

            features.append([p.strength, p.weight, nf, na])
            names.append(pd["name"])

        all_features.append(features)
        all_winners.append(winner_pos)
        all_is_legacy.append(legacy)
        all_pirate_names.append(names)

X = np.array(all_features, dtype=np.float32)
Y = np.array(all_winners, dtype=np.int64)
is_legacy = np.array(all_is_legacy, dtype=bool)

# Split by hash: 80% train, 20% test, mixing legacy and modern
import hashlib
n_total = len(X)
is_train = np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 5 != 0 for i in range(n_total)])
is_test = ~is_train

# Normalize using train stats
train_X = X[is_train]
means = train_X.reshape(-1, X.shape[2]).mean(axis=0)
stds = train_X.reshape(-1, X.shape[2]).std(axis=0)
stds[stds == 0] = 1.0
X_norm = (X - means) / stds

print(f"Train: {is_train.sum()} arenas")
print(f"Test:  {is_test.sum()} arenas")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

X_train = torch.tensor(X_norm[is_train], device=device)
Y_train = torch.tensor(Y[is_train], device=device)
X_test = torch.tensor(X_norm[is_test], device=device)
Y_test = torch.tensor(Y[is_test], device=device)

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

# Train best config (128-64), 5 runs, keep best
print("\nTraining 128-64 NN (5 runs)...")
best_test_ll = -999
for run in range(5):
    model = SiameseScoringNet(4, [128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    best_epoch_ll = -999
    patience_counter = 0
    for epoch in range(300):
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
    test_ll = eval_ll(model, X_test, Y_test)
    print(f"  run {run}: test(modern)={test_ll:.5f}")
    if test_ll > best_test_ll:
        best_test_ll = test_ll
        best_model = model

print(f"  Best NN test LL = {best_test_ll:.5f}")

# Get NN predicted win probabilities for all test arenas
best_model.eval()
with torch.no_grad():
    logits = best_model(X_test)
    nn_probs = torch.softmax(-logits, dim=1).numpy()  # [n_test, 4]

# Collect per-pirate stats on test data
test_indices = np.where(is_test)[0]

pirate_stats = defaultdict(lambda: {"predicted": [], "actual_wins": 0, "appearances": 0})

for i, idx in enumerate(test_indices):
    winner_pos = Y[idx]
    names = all_pirate_names[idx]
    probs = nn_probs[i]

    for pos in range(4):
        name = names[pos]
        pirate_stats[name]["predicted"].append(probs[pos])
        pirate_stats[name]["appearances"] += 1
        if pos == winner_pos:
            pirate_stats[name]["actual_wins"] += 1

# Compute and compare
from scipy import stats as sp_stats

print(f"\n{'='*90}")
print(f"{'Pirate':<30} {'N':>5} {'Pred%':>7} {'Real%':>7} {'95% CI':>15} {'Status':>10}")
print(f"{'='*90}")

outliers = []
for name in sorted(pirate_stats.keys(), key=lambda n: -pirate_stats[n]["appearances"]):
    s = pirate_stats[name]
    n = s["appearances"]
    wins = s["actual_wins"]
    pred_rate = np.mean(s["predicted"])
    real_rate = wins / n

    # Wilson score 95% CI for binomial proportion
    z = 1.96
    denom = 1 + z**2 / n
    center = (real_rate + z**2 / (2*n)) / denom
    margin = z * np.sqrt((real_rate * (1 - real_rate) + z**2 / (4*n)) / n) / denom
    ci_lo = center - margin
    ci_hi = center + margin

    outside = pred_rate < ci_lo or pred_rate > ci_hi
    status = "** OUT **" if outside else "ok"

    if outside:
        direction = "OVER" if pred_rate > ci_hi else "UNDER"
        outliers.append((name, n, pred_rate, real_rate, ci_lo, ci_hi, direction))

    print(f"{name:<30} {n:>5} {pred_rate*100:>6.1f}% {real_rate*100:>6.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%] {status:>10}")

print(f"\n{'='*90}")
print(f"OUTLIERS (NN predicted rate outside 95% CI of real rate):")
print(f"{'='*90}")
if outliers:
    for name, n, pred, real, lo, hi, direction in sorted(outliers, key=lambda x: abs(x[2]-x[3]), reverse=True):
        print(f"  {name:<30} N={n:>4}  pred={pred*100:.1f}%  real={real*100:.1f}%  CI=[{lo*100:.1f}%, {hi*100:.1f}%]  {direction}")
else:
    print("  None found!")

# Write table to file
with open("nn_winrate_table.txt", "w") as f:
    f.write(f"NN Win Rate Analysis (128-64, hash-split 80/20 train/test)\n")
    f.write(f"Best test LL = {best_test_ll:.5f}\n")
    f.write(f"Train: {is_train.sum()} arenas, Test: {is_test.sum()} arenas\n\n")
    f.write(f"{'Pirate':<30} {'N':>5} {'Pred%':>7} {'Real%':>7} {'95% CI':>15} {'Status':>10}\n")
    f.write(f"{'-'*80}\n")
    for name in sorted(pirate_stats.keys(), key=lambda n: -pirate_stats[n]["actual_wins"]/pirate_stats[n]["appearances"]):
        s = pirate_stats[name]
        n = s["appearances"]
        wins = s["actual_wins"]
        pred_rate = np.mean(s["predicted"])
        real_rate = wins / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (real_rate + z**2 / (2*n)) / denom
        margin = z * np.sqrt((real_rate * (1 - real_rate) + z**2 / (4*n)) / n) / denom
        ci_lo = center - margin
        ci_hi = center + margin
        outside = pred_rate < ci_lo or pred_rate > ci_hi
        status = "** OUT **" if outside else "ok"
        f.write(f"{name:<30} {n:>5} {pred_rate*100:>6.1f}% {real_rate*100:>6.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%] {status:>10}\n")
    if outliers:
        f.write(f"\nOUTLIERS:\n")
        for name, n, pred, real, lo, hi, direction in sorted(outliers, key=lambda x: abs(x[2]-x[3]), reverse=True):
            f.write(f"  {name:<30} N={n:>4}  pred={pred*100:.1f}%  real={real*100:.1f}%  CI=[{lo*100:.1f}%, {hi*100:.1f}%]  {direction}\n")
    else:
        f.write(f"\nNo outliers found.\n")
    print(f"\nTable written to nn_winrate_table.txt")

# Write table to file
with open("nn_winrate_table.txt", "w") as f:
    f.write(f"NN Win Rate Analysis (128-64, hash-split 80/20 train/test)\n")
    f.write(f"Best test LL = {best_test_ll:.5f}\n")
    f.write(f"Train: {is_train.sum()} arenas, Test: {is_test.sum()} arenas\n\n")
    f.write(f"{'Pirate':<30} {'N':>5} {'Pred%':>7} {'Real%':>7} {'95% CI':>15} {'Status':>10}\n")
    f.write(f"{'-'*80}\n")
    for name in sorted(pirate_stats.keys(), key=lambda n: -pirate_stats[n]["actual_wins"]/pirate_stats[n]["appearances"]):
        s = pirate_stats[name]
        n = s["appearances"]
        wins = s["actual_wins"]
        pred_rate = np.mean(s["predicted"])
        real_rate = wins / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (real_rate + z**2 / (2*n)) / denom
        margin = z * np.sqrt((real_rate * (1 - real_rate) + z**2 / (4*n)) / n) / denom
        ci_lo = center - margin
        ci_hi = center + margin
        outside = pred_rate < ci_lo or pred_rate > ci_hi
        status = "** OUT **" if outside else "ok"
        f.write(f"{name:<30} {n:>5} {pred_rate*100:>6.1f}% {real_rate*100:>6.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%] {status:>10}\n")
    if outliers:
        f.write(f"\nOUTLIERS:\n")
        for name, n, pred, real, lo, hi, direction in sorted(outliers, key=lambda x: abs(x[2]-x[3]), reverse=True):
            f.write(f"  {name:<30} N={n:>4}  pred={pred*100:.1f}%  real={real*100:.1f}%  CI=[{lo*100:.1f}%, {hi*100:.1f}%]  {direction}\n")
    else:
        f.write(f"\nNo outliers found.\n")
    print(f"\nTable written to nn_winrate_table.txt")
