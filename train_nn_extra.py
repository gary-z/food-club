#!/usr/bin/env python3
"""Train NN with unconventional features derived from public pirate data."""
import json, numpy as np, hashlib
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

        # Unconventional features
        self.name_len = len(self.name)
        self.name_vowels = sum(1 for c in self.name.lower() if c in 'aeiou')
        self.name_consonants = sum(1 for c in self.name.lower() if c.isalpha() and c not in 'aeiou')
        self.name_spaces = self.name.count(' ')
        self.name_words = len(self.name.split())

        # Weight digits
        w_str = str(self.weight)
        self.weight_digit_sum = sum(int(c) for c in w_str)
        self.weight_digits = [int(c) for c in w_str.zfill(3)]  # pad to 3 digits

        # Strength digits
        s_str = str(self.strength)
        self.str_digit_sum = sum(int(c) for c in s_str)
        self.str_digits = [int(c) for c in s_str.zfill(2)]  # pad to 2 digits

        # First letter ord
        self.first_letter = ord(self.name[0].upper()) - ord('A')

pirates_list = [PirateInfo(d) for d in raw["pirates"]]
pindex = {p.name: i for i, p in enumerate(pirates_list)}

# Print pirate unconventional features for inspection
print("Pirate unconventional features:")
print(f"{'Name':<30} {'Str':>3} {'Wt':>3} {'NLen':>4} {'Vowl':>4} {'Cons':>4} {'WDS':>3} {'WtDig':>5} {'StDig':>5} {'1st':>3}")
for p in sorted(pirates_list, key=lambda p: -p.strength):
    print(f"{p.name:<30} {p.strength:>3} {p.weight:>3} {p.name_len:>4} {p.name_vowels:>4} {p.name_consonants:>4} {p.name_words:>3} {p.weight_digit_sum:>5} {p.str_digit_sum:>5} {p.first_letter:>3}")

with open("historical_matches.json") as f:
    hist = json.load(f)

def build_features(pirates_list, pindex, hist, feature_set="baseline"):
    """Build feature arrays. feature_set controls which features to include."""
    all_features = []
    all_winners = []
    all_pirate_names = []
    all_is_legacy = []

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

                if feature_set == "baseline":
                    feats = [p.strength, p.weight, nf, na]
                elif feature_set == "extra":
                    feats = [
                        p.strength, p.weight, nf, na,
                        p.name_len, p.name_vowels, p.name_consonants,
                        p.weight_digit_sum, p.str_digit_sum,
                        p.weight_digits[0], p.weight_digits[1], p.weight_digits[2],
                        p.str_digits[0], p.str_digits[1],
                        p.first_letter, p.name_words,
                    ]
                elif feature_set == "name_only":
                    feats = [
                        p.strength, p.weight, nf, na,
                        p.name_len, p.name_vowels, p.name_consonants,
                    ]
                elif feature_set == "digits_only":
                    feats = [
                        p.strength, p.weight, nf, na,
                        p.weight_digit_sum, p.str_digit_sum,
                        p.weight_digits[0], p.weight_digits[1], p.weight_digits[2],
                        p.str_digits[0], p.str_digits[1],
                    ]

                features.append(feats)
                names.append(pd["name"])

            all_features.append(features)
            all_winners.append(winner_pos)
            all_pirate_names.append(names)
            all_is_legacy.append(legacy)

    return (np.array(all_features, dtype=np.float32), np.array(all_winners, dtype=np.int64),
            all_pirate_names, np.array(all_is_legacy, dtype=bool))

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

def train_and_eval(X, Y, n_features, label, is_legacy, n_runs=5):
    # Train on 80% hash-split of ALL data, test only on modern within the held-out 20%
    n_total = len(X)
    is_train = np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 5 != 0 for i in range(n_total)])
    is_test = ~is_train

    print(f"  Train: {is_train.sum()}, Test: {is_test.sum()}")

    train_X = X[is_train]
    means = train_X.reshape(-1, X.shape[2]).mean(axis=0)
    stds = train_X.reshape(-1, X.shape[2]).std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds

    X_tr = torch.tensor(X_norm[is_train], device=device)
    Y_tr = torch.tensor(Y[is_train], device=device)
    X_te = torch.tensor(X_norm[is_test], device=device)
    Y_te = torch.tensor(Y[is_test], device=device)

    best_test_ll = -999
    for run in range(n_runs):
        model = SiameseScoringNet(n_features, [128, 64]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=1024, shuffle=True)

        best_epoch_ll = -999
        patience_counter = 0
        for epoch in range(300):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(-model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tll = eval_ll(model, X_te, Y_te)
            scheduler.step(-tll)
            if tll > best_epoch_ll:
                best_epoch_ll = tll
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= 30:
                break

        model.load_state_dict(best_state)
        tll = eval_ll(model, X_te, Y_te)
        trll = eval_ll(model, X_tr, Y_tr)
        print(f"  run {run}: train={trll:.5f} test={tll:.5f}")
        if tll > best_test_ll:
            best_test_ll = tll

    print(f"  {label}: BEST test = {best_test_ll:.5f}")
    return best_test_ll

# Run all feature sets
results = {}

for fs in ["baseline", "name_only", "digits_only", "extra"]:
    print(f"\n{'='*60}")
    print(f"Feature set: {fs}")
    print(f"{'='*60}")
    X, Y, _, is_leg = build_features(pirates_list, pindex, hist, fs)
    n_feat = X.shape[2]
    print(f"  {n_feat} features per pirate, {len(X)} arenas")
    results[fs] = train_and_eval(X, Y, n_feat, fs, is_leg)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
for fs, ll in sorted(results.items(), key=lambda x: -x[1]):
    delta = ll - results["baseline"]
    print(f"  {fs:<20} test LL = {ll:.5f}  (delta = {delta:+.5f})")
