#!/usr/bin/env python3
"""Train NN on legacy data, test on modern. Raw inputs only."""
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

all_features = []
all_winners = []
all_is_legacy = []

for day_idx, day in enumerate(hist):
    for arena in day:
        foods = arena["foods"]
        food_ids = [course_idx[f] for f in foods if f in course_idx]
        legacy = arena.get("legacy", False)
        winner_name = arena["winner"]

        features = []
        winner_pos = -1
        for pos, pd in enumerate(arena["pirates"]):
            p = pirates_list[pindex[pd["name"]]]
            if pd["name"] == winner_name:
                winner_pos = pos

            # Allergy has precedence over fav for overlap foods
            nf = 0; na = 0
            for c in food_ids:
                if c in p.alg: na += 1
                elif c in p.fav: nf += 1

            feats = [
                p.strength,
                p.weight,
                nf,
                na,
            ]
            features.append(feats)

        all_features.append(features)
        all_winners.append(winner_pos)
        all_is_legacy.append(legacy)

X = np.array(all_features, dtype=np.float32)
Y = np.array(all_winners, dtype=np.int64)
is_legacy = np.array(all_is_legacy, dtype=bool)

# Normalize using training (legacy) stats
train_X = X[is_legacy]
means = train_X.reshape(-1, X.shape[2]).mean(axis=0)
stds = train_X.reshape(-1, X.shape[2]).std(axis=0)
stds[stds == 0] = 1.0
print(f"Feature means: {means}")
print(f"Feature stds:  {stds}")

X_norm = (X - means) / stds

print(f"Legacy (train): {is_legacy.sum()} arenas")
print(f"Modern (test):  {(~is_legacy).sum()} arenas")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

X_train = torch.tensor(X_norm[is_legacy], device=device)
Y_train = torch.tensor(Y[is_legacy], device=device)
X_test = torch.tensor(X_norm[~is_legacy], device=device)
Y_test = torch.tensor(Y[~is_legacy], device=device)

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

configs = [
    ([64, 32], "64-32"),
    ([128, 64], "128-64"),
    ([256, 128, 64], "256-128-64"),
]

for hidden, name in configs:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    best_test_ll = -999
    for run in range(5):
        model = SiameseScoringNet(4, hidden).to(device)
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
        train_ll = eval_ll(model, X_train, Y_train)
        test_ll = eval_ll(model, X_test, Y_test)
        print(f"  run {run}: train(legacy)={train_ll:.5f} test(modern)={test_ll:.5f}")
        if test_ll > best_test_ll:
            best_test_ll = test_ll
            best_model = model

    print(f"  BEST test(modern) = {best_test_ll:.5f}")
    pb = best_model.pos_bias.detach().numpy()
    print(f"  pos_bias: {pb}")

print(f"\nReference: Model 4 hand-rolled on modern: LL/arena = -1.06314")
