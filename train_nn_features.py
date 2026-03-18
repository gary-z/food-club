#!/usr/bin/env python3
"""Test unconventional features as 5th input to the NN."""
import json, numpy as np, zlib
from collections import defaultdict

with open("pirates.json") as f:
    raw = json.load(f)

course_names = list(raw["courses"].keys())
course_idx = {n: i for i, n in enumerate(course_names)}
cat_courses = defaultdict(set)
for cname, cats in raw["courses"].items():
    for cat in cats: cat_courses[cat].add(course_idx[cname])

class PirateInfo:
    def __init__(self, d, idx):
        self.name = d["name"]
        self.strength = d["strength"]
        self.weight = d["weight"]
        self.idx = idx
        self.fav = set()
        self.fav_categories = d["favorites"]
        for c in d["favorites"]: self.fav |= cat_courses.get(c, set())
        self.alg = set()
        self.allergy_categories = d["allergies"]
        for c in d["allergies"]: self.alg |= cat_courses.get(c, set())
        # Unconventional features
        self.name_len = len(d["name"])
        self.word_count = len(d["name"].split())
        self.first_char_ascii = ord(d["name"][0])
        self.crc32 = zlib.crc32(d["name"].encode()) & 0xffffffff
        self.n_fav_cats = len(d["favorites"])
        self.n_alg_cats = len(d["allergies"])
        # str+wt derived
        self.str_plus_wt = d["strength"] + d["weight"]
        self.str_times_wt = d["strength"] * d["weight"]
        # digit sum of strength
        self.str_digit_sum = sum(int(c) for c in str(d["strength"]))
        self.wt_digit_sum = sum(int(c) for c in str(d["weight"]))
        # Neopets pirate IDs might be sequential - use index as proxy
        self.array_idx = idx

pirates_list = [PirateInfo(d, i) for i, d in enumerate(raw["pirates"])]
pindex = {p.name: i for i, p in enumerate(pirates_list)}

# Print pirate unconventional features for inspection
print("Pirate unconventional features:")
print(f"{'Name':<28} str  wt  nlen wc  asc   crc32     nfc nac s+w   s*w   sds wds idx")
for p in pirates_list:
    print(f"{p.name:<28} {p.strength:3} {p.weight:3} {p.name_len:3}  {p.word_count}  {p.first_char_ascii:3}  {p.crc32:10}  {p.n_fav_cats}   {p.n_alg_cats}  {p.str_plus_wt:4} {p.str_times_wt:5} {p.str_digit_sum:3} {p.wt_digit_sum:3} {p.array_idx:3}")

with open("historical_matches.json") as f:
    hist = json.load(f)

# Test drafting-order features
# For each pirate: count favs/allergies not shared with higher-indexed pirates
configs = [
    "baseline (str,wt,nf,na)",
    "+excl_fav_hi",            # nf not also fav of higher-index pirate
    "+excl_alg_hi",            # na not also fav of higher-index pirate
    "+both_excl_hi",           # both above
    "+excl_fav_lo",            # nf not also fav of lower-index pirate
    "+excl_alg_lo",            # na not also fav of lower-index pirate
    "+both_excl_lo",           # both above
    "+shared_fav",             # nf that ARE shared with any other pirate
    "+shared_alg",             # na that ARE shared with any other pirate
]

def hash_day(idx):
    h = idx * 0x517cc1b727220a95 & 0xffffffffffffffff
    h ^= h >> 32
    h = h * 0x6c62272e07bb0142 & 0xffffffffffffffff
    h ^= h >> 32
    return h

# Build all datasets at once
all_data = {}
for fname in configs:
    all_data[fname] = {"features": [], "winners": [], "is_train": []}

for day_idx, day in enumerate(hist):
    is_train = (hash_day(day_idx) % 2 == 0)
    for arena in day:
        foods = arena["foods"]
        food_ids = [course_idx[f] for f in foods if f in course_idx]
        winner_name = arena["winner"]

        # First pass: compute per-pirate fav/alg food sets
        arena_pirates = []
        arena_fav_foods = []  # list of sets: which food_ids are fav (not alg) for each pirate
        arena_alg_foods = []  # list of sets: which food_ids are alg for each pirate
        for pd in arena["pirates"]:
            p = pirates_list[pindex[pd["name"]]]
            arena_pirates.append(p)
            fav_f = set()
            alg_f = set()
            for c in food_ids:
                if c in p.alg:
                    alg_f.add(c)
                elif c in p.fav:
                    fav_f.add(c)
            arena_fav_foods.append(fav_f)
            arena_alg_foods.append(alg_f)

        per_config_features = {fname: [] for fname in configs}
        winner_pos = -1
        for pos, pd in enumerate(arena["pirates"]):
            if pd["name"] == winner_name:
                winner_pos = pos

            nf = len(arena_fav_foods[pos])
            na = len(arena_alg_foods[pos])

            # Exclusive favs: not also fav for any higher-index pirate
            excl_fav_hi = sum(1 for f in arena_fav_foods[pos]
                              if not any(f in arena_fav_foods[j] for j in range(pos+1, 4)))
            # Exclusive alg: not also fav for any higher-index pirate
            excl_alg_hi = sum(1 for f in arena_alg_foods[pos]
                              if not any(f in arena_fav_foods[j] for j in range(pos+1, 4)))
            # Same but relative to lower-index pirates
            excl_fav_lo = sum(1 for f in arena_fav_foods[pos]
                              if not any(f in arena_fav_foods[j] for j in range(0, pos)))
            excl_alg_lo = sum(1 for f in arena_alg_foods[pos]
                              if not any(f in arena_fav_foods[j] for j in range(0, pos)))
            # Shared with ANY other pirate
            shared_fav = sum(1 for f in arena_fav_foods[pos]
                             if any(f in arena_fav_foods[j] for j in range(4) if j != pos))
            shared_alg = sum(1 for f in arena_alg_foods[pos]
                             if any(f in arena_alg_foods[j] for j in range(4) if j != pos))

            p = arena_pirates[pos]
            base = [p.strength, p.weight, nf, na]

            per_config_features["baseline (str,wt,nf,na)"].append(base)
            per_config_features["+excl_fav_hi"].append(base + [excl_fav_hi])
            per_config_features["+excl_alg_hi"].append(base + [excl_alg_hi])
            per_config_features["+both_excl_hi"].append(base + [excl_fav_hi, excl_alg_hi])
            per_config_features["+excl_fav_lo"].append(base + [excl_fav_lo])
            per_config_features["+excl_alg_lo"].append(base + [excl_alg_lo])
            per_config_features["+both_excl_lo"].append(base + [excl_fav_lo, excl_alg_lo])
            per_config_features["+shared_fav"].append(base + [shared_fav])
            per_config_features["+shared_alg"].append(base + [shared_alg])

        for fname in configs:
            all_data[fname]["features"].append(per_config_features[fname])
            all_data[fname]["winners"].append(winner_pos)
            all_data[fname]["is_train"].append(is_train)

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

hidden = [128, 64]
n_runs = 5
n_epochs = 300

print(f"\n{'='*70}")
print(f"Testing extra features with {hidden} network, {n_runs} runs each")
print(f"{'='*70}")

results = {}
for fname in configs:
    X = np.array(all_data[fname]["features"], dtype=np.float32)
    Y = np.array(all_data[fname]["winners"], dtype=np.int64)
    is_train = np.array(all_data[fname]["is_train"], dtype=bool)
    if fname == "baseline (str,wt,nf,na)":
        print(f"  Train: {is_train.sum()} arenas, Test: {(~is_train).sum()} arenas (mixed hash split)")

    # Normalize using training split stats
    train_X = X[is_train]
    means = train_X.reshape(-1, X.shape[2]).mean(axis=0)
    stds = train_X.reshape(-1, X.shape[2]).std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds

    X_train = torch.tensor(X_norm[is_train], device=device)
    Y_train = torch.tensor(Y[is_train], device=device)
    X_test = torch.tensor(X_norm[~is_train], device=device)
    Y_test = torch.tensor(Y[~is_train], device=device)

    n_feats = X.shape[2]
    best_test_ll = -999
    best_train_ll = -999

    for run in range(n_runs):
        model = SiameseScoringNet(n_feats, hidden).to(device)
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

    delta = best_test_ll - results.get("baseline (str,wt,nf,na)", (-999, -999))[1] if "baseline (str,wt,nf,na)" in results else 0
    results[fname] = (best_train_ll, best_test_ll)
    print(f"  {fname:<30} feats={n_feats} train={best_train_ll:.5f} test={best_test_ll:.5f}")

# Summary sorted by test LL
print(f"\n{'='*70}")
print(f"Summary (sorted by test LL)")
print(f"{'='*70}")
baseline_ll = results["baseline (str,wt,nf,na)"][1]
for fname, (tr, te) in sorted(results.items(), key=lambda x: -x[1][1]):
    delta = te - baseline_ll
    print(f"  {fname:<30} test={te:.5f} (delta={delta:+.5f})")

print(f"\nReference: Model 4 hand-rolled on modern: LL/arena = -1.06314")
