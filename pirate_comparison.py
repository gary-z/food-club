#!/usr/bin/env python3
"""Per-pirate win rate comparison: historical vs NN vs hand-rolled Model 1."""
import json, math, numpy as np
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
        self.wo = min((221 - self.weight) // 2, 7) if self.weight < 221 else 0
        self.fav = set()
        for c in d["favorites"]: self.fav |= cat_courses.get(c, set())
        self.alg = set()
        for c in d["allergies"]: self.alg |= cat_courses.get(c, set())

pirates_list = [PirateInfo(d) for d in raw["pirates"]]
pindex = {p.name: i for i, p in enumerate(pirates_list)}

with open("historical_matches.json") as f:
    hist = json.load(f)

# ── Model 1 exact PMF ──
def dice_sum_pmf(n, d):
    if d == 0 or n == 0:
        return np.array([1.0])
    mx = n * d
    inv_d = 1.0 / d
    pmf = np.zeros(mx + 1)
    pmf[1:d+1] = inv_d
    for _ in range(1, n):
        new = np.zeros(mx + 1)
        s = 0.0
        for k in range(mx + 1):
            if k >= 1: s += pmf[k-1]
            if k > d: s -= pmf[k - d - 1]
            new[k] = s * inv_d
        pmf = new
    return pmf

# Precompute roll tables for Model 1
MAX_UPPER = 122
roll_table = [dice_sum_pmf(4, d) for d in range(MAX_UPPER + 1)]

def model1_score_pmf(strength, weight, nf, na):
    raw_wo = (221 - min(weight, 221)) // 2
    wo = min(raw_wo, 7)

    dmg_pmf = dice_sum_pmf(na, wo) if (na > 0 and wo > 0) else np.array([1.0])

    max_raw = 4 * MAX_UPPER
    max_q = max_raw // 14
    qpmf = np.zeros(max_q + 1)

    for dmg_val, dp in enumerate(dmg_pmf):
        if dp < 1e-15: continue
        eff_str = max(0, strength - dmg_val)
        raw_upper = max(1, 112 - eff_str)
        red = raw_upper // 15
        upper = max(1, raw_upper - nf * red)

        if upper <= MAX_UPPER:
            rpmf = roll_table[upper]
            for k, rp in enumerate(rpmf):
                if rp > 0:
                    qk = k // 14
                    if qk <= max_q:
                        qpmf[qk] += dp * rp
    return qpmf

def win_probs_later_wins(pmfs):
    max_t = max(len(p) for p in pmfs)
    # survival functions
    surv = []
    for pm in pmfs:
        s = np.zeros(max_t + 1)
        acc = 0.0
        for t in range(len(pm)-1, -1, -1):
            s[t] = acc
            acc += pm[t]
        surv.append(s)

    def f(i, t): return pmfs[i][t] if t < len(pmfs[i]) else 0.0
    def s(i, t): return surv[i][t] if t < len(surv[i]) else 0.0
    def g(i, t): return 1.0 if t == 0 else s(i, t-1)

    probs = np.zeros(4)
    for t in range(max_t):
        probs[3] += f(3,t) * g(0,t) * g(1,t) * g(2,t)
        probs[2] += f(2,t) * g(0,t) * g(1,t) * s(3,t)
        probs[1] += f(1,t) * g(0,t) * s(2,t) * s(3,t)
        probs[0] += f(0,t) * s(1,t) * s(2,t) * s(3,t)
    return probs

# ── Build data ──
print("Building arena data...")
arenas = []  # list of (pirate_names, pirate_features, nf_na_list, winner_pos, is_legacy)

for day_idx, day in enumerate(hist):
    for arena in day:
        foods = arena["foods"]
        food_ids = [course_idx[f] for f in foods if f in course_idx]
        legacy = arena.get("legacy", False)
        winner_name = arena["winner"]

        names = []
        features = []
        nf_na = []
        winner_pos = -1
        for pos, pd in enumerate(arena["pirates"]):
            p = pirates_list[pindex[pd["name"]]]
            if pd["name"] == winner_name:
                winner_pos = pos

            nf = 0; na = 0
            for c in food_ids:
                if c in p.alg: na += 1
                elif c in p.fav: nf += 1

            names.append(pd["name"])
            features.append([p.strength, p.weight, nf, na])
            nf_na.append((nf, na, p.strength, p.weight))

        arenas.append((names, features, nf_na, winner_pos, legacy))

legacy_arenas = [(i, a) for i, a in enumerate(arenas) if a[4]]
modern_arenas = [(i, a) for i, a in enumerate(arenas) if not a[4]]
print(f"Legacy: {len(legacy_arenas)}, Modern: {len(modern_arenas)}")

# ── Train NN on legacy ──
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

# Build tensors
X_legacy = np.array([a[1] for _, a in legacy_arenas], dtype=np.float32)
Y_legacy = np.array([a[3] for _, a in legacy_arenas], dtype=np.int64)
X_modern = np.array([a[1] for _, a in modern_arenas], dtype=np.float32)
Y_modern = np.array([a[3] for _, a in modern_arenas], dtype=np.int64)

# Normalize
means = X_legacy.reshape(-1, 4).mean(axis=0)
stds = X_legacy.reshape(-1, 4).std(axis=0)
stds[stds == 0] = 1.0
X_legacy_n = (X_legacy - means) / stds
X_modern_n = (X_modern - means) / stds

X_train_t = torch.tensor(X_legacy_n, device=device)
Y_train_t = torch.tensor(Y_legacy, device=device)
X_test_t = torch.tensor(X_modern_n, device=device)
Y_test_t = torch.tensor(Y_modern, device=device)

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

def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(-logits, dim=1)
    return probs.numpy()

print("\nTraining NN (128-64) on legacy...")
best_ll = -999
for run in range(5):
    model = SiameseScoringNet(4, [128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    dataset = TensorDataset(X_train_t, Y_train_t)
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
        test_ll = eval_ll(model, X_test_t, Y_test_t)
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
    ft = eval_ll(model, X_test_t, Y_test_t)
    print(f"  run {run}: modern LL={ft:.5f}")
    if ft > best_ll:
        best_ll = ft
        best_model = model

print(f"Best NN modern LL: {best_ll:.5f}")

# ── Get NN predictions on modern ──
print("\nComputing NN predictions on modern...")
nn_probs = get_probs(best_model, X_test_t)  # (n_modern, 4)

# ── Get Model 1 predictions on modern ──
print("Computing Model 1 predictions on modern...")
m1_probs = np.zeros((len(modern_arenas), 4))
for idx, (_, arena) in enumerate(modern_arenas):
    names, features, nf_na, winner_pos, legacy = arena
    pmfs = []
    for pos in range(4):
        nf, na, strength, weight = nf_na[pos]
        pmfs.append(model1_score_pmf(strength, weight, nf, na))
    probs = win_probs_later_wins(pmfs)
    m1_probs[idx] = probs
    if idx % 1000 == 0:
        print(f"  {idx}/{len(modern_arenas)}")

# ── Per-pirate aggregation ──
print("\nAggregating per-pirate stats on modern data...")

pirate_stats = defaultdict(lambda: {"wins": 0, "appearances": 0, "nn_prob_sum": 0.0, "m1_prob_sum": 0.0})

for idx, (_, arena) in enumerate(modern_arenas):
    names, features, nf_na, winner_pos, legacy = arena
    for pos in range(4):
        pname = names[pos]
        pirate_stats[pname]["appearances"] += 1
        pirate_stats[pname]["nn_prob_sum"] += nn_probs[idx, pos]
        pirate_stats[pname]["m1_prob_sum"] += m1_probs[idx, pos]
        if pos == winner_pos:
            pirate_stats[pname]["wins"] += 1

print(f"\n{'Pirate':<25} {'Str':>3} {'Apps':>5} {'Wins':>5} {'Hist%':>6} {'±95%':>5} {'M1%':>6} {'NN%':>6} {'M1err':>6} {'NNerr':>6} {'Better':>7}")
print("-" * 110)

rows = []
for pname, stats in pirate_stats.items():
    p = pirates_list[pindex[pname]]
    n = stats["appearances"]
    w = stats["wins"]
    hist_rate = w / n
    # 95% CI using normal approx
    se = math.sqrt(hist_rate * (1 - hist_rate) / n) if n > 0 else 0
    ci95 = 1.96 * se
    m1_rate = stats["m1_prob_sum"] / n
    nn_rate = stats["nn_prob_sum"] / n
    m1_err = m1_rate - hist_rate
    nn_err = nn_rate - hist_rate
    better = "NN" if abs(nn_err) < abs(m1_err) else "M1" if abs(m1_err) < abs(nn_err) else "TIE"
    rows.append((pname, p.strength, n, w, hist_rate, ci95, m1_rate, nn_rate, m1_err, nn_err, better))

rows.sort(key=lambda r: r[1])  # sort by strength
nn_better = 0
m1_better = 0
for r in rows:
    pname, strength, n, w, hist_rate, ci95, m1_rate, nn_rate, m1_err, nn_err, better = r
    # Flag if model prediction is outside 95% CI
    m1_flag = "*" if abs(m1_err) > ci95 else " "
    nn_flag = "*" if abs(nn_err) > ci95 else " "
    print(f"{pname:<25} {strength:>3} {n:>5} {w:>5} {hist_rate:>6.1%} {ci95:>4.1%} {m1_rate:>6.1%}{m1_flag} {nn_rate:>6.1%}{nn_flag} {m1_err:>+5.1%}  {nn_err:>+5.1%}  {better:>5}")
    if better == "NN": nn_better += 1
    elif better == "M1": m1_better += 1

print(f"\nNN closer: {nn_better}, M1 closer: {m1_better}")

# Summary: which pirates does each model get significantly wrong?
print(f"\nPirates where model is outside 95% CI (* above):")
print(f"{'Pirate':<25} {'M1 sig?':>7} {'NN sig?':>7} {'M1 err':>7} {'NN err':>7}")
print("-" * 60)
for r in rows:
    pname, strength, n, w, hist_rate, ci95, m1_rate, nn_rate, m1_err, nn_err, better = r
    m1_sig = abs(m1_err) > ci95
    nn_sig = abs(nn_err) > ci95
    if m1_sig or nn_sig:
        print(f"{pname:<25} {'YES' if m1_sig else 'no':>7} {'YES' if nn_sig else 'no':>7} {m1_err:>+6.2%} {nn_err:>+6.2%}")
