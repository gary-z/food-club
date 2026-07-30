#!/usr/bin/env python3
"""Exact win probabilities for the dice-race game model, vectorised over arenas.

Mirrors the PMF engine in sim/src/food_unified.rs and docs/foodclub.js: each
pirate rolls n_rolls dice of size `upper`, the sum is optionally quantised by a
divisor, lowest total wins, and allergy damage is marginalised out rather than
sampled.  Because only (strength, weight, nf, na, position) enter, there are a
couple of thousand distinct pirate states in the whole data set, so every score
PMF is computed once and then gathered per arena.
"""
import numpy as np

MAX_WEIGHT = 221


def dice_sum_pmf(n, d):
    """PMF of the sum of n dice uniform on 1..d, indexed by the sum."""
    if d <= 0 or n <= 0:
        return np.array([1.0])
    pmf = np.zeros(n * d + 1)
    pmf[1:d + 1] = 1.0 / d
    for _ in range(n - 1):
        c = np.concatenate(([0.0], np.cumsum(pmf)))
        # rolling window sum of length d, divided by d
        out = np.zeros_like(pmf)
        idx = np.arange(len(pmf))
        lo = np.maximum(idx - d, 0)
        out = (c[idx] - c[lo]) / d
        pmf = out
    return pmf


class Model:
    """Parameters of one hand-rolled game model (see best_models.txt)."""

    def __init__(self, base, n_rolls, fav_mode="bulk", fav_param=15, pos_step=0,
                 divisor=0, max_effect=7, allergy_order="before", tiebreak="later",
                 wo_min=0):
        self.base = base
        self.n_rolls = n_rolls
        self.fav_mode = fav_mode        # bulk | iterative | mul
        self.fav_param = fav_param
        self.pos_step = pos_step        # die *= (100 - pos*pos_step)/100
        self.divisor = divisor          # 0 = no quantisation
        self.max_effect = max_effect    # cap on weight_offset
        self.allergy_order = allergy_order  # before (life -= dmg) | after (die += dmg)
        self.tiebreak = tiebreak        # later | earlier
        self.wo_min = wo_min            # dice(1,0) -> wo_min (1 reproduces old PHP)

    def wo(self, weight):
        v = (MAX_WEIGHT - min(weight, MAX_WEIGHT)) // 2
        if self.max_effect > 0:
            v = min(v, self.max_effect)
        return max(v, self.wo_min)

    def die_from_life(self, life, nf, pos):
        u = max(1, self.base - life)
        if self.fav_mode == "bulk":
            u = max(1, u - nf * (u // self.fav_param))
        elif self.fav_mode == "iterative":
            for _ in range(nf):
                u = max(1, u - u // self.fav_param)
        elif self.fav_mode == "mul":
            x = float(u)
            for _ in range(nf):
                x *= self.fav_param / 100.0
            u = max(1, int(np.floor(x)))
        if self.pos_step:
            u = max(1, (u * (100 - pos * self.pos_step)) // 100)
        return u

    def dice_of_state(self, strength, weight, nf, na, pos):
        """[(prob, die size)] after marginalising allergy damage"""
        w = self.wo(weight)
        if na == 0 or w <= 0:
            return [(1.0, self.die_from_life(strength, nf, pos))]
        dmg = dice_sum_pmf(na, w)
        out = []
        for v, pr in enumerate(dmg):
            if pr <= 0:
                continue
            if self.allergy_order == "before":
                out.append((pr, self.die_from_life(strength - v, nf, pos)))
            else:
                out.append((pr, max(1, self.die_from_life(strength, nf, pos) + v)))
        return out


def score_pmfs(states, model):
    """states: list of (strength, weight, nf, na, pos) -> (S,) PMF each, padded."""
    dice_needed = set()
    per_state = []
    for st in states:
        combo = model.dice_of_state(*st)
        per_state.append(combo)
        dice_needed.update(d for _, d in combo)
    tbl = {}
    for d in dice_needed:
        raw = dice_sum_pmf(model.n_rolls, d)
        if model.divisor and model.divisor > 1:
            idx = np.arange(len(raw)) // model.divisor
            q = np.bincount(idx, weights=raw)
            tbl[d] = q
        else:
            tbl[d] = raw
    S = max(len(v) for v in tbl.values())
    out = np.zeros((len(states), S))
    for i, combo in enumerate(per_state):
        for pr, d in combo:
            v = tbl[d]
            out[i, :len(v)] += pr * v
    return out


def win_probs(pmf_states, cls, tiebreak="later", chunk=2000):
    """cls: (n,4) indices into pmf_states.  Returns (n,4) win probabilities."""
    S = pmf_states.shape[1]
    ge = np.cumsum(pmf_states[:, ::-1], axis=1)[:, ::-1]          # P(score >= s)
    gt = ge - pmf_states                                          # P(score >  s)
    n = cls.shape[0]
    out = np.empty((n, 4))
    for a0 in range(0, n, chunk):
        a1 = min(a0 + chunk, n)
        c = cls[a0:a1]
        P = pmf_states[c]        # (m,4,S)
        GE = ge[c]
        GT = gt[c]
        if tiebreak == "later":
            # a later position wins ties, so pirate i only needs to match the
            # positions before it and must strictly beat the ones after it
            before, after = GE, GT
        else:
            before, after = GT, GE
        for i in range(4):
            acc = P[:, i, :].copy()
            for j in range(4):
                if j == i:
                    continue
                acc *= before[:, j, :] if j < i else after[:, j, :]
            out[a0:a1, i] = acc.sum(axis=1)
    return out


def arena_win_probs(feat, model, chunk=2000):
    """feat: (n,4,4) strength, weight, nf, na -> (n,4) exact win probabilities."""
    n = feat.shape[0]
    key = np.empty((n, 4), dtype=np.int64)
    states = {}
    lst = []
    f = feat.astype(np.int64)
    for pos in range(4):
        block = f[:, pos, :]
        for a in range(n):
            st = (block[a, 0], block[a, 1], block[a, 2], block[a, 3], pos)
            c = states.get(st)
            if c is None:
                c = len(lst)
                states[st] = c
                lst.append(st)
            key[a, pos] = c
    pmfs = score_pmfs(lst, model)
    return win_probs(pmfs, key, model.tiebreak, chunk=chunk), key, lst


MODELS = {
    # from best_models.txt
    "M1": Model(base=112, n_rolls=4, fav_mode="bulk", fav_param=15, divisor=14,
                max_effect=7, allergy_order="before", tiebreak="later"),
    "M2": Model(base=109, n_rolls=3, fav_mode="mul", fav_param=93, pos_step=7,
                max_effect=7, allergy_order="before", tiebreak="later"),
    "M4": Model(base=120, n_rolls=6, fav_mode="iterative", fav_param=16, divisor=22,
                max_effect=6, allergy_order="after", tiebreak="later"),
    "M5": Model(base=118, n_rolls=5, fav_mode="mul", fav_param=94, pos_step=2,
                divisor=10, max_effect=6, allergy_order="after", tiebreak="later"),
}


def mc_win_probs(feat, model, iters, seed=0):
    """Plain monte carlo of the same model, to sanity check the PMF path."""
    rng = np.random.default_rng(seed)
    n = feat.shape[0]
    wins = np.zeros((n, 4))
    f = feat.astype(np.int64)
    for a in range(n):
        for _ in range(iters):
            times = []
            for pos in range(4):
                strength, weight, nf, na = f[a, pos]
                wo = model.wo(weight)
                dmg = int(rng.integers(1, wo + 1, size=na).sum()) if (na and wo > 0) else 0
                if model.allergy_order == "before":
                    die = model.die_from_life(strength - dmg, nf, pos)
                else:
                    die = max(1, model.die_from_life(strength, nf, pos) + dmg)
                t = int(rng.integers(1, die + 1, size=model.n_rolls).sum())
                if model.divisor and model.divisor > 1:
                    t //= model.divisor
                times.append(t)
            best = min(times)
            # later position wins ties
            wins[a, max(i for i in range(4) if times[i] == best)] += 1
    return wins / iters


if __name__ == "__main__":
    import fc_data

    d = fc_data.load_arenas()
    win = np.zeros((d["feat"].shape[0], 4), dtype=bool)
    win[np.arange(len(win)), d["winner"]] = True
    for name, m in MODELS.items():
        p, key, lst = arena_win_probs(d["feat"], m)
        ll_mod = np.log(np.maximum(p[~d["legacy"]][win[~d["legacy"]]], 1e-12)).mean()
        ll_leg = np.log(np.maximum(p[d["legacy"]][win[d["legacy"]]], 1e-12)).mean()
        pred = np.clip(np.floor(1.0 / p), 2, 13).astype(np.int64)
        print(f"{name}: states={len(lst)} sum p={p.sum(axis=1).mean():.6f} "
              f"LL legacy={ll_leg:.5f} modern={ll_mod:.5f} "
              f"odds slot-exact={100*(pred == d['odds']).mean():.2f}%")

    # sanity check the PMF path against a plain monte carlo on a sample of arenas
    print("\nPMF vs plain monte carlo (400 arenas x 20000 draws):")
    sel = np.random.default_rng(0).choice(len(d["feat"]), 400, replace=False)
    for name, m in MODELS.items():
        p, _, _ = arena_win_probs(d["feat"][sel], m)
        pm = mc_win_probs(d["feat"][sel], m, 20000, seed=1)
        err = np.abs(p - pm)
        se = np.sqrt(np.maximum(pm * (1 - pm), 1e-9) / 20000)
        print(f"  {name}: max|diff|={err.max():.4f} mean|diff|={err.mean():.5f} "
              f"max z={(err / se).max():.2f}")
