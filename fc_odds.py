#!/usr/bin/env python3
"""Publishing rules for the opening line, and the fit of a per-state rating rule.

publish(p, step, mode)  quantises a probability vector and turns it into odds,
    odds_i = clamp(floor(1/quantise(p_i)), 2, 13)

tiles(step, mode)  inverts that: the probability interval each published odds
    value implies.

fit_separable(...)  fits one weight per pirate state under the Luce/ratio rule
    p_i = w_i / sum_j w_j so as to maximise the number of exactly reproduced
    odds values.  Every step is an exact 1-D max-coverage problem over the
    intervals a single weight has to satisfy, so the match count is monotone.
"""
import numpy as np


def clamp_floor(x):
    return np.clip(np.floor(x), 2, 13)


def publish(p, step=None, mode="round"):
    """probability vector -> published odds"""
    if step is None:
        q = np.asarray(p, dtype=float)
    elif mode == "round":
        q = np.round(np.asarray(p, dtype=float) / step) * step
    else:
        q = np.floor(np.asarray(p, dtype=float) / step) * step
    with np.errstate(divide="ignore"):
        o = np.where(q <= 0, 13, clamp_floor(1.0 / np.maximum(q, 1e-15)))
    return o.astype(np.int64)


def tiles(step=None, mode="round"):
    """published odds value -> (lo, hi) on the underlying probability"""
    if step is None:
        return {v: (0.0 if v == 13 else 1.0 / (v + 1),
                    1.0 if v == 2 else 1.0 / v) for v in range(2, 14)}
    M = int(round(1.0 / step))
    grid = {}
    for m in range(M + 1):
        q = m * step
        v = 13 if m == 0 else int(min(13, max(2, np.floor(1.0 / q))))
        grid.setdefault(v, []).append(m)
    out = {}
    for v in range(2, 14):
        ms = grid.get(v)
        if not ms:
            out[v] = None
            continue
        lo_m, hi_m = min(ms), max(ms)
        if mode == "round":
            lo, hi = (lo_m - 0.5) * step, (hi_m + 0.5) * step
        else:
            lo, hi = lo_m * step, (hi_m + 1) * step
        out[v] = (max(lo, 0.0), min(hi, 1.0))
    return out


def tile_arrays(step=None, mode="round"):
    T = tiles(step, mode)
    lo = np.array([T[v][0] for v in range(2, 14)])
    hi = np.array([T[v][1] for v in range(2, 14)])
    return lo, hi


def _max_coverage(los, his):
    """point covered by the most half-open [lo,hi) intervals"""
    ev = []
    for lo, hi in zip(los, his):
        if hi <= lo:
            continue
        ev.append((lo, 1))
        ev.append((hi, -1))
    if not ev:
        return None, 0
    ev.sort(key=lambda t: (t[0], -t[1]))
    cur = 0
    best_v, best_c = None, 0
    for val, delta in ev:
        cur += delta
        if delta > 0 and cur > best_c:
            best_v, best_c = val, cur
    return best_v, best_c


def fit_separable(ids, n_cls, target, lo_t, hi_t, train, w0=None, sweeps=8,
                  verbose=True, log=print):
    """Luce weights per class, coordinate ascent on exact reproduction.

    ids     (n,4) class index per slot
    target  (n,4) published odds to reproduce
    lo_t/hi_t  (12,) probability tile per odds value 2..13
    """
    n = ids.shape[0]
    lo = lo_t[target - 2]
    hi = hi_t[target - 2]
    w = np.ones(n_cls) if w0 is None else np.maximum(w0.copy(), 1e-9)

    slots = [[] for _ in range(n_cls)]
    for a in np.where(train)[0]:
        for i in range(4):
            slots[ids[a, i]].append((a, i))

    order = np.argsort([-len(s) for s in slots])
    for sweep in range(sweeps):
        S = w[ids].sum(axis=1)
        for c in order:
            if not slots[c]:
                continue
            los, his = [], []
            for a, i in slots[c]:
                T = S[a] - w[ids[a, i]]          # other three weights
                l, h = lo[a, i], hi[a, i]
                # own constraint: w/(w+T) in [l,h)
                if l > 0:
                    los.append(l * T / (1 - l))
                else:
                    los.append(0.0)
                his.append(np.inf if h >= 1 else h * T / (1 - h))
                # the other three slots also move when w changes
                for j in range(4):
                    if j == i:
                        continue
                    wj = w[ids[a, j]]
                    Tj = S[a] - w[ids[a, i]]     # total minus this class' weight
                    lj, hj = lo[a, j], hi[a, j]
                    if hj >= 1:
                        low = 0.0
                    else:
                        low = wj / hj - Tj
                    high = np.inf if lj <= 0 else wj / lj - Tj
                    los.append(max(low, 0.0))
                    his.append(high)
            val, cnt = _max_coverage(los, his)
            if val is None:
                continue
            new = max(val * (1 + 1e-12), 1e-12)
            if np.isfinite(new) and new > 0:
                for a, i in slots[c]:
                    S[a] += new - w[c]
                w[c] = new
        if verbose:
            pred = predict_separable(w, ids)
            log(f"      sweep {sweep}: train slot-exact "
                f"{100*(pred[train] == target[train]).mean():.2f}%")
    return w


def predict_separable(w, ids, step=None, mode="round"):
    ww = w[ids]
    p = ww / ww.sum(axis=1, keepdims=True)
    return publish(p, step, mode)


def separable_probs(w, ids):
    ww = w[ids]
    return ww / ww.sum(axis=1, keepdims=True)


def class_ids(keys, n):
    """keys: list of (n,4) integer arrays -> (n,4) class ids, dict of key->id"""
    flat = list(zip(*[np.asarray(k).ravel() for k in keys]))
    uniq = {}
    out = np.empty(len(flat), dtype=np.int64)
    for i, k in enumerate(flat):
        out[i] = uniq.setdefault(k, len(uniq))
    return out.reshape(n, 4), uniq
