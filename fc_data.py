#!/usr/bin/env python3
"""Shared data loading for Food Club analyses.

Produces, for every historical arena, the per-position features the game
algorithm actually consumes (strength, weight, fav count, allergy count) plus
the opening odds, the winner and the legacy flag.

Findings 7/8/37 establish that only the *counts* of favourite/allergy courses
matter and that overlap courses count as allergy only (elseif ordering), so
(pirate, nf, na) is the pirate's complete state and
((pirate, nf, na) x 4, in position order) is the arena's complete state.
"""
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_pirates():
    with open(os.path.join(ROOT, "pirates.json")) as f:
        raw = json.load(f)

    course_names = list(raw["courses"].keys())
    course_idx = {n: i for i, n in enumerate(course_names)}
    cat_courses = defaultdict(set)
    for cname, cats in raw["courses"].items():
        for cat in cats:
            cat_courses[cat].add(course_idx[cname])

    pirates = []
    for d in raw["pirates"]:
        fav = set()
        for c in d["favorites"]:
            fav |= cat_courses.get(c, set())
        alg = set()
        for c in d["allergies"]:
            alg |= cat_courses.get(c, set())
        pirates.append(dict(name=d["name"], strength=d["strength"],
                            weight=d["weight"], fav=fav, alg=alg))
    return pirates, course_idx


def load_arenas():
    """Returns a dict of parallel arrays, one row per arena.

    feat        (n, 4, 4) float32   strength, weight, nf, na  per position
    pirate_ix   (n, 4) int8         index into the pirate table
    odds        (n, 4) int16        opening odds
    cur_odds    (n, 4) int16        current (end of day) odds, -1 when absent
    winner      (n,)  int8          winning position
    legacy      (n,)  bool
    day         (n,)  int32         index into historical_matches.json
    arena       (n,)  int8          arena slot within the day (0-4)
    foods       (n, 10) int16       course indices
    """
    pirates, course_idx = load_pirates()
    pindex = {p["name"]: i for i, p in enumerate(pirates)}

    with open(os.path.join(ROOT, "historical_matches.json")) as f:
        hist = json.load(f)

    feat, pirate_ix, odds, cur_odds = [], [], [], []
    winner, legacy, day, arena_no, foods = [], [], [], [], []

    for di, dayrec in enumerate(hist):
        for ai, arena in enumerate(dayrec):
            food_ids = [course_idx[f] for f in arena["foods"] if f in course_idx]
            f_row, p_row, o_row, c_row = [], [], [], []
            win = -1
            for pos, pd in enumerate(arena["pirates"]):
                p = pirates[pindex[pd["name"]]]
                nf = na = 0
                for c in food_ids:
                    if c in p["alg"]:
                        na += 1
                    elif c in p["fav"]:
                        nf += 1
                f_row.append([p["strength"], p["weight"], nf, na])
                p_row.append(pindex[pd["name"]])
                o_row.append(pd["odds"])
                c_row.append(pd.get("current_odds", -1))
                if pd["name"] == arena["winner"]:
                    win = pos
            feat.append(f_row)
            pirate_ix.append(p_row)
            odds.append(o_row)
            cur_odds.append(c_row)
            winner.append(win)
            legacy.append(arena.get("legacy", False))
            day.append(di)
            arena_no.append(ai)
            foods.append(food_ids)

    return dict(
        feat=np.array(feat, dtype=np.float32),
        pirate_ix=np.array(pirate_ix, dtype=np.int8),
        odds=np.array(odds, dtype=np.int16),
        cur_odds=np.array(cur_odds, dtype=np.int16),
        winner=np.array(winner, dtype=np.int8),
        legacy=np.array(legacy, dtype=bool),
        day=np.array(day, dtype=np.int32),
        arena=np.array(arena_no, dtype=np.int8),
        foods=np.array(foods, dtype=np.int16),
        pirates=pirates,
    )


if __name__ == "__main__":
    d = load_arenas()
    print(f"arenas {d['feat'].shape[0]}  legacy {d['legacy'].sum()}  "
          f"modern {(~d['legacy']).sum()}")
    print("odds histogram:")
    vals, cnts = np.unique(d["odds"], return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  {v:>3}: {c:>6} ({100*c/d['odds'].size:.2f}%)")
    nf = d["feat"][:, :, 2].astype(int)
    na = d["feat"][:, :, 3].astype(int)
    print(f"nf range {nf.min()}..{nf.max()}   na range {na.min()}..{na.max()}")
