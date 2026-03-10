# Food Club Reverse Engineering Findings

All simulations use 20M days (Rust) unless noted. Log ratio avg is the primary metric (lower = better).
Historical data: ~8000 days, 4835 Orvinn appearances.

---

## Baseline

**Python model** (`simple_win_rate_model.py`): log ratio **0.072**

Formula: `score = -sum((112.5 - effective_strength) * rand() for _ in range(3))`
where `effective_strength = strength + 2.7 * n_fav - 3.0 * n_allergy`
Winner = max score (lowest total eating time).
Note: the loop runs exactly 3 times regardless of course count — not per-course.

---

## Code Leak

```php
$weight_offset = floor(($FC_PIRATE_MAX_WEIGHT - $pirate_weight) / 2);
$weight_offset = min($weight_offset, $FC_WEIGHT_MAX_EFFECT);
$weight_loss = dice(1, $weight_offset);   // uniform int in [1, weight_offset]
$pirate[$id]["life"] -= $weight_loss;
```

Context: "This might apply per allergy food." `dice(1, N)` = `rand(1..N)`.
Two unknown global constants: `FC_PIRATE_MAX_WEIGHT` (≥221), `FC_WEIGHT_MAX_EFFECT`.

---

## Life-Based Hypotheses (H1–H6)

All use `life = strength` as starting value. Winner = highest remaining life.
Weight offset formula: `wo = min(floor((max_weight - pirate_weight) / 2), max_effect)`.
`roll(n)` = uniform int in [1, n], 0 if n=0.

Important: favorites only apply if course is NOT also an allergy (matching Python logic).

| Hypothesis | Description | Best params | Log ratio |
|---|---|---|---|
| H1 | Allergy penalty only: `life -= roll(wo)` per allergy course | max_w=250, max_e=25 | 0.392 |
| H2 | H1 + fixed fav bonus per fav course | max_w=250, max_e=25, fav=5 | 0.273 |
| H3 | Symmetric dice: allergies lose `roll(wo)`, favs gain `roll(wo)` | max_w=300, max_e=20 | 0.408 |
| H4 | Weight penalty on EVERY course + extra for allergies + fixed fav bonus | max_w=250, max_e=15, fav=3 | 0.126 |
| H5 | H4 + weight-scaled fav dice: heavier pirates gain more from favs | max_w=250, max_e=15, max_fav_e=5 | 0.161 |
| H6 | H4 + weight-based variance on initial life: `life += roll(weight / var_div)` | max_w=250, max_e=15, fav=3, var_div=15 | 0.111 |

**Key finding:** H4 (weight penalty on ALL courses, not just allergies) is the decisive improvement over H1–H3. The code leak likely describes a general eating mechanic, not just an allergy-specific one.

H6's weight-based initial life variance helps but doesn't resolve the core gap.
H5 (heavier pirates gaining more from favorites) does not help.

Consistent best params across all life-based models: **max_weight≈250, max_effect≈15**.

---

## Bulk-Roll Hypotheses (H7–H8)

Hypothesis: only the COUNT of favorites/allergies/regular courses matters, not per-course iteration. One roll per category.

| Hypothesis | Description | Best params | Log ratio |
|---|---|---|---|
| H7 | One bulk roll for favs + one for allergies (no regular) | max_w=250, max_e=30, fav_p=8 | 0.176 |
| H8 | Three bulk rolls: one each for favs, allergies, regular courses | max_w=220, max_e=25, fav_p=3, reg_p=5 | 0.118 |

H7 without a regular-course penalty loses too much signal (strength stops mattering enough).
H8 is comparable to H6 but not better. Bulk vs per-course doesn't make a meaningful difference.

---

## 3-Roll Category Hypotheses (H9–H11)

Motivated by Python's `for _ in range(3)` loop — exactly 3 rolls regardless of courses.
Each roll has its own upper bound based on category count. Winner = lowest total time.

| Hypothesis | Description | Best params | Log ratio |
|---|---|---|---|
| H9 (additive) | `fav_upper = center - n_fav*s`, `allergy_upper = center + n_allergy*s`, `normal_upper = center` where `center = base - strength` | base=110, fav_s=8, all_s=10 | **0.0737** |
| H10 (mult, 3-roll) | Per-category multiplicative: `fav_upper = center * fav_pct^n_fav`, etc. | base=110, fav%=70, all%=120 | 0.0782 |
| **H11 (mult, shared)** | All 3 rolls share one bound: `upper = (base-strength) * fav_pct^n_fav * allergy_pct^n_allergy` | **base=110, fav%=92, all%=115** | **0.0709** |

**H11 beats the Python baseline (0.072).** Current best model.

H9's mean is nearly identical to the Python model (base=110≈112.5, fav_s=8≈2.7×3, all_s=10≈3.0×3) — the structural difference is variance per roll by category. Minimal improvement over Python.

H10 (separate multiplicative bounds per category) is worse than H11 — the shared bound is the right structure.

**H11 interpretation:** Each favorite multiplies eating time by 0.92 (8% faster per favorite). Each allergy multiplies eating time by 1.15 (15% slower per allergy). Non-linear: 7 favorites → 0.92^7 = 0.56× time (44% faster). This is the key improvement over additive models.

---

## Orvinn Analysis

Orvinn the First Mate: weight=221 (heaviest), strength=52 (lowest), 14 favorite courses, 5 allergy courses.
Historical win rate: **10.8%**. Best model (H11) predicts: **6.5%**. Gap: **+4.1pp**.

### Win rate by # of favorites in arena (historical):

| n_fav | win rate |
|---|---|
| 0 | 3.4% |
| 2 | 6.6% |
| 3 | 9.8% |
| 4 | 11.4% |
| 5 | 14.4% |
| 6 | 16.2% |
| 7 | 24.0% |

Non-linear: at 7 favorites Orvinn wins 24% — far more than any additive model can capture.
The multiplicative H11 improves this but still underpredicts the 7-favorite case significantly.

### Orvinn's in-game odds vs actual win rate:

| odds | actual win rate | implied prob |
|---|---|---|
| 2 | 31.9% | 50% |
| 3 | 29.6% | 33% |
| 13 (44% of arenas) | 4.8% | 7.7% |

The game itself assigns Orvinn odds=13 in 44% of arenas, and at those odds he wins 4.8% (below model). At odds=2–3 he wins ~30% (well above model). This suggests the game's own odds engine is capturing something about arena food composition that our model misses.

### Model gap by opponent strength tier (H11 expected vs actual):

| Strongest opponent | actual | H11 expected | ratio (actual/expected) |
|---|---|---|---|
| Weakest (avg str 76.9) | 15.3% | 10.2% | 1.51× |
| Weak-mid (81.1) | 12.3% | 7.2% | 1.71× |
| Strong-mid (85.5) | 9.3% | 5.9% | 1.59× |
| **Strongest (91.3)** | **6.0%** | **3.3%** | **1.82×** |

**The proportional gap is largest against the strongest opponents.** Against Gooblah specifically: actual=4.9%, expected=2.4% (2.0× miss).

### Root cause hypothesis:

The `(base - strength)` term creates a very steep spread: Gooblah's base upper = 17, Orvinn's = 58 (3.4× ratio). This compounds over 3 rolls into ~40× win probability ratio — too extreme. When Gooblah hits his Slushies allergy, his upper jumps to 30 (≈ Orvinn's upper), making them near-equal, but the model may not weight this correctly.

**Proposed next step:** Sub-linear strength scaling. Instead of `(base - strength)`, try `(base - strength)^p` with p < 1 (e.g., 0.7–0.9), or `sqrt(base - strength)`. This compresses the strength gap without changing the ordering, and would let Orvinn compete better against the top without over-inflating his win rate against weak opponents.

---

## Infrastructure

- **Rust simulation** (`sim/`): 10.6× faster than Python (0.59s vs 6.3s for 2M days)
- Data in `pirates.json`, loaded by both Python and Rust
- Grid search parallelised with rayon; final runs at 20M days
- Consistent bug to watch: favorites must exclude courses that are also allergies (matching Python's `num_favs` logic)
