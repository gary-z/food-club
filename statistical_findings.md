# Odds Maker Reverse Engineering

## Summary

The odds maker uses **Monte Carlo simulation** (or exact probability computation using the game's own contest engine), not a simple heuristic formula. It is aware of favorites, allergies, position, overlap interactions, and opponent matchups. Odds are output as integers 2-13, with 13 acting as a catch-all floor for weak pirates.

---

## Evidence

### 1. The odds maker knows about favorites and allergies

Same pirate receives different odds depending on the arena's food lineup. Controlling for pirate identity (and therefore strength/weight), odds shift predictably with the number of favorites and allergies present.

**Gooblah the Grarrl (str=93):**

| (nFav, nAllergy) | Avg Odds | Win Rate |
|-------------------|----------|----------|
| (4, 0)            | 2.00     | 85.0%    |
| (2, 0)            | 2.00     | 78.1%    |
| (0, 0)            | 2.00     | 69.9%    |
| (0, 2)            | 2.32     | 42.5%    |
| (0, 3)            | 2.45     | 39.4%    |

**Linear regression** (odds ~ strength + nFav + nAllergy + position, R^2=0.56):
- Each favorite: **-0.84 odds**
- Each allergy: **+1.09 odds**
- Each position step (0->3): **-1.09 odds**
- Each strength point: **-0.27 odds**

The odds maker adjusts for the specific food courses served in each arena, not just static pirate attributes.

### 2. The odds maker knows about position

Every pirate receives lower (better) odds at position 3 than position 0. Selected examples:

| Pirate              | Str | Pos 0 | Pos 3 | Shift  |
|---------------------|-----|-------|-------|--------|
| Peg Leg Percival    | 73  | 10.45 | 5.44  | -5.01  |
| Captain Crossblades | 66  | 11.86 | 7.01  | -4.85  |
| Ol' Stripey         | 74  | 8.45  | 3.99  | -4.46  |
| Orvinn              | 52  | 11.47 | 7.13  | -4.34  |
| Gooblah             | 93  | 2.11  | 2.01  | -0.11  |

Gooblah's small shift is due to the odds=2 floor. Weaker pirates show shifts of 3-5 full odds points across positions.

**Residual position effect:** After controlling for (pirate, odds, nFav, nAllergy), a small excess win rate remains by position (pos 0: -3.1%, pos 3: +2.9%). This residual is likely a quantization artifact from mapping continuous probabilities to only 12 integer values (2-13).

### 3. The odds maker knows allergy overrides favorite on overlap foods

When a food triggers both a pirate's favorite and allergy categories, the odds maker treats it as harmful. Comparing arenas with vs without overlap foods, controlling for the number of would-be favorites:

- **Average odds increase when overlap present: +1.09**
- **Average win rate decrease when overlap present: -4.4%**

**Sir Edmund Ogletree** (2 would-be-favs, 1 allergy):

| Condition    | N   | Avg Odds | Win Rate |
|--------------|-----|----------|----------|
| With overlap | 216 | 5.18     | 22.7%    |
| Pure (no ov) | 427 | 3.99     | 31.6%    |

The odds maker correctly penalizes overlap situations, consistent with the allergy-overrides-favorite rule.

### 4. The odds maker considers opponents

For the same pirate with the same food context (nFav, nAllergy), odds correlate with the average strength of the three opponents in the arena.

- **Mean correlation (odds vs avg opponent strength): 0.49**
- **Median: 0.52**
- Individual combos range up to r=0.60+

A pure heuristic based only on a pirate's own stats would show zero correlation with opponent composition. The strong dependence on opponents proves the system evaluates the full 4-pirate matchup in each arena.

### 5. Odds are not a deterministic formula

For 97.8% of (pirate, nFav, nAllergy) combinations with 20+ observations, the assigned odds are **not constant** -- they vary depending on the specific matchup. Even adding position, 94.6% of combos still produce variable odds.

Example -- Ol' Stripey at nf=2, na=1 (N=606): odds range across all 12 values (2-13), with no single value exceeding 19% frequency.

This rules out any simple mapping like `odds = f(strength, nFav, nAllergy, position)`.

---

## Heuristic vs Monte Carlo

| Evidence | Implication |
|----------|-------------|
| Odds depend on opponent identity (r=0.49 correlation) | System evaluates full matchup, not individual stats |
| Same (pirate, nFav, nAllergy, pos) produces different odds | Odds are context-dependent, not from a lookup table |
| Overround (sum of 1/odds per arena) varies widely: 0.73 to 1.41, std=0.15 | Not a fixed-margin formula |
| All known game mechanics (strength, favs, allergies, position, overlap) are reflected in odds | System has access to game internals |
| Odds are integers 2-13 with 13 as a catch-all | Continuous probabilities quantized to integer odds |

**Conclusion:** The odds are derived from the game's own contest simulation. The system likely runs the contest (or computes exact probabilities using the same mechanics that determine winners), obtains each pirate's win probability against the specific opponents with the specific foods, then maps those probabilities to integer odds in the range 2-13.

---

## Odds calibration

| Odds | Count  | Actual WR | Implied WR | Ratio |
|------|--------|-----------|------------|-------|
| 2    | 27,587 | 52.2%     | 50.0%      | 1.04  |
| 3    | 10,394 | 28.6%     | 33.3%      | 0.86  |
| 4    | 8,106  | 22.8%     | 25.0%      | 0.91  |
| 5    | 7,531  | 18.4%     | 20.0%      | 0.92  |
| 6    | 4,248  | 15.8%     | 16.7%      | 0.95  |
| 7    | 4,618  | 13.7%     | 14.3%      | 0.96  |
| 8    | 2,440  | 12.5%     | 12.5%      | 1.00  |
| 9    | 2,415  | 11.7%     | 11.1%      | 1.05  |
| 10   | 2,626  | 10.9%     | 10.0%      | 1.09  |
| 11   | 2,694  | 9.7%      | 9.1%       | 1.07  |
| 12   | 2,824  | 8.5%      | 8.3%       | 1.02  |
| 13   | 21,217 | 4.1%      | 7.7%       | 0.54  |

Odds 2-12 are reasonably well calibrated (ratio 0.86-1.09). The odds=13 bucket is heavily miscalibrated (4.1% actual vs 7.7% implied) because it acts as a catch-all for all pirates below ~8% win probability -- many odds=13 pirates have true win rates of 1-3%.

---

## Overround

The sum of implied probabilities (1/odds) per arena averages **1.04** with high variance (std=0.15, range 0.73-1.41). This is unusually noisy for a betting system, suggesting the integer quantization introduces substantial rounding error rather than a deliberate margin. A traditional bookmaker targets a stable overround (e.g. 1.10); the wild variance here is consistent with independent rounding of each pirate's probability to the nearest integer odds value.
