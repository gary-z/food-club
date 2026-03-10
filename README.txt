# Neopets Food Club Solver



Neopets Food Club is a betting game around the results of a daily eating contest between pirates. 



## Game Description

- There are 20 pirates. Each has a strength score, weight, favorite food types, and allergy food types.
- There are 40 possible food courses. Each course belongs to up to 3 food types.
- Each day, the pirates are randomly divided into 5 arenas of 4 pirates.
- Each arena randomly receives 10 distinct courses. Courses can be repeated across arenas.



## Betting

Each arena contains odds for each pirate.

- Each pirate's odds are 1:N, where 1 <= N <= 13.
- Players can place parley bets on any number of arenas. The total payout multiplies the odds in each individual arena.
- Players can place up to 10 bets per day.
- Players have a wager limit per bet based on their account age.
- Each bet payout is capped at 1,000,000 points.



## Historical Data

Over 8000 days of complete historical data is available. This can be used to reverse engineer the random process that determines winners in each arena. Pirates' win rates are between 6% and 65%.



## Random Processes

The following is known about the random process:
- A pirate's strength is the main contributing factor to their win rate.
- Favorite foods increase win likelihood, and allergies decrease likelihood.



A code leak reveals how pirate weight contributes. Heavier pirates suffer less penalty for certain things. This might apply per allergy food, but the context around this code is unclear.

```
$weight_offset = floor( ($FC_PIRATE_MAX_WEIGHT - $FC_PIRATE[$pirate_id]["weight"]) / 2);
$weight_offset = ($weight_offset > $FC_WEIGHT_MAX_EFFECT)? $FC_WEIGHT_MAX_EFFECT:$weight_offset;
$weight_loss = dice(1, $weight_offset);
$FC_PIRATE[$pirate_id]["life"] -= $weight_loss;

```