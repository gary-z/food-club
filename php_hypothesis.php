<?php
// Hypothetical Food Club arena resolution code (Neopets, ~2005)
// This reconstructs what the PHP source likely looks like,
// based on reverse-engineering from 4835 days of historical data.
//
// Key insight: a variable scoping bug in PHP (no block scope)
// causes $adj to accumulate across the pirate loop, creating
// an unintended position advantage for later pirates.
//
// Known constants (from leaked code + statistical fitting):
//   $FC_PIRATE_MAX_WEIGHT  = 221
//   $FC_WEIGHT_MAX_EFFECT  = 7
//   $FC_BASE               = 109
//   $FC_FAV_SPEED          = 93

function resolve_arena($arena_id) {
    global $FC_PIRATE, $FC_PIRATE_MAX_WEIGHT, $FC_WEIGHT_MAX_EFFECT,
           $FC_BASE, $FC_FAV_SPEED;

    $pirates = get_arena_pirates($arena_id);
    $foods   = get_arena_foods($arena_id);

    // Reset life for this contest
    foreach ($pirates as $pirate_id) {
        $FC_PIRATE[$pirate_id]["life"] = $FC_PIRATE[$pirate_id]["strength"];
    }

    // BUG: $adj declared outside pirate loop — never reset per pirate
    $adj = 0;

    foreach ($pirates as $pirate_id) {
        // Count favorites; apply allergy damage to life
        $n_fav = 0;
        foreach ($foods as $food_id) {
            if (is_allergy($pirate_id, $food_id)) {
                // --- LEAKED CODE (confirmed) ---
                $weight_offset = floor(($FC_PIRATE_MAX_WEIGHT
                    - $FC_PIRATE[$pirate_id]["weight"]) / 2);
                $weight_offset = ($weight_offset > $FC_WEIGHT_MAX_EFFECT)
                    ? $FC_WEIGHT_MAX_EFFECT : $weight_offset;
                $weight_loss = dice(1, $weight_offset);
                $FC_PIRATE[$pirate_id]["life"] -= $weight_loss;
                // --- END LEAKED CODE ---
            } elseif (is_favorite($pirate_id, $food_id)) {
                $n_fav++;
            }
            // overlap foods (both fav AND allergy) hit the allergy branch
            // via elseif — the favorite is silently skipped
        }

        // Eating die size: weaker pirate -> bigger die -> slower
        // Favorites shrink the die multiplicatively (93% per fav)
        $upper = max(1, floor(
            ($FC_BASE - $FC_PIRATE[$pirate_id]["life"])
            * pow($FC_FAV_SPEED / 100, $n_fav)
        ));

        // BUG: $adj carries from previous pirate
        //   pirate 0 (pos 0): adj=0  -> upper *= 100/100 = 1.00
        //   pirate 1 (pos 1): adj=7  -> upper *= 93/100  = 0.93
        //   pirate 2 (pos 2): adj=14 -> upper *= 86/100  = 0.86
        //   pirate 3 (pos 3): adj=21 -> upper *= 79/100  = 0.79
        $upper = max(1, floor($upper * (100 - $adj) / 100));

        // Roll eating time (3 rolls, lowest total wins)
        $time = 0;
        for ($i = 0; $i < 3; $i++) {
            $time += dice(1, $upper);
        }
        $FC_PIRATE[$pirate_id]["time"] = $time;

        // BUG: accumulates across pirates instead of resetting
        $adj += $FC_WEIGHT_MAX_EFFECT;   // += 7 each pirate
    }

    // Determine winner: lowest eating time
    // Uses strict < so first pirate with min wins ties
    // (ties are rare with raw score comparison)
    $min_time = PHP_INT_MAX;
    $winner   = -1;
    foreach ($pirates as $pirate_id) {
        if ($FC_PIRATE[$pirate_id]["time"] < $min_time) {
            $min_time = $FC_PIRATE[$pirate_id]["time"];
            $winner   = $pirate_id;
        }
    }
    return $winner;
}
