// Food Club PMF Engine + Betting Strategy
// Model 4: Iterative Fav + Allergy-After (modern LL=-1.06314)

const MODEL = {
  BASE: 120,
  FAV_DIV: 16,
  N_ROLLS: 6,
  DIVISOR: 22,
  MAX_WEIGHT: 221,
  MAX_EFFECT: 6,
};

// Static pirate data (1-indexed: pirate ID N = PIRATES[N-1])
const PIRATES = [
  { name: "Scurvy Dan the Blade", weight: 166, strength: 87, favorites: ["Salty foods", "Meats"], allergies: ["Candy"] },
  { name: "Young Sproggie", weight: 112, strength: 73, favorites: ["Meats", "Neggs"], allergies: ["Gross foods"] },
  { name: "Orvinn the First Mate", weight: 221, strength: 52, favorites: ["Candy", "Slushies", "Pizza"], allergies: ["Fruits"] },
  { name: "Lucky McKyriggan", weight: 182, strength: 82, favorites: ["Gross foods"], allergies: ["Pizza"] },
  { name: "Sir Edmund Ogletree", weight: 177, strength: 79, favorites: ["Dairy"], allergies: ["Breads"] },
  { name: "Peg Leg Percival", weight: 202, strength: 73, favorites: ["Spicy foods"], allergies: ["Smoothies"] },
  { name: "Bonnie Pip Culliford", weight: 116, strength: 76, favorites: ["Candy", "Smoothies"], allergies: ["Spicy foods"] },
  { name: "Puffo the Waister", weight: 180, strength: 68, favorites: ["Candy", "Smoothies", "Slushies"], allergies: ["Meats"] },
  { name: "Stuff-A-Roo", weight: 211, strength: 59, favorites: ["Pizza"], allergies: ["Neggs"] },
  { name: "Squire Venable", weight: 213, strength: 61, favorites: ["Breads"], allergies: ["Fruits"] },
  { name: "Captain Crossblades", weight: 185, strength: 66, favorites: ["Slushies", "Pizza"], allergies: ["Salty foods"] },
  { name: "Ol' Stripey", weight: 189, strength: 74, favorites: ["Meats", "Slushies"], allergies: ["Breads"] },
  { name: "Ned the Skipper", weight: 169, strength: 79, favorites: ["Meats"], allergies: ["Dairy"] },
  { name: "Fairfax the Deckhand", weight: 151, strength: 71, favorites: ["Vegetables", "Fruits"], allergies: ["Salty foods"] },
  { name: "Gooblah the Grarrl", weight: 199, strength: 93, favorites: ["Meats"], allergies: ["Slushies"] },
  { name: "Franchisco Corvallio", weight: 165, strength: 81, favorites: ["Spicy foods", "Meats"], allergies: ["Candy"] },
  { name: "Federismo Corvallio", weight: 166, strength: 81, favorites: ["Gross foods", "Pizza"], allergies: ["Smoothies"] },
  { name: "Admiral Blackbeard", weight: 171, strength: 76, favorites: ["Vegetables", "Fruits"], allergies: ["Dairy"] },
  { name: "Buck Cutlass", weight: 189, strength: 89, favorites: ["Candy"], allergies: ["Vegetables"] },
  { name: "The Tailhook Kid", weight: 207, strength: 81, favorites: ["Vegetables"], allergies: ["Neggs"] },
];

// Static food data (1-indexed: food ID N = FOODS[N-1])
const FOODS = [
  { name: "Hotfish", categories: ["Salty foods", "Meats"] },
  { name: "Broccoli", categories: ["Vegetables"] },
  { name: "Wriggling Grub", categories: ["Gross foods"] },
  { name: "Joint Of Ham", categories: ["Meats"] },
  { name: "Rainbow Negg", categories: ["Neggs"] },
  { name: "Streaky Bacon", categories: ["Meats"] },
  { name: "Ultimate Burger", categories: ["Meats"] },
  { name: "Bacon Muffin", categories: ["Meats", "Breads"] },
  { name: "Hot Cakes", categories: ["Breads"] },
  { name: "Spicy Wings", categories: ["Spicy foods", "Meats"] },
  { name: "Apple Onion Rings", categories: ["Fruits", "Gross foods"] },
  { name: "Sushi", categories: ["Salty foods", "Meats"] },
  { name: "Negg Stew", categories: ["Neggs"] },
  { name: "Ice Chocolate Cake", categories: ["Candy"] },
  { name: "Strochal", categories: ["Candy"] },
  { name: "Mallowicious Bar", categories: ["Candy"] },
  { name: "Fungi Pizza", categories: ["Gross foods", "Pizza"] },
  { name: "Broccoli and Cheese Pizza", categories: ["Vegetables", "Dairy", "Pizza"] },
  { name: "Bubbling Blueberry Pizza", categories: ["Fruits", "Pizza"] },
  { name: "Grapity Slush", categories: ["Slushies"] },
  { name: "Rainborific Slush", categories: ["Slushies"] },
  { name: "Tangy Tropic Slush", categories: ["Slushies"] },
  { name: "Blueberry Tomato Blend", categories: ["Fruits", "Dairy", "Smoothies"] },
  { name: "Lemon Blitz", categories: ["Fruits", "Dairy", "Smoothies"] },
  { name: "Fresh Seaweed Pie", categories: ["Salty foods", "Gross foods"] },
  { name: "Flaming Burnumup", categories: ["Spicy foods", "Vegetables"] },
  { name: "Hot Tyrannian Pepper", categories: ["Spicy foods", "Vegetables"] },
  { name: "Eye Candy", categories: ["Candy", "Gross foods"] },
  { name: "Cheese and Tomato Sub", categories: ["Fruits", "Breads", "Dairy"] },
  { name: "Asparagus Pie", categories: ["Vegetables"] },
  { name: "Wild Chocomato", categories: ["Dairy", "Smoothies"] },
  { name: "Cinnamon Swirl", categories: ["Candy", "Breads"] },
  { name: "Anchovies", categories: ["Salty foods", "Meats"] },
  { name: "Flaming Fire Faerie Pizza", categories: ["Spicy foods", "Vegetables", "Pizza"] },
  { name: "Orange Negg", categories: ["Neggs"] },
  { name: "Fish Negg", categories: ["Neggs"] },
  { name: "Super Lemon Grape Slush", categories: ["Slushies"] },
  { name: "Rasmelon", categories: ["Smoothies"] },
  { name: "Mustard Ice Cream", categories: ["Dairy", "Gross foods"] },
  { name: "Worm and Leech Pizza", categories: ["Gross foods", "Pizza"] },
];

const ARENA_NAMES = ["Shipwreck", "Lagoon", "Treasure Island", "Hidden Cove", "Harpoon Harry's"];

// ==================== PMF Engine ====================

// Count favs and allergies for a pirate given food IDs (1-indexed)
function courseCounts(pirateId, foodIds) {
  const pirate = PIRATES[pirateId - 1];
  const favSet = new Set(pirate.favorites);
  const allergySet = new Set(pirate.allergies);
  let nf = 0, na = 0;
  for (const foodId of foodIds) {
    const food = FOODS[foodId - 1];
    let isFav = false, isAllergy = false;
    for (const cat of food.categories) {
      if (favSet.has(cat)) isFav = true;
      if (allergySet.has(cat)) isAllergy = true;
    }
    // Overlap: allergy takes priority
    if (isFav && isAllergy) { na++; }
    else if (isFav) { nf++; }
    else if (isAllergy) { na++; }
  }
  return { nf, na };
}

// PMF of sum of n dice, each uniform on {1, ..., d}
function diceSumPmf(n, d) {
  if (d === 0 || n === 0) return [1.0];
  const max = n * d;
  const invD = 1.0 / d;
  let pmf = new Float64Array(max + 1);
  for (let k = 1; k <= d; k++) pmf[k] = invD;
  for (let i = 1; i < n; i++) {
    const newPmf = new Float64Array(max + 1);
    let s = 0;
    for (let k = 0; k <= max; k++) {
      if (k >= 1) s += pmf[k - 1];
      if (k > d) s -= pmf[k - d - 1];
      newPmf[k] = s * invD;
    }
    pmf = newPmf;
  }
  return pmf;
}

// Precomputed roll table: rollTable[d] = PMF of sum of N_ROLLS dice each 1..d
let _rollTable = null;
function getRollTable() {
  if (_rollTable) return _rollTable;
  const maxUpper = 200;
  _rollTable = new Array(maxUpper + 1);
  for (let d = 0; d <= maxUpper; d++) {
    _rollTable[d] = diceSumPmf(MODEL.N_ROLLS, d);
  }
  return _rollTable;
}

// Compute a pirate's quantized score PMF
function pirateScorePmf(pirateId, foodIds) {
  const rollTable = getRollTable();
  const pirate = PIRATES[pirateId - 1];
  const { nf, na } = courseCounts(pirateId, foodIds);

  const rawWo = Math.floor((MODEL.MAX_WEIGHT - Math.min(pirate.weight, MODEL.MAX_WEIGHT)) / 2);
  const wo = MODEL.MAX_EFFECT > 0 ? Math.min(rawWo, MODEL.MAX_EFFECT) : rawWo;

  // Allergy damage PMF
  const dmgPmf = (na > 0 && wo > 0) ? diceSumPmf(na, wo) : [1.0];

  const maxRawScore = MODEL.N_ROLLS * (rollTable.length - 1);
  const rawPmf = new Float64Array(maxRawScore + 1);

  for (let dmgVal = 0; dmgVal < dmgPmf.length; dmgVal++) {
    const dp = dmgPmf[dmgVal];
    if (dp < 1e-15) continue;

    // Die size from strength
    let upper = MODEL.BASE > pirate.strength ? MODEL.BASE - pirate.strength : 1;
    if (upper < 1) upper = 1;

    // Iterative fav reduction
    for (let i = 0; i < nf; i++) {
      const red = Math.floor(upper / MODEL.FAV_DIV);
      upper = Math.max(upper - red, 1);
    }

    // Allergy damage AFTER fav
    upper += dmgVal;
    if (upper < 1) upper = 1;

    if (upper < rollTable.length) {
      const rpmf = rollTable[upper];
      for (let k = 0; k < rpmf.length; k++) {
        if (rpmf[k] > 0 && k < rawPmf.length) {
          rawPmf[k] += dp * rpmf[k];
        }
      }
    }
  }

  // Floor quantization
  const maxQ = Math.floor(maxRawScore / MODEL.DIVISOR);
  const qpmf = new Float64Array(maxQ + 1);
  for (let k = 0; k < rawPmf.length; k++) {
    if (rawPmf[k] < 1e-15) continue;
    const qk = Math.floor(k / MODEL.DIVISOR);
    if (qk <= maxQ) qpmf[qk] += rawPmf[k];
  }
  return qpmf;
}

// Compute win probabilities from 4 score PMFs. Later position wins ties.
function winProbsFromPmfs(pmfs) {
  const maxT = Math.max(...pmfs.map(p => p.length));

  // Survival functions: P(score > t)
  const surv = pmfs.map(pmf => {
    const s = new Float64Array(maxT + 1);
    let acc = 0;
    for (let t = pmf.length - 1; t >= 0; t--) {
      s[t] = acc;
      acc += pmf[t];
    }
    return s;
  });

  const f = (i, t) => t < pmfs[i].length ? pmfs[i][t] : 0;
  const s = (i, t) => t < surv[i].length ? surv[i][t] : 0;
  const g = (i, t) => t === 0 ? 1.0 : s(i, t - 1);

  const probs = [0, 0, 0, 0];
  for (let t = 0; t < maxT; t++) {
    // Later position wins ties
    probs[3] += f(3,t) * g(0,t) * g(1,t) * g(2,t);
    probs[2] += f(2,t) * g(0,t) * g(1,t) * s(3,t);
    probs[1] += f(1,t) * g(0,t) * s(2,t) * s(3,t);
    probs[0] += f(0,t) * s(1,t) * s(2,t) * s(3,t);
  }
  return probs;
}

// Compute win probabilities for an arena
// pirateIds: array of 4 pirate IDs (1-indexed)
// foodIds: array of 10 food IDs (1-indexed)
function arenaWinProbs(pirateIds, foodIds) {
  const pmfs = pirateIds.map(pid => pirateScorePmf(pid, foodIds));
  return winProbsFromPmfs(pmfs);
}

// ==================== Round Processing ====================

// Parse round JSON from neofood.club API into our format
function parseRound(roundData) {
  const arenas = [];
  for (let a = 0; a < 5; a++) {
    const pirateIds = roundData.pirates[a]; // 4 pirate IDs (1-indexed)
    const openingOdds = roundData.openingOdds[a].slice(1); // skip leading 1
    const currentOdds = roundData.currentOdds[a].slice(1);
    const foodIds = roundData.foods[a]; // 10 food IDs (1-indexed)
    arenas.push({ pirateIds, openingOdds, currentOdds, foodIds });
  }
  return arenas;
}

// Compute win probabilities for all 5 arenas
function computeAllProbs(arenas) {
  return arenas.map(arena => arenaWinProbs(arena.pirateIds, arena.foodIds));
}

// ==================== Betting Strategy ====================

// Generate bets using the current_exploit strategy:
// Anchors: opening_odds=2 with model p >= min2sProb, OR current_odds >= opening + minJump
// All payouts use current odds. Only keep bets with EV >= 1.0. Top N by EV.
function generateBets(arenas, probs, options = {}) {
  const {
    maxBets = 10,
    maxPayoutRatio = 60, // max payout multiplier per bet
    min2sProb = 0.55,
    minJump = 1,
  } = options;

  const n = arenas.length;

  // Precompute anchors and jump status
  const pirateIsJump = arenas.map((arena, ai) =>
    arena.pirateIds.map((pid, pi) => {
      const curOdds = arena.currentOdds[pi];
      const openOdds = arena.openingOdds[pi];
      return curOdds >= openOdds + minJump;
    })
  );

  const pirateIsAnchor = arenas.map((arena, ai) =>
    arena.pirateIds.map((pid, pi) => {
      const prob = probs[ai][pi];
      const openOdds = arena.openingOdds[pi];
      return (openOdds === 2 && prob >= min2sProb) || pirateIsJump[ai][pi];
    })
  );

  const hasAnchor = pirateIsAnchor.map(a => a.some(x => x));

  const possibleBets = [];

  for (let mask = 1; mask < (1 << n); mask++) {
    const arenaIndices = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) arenaIndices.push(i);
    }

    // Must include at least one arena with an anchor
    if (!arenaIndices.some(i => hasAnchor[i])) continue;

    // Enumerate all pirate combinations for selected arenas
    const comboIndices = new Array(arenaIndices.length).fill(0);
    while (true) {
      // Check: at least one pirate in combo is an anchor
      const comboHasAnchor = arenaIndices.some((ai, j) => pirateIsAnchor[ai][comboIndices[j]]);

      if (comboHasAnchor) {
        let winProb = 1;
        let payout = 1;
        const selections = [];

        for (let j = 0; j < arenaIndices.length; j++) {
          const ai = arenaIndices[j];
          const pi = comboIndices[j];
          // Floor-for-jumped: use odds-maker floor 1/(opening+1) for jumped pirates
          const prob = pirateIsJump[ai][pi]
            ? 1.0 / (arenas[ai].openingOdds[pi] + 1)
            : probs[ai][pi];
          winProb *= prob;
          payout = Math.min(payout * arenas[ai].currentOdds[pi], maxPayoutRatio);
          selections.push({
            arena: ai,
            pirateIdx: pi,
            pirateId: arenas[ai].pirateIds[pi],
            pirateName: PIRATES[arenas[ai].pirateIds[pi] - 1].name,
            openingOdds: arenas[ai].openingOdds[pi],
            currentOdds: arenas[ai].currentOdds[pi],
          });
        }

        const ev = winProb * payout;
        if (ev >= 1.0) {
          possibleBets.push({ selections, winProb, payout, ev });
        }
      }

      // Advance combo indices
      let carry = true;
      for (let j = arenaIndices.length - 1; j >= 0; j--) {
        if (carry) {
          comboIndices[j]++;
          if (comboIndices[j] >= 4) {
            comboIndices[j] = 0;
          } else {
            carry = false;
          }
        }
      }
      if (carry) break;
    }
  }

  // Sort by EV descending, take top N
  possibleBets.sort((a, b) => b.ev - a.ev);
  return possibleBets.slice(0, maxBets);
}

// ==================== Exports ====================

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    MODEL, PIRATES, FOODS, ARENA_NAMES,
    courseCounts, diceSumPmf, getRollTable, pirateScorePmf,
    winProbsFromPmfs, arenaWinProbs,
    parseRound, computeAllProbs, generateBets,
  };
}
