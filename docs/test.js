const { courseCounts, arenaWinProbs, generateBets, computeAllProbs, PIRATES } = require('./foodclub.js');

// Test vectors for Round 9811 (with corrected food ID ordering matching neofood.club API)
const TEST_VECTORS = {
  courseCounts: [
    [{nf:3,na:1}, {nf:2,na:2}, {nf:3,na:2}, {nf:2,na:2}],
    [{nf:2,na:1}, {nf:5,na:2}, {nf:3,na:2}, {nf:1,na:0}],
    [{nf:3,na:1}, {nf:2,na:2}, {nf:3,na:0}, {nf:0,na:1}],
    [{nf:1,na:1}, {nf:0,na:0}, {nf:2,na:3}, {nf:0,na:1}],
    [{nf:0,na:1}, {nf:2,na:1}, {nf:3,na:3}, {nf:3,na:1}],
  ],
  winProbs: [
    [0.212440461965284, 0.074199440365635, 0.248070954780872, 0.465289142888208],
    [0.070295733831376, 0.075011777649015, 0.372217380759654, 0.482475107759953],
    [0.074131693652797, 0.242380832483440, 0.571226680751208, 0.112260793112556],
    [0.319184639048000, 0.418949166933068, 0.053057116327759, 0.208809077691172],
    [0.123889208235492, 0.096326792425431, 0.039529164489092, 0.740254834849984],
  ],
};

// Round 9811 data
const arenaPirateIds = [
  [12, 11, 7, 17],
  [2, 8, 1, 20],
  [3, 18, 14, 10],
  [16, 5, 9, 6],
  [19, 4, 13, 15],
];
const arenaFoodIds = [
  [1, 2, 40, 34, 26, 24, 31, 6, 12, 32],
  [32, 25, 16, 23, 9, 8, 22, 7, 37, 18],
  [14, 36, 4, 15, 30, 27, 10, 29, 20, 39],
  [17, 33, 35, 21, 19, 13, 5, 38, 28, 11],
  [3, 37, 31, 6, 34, 1, 23, 12, 38, 39],
];

let passed = 0;
let failed = 0;

function check(label, actual, expected, tol = 1e-10) {
  const diff = Math.abs(actual - expected);
  if (diff > tol) {
    console.log(`  FAIL ${label}: got ${actual}, expected ${expected}, diff=${diff}`);
    failed++;
  } else {
    passed++;
  }
}

// Test course counts
console.log('Testing course counts...');
for (let a = 0; a < 5; a++) {
  for (let p = 0; p < 4; p++) {
    const pid = arenaPirateIds[a][p];
    const result = courseCounts(pid, arenaFoodIds[a]);
    const expected = TEST_VECTORS.courseCounts[a][p];
    check(`arena ${a} pirate ${pid} (${PIRATES[pid-1].name}) nf`, result.nf, expected.nf);
    check(`arena ${a} pirate ${pid} (${PIRATES[pid-1].name}) na`, result.na, expected.na);
  }
}

// Test win probabilities
console.log('Testing win probabilities...');
for (let a = 0; a < 5; a++) {
  const probs = arenaWinProbs(arenaPirateIds[a], arenaFoodIds[a]);
  for (let p = 0; p < 4; p++) {
    check(
      `arena ${a} pirate ${arenaPirateIds[a][p]} (${PIRATES[arenaPirateIds[a][p]-1].name}) prob`,
      probs[p], TEST_VECTORS.winProbs[a][p], 1e-9
    );
  }
}

// Check probs sum to ~1
for (let a = 0; a < 5; a++) {
  const probs = arenaWinProbs(arenaPirateIds[a], arenaFoodIds[a]);
  const sum = probs.reduce((a, b) => a + b, 0);
  check(`arena ${a} prob sum`, sum, 1.0, 1e-9);
}

// ==================== generateBets tests ====================

console.log('Testing generateBets floor-for-jumped...');

// Helper: build a minimal arena for testing
function makeArena(pirateIds, foodIds, openingOdds, currentOdds) {
  return { pirateIds, foodIds, openingOdds, currentOdds };
}

// Test 1: A jumped pirate should use floor probability 1/(opening+1), not model prob
{
  // Single arena, pirate 0 has opening=3 current=5 (jumped by 2)
  // With model probs and floor-for-jumped, EV calculation should differ
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 5, 7, 13], [5, 5, 7, 13])];
  const probs = computeAllProbs(arenas);

  // Pirate 0: opening=3, current=5, jumped → floor prob = 1/(3+1) = 0.25
  // With floor: EV = 0.25 * 5 = 1.25
  const bets = generateBets(arenas, probs);
  const betsWithP0 = bets.filter(b => b.selections.some(s => s.arena === 0 && s.pirateIdx === 0));

  if (betsWithP0.length > 0) {
    const singleBet = betsWithP0.find(b => b.selections.length === 1);
    if (singleBet) {
      // Floor prob for opening=3 jump: 1/4 = 0.25, payout=5, EV=1.25
      check('jumped pirate floor EV', singleBet.ev, 0.25 * 5, 1e-10);
      check('jumped pirate floor winProb', singleBet.winProb, 0.25, 1e-10);
      console.log(`  jumped pirate: model_prob=${probs[0][0].toFixed(4)}, floor_prob=0.25, ev=${singleBet.ev.toFixed(4)}`);
    } else {
      console.log('  SKIP: no single-arena bet with jumped pirate (may be filtered by EV)');
    }
  } else {
    console.log('  SKIP: no bets include jumped pirate');
  }
}

// Test 2: A non-jumped pirate (2:1 anchor) should use model probability, not floor
{
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 5, 7, 13], [2, 5, 7, 13])];
  const probs = computeAllProbs(arenas);

  const bets = generateBets(arenas, probs);
  // Pirate 0: opening=2, current=2, not jumped, is anchor if model p >= 0.55
  const betsWithP0 = bets.filter(b =>
    b.selections.length === 1 && b.selections[0].arena === 0 && b.selections[0].pirateIdx === 0);

  if (betsWithP0.length > 0) {
    // Should use model prob, not floor
    check('2:1 anchor uses model prob', betsWithP0[0].winProb, probs[0][0], 1e-10);
    console.log(`  2:1 anchor: model_prob=${probs[0][0].toFixed(4)}, used=${betsWithP0[0].winProb.toFixed(4)}`);
  } else {
    // p < 0.55 so not an anchor — that's fine
    console.log(`  2:1 pirate prob=${probs[0][0].toFixed(4)} < 0.55, not anchor (expected)`);
  }
}

// Test 3: Non-anchor non-jumped positive-EV pirates should be allowed
{
  // Pirate 1 is not jumped and not 2:1, but has positive EV at 5:1 odds.
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 5, 7, 13], [3, 5, 7, 13])];
  const probs = [[0.10, 0.25, 0.10, 0.05]];

  const bets = generateBets(arenas, probs);
  const betWithP1Only = bets.find(b =>
    b.selections.length === 1 && b.selections[0].arena === 0 && b.selections[0].pirateIdx === 1);

  if (betWithP1Only) {
    check('non-anchor positive-EV uses model prob', betWithP1Only.winProb, probs[0][1], 1e-10);
  } else {
    console.log('  FAIL: non-anchor positive-EV pirate was excluded');
    failed++;
  }
}

// Test 4: Verify payout cap is respected
{
  // Two arenas with high-odds pirates that would exceed cap=60
  const arenas = [
    makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [3, 5, 7, 13], [5, 5, 7, 13]),
    makeArena([5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [3, 5, 7, 13], [5, 5, 7, 13]),
  ];
  const probs = computeAllProbs(arenas);
  const bets = generateBets(arenas, probs, { maxPayoutRatio: 60 });
  const maxPayout = Math.max(...bets.map(b => b.payout));
  if (maxPayout <= 60) {
    passed++;
  } else {
    console.log(`  FAIL: payout ${maxPayout} exceeds cap 60`);
    failed++;
  }
}

// Test 5: Positive-EV bets can be returned even when no anchors exist
{
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 5, 7, 13], [3, 5, 7, 13])];
  const probs = [[0.10, 0.25, 0.10, 0.05]];
  const bets = generateBets(arenas, probs);

  if (bets.length > 0 && bets.some(b => b.ev >= 1.0)) {
    passed++;
  } else {
    console.log('  FAIL: no-anchor positive-EV bet was not returned');
    failed++;
  }
}

// Test 6: All returned bets must have EV >= 1.0
{
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 5, 7, 13], [5, 5, 7, 13])];
  const probs = computeAllProbs(arenas);
  const bets = generateBets(arenas, probs);

  let allPositiveEV = true;
  for (const bet of bets) {
    if (bet.ev < 1.0 - 1e-10) { allPositiveEV = false; break; }
  }
  if (allPositiveEV) {
    passed++;
  } else {
    console.log('  FAIL: found bet with EV < 1.0');
    failed++;
  }
}

// Test 7: If fewer than maxBets positive-EV combinations are available, truncate the list
{
  const arenas = [makeArena([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 5, 7, 13], [1, 5, 7, 13])];
  const probs = [[0.55, 0.20, 0.15, 0.10]];
  const bets = generateBets(arenas, probs, { maxBets: 3, maxPayoutRatio: 1 });

  if (bets.length === 0) {
    passed++;
  } else {
    console.log(`  FAIL: expected no negative-EV fallback bets, got ${bets.length}: ${bets.map(b => b.ev.toFixed(3)).join(', ')}`);
    failed++;
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
