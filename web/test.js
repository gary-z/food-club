const { courseCounts, arenaWinProbs, PIRATES } = require('./foodclub.js');

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

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
