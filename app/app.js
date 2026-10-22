/**
 * FootyPredict: Dual-Stage Match Simulation & Inference Engine
 * Client-side execution of GBDT Classification and Bivariate Poisson Scoreline Formulation.
 */

const TEAMS_DATABASE = {
  "Premier League": [
    { name: "Arsenal", elo: 1890, att: 2.15, defn: 0.75, form: ["L", "W", "W", "W", "W"], gd: 6, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/359.png" },
    { name: "Manchester City", elo: 1920, att: 2.30, defn: 0.80, form: ["W", "W", "W", "W", "W"], gd: 10, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/382.png" },
    { name: "Liverpool", elo: 1905, att: 2.25, defn: 0.82, form: ["W", "D", "W", "D", "D"], gd: 4, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/364.png" },
    { name: "Chelsea", elo: 1780, att: 1.75, defn: 1.10, form: ["L", "D", "L", "W", "W"], gd: 0, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/363.png" },
    { name: "Tottenham", elo: 1765, att: 1.80, defn: 1.25, form: ["L", "D", "D", "L", "L"], gd: -4, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/367.png" },
    { name: "Aston Villa", elo: 1770, att: 1.70, defn: 1.15, form: ["W", "L", "D", "L", "L"], gd: -2, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/362.png" },
    { name: "Newcastle United", elo: 1750, att: 1.65, defn: 1.18, form: ["W", "L", "D", "W", "D"], gd: 2, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/361.png" },
    { name: "Manchester United", elo: 1740, att: 1.55, defn: 1.22, form: ["D", "L", "D", "W", "L"], gd: -2, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/360.png" }
  ],
  "La Liga": [
    { name: "Real Madrid", elo: 1930, att: 2.35, defn: 0.78, form: ["W", "L", "W", "W", "W"], gd: 6, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/86.png" },
    { name: "Barcelona", elo: 1910, att: 2.40, defn: 0.85, form: ["W", "W", "W", "W", "W"], gd: 10, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/83.png" },
    { name: "Atletico Madrid", elo: 1820, att: 1.75, defn: 0.80, form: ["W", "L", "W", "D", "W"], gd: 4, rest: 7, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/1068.png" },
    { name: "Athletic Bilbao", elo: 1740, att: 1.55, defn: 1.05, form: ["D", "W", "W", "L", "L"], gd: 0, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/93.png" },
    { name: "Real Sociedad", elo: 1725, att: 1.45, defn: 1.10, form: ["W", "D", "W", "L", "L"], gd: 0, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/89.png" }
  ],
  "Bundesliga": [
    { name: "Bayern Munich", elo: 1915, att: 2.50, defn: 0.90, form: ["D", "W", "W", "D", "W"], gd: 6, rest: 7, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/132.png" },
    { name: "Bayer Leverkusen", elo: 1860, att: 2.10, defn: 0.92, form: ["W", "W", "D", "W", "L"], gd: 4, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/131.png" },
    { name: "Borussia Dortmund", elo: 1800, att: 1.95, defn: 1.20, form: ["D", "W", "W", "W", "W"], gd: 8, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/124.png" },
    { name: "RB Leipzig", elo: 1785, att: 1.85, defn: 1.15, form: ["W", "L", "W", "L", "W"], gd: 2, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/11420.png" }
  ],
  "Serie A": [
    { name: "Inter Milan", elo: 1875, att: 2.05, defn: 0.76, form: ["D", "W", "W", "W", "W"], gd: 8, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/110.png" },
    { name: "Juventus", elo: 1810, att: 1.65, defn: 0.82, form: ["W", "L", "D", "W", "W"], gd: 4, rest: 7, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/111.png" },
    { name: "AC Milan", elo: 1800, att: 1.80, defn: 1.15, form: ["W", "D", "D", "W", "W"], gd: 4, rest: 5, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/103.png" },
    { name: "Napoli", elo: 1795, att: 1.75, defn: 1.05, form: ["D", "W", "L", "L", "W"], gd: 0, rest: 6, logo: "https://a.espncdn.com/i/teamlogos/soccer/500/114.png" }
  ]
};

// Mathematical helper for Poisson probability: P(k; λ) = (λ^k * e^-λ) / k!
function factorial(n) {
  let res = 1;
  for (let i = 2; i <= n; i++) res *= i;
  return res;
}

function poisson(k, lambda) {
  return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
}

// DOM Elements
const leagueSelect = document.getElementById("league-select");
const homeTeamSelect = document.getElementById("home-team-select");
const awayTeamSelect = document.getElementById("away-team-select");
const swapBtn = document.getElementById("swap-teams-btn");
const simulateBtn = document.getElementById("simulate-btn");
const presetChips = document.querySelectorAll(".preset-chip");

// Populating Team Selectors
function populateTeams(leagueName, defaultHome = null, defaultAway = null) {
  const teams = TEAMS_DATABASE[leagueName];
  homeTeamSelect.innerHTML = "";
  awayTeamSelect.innerHTML = "";

  teams.forEach((team, idx) => {
    const optHome = document.createElement("option");
    optHome.value = team.name;
    optHome.textContent = team.name;
    homeTeamSelect.appendChild(optHome);

    const optAway = document.createElement("option");
    optAway.value = team.name;
    optAway.textContent = team.name;
    awayTeamSelect.appendChild(optAway);
  });

  if (defaultHome && teams.some(t => t.name === defaultHome)) {
    homeTeamSelect.value = defaultHome;
  } else {
    homeTeamSelect.selectedIndex = 0;
  }

  if (defaultAway && teams.some(t => t.name === defaultAway)) {
    awayTeamSelect.value = defaultAway;
  } else {
    awayTeamSelect.selectedIndex = Math.min(1, teams.length - 1);
  }
}

function getTeam(name) {
  for (const league in TEAMS_DATABASE) {
    const found = TEAMS_DATABASE[league].find(t => t.name === name);
    if (found) return found;
  }
  return null;
}

// Core Dual-Stage Computation
function computeMatchInference() {
  const homeName = homeTeamSelect.value;
  const awayName = awayTeamSelect.value;

  if (homeName === awayName) {
    alert("Please select two distinct teams for the match simulation.");
    return;
  }

  const home = getTeam(homeName);
  const away = getTeam(awayName);

  if (!home || !away) return;

  // 1. Feature Engineering
  const homeAdvantageElo = 65; // Statistical home pitch advantage
  const eloDiff = (home.elo - away.elo) + homeAdvantageElo;

  const restAdvantageHome = home.rest >= 6 ? 0.05 : -0.05;
  const restAdvantageAway = away.rest >= 6 ? 0.05 : -0.05;

  // 2. Stage 2: Bivariate Poisson Expected Goals (xG)
  const lamHome = Math.max(0.4, (home.att * 0.70 + away.defn * 0.45) + 0.28 + restAdvantageHome + (eloDiff / 650.0) * 0.4);
  const lamAway = Math.max(0.3, (away.att * 0.65 + home.defn * 0.40) + restAdvantageAway - (eloDiff / 650.0) * 0.35);

  // Generate Scoreline Matrix (0-0 to 5-5)
  const maxGoals = 6;
  const scoreMatrix = [];
  let sumPoisHome = 0;
  let sumPoisDraw = 0;
  let sumPoisAway = 0;
  const flatScores = [];

  for (let h = 0; h < maxGoals; h++) {
    const pH = poisson(h, lamHome);
    for (let a = 0; a < maxGoals; a++) {
      const pA = poisson(a, lamAway);
      const prob = pH * pA;
      scoreMatrix.push({ h, a, prob });
      flatScores.push({ score: `${h} - ${a}`, h, a, prob });

      if (h > a) sumPoisHome += prob;
      else if (h === a) sumPoisDraw += prob;
      else sumPoisAway += prob;
    }
  }

  // Normalize Poisson outcomes
  const totalPois = sumPoisHome + sumPoisDraw + sumPoisAway;
  sumPoisHome /= totalPois;
  sumPoisDraw /= totalPois;
  sumPoisAway /= totalPois;

  // 3. Stage 1: GBDT Sigmoid Outcome Classification
  // Logistic link on EloDiff and Form Differential
  const formDiff = (home.form.filter(x => x === 'W').length * 3) - (away.form.filter(x => x === 'W').length * 3);
  const zHome = (eloDiff / 220.0) + (formDiff * 0.08) + 0.35;
  const zAway = -(eloDiff / 250.0) - (formDiff * 0.08) - 0.25;

  let clfHome = 1.0 / (1.0 + Math.exp(-zHome));
  let clfAway = 1.0 / (1.0 + Math.exp(-zAway));
  let clfDraw = Math.max(0.18, 1.0 - (clfHome + clfAway) * 0.55);

  const clfTotal = clfHome + clfDraw + clfAway;
  clfHome /= clfTotal;
  clfDraw /= clfTotal;
  clfAway /= clfTotal;

  // 4. Dual-Stage Ensemble Blending (55% GBDT Classifier, 45% Poisson Formulation)
  const alpha = 0.55;
  let finalHomeProb = alpha * clfHome + (1 - alpha) * sumPoisHome;
  let finalDrawProb = alpha * clfDraw + (1 - alpha) * sumPoisDraw;
  let finalAwayProb = alpha * clfAway + (1 - alpha) * sumPoisAway;

  const finalTotal = finalHomeProb + finalDrawProb + finalAwayProb;
  finalHomeProb /= finalTotal;
  finalDrawProb /= finalTotal;
  finalAwayProb /= finalTotal;

  // Sort scorelines by probability
  flatScores.sort((a, b) => b.prob - a.prob);
  const topScorelines = flatScores.slice(0, 5);

  // Update UI State
  updateUI({
    home,
    away,
    eloDiff,
    finalHomeProb,
    finalDrawProb,
    finalAwayProb,
    lamHome,
    lamAway,
    topScorelines
  });
}

function updateUI(data) {
  const {
    home, away, eloDiff,
    finalHomeProb, finalDrawProb, finalAwayProb,
    lamHome, lamAway, topScorelines
  } = data;

  // Update Team Names with official logos
  document.getElementById("stat-home-name").innerHTML = `<img src="${home.logo}" alt="${home.name}" class="inline-crest" /> ${home.name}`;
  document.getElementById("stat-away-name").innerHTML = `<img src="${away.logo}" alt="${away.name}" class="inline-crest" /> ${away.name}`;
  document.getElementById("xg-home-team").innerHTML = `<img src="${home.logo}" alt="${home.name}" class="inline-crest" /> ${home.name} xG`;
  document.getElementById("xg-away-team").innerHTML = `<img src="${away.logo}" alt="${away.name}" class="inline-crest" /> ${away.name} xG`;
  document.getElementById("diag-home-name").innerHTML = `<img src="${home.logo}" alt="${home.name}" class="inline-crest" /> ${home.name}`;
  document.getElementById("diag-away-name").innerHTML = `<img src="${away.logo}" alt="${away.name}" class="inline-crest" /> ${away.name}`;

  // Update Elo & Ratings
  document.getElementById("diag-home-elo").textContent = `${home.elo} Elo`;
  document.getElementById("diag-away-elo").textContent = `${away.elo} Elo`;
  const eloDeltaElem = document.getElementById("val-elo-delta");
  eloDeltaElem.textContent = (eloDiff >= 0 ? `+${Math.round(eloDiff)}` : `${Math.round(eloDiff)}`);
  eloDeltaElem.style.color = eloDiff >= 0 ? "var(--accent-pitch)" : "var(--accent-away)";

  // Update Probabilities
  const pHPct = (finalHomeProb * 100).toFixed(1);
  const pDPct = (finalDrawProb * 100).toFixed(1);
  const pAPct = (finalAwayProb * 100).toFixed(1);

  document.getElementById("val-home-prob").textContent = `${pHPct}%`;
  document.getElementById("val-draw-prob").textContent = `${pDPct}%`;
  document.getElementById("val-away-prob").textContent = `${pAPct}%`;

  document.getElementById("val-home-odds").textContent = `${(1 / finalHomeProb).toFixed(2)} Fair Odds`;
  document.getElementById("val-draw-odds").textContent = `${(1 / finalDrawProb).toFixed(2)} Fair Odds`;
  document.getElementById("val-away-odds").textContent = `${(1 / finalAwayProb).toFixed(2)} Fair Odds`;

  // Segmented Bar Widths
  document.getElementById("bar-home").style.width = `${pHPct}%`;
  document.getElementById("bar-draw").style.width = `${pDPct}%`;
  document.getElementById("bar-away").style.width = `${pAPct}%`;

  // Dominant Outcome Badge
  const domBadge = document.getElementById("dominant-outcome");
  if (finalHomeProb > 0.48) {
    domBadge.textContent = `${home.name} Favored`;
    domBadge.style.color = "var(--accent-pitch)";
    domBadge.style.borderColor = "rgba(16, 185, 129, 0.3)";
  } else if (finalAwayProb > 0.42) {
    domBadge.textContent = `${away.name} Favored`;
    domBadge.style.color = "var(--accent-away)";
    domBadge.style.borderColor = "rgba(56, 189, 248, 0.3)";
  } else {
    domBadge.textContent = "High Variance / Even Contest";
    domBadge.style.color = "var(--accent-draw)";
    domBadge.style.borderColor = "rgba(245, 158, 11, 0.3)";
  }

  // Expected Goals (xG)
  document.getElementById("val-home-xg").textContent = lamHome.toFixed(2);
  document.getElementById("val-away-xg").textContent = lamAway.toFixed(2);

  // Top 5 Scorelines
  const scorelinesList = document.getElementById("scorelines-list");
  scorelinesList.innerHTML = "";
  const maxScoreProb = topScorelines[0].prob;

  topScorelines.forEach(item => {
    const row = document.createElement("div");
    row.className = "scoreline-row";

    const pct = (item.prob * 100).toFixed(1);
    const fillWidth = Math.min(100, Math.round((item.prob / maxScoreProb) * 100));

    row.innerHTML = `
      <span class="score-tag">${item.score}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width: ${fillWidth}%;"></div>
      </div>
      <span class="score-pct">${pct}%</span>
    `;
    scorelinesList.appendChild(row);
  });

  // Render Form Badges
  renderFormBadges("home-form-badges", home.form);
  renderFormBadges("away-form-badges", away.form);

  document.getElementById("home-gd-val").textContent = (home.gd > 0 ? `+${home.gd}` : `${home.gd}`);
  document.getElementById("away-gd-val").textContent = (away.gd > 0 ? `+${away.gd}` : `${away.gd}`);
  document.getElementById("home-rest-val").textContent = `${home.rest} Days`;
  document.getElementById("away-rest-val").textContent = `${away.rest} Days`;

  // Prepare dynamic randomized match simulation for this fixture
  currentSimData = generateMatchSimulation(home, away, lamHome, lamAway, finalHomeProb, finalDrawProb, finalAwayProb);
  setupPreMatchScoreboard(currentSimData);

  // Update squad view buttons
  document.getElementById("squad-btn-home-name").textContent = `${home.name} Squad`;
  document.getElementById("squad-btn-away-name").textContent = `${away.name} Squad`;
  renderSquadCards(currentSquadTeam === "away" ? away.name : home.name);
}

function renderFormBadges(containerId, formArr) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  formArr.forEach(result => {
    const b = document.createElement("span");
    b.className = `form-badge badge-${result.toLowerCase()}`;
    b.textContent = result;
    container.appendChild(b);
  });
}

// Team Crest Code Helper
function getTeamCode(name) {
  const codes = {
    "Arsenal": "ARS", "Manchester City": "MCI", "Liverpool": "LIV", "Chelsea": "CHE",
    "Tottenham": "TOT", "Aston Villa": "AVL", "Newcastle United": "NEW", "Manchester United": "MUN",
    "Real Madrid": "RMA", "Barcelona": "BAR", "Atletico Madrid": "ATM", "Athletic Bilbao": "ATH", "Real Sociedad": "RSO",
    "Bayern Munich": "BAY", "Bayer Leverkusen": "B04", "Borussia Dortmund": "BVB", "RB Leipzig": "RBL",
    "Inter Milan": "INT", "Juventus": "JUV", "AC Milan": "ACM", "Napoli": "NAP"
  };
  return codes[name] || name.substring(0, 3).toUpperCase();
}

// Fallback Squad Generator if needed
function getSquad(teamName) {
  if (typeof SQUADS_DATABASE !== "undefined" && SQUADS_DATABASE[teamName]) {
    return SQUADS_DATABASE[teamName];
  }
  return [
    { name: `${teamName} Keeper`, num: 1, pos: "GK", role: "Goalkeeper", rating: 8.0, goals: 0, assists: 0, cleanSheets: 10, trait: "Shot Stopper" },
    { name: `${teamName} Captain`, num: 4, pos: "DF", role: "Centre Back", rating: 8.1, goals: 2, assists: 1, trait: "Commanding Leader" },
    { name: `${teamName} Defender`, num: 5, pos: "DF", role: "Centre Back", rating: 7.9, goals: 1, assists: 0, trait: "Strong Tackler" },
    { name: `${teamName} Fullback`, num: 2, pos: "DF", role: "Right Back", rating: 7.8, goals: 1, assists: 4, trait: "Pacy Runner" },
    { name: `${teamName} Leftback`, num: 3, pos: "DF", role: "Left Back", rating: 7.8, goals: 0, assists: 3, trait: "Solid Defending" },
    { name: `${teamName} Playmaker`, num: 8, pos: "MF", role: "Attacking Mid", rating: 8.4, goals: 8, assists: 10, trait: "Creative Spark" },
    { name: `${teamName} Anchor`, num: 6, pos: "MF", role: "Defensive Mid", rating: 8.1, goals: 3, assists: 2, trait: "Ball Winner" },
    { name: `${teamName} Winger L`, num: 11, pos: "FW", role: "Left Wing", rating: 8.3, goals: 11, assists: 7, trait: "Speed Merchant" },
    { name: `${teamName} Winger R`, num: 7, pos: "FW", role: "Right Wing", rating: 8.4, goals: 12, assists: 8, trait: "Cut-Inside Threat" },
    { name: `${teamName} Striker`, num: 9, pos: "FW", role: "Centre Forward", rating: 8.6, goals: 18, assists: 4, trait: "Lethal Finisher" },
    { name: `${teamName} Midfielder`, num: 10, pos: "MF", role: "Central Mid", rating: 8.0, goals: 5, assists: 5, trait: "Box to Box" }
  ];
}

// Sample Poisson random integer with Knuth's algorithm
function samplePoisson(lambda) {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1.0;
  do {
    k++;
    p *= Math.random();
  } while (p > L);
  return k - 1;
}

// -------------------------------------------------------------
// Interactive Live Match Simulation Engine
// -------------------------------------------------------------
let currentSimData = null;
let simIntervalId = null;
let isSimulating = false;
let currentSimMinute = 0;
let simSpeedMultiplier = 3; // Default 3x ticker

const goalDescriptions = [
  "unleashes an unstoppable first-time strike into the top corner!",
  "rises highest inside the box to power a thunderous header into the net!",
  "coolly slots the ball past the rushing goalkeeper from 12 yards out!",
  "beats two defenders with silky footwork and curls home a beauty!",
  "reacts quickest to the loose ball in the six-yard box and taps in!",
  "drives a ferocious low shot through a crowded penalty area!",
  "finishes off a scintillating team counter-attack with a calm finish!"
];

function pickWeightedScorer(squad) {
  const fw = squad.filter(p => p.pos === "FW");
  const mf = squad.filter(p => p.pos === "MF");
  const df = squad.filter(p => p.pos === "DF");

  const roll = Math.random();
  if (roll < 0.65 && fw.length) return fw[Math.floor(Math.random() * fw.length)];
  if (roll < 0.90 && mf.length) return mf[Math.floor(Math.random() * mf.length)];
  if (df.length) return df[Math.floor(Math.random() * df.length)];
  return squad[0];
}

function pickAssist(squad, scorer) {
  if (Math.random() > 0.85) return null; // Solo goal
  const teammates = squad.filter(p => p.name !== scorer.name && p.pos !== "GK");
  if (!teammates.length) return null;
  return teammates[Math.floor(Math.random() * teammates.length)];
}

function generateMatchSimulation(home, away, lamHome, lamAway, homeProb, drawProb, awayProb) {
  const homeSquad = getSquad(home.name);
  const awaySquad = getSquad(away.name);

  // Sample actual score from Poisson intensities with random variation
  let hGoals = samplePoisson(lamHome);
  let aGoals = samplePoisson(lamAway);

  // Realistic boundary
  hGoals = Math.min(5, Math.max(0, hGoals));
  aGoals = Math.min(5, Math.max(0, aGoals));

  const events = [];

  // Generate Home Goals
  for (let g = 0; g < hGoals; g++) {
    const minute = Math.floor(Math.random() * 88) + 2;
    const scorer = pickWeightedScorer(homeSquad);
    const assister = pickAssist(homeSquad, scorer);
    const descText = goalDescriptions[Math.floor(Math.random() * goalDescriptions.length)];

    events.push({
      minute,
      team: home.name,
      side: "home",
      type: "goal",
      icon: "⚽",
      title: `GOAL! ${scorer.name} (${scorer.num})`,
      desc: assister ? `${scorer.name} ${descText} (Assist: ${assister.name})` : `${scorer.name} ${descText} (Unassisted solo strike)`,
      player: scorer,
      assister: assister
    });
  }

  // Generate Away Goals
  for (let g = 0; g < aGoals; g++) {
    const minute = Math.floor(Math.random() * 88) + 2;
    const scorer = pickWeightedScorer(awaySquad);
    const assister = pickAssist(awaySquad, scorer);
    const descText = goalDescriptions[Math.floor(Math.random() * goalDescriptions.length)];

    events.push({
      minute,
      team: away.name,
      side: "away",
      type: "goal",
      icon: "⚽",
      title: `GOAL! ${scorer.name} (${scorer.num})`,
      desc: assister ? `${scorer.name} ${descText} (Assist: ${assister.name})` : `${scorer.name} ${descText} (Unassisted solo strike)`,
      player: scorer,
      assister: assister
    });
  }

  // Heroic Goalkeeper Saves
  const homeGk = homeSquad.find(p => p.pos === "GK") || homeSquad[0];
  const awayGk = awaySquad.find(p => p.pos === "GK") || awaySquad[0];

  if (Math.random() > 0.25) {
    const minSave = Math.floor(Math.random() * 82) + 6;
    events.push({
      minute: minSave,
      team: home.name,
      side: "home",
      type: "save",
      icon: "🧤",
      title: `Heroic Save! ${homeGk.name}`,
      desc: `${homeGk.name} makes a reflex fingertip save pushing the strike past the post.`,
      player: homeGk
    });
  }

  if (Math.random() > 0.3) {
    const minSave2 = Math.floor(Math.random() * 82) + 6;
    events.push({
      minute: minSave2,
      team: away.name,
      side: "away",
      type: "save",
      icon: "🧤",
      title: `Crucial Stop! ${awayGk.name}`,
      desc: `${awayGk.name} charges off the line to smother a dangerous 1-on-1 chance.`,
      player: awayGk
    });
  }

  // Yellow Cards
  if (Math.random() > 0.25) {
    const minCard = Math.floor(Math.random() * 75) + 15;
    const homeDefenders = homeSquad.filter(p => p.pos === "DF" || p.pos === "MF");
    const bookedPlayer = homeDefenders[Math.floor(Math.random() * homeDefenders.length)] || homeSquad[1];
    events.push({
      minute: minCard,
      team: home.name,
      side: "home",
      type: "card",
      icon: "🟨",
      title: `Yellow Card: ${bookedPlayer.name}`,
      desc: `Referee books ${bookedPlayer.name} for a late sliding challenge stopping a counter-attack.`,
      player: bookedPlayer
    });
  }

  if (Math.random() > 0.3) {
    const minCard2 = Math.floor(Math.random() * 75) + 20;
    const awayDefenders = awaySquad.filter(p => p.pos === "DF" || p.pos === "MF");
    const bookedPlayer = awayDefenders[Math.floor(Math.random() * awayDefenders.length)] || awaySquad[1];
    events.push({
      minute: minCard2,
      team: away.name,
      side: "away",
      type: "card",
      icon: "🟨",
      title: `Yellow Card: ${bookedPlayer.name}`,
      desc: `Tactical foul by ${bookedPlayer.name} bringing down the winger in transition.`,
      player: bookedPlayer
    });
  }

  // Woodwork strike
  if (Math.random() > 0.5) {
    const minWood = Math.floor(Math.random() * 85) + 8;
    const isHome = Math.random() > 0.5;
    const squad = isHome ? homeSquad : awaySquad;
    const attacker = pickWeightedScorer(squad);
    events.push({
      minute: minWood,
      team: isHome ? home.name : away.name,
      side: isHome ? "home" : "away",
      type: "woodwork",
      icon: "🥅",
      title: `Rattles the Crossbar! ${attacker.name}`,
      desc: `A blistering effort from ${attacker.name} beats the keeper but smashes against the bar!`,
      player: attacker
    });
  }

  // Sort events chronologically
  events.sort((a, b) => a.minute - b.minute);

  // Compute Dynamic Match Ratings for Players
  const allMatchPlayers = [];
  const processSquadRatings = (squad, goalsScored, goalsConceded, teamName) => {
    squad.forEach(p => {
      let r = p.rating;
      const goalsInMatch = events.filter(e => e.type === "goal" && e.player && e.player.name === p.name).length;
      r += goalsInMatch * 0.95;
      const assistsInMatch = events.filter(e => e.type === "goal" && e.assister && e.assister.name === p.name).length;
      r += assistsInMatch * 0.55;
      if (goalsConceded === 0 && (p.pos === "GK" || p.pos === "DF")) r += 0.5;
      if (goalsConceded >= 2 && (p.pos === "GK" || p.pos === "DF")) r -= 0.3;
      const savesInMatch = events.filter(e => e.type === "save" && e.player && e.player.name === p.name).length;
      r += savesInMatch * 0.4;
      const cardsInMatch = events.filter(e => e.type === "card" && e.player && e.player.name === p.name).length;
      r -= cardsInMatch * 0.35;
      r = Math.min(9.8, Math.max(5.5, r));

      allMatchPlayers.push({
        name: p.name,
        num: p.num,
        pos: p.pos,
        team: teamName,
        matchRating: parseFloat(r.toFixed(1)),
        baseRating: p.rating,
        goalsInMatch,
        assistsInMatch,
        seasonGoals: p.goals,
        seasonAssists: p.assists,
        trait: p.trait
      });
    });
  };

  processSquadRatings(homeSquad, hGoals, aGoals, home.name);
  processSquadRatings(awaySquad, aGoals, hGoals, away.name);

  allMatchPlayers.sort((a, b) => b.matchRating - a.matchRating);
  const motm = allMatchPlayers[0];

  const homePoss = Math.round(48 + (home.elo - away.elo) / 35 + 3);
  const clampedPoss = Math.max(38, Math.min(65, homePoss));
  const hShots = Math.round(lamHome * 5.2 + 3);
  const hTarget = Math.max(hGoals, Math.round(hShots * 0.42));
  const aShots = Math.round(lamAway * 4.8 + 2);
  const aTarget = Math.max(aGoals, Math.round(aShots * 0.40));

  return {
    home,
    away,
    lamHome,
    lamAway,
    homeProb,
    drawProb,
    awayProb,
    hGoals,
    aGoals,
    events,
    allMatchPlayers,
    motm,
    possHome: clampedPoss,
    possAway: 100 - clampedPoss,
    hShots,
    hTarget,
    aShots,
    aTarget
  };
}

function setupPreMatchScoreboard(simData) {
  if (simIntervalId) {
    clearInterval(simIntervalId);
    simIntervalId = null;
  }
  isSimulating = false;
  currentSimMinute = 0;

  const playBtn = document.getElementById("btn-play-sim");
  if (playBtn) {
    playBtn.classList.remove("simulating");
    const playText = document.getElementById("sim-play-text");
    if (playText) playText.textContent = "Play Live Match";
  }

  const clockBadge = document.getElementById("sim-clock");
  if (clockBadge) {
    clockBadge.textContent = "READY";
    clockBadge.classList.remove("live-active");
  }

  document.getElementById("sim-home-crest").innerHTML = `<img src="${simData.home.logo}" alt="${simData.home.name}" class="crest-img" />`;
  document.getElementById("sim-away-crest").innerHTML = `<img src="${simData.away.logo}" alt="${simData.away.name}" class="crest-img" />`;
  document.getElementById("sim-home-name").textContent = simData.home.name;
  document.getElementById("sim-away-name").textContent = simData.away.name;
  document.getElementById("sim-home-elo").textContent = `${simData.home.elo} Elo • Home`;
  document.getElementById("sim-away-elo").textContent = `${simData.away.elo} Elo • Away`;

  document.getElementById("sim-score-h").textContent = "0";
  document.getElementById("sim-score-a").textContent = "0";
  document.getElementById("sim-status-label").textContent = "Ready for Kick-Off — Click 'Play Live Match'";

  document.getElementById("sim-stat-xg").textContent = `${simData.lamHome.toFixed(2)} — ${simData.lamAway.toFixed(2)}`;
  document.getElementById("sim-stat-poss").textContent = `${simData.possHome}% — ${simData.possAway}%`;
  document.getElementById("sim-stat-shots").textContent = `0 (0) — 0 (0)`;

  // Progress Bar reset
  const progressFill = document.getElementById("sim-progress-fill");
  if (progressFill) progressFill.style.width = "0%";
  const markersContainer = document.getElementById("sim-progress-markers");
  if (markersContainer) markersContainer.innerHTML = "";

  // Timeline feed reset
  const feed = document.getElementById("match-timeline-feed");
  if (feed) {
    feed.innerHTML = `
      <div class="timeline-event-row home-event">
        <div class="event-min">KO</div>
        <div class="event-icon">⚽</div>
        <div class="event-details">
          <div class="event-title">Match Ready for Kick-Off</div>
          <div class="event-desc">Click <strong>'Play Live Match'</strong> to simulate real-time minute-by-minute action, or <strong>'Instant Result'</strong> to skip directly to full-time.</div>
        </div>
      </div>
    `;
  }
  document.getElementById("sim-events-count").textContent = `Ready (0/${simData.events.length} Events)`;

  // Placeholder MOTM
  const motmDisplay = document.getElementById("motm-hero-display");
  if (motmDisplay) {
    motmDisplay.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 13px;">
        ⏳ Match underway — Man of the Match and performance ratings unlock at the final whistle.
      </div>
    `;
  }

  const topList = document.getElementById("top-performers-list");
  if (topList) {
    topList.innerHTML = `
      <div style="padding: 14px; text-align: center; color: var(--text-muted); font-size: 12px;">
        Player match ratings will be calculated post-match.
      </div>
    `;
  }
}

function appendTimelineEvent(feed, ev) {
  const row = document.createElement("div");
  row.className = `timeline-event-row ${ev.side}-event`;
  row.innerHTML = `
    <div class="event-min">${ev.minute}'</div>
    <div class="event-icon">${ev.icon}</div>
    <div class="event-details">
      <div class="event-title">
        <span>${ev.title}</span>
        <span class="event-team-pill">${ev.team}</span>
      </div>
      <div class="event-desc">${ev.desc}</div>
    </div>
  `;
  feed.insertBefore(row, feed.firstChild);
}

function addProgressMarker(container, ev) {
  const marker = document.createElement("div");
  const pct = Math.min(98, Math.max(2, (ev.minute / 90) * 100));
  marker.style.left = `${pct}%`;
  marker.className = `sim-marker-dot ${ev.type === "goal" ? (ev.side === "home" ? "sim-marker-goal-home" : "sim-marker-goal-away") : "sim-marker-card"}`;
  marker.textContent = ev.icon;
  marker.title = `${ev.minute}' ${ev.title}`;
  container.appendChild(marker);
}

function triggerGoalAnimation(elem) {
  elem.classList.remove("goal-flash");
  void elem.offsetWidth; // trigger reflow
  elem.classList.add("goal-flash");
}

function renderMOTMAndPerformers(simData) {
  const motm = simData.motm;
  const motmDisplay = document.getElementById("motm-hero-display");
  if (motmDisplay && motm) {
    motmDisplay.innerHTML = `
      <div class="motm-player-info">
        <div class="motm-number-box">#${motm.num}</div>
        <div>
          <span class="motm-name">${motm.name}</span>
          <span class="motm-team-role">${motm.team} • ${motm.pos} (${motm.trait})</span>
          <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 3px;">
            Match Impact: ${motm.goalsInMatch > 0 ? `⚽ ${motm.goalsInMatch} Goals ` : ""}${motm.assistsInMatch > 0 ? `👟 ${motm.assistsInMatch} Assists ` : ""}Season: ${motm.seasonGoals}G / ${motm.seasonAssists}A
          </div>
        </div>
      </div>
      <div class="motm-rating-badge">
        <div class="motm-score">${motm.matchRating.toFixed(1)}</div>
        <div class="motm-caption">Official MOTM</div>
      </div>
    `;
  }

  const topList = document.getElementById("top-performers-list");
  if (topList) {
    topList.innerHTML = "";
    simData.allMatchPlayers.slice(0, 5).forEach(p => {
      const row = document.createElement("div");
      row.className = "performer-row";
      const ratingClass = p.matchRating >= 8.5 ? "rating-elite" : (p.matchRating >= 7.8 ? "rating-good" : "rating-avg");
      row.innerHTML = `
        <div class="performer-meta">
          <span class="performer-pos">${p.pos}</span>
          <div>
            <span class="performer-name">${p.name}</span>
            <span class="performer-team">${p.team}</span>
          </div>
        </div>
        <div style="display: flex; align-items: center;">
          <span class="performer-stats">${p.goalsInMatch > 0 ? `⚽${p.goalsInMatch} ` : ""}${p.assistsInMatch > 0 ? `👟${p.assistsInMatch}` : ""}</span>
          <span class="performer-rating ${ratingClass}">${p.matchRating.toFixed(1)}</span>
        </div>
      `;
      topList.appendChild(row);
    });
  }
}

function startLiveSimulation(simData, instant = false) {
  if (simIntervalId) {
    clearInterval(simIntervalId);
    simIntervalId = null;
  }

  const playBtn = document.getElementById("btn-play-sim");
  const playText = document.getElementById("sim-play-text");
  const clockBadge = document.getElementById("sim-clock");
  const feed = document.getElementById("match-timeline-feed");
  const progressFill = document.getElementById("sim-progress-fill");
  const markersContainer = document.getElementById("sim-progress-markers");
  const scoreH = document.getElementById("sim-score-h");
  const scoreA = document.getElementById("sim-score-a");

  // Instant finish mode
  if (instant) {
    isSimulating = false;
    currentSimMinute = 90;
    if (clockBadge) {
      clockBadge.textContent = "FT";
      clockBadge.classList.remove("live-active");
    }
    if (progressFill) progressFill.style.width = "100%";

    if (scoreH) scoreH.textContent = simData.hGoals;
    if (scoreA) scoreA.textContent = simData.aGoals;
    document.getElementById("sim-stat-shots").textContent = `${simData.hShots} (${simData.hTarget}) — ${simData.aShots} (${simData.aTarget})`;

    if (feed) feed.innerHTML = "";
    if (markersContainer) markersContainer.innerHTML = "";
    simData.events.forEach(ev => {
      appendTimelineEvent(feed, ev);
      addProgressMarker(markersContainer, ev);
    });

    renderMOTMAndPerformers(simData);

    const winnerText = simData.hGoals > simData.aGoals ? `${simData.home.name} Victory` : (simData.hGoals < simData.aGoals ? `${simData.away.name} Victory` : "Honours Even (Draw)");
    document.getElementById("sim-status-label").textContent = `Full Time — ${winnerText} (${simData.hGoals} - ${simData.aGoals})`;
    document.getElementById("sim-events-count").textContent = `${simData.events.length} Events Logged`;

    if (playBtn) {
      playBtn.classList.remove("simulating");
      if (playText) playText.textContent = "🔄 Re-Simulate Match";
    }
    return;
  }

  // Live simulation mode
  isSimulating = true;
  currentSimMinute = 0;
  let liveHScore = 0;
  let liveAScore = 0;
  const revealedEvents = new Set();

  if (scoreH) scoreH.textContent = "0";
  if (scoreA) scoreA.textContent = "0";
  if (feed) feed.innerHTML = "";
  if (markersContainer) markersContainer.innerHTML = "";
  if (progressFill) progressFill.style.width = "0%";

  if (clockBadge) clockBadge.classList.add("live-active");
  if (playBtn) {
    playBtn.classList.add("simulating");
    if (playText) playText.textContent = "⏱ Simulating...";
  }

  // Initial Kickoff event
  if (feed) {
    feed.innerHTML = `
      <div class="timeline-event-row home-event">
        <div class="event-min">0'</div>
        <div class="event-icon">⏱</div>
        <div class="event-details">
          <div class="event-title">Match Kick-Off</div>
          <div class="event-desc">The referee blows the whistle. The match is underway at ${simData.home.name}'s stadium!</div>
        </div>
      </div>
    `;
  }

  // Tick delay based on speed multiplier
  const tickDelay = Math.round(90 / simSpeedMultiplier);

  simIntervalId = setInterval(() => {
    currentSimMinute++;

    // Update clock
    if (clockBadge) {
      if (currentSimMinute < 45) {
        clockBadge.textContent = `⏱ ${currentSimMinute}'`;
      } else if (currentSimMinute === 45) {
        clockBadge.textContent = `45' HT`;
      } else if (currentSimMinute < 90) {
        clockBadge.textContent = `⏱ ${currentSimMinute}'`;
      } else {
        clockBadge.textContent = `FT`;
      }
    }

    // Update progress bar
    if (progressFill) {
      progressFill.style.width = `${Math.min(100, (currentSimMinute / 90) * 100)}%`;
    }

    // Dynamic shots counter
    const currentHShots = Math.round((currentSimMinute / 90) * simData.hShots);
    const currentAShots = Math.round((currentSimMinute / 90) * simData.aShots);
    const currentHTarget = Math.min(currentHShots, Math.round((currentSimMinute / 90) * simData.hTarget));
    const currentATarget = Math.min(currentAShots, Math.round((currentSimMinute / 90) * simData.aTarget));
    document.getElementById("sim-stat-shots").textContent = `${currentHShots} (${currentHTarget}) — ${currentAShots} (${currentATarget})`;

    // Check for match events
    simData.events.forEach((ev, idx) => {
      if (ev.minute <= currentSimMinute && !revealedEvents.has(idx)) {
        revealedEvents.add(idx);
        appendTimelineEvent(feed, ev);
        addProgressMarker(markersContainer, ev);

        if (ev.type === "goal") {
          if (ev.side === "home") {
            liveHScore++;
            if (scoreH) {
              scoreH.textContent = liveHScore;
              triggerGoalAnimation(scoreH);
            }
          } else {
            liveAScore++;
            if (scoreA) {
              scoreA.textContent = liveAScore;
              triggerGoalAnimation(scoreA);
            }
          }
        }
      }
    });

    document.getElementById("sim-events-count").textContent = `${revealedEvents.size} / ${simData.events.length} Events`;

    // Half time note
    if (currentSimMinute === 45 && feed) {
      const htRow = document.createElement("div");
      htRow.className = "timeline-event-row";
      htRow.innerHTML = `
        <div class="event-min">45'</div>
        <div class="event-icon">⏸</div>
        <div class="event-details">
          <div class="event-title">Half-Time Whistle</div>
          <div class="event-desc">Teams head into the dressing rooms. Score at the break: ${liveHScore} - ${liveAScore}.</div>
        </div>
      `;
      feed.insertBefore(htRow, feed.firstChild);
    }

    // Full time
    if (currentSimMinute >= 90) {
      clearInterval(simIntervalId);
      simIntervalId = null;
      isSimulating = false;
      if (clockBadge) {
        clockBadge.classList.remove("live-active");
        clockBadge.textContent = "FT";
      }

      // Final whistle event
      if (feed) {
        const ftRow = document.createElement("div");
        ftRow.className = "timeline-event-row";
        ftRow.innerHTML = `
          <div class="event-min">90'</div>
          <div class="event-icon">🏁</div>
          <div class="event-details">
            <div class="event-title">Full-Time Whistle!</div>
            <div class="event-desc">The referee blows the whistle. Final score: ${simData.home.name} ${liveHScore} - ${liveAScore} ${simData.away.name}.</div>
          </div>
        `;
        feed.insertBefore(ftRow, feed.firstChild);
      }

      // Finalize MOTM and Performers
      renderMOTMAndPerformers(simData);

      const winnerText = liveHScore > liveAScore ? `${simData.home.name} Victory` : (liveHScore < liveAScore ? `${simData.away.name} Victory` : "Honours Even (Draw)");
      document.getElementById("sim-status-label").textContent = `Full Time — ${winnerText} (${liveHScore} - ${liveAScore})`;

      if (playBtn) {
        playBtn.classList.remove("simulating");
        if (playText) playText.textContent = "🔄 Re-Simulate Match";
      }
    }
  }, tickDelay);
}

// -------------------------------------------------------------
// Squads & Player Cards Display
// -------------------------------------------------------------
let currentSquadTeam = "home"; // "home" or "away"
let currentPosFilter = "ALL";

function renderSquadCards(teamName, filterPos = "ALL") {
  const squad = getSquad(teamName);
  const container = document.getElementById("squad-cards-grid");
  container.innerHTML = "";

  const filtered = filterPos === "ALL" ? squad : squad.filter(p => p.pos === filterPos);

  filtered.forEach(p => {
    const card = document.createElement("div");
    card.className = "player-card";
    const ratingClass = p.rating >= 8.5 ? "rating-elite" : (p.rating >= 7.9 ? "rating-good" : "rating-avg");

    card.innerHTML = `
      <div class="player-card-header">
        <div class="player-main-info">
          <span class="player-number">#${p.num}</span>
          <div class="player-name-block">
            <span class="card-player-name">${p.name}</span>
            <span class="card-player-role">${p.role} • ${p.pos}</span>
          </div>
        </div>
        <span class="player-rating-pill ${ratingClass}">${p.rating.toFixed(1)}</span>
      </div>

      <div class="player-stats-row">
        <div class="stat-item">
          <span>Goals:</span>
          <strong>${p.goals}</strong>
        </div>
        <div class="stat-item">
          <span>Assists:</span>
          <strong>${p.assists}</strong>
        </div>
        ${p.cleanSheets !== undefined ? `
        <div class="stat-item">
          <span>Clean Sheets:</span>
          <strong>${p.cleanSheets}</strong>
        </div>` : ""}
      </div>

      <div class="player-trait-tag">
        Trait: ${p.trait}
      </div>
    `;
    container.appendChild(card);
  });
}

// -------------------------------------------------------------
// View Switching & Tab Navigation
// -------------------------------------------------------------
const viewTabs = document.querySelectorAll(".view-tab");
const viewContainers = document.querySelectorAll(".view-container");

viewTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    viewTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");

    const targetId = tab.getAttribute("data-target");
    viewContainers.forEach(c => {
      if (c.id === targetId) {
        c.classList.add("active");
      } else {
        c.classList.remove("active");
      }
    });

    if (targetId === "squads-view") {
      const activeTeam = currentSquadTeam === "away" ? awayTeamSelect.value : homeTeamSelect.value;
      renderSquadCards(activeTeam, currentPosFilter);
    }
  });
});

// Squad team toggle buttons
const squadHomeBtn = document.getElementById("squad-tab-home");
const squadAwayBtn = document.getElementById("squad-tab-away");

if (squadHomeBtn && squadAwayBtn) {
  squadHomeBtn.addEventListener("click", () => {
    squadHomeBtn.classList.add("active");
    squadAwayBtn.classList.remove("active");
    currentSquadTeam = "home";
    renderSquadCards(homeTeamSelect.value, currentPosFilter);
  });

  squadAwayBtn.addEventListener("click", () => {
    squadAwayBtn.classList.add("active");
    squadHomeBtn.classList.remove("active");
    currentSquadTeam = "away";
    renderSquadCards(awayTeamSelect.value, currentPosFilter);
  });
}

// Position filter chips
const posChips = document.querySelectorAll(".pos-chip");
posChips.forEach(chip => {
  chip.addEventListener("click", () => {
    posChips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentPosFilter = chip.getAttribute("data-pos");
    const activeTeam = currentSquadTeam === "away" ? awayTeamSelect.value : homeTeamSelect.value;
    renderSquadCards(activeTeam, currentPosFilter);
  });
});

// Match Simulation Action Buttons
const playSimBtn = document.getElementById("btn-play-sim");
if (playSimBtn) {
  playSimBtn.addEventListener("click", () => {
    if (!currentSimData) {
      computeMatchInference();
    }
    // If finished, generate fresh randomized simulation
    if (currentSimMinute >= 90) {
      currentSimData = generateMatchSimulation(
        currentSimData.home,
        currentSimData.away,
        currentSimData.lamHome,
        currentSimData.lamAway,
        currentSimData.homeProb,
        currentSimData.drawProb,
        currentSimData.awayProb
      );
    }
    startLiveSimulation(currentSimData, false);
  });
}

const instantSimBtn = document.getElementById("btn-instant-sim");
if (instantSimBtn) {
  instantSimBtn.addEventListener("click", () => {
    if (!currentSimData) {
      computeMatchInference();
    }
    if (currentSimMinute >= 90) {
      currentSimData = generateMatchSimulation(
        currentSimData.home,
        currentSimData.away,
        currentSimData.lamHome,
        currentSimData.lamAway,
        currentSimData.homeProb,
        currentSimData.drawProb,
        currentSimData.awayProb
      );
    }
    startLiveSimulation(currentSimData, true);
  });
}

// Speed Toggle Buttons
const speedButtons = document.querySelectorAll(".speed-btn");
speedButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    speedButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    simSpeedMultiplier = parseFloat(btn.getAttribute("data-speed")) || 3;
    if (isSimulating && currentSimData) {
      startLiveSimulation(currentSimData, false);
    }
  });
});

// Event Listeners
leagueSelect.addEventListener("change", (e) => {
  populateTeams(e.target.value);
  computeMatchInference();
});

homeTeamSelect.addEventListener("change", () => {
  computeMatchInference();
});

awayTeamSelect.addEventListener("change", () => {
  computeMatchInference();
});

swapBtn.addEventListener("click", () => {
  const hVal = homeTeamSelect.value;
  const aVal = awayTeamSelect.value;
  homeTeamSelect.value = aVal;
  awayTeamSelect.value = hVal;
  computeMatchInference();
});

simulateBtn.addEventListener("click", computeMatchInference);

presetChips.forEach(chip => {
  chip.addEventListener("click", () => {
    presetChips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");

    const league = chip.getAttribute("data-league");
    const home = chip.getAttribute("data-home");
    const away = chip.getAttribute("data-away");

    leagueSelect.value = league;
    populateTeams(league, home, away);
    computeMatchInference();
  });
});

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  populateTeams("Premier League", "Arsenal", "Chelsea");
  computeMatchInference();
});

