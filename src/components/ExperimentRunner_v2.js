/**
 * ExperimentRunner v2.0
 * Comprehensive experiment framework for Experiments 1-6.
 *
 * KEY DIFFERENCE FROM v1 (ExperimentRunner.js):
 *   - Self-contained physics simulation — does NOT need App.jsx's animation loop
 *   - Async execution with event-loop yielding every 50 frames (non-blocking UI)
 *   - Supports circular obstacle objects for environmental complexity (Exp 2)
 *   - onProgressUpdate callback for React UI updates
 *
 * USAGE (from App.jsx):
 *   const runner = new ExperimentRunner();
 *   runner.onProgressUpdate = (info) => setProgress(info);
 *   await runner.runExperiment(2);   // runs runExp2() then auto-saves JSON
 *   runner.stop();                   // abort mid-run
 *
 * Author: Santeri
 * Version: 2.0
 * Date: May 2026
 */

import { ShortTermMemory, STMFrame } from './memory/stm.js';
import { LongTermMemory, LTMPattern } from './memory/LTM.js';
import { ConsolidationEngine } from './memory/ConsolidationEngine.js';
import { DualMemoryController } from './memory/DualMemoryController.js';
import {
  SIMULATION_CONFIG,
  MEMORY_CONFIG,
  CONTROLLER_CONFIG,
  EXPERIMENT_CONFIG,
  EXP3_ROBUSTNESS_CONFIG,
  EXP4_MULTI_AGENT_CONFIG,
  EXP4_5_SHARED_LTM_CONFIG,
  EXP5_REWARD_VARIATION_CONFIG,
  EXP5_5_MULTI_OBJECTIVE_CONFIG,
  EXP5_5_5_WEIGHT_OPTIMIZATION_CONFIG,
  EXP6_TRANSFER_LEARNING_CONFIG,
  EXP8_WEIGHT_OPTIMIZATION_CONFIG,
  EXP9_LEARNING_DYNAMICS_CONFIG,
  getEXP3Config,
  getAgentStartPosition,
  createAgentForCondition,
} from './EXPERIMENT_CONFIG.js';

// ── Simulation constants (from central config) ────────────────────────────────

const SIM_N       = SIMULATION_CONFIG.HOPFIELD_NEURONS;
const SIM_W       = SIMULATION_CONFIG.WORLD_WIDTH;
const SIM_H       = SIMULATION_CONFIG.WORLD_HEIGHT;
const SIM_OBS_R   = SIMULATION_CONFIG.OBSTACLE_DETECTION_RANGE;
const SIM_FOOD_R  = SIMULATION_CONFIG.FOOD_DETECTION_RANGE;
const SIM_AGENT_R = SIMULATION_CONFIG.AGENT_RADIUS;
const SIM_FOOD_PX = SIMULATION_CONFIG.FOOD_RADIUS;
const SIM_SPD     = SIMULATION_CONFIG.AGENT_SPEED;
const SIM_TURN    = SIMULATION_CONFIG.AGENT_TURN_RATE;
const SIM_BETA    = SIMULATION_CONFIG.HOPFIELD_BETA;
const SIM_NOISE   = SIMULATION_CONFIG.HOPFIELD_NOISE;
const YIELD_EVERY = SIMULATION_CONFIG.YIELD_EVERY_N_FRAMES;
const SIM_SA      = SIMULATION_CONFIG.SENSOR_ANGLES;

// ── PERFECT_PATS (exact copy of App.jsx) ─────────────────────────────────────

function _makePat(obs, food, motor) {
  const d = Array(SIM_N).fill(-1);
  obs.forEach(i  => { d[i]     = 1; });
  food.forEach(i => { d[i + 5] = 1; });
  if (motor === 'L') d[15] = 1;
  if (motor === 'F') d[16] = 1;
  if (motor === 'R') d[17] = 1;
  return d;
}

const PERFECT_PATS = [
  { name: 'F→F',  data: _makePat([],  [0], 'F') },
  { name: 'FR→R', data: _makePat([],  [1], 'R') },
  { name: 'R→R',  data: _makePat([],  [2], 'R') },
  { name: 'FL→L', data: _makePat([],  [3], 'L') },
  { name: 'L→L',  data: _makePat([],  [4], 'L') },
  { name: 'W→R',  data: _makePat([0], [],  'R') },
  { name: 'WR→L', data: _makePat([1], [],  'L') },
  { name: 'WL→R', data: _makePat([3], [],  'R') },
  { name: '○→F',  data: _makePat([],  [],  'F') },
];

// ── Physics helpers ───────────────────────────────────────────────────────────

function _modernStep(state, patterns, beta, noiseLevel) {
  if (!patterns.length) return { newState: [...state], attn: [] };
  const noisyState = state.map(v =>
    Math.random() < noiseLevel ? (Math.random() < 0.5 ? 1 : -1) : v
  );
  const sims = patterns.map(({ data }) => {
    let d = 0;
    for (let i = 0; i < SIM_N; i++) d += data[i] * noisyState[i];
    return beta * d;
  });
  const mx = Math.max(...sims);
  const ex = sims.map(s => Math.exp(s - mx));
  const sm = ex.reduce((a, b) => a + b, 0);
  const attn = ex.map(e => e / sm);
  const raw = new Float32Array(SIM_N);
  for (let k = 0; k < patterns.length; k++)
    for (let i = 0; i < SIM_N; i++) raw[i] += patterns[k].data[i] * attn[k];
  return { newState: Array.from(raw, v => (v >= 0 ? 1 : -1)), attn };
}

function _decodeMotor(state) {
  const L = state[15], F = state[16], R = state[17];
  if (F === 1 && L !== 1 && R !== 1) return 'F';
  if (R === 1 && L !== 1)            return 'R';
  if (L === 1)                       return 'L';
  return 'F';
}

function _encodeSensors(obs, food) {
  const s = Array(SIM_N).fill(-1);
  for (let i = 0; i < 5; i++) {
    if (obs[i])  s[i]     = 1;
    if (food[i]) s[i + 5] = 1;
  }
  return s;
}

function _determineContext(sensors) {
  if (sensors.food.some(f => f)) return 'foraging';
  if (sensors.obs.some(o => o))  return 'avoidance';
  return 'exploration';
}

function _wallDist(ox, oy, angle, range, worldW = SIM_W, worldH = SIM_H) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let t = range;
  if (dx < 0) t = Math.min(t, -ox / dx);
  if (dx > 0) t = Math.min(t, (worldW - ox) / dx);
  if (dy < 0) t = Math.min(t, -oy / dy);
  if (dy > 0) t = Math.min(t, (worldH - oy) / dy);
  return t;
}

/** Minimum positive t where ray (ox,oy)+(dx,dy)*t intersects circle (cx,cy,r). */
function _raySphereHit(ox, oy, dx, dy, cx, cy, r) {
  const fx = ox - cx, fy = oy - cy;
  const a   = dx * dx + dy * dy;
  const b   = 2 * (fx * dx + fy * dy);
  const c   = fx * fx + fy * fy - r * r;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  const sq = Math.sqrt(disc);
  const t1 = (-b - sq) / (2 * a);
  const t2 = (-b + sq) / (2 * a);
  if (t1 >= 0) return t1;
  if (t2 >= 0) return t2;
  return Infinity;
}

function _getSensors(ax, ay, angle, foods, obstacles, worldW = SIM_W, worldH = SIM_H) {
  const obs  = Array(5).fill(false);
  const food = Array(5).fill(false);
  const DETECT = SIM_OBS_R * 0.88;

  for (let s = 0; s < 5; s++) {
    const ra = angle + SIM_SA[s];
    const dx = Math.cos(ra), dy = Math.sin(ra);

    if (_wallDist(ax, ay, ra, SIM_OBS_R, worldW, worldH) < DETECT) {
      obs[s] = true;
    }
    if (!obs[s]) {
      for (const o of obstacles) {
        if (_raySphereHit(ax, ay, dx, dy, o.x, o.y, o.r + SIM_AGENT_R) < DETECT) {
          obs[s] = true;
          break;
        }
      }
    }
    for (const f of foods) {
      const fdx = f.x - ax, fdy = f.y - ay;
      const dist = Math.sqrt(fdx * fdx + fdy * fdy);
      if (dist < SIM_FOOD_R) {
        let rel = Math.atan2(fdy, fdx) - angle;
        while (rel >  Math.PI) rel -= 2 * Math.PI;
        while (rel < -Math.PI) rel += 2 * Math.PI;
        if (Math.abs(rel - SIM_SA[s]) < Math.PI / 4) food[s] = true;
      }
    }
  }
  return { obs, food };
}

function _moveAgent(x, y, angle, action, obstacles, speed = SIM_SPD, turn = SIM_TURN, worldW = SIM_W, worldH = SIM_H) {
  let na = angle;
  if (action === 'L') na -= turn;
  if (action === 'R') na += turn;

  let nx = x + Math.cos(na) * speed;
  let ny = y + Math.sin(na) * speed;

  let bounced = false;
  const mg = SIM_AGENT_R + 3;
  if (nx < mg)           { nx = mg;            na = Math.PI - na; bounced = true; }
  if (nx > worldW - mg)  { nx = worldW - mg;   na = Math.PI - na; bounced = true; }
  if (ny < mg)           { ny = mg;            na = -na;          bounced = true; }
  if (ny > worldH - mg)  { ny = worldH - mg;   na = -na;          bounced = true; }

  for (const o of obstacles) {
    const odx = nx - o.x, ody = ny - o.y;
    const dist = Math.sqrt(odx * odx + ody * ody);
    const minD = SIM_AGENT_R + o.r;
    if (dist < minD && dist > 0) {
      bounced = true;
      const nx_n = odx / dist, ny_n = ody / dist;
      nx = o.x + nx_n * (minD + 1);
      ny = o.y + ny_n * (minD + 1);
      const dot = Math.cos(na) * nx_n + Math.sin(na) * ny_n;
      na = Math.atan2(Math.sin(na) - 2 * dot * ny_n, Math.cos(na) - 2 * dot * nx_n);
    }
  }
  return { x: nx, y: ny, angle: na, bounced };
}

function _checkFoodCollision(ax, ay, foods, worldW = SIM_W, worldH = SIM_H) {
  let eaten = 0;
  for (const f of foods) {
    const dx = f.x - ax, dy = f.y - ay;
    if (Math.sqrt(dx * dx + dy * dy) < SIM_AGENT_R + SIM_FOOD_PX + 2) {
      f.x = 50 + Math.random() * (worldW - 100);
      f.y = 50 + Math.random() * (worldH - 100);
      eaten++;
    }
  }
  return eaten;
}

/**
 * Detect pairwise agent-agent proximity and push overlapping agents apart.
 * Increments `agent.collisions` for every frame two agents are too close.
 *
 * @param {Array<{x:number, y:number, collisions:number}>} agents
 */
function _resolveAgentCollisions(agents) {
  const minDist = SIM_AGENT_R * 2 + EXP4_MULTI_AGENT_CONFIG.COLLISION_DISTANCE_THRESHOLD;
  const force   = EXP4_MULTI_AGENT_CONFIG.COLLISION_SEPARATION_FORCE;
  for (let i = 0; i < agents.length; i++) {
    for (let j = i + 1; j < agents.length; j++) {
      const dx   = agents[j].x - agents[i].x;
      const dy   = agents[j].y - agents[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist && dist > 0) {
        agents[i].collisions++;
        agents[j].collisions++;
        const ang = Math.atan2(dy, dx);
        const fx  = Math.cos(ang) * force;
        const fy  = Math.sin(ang) * force;
        agents[i].x -= fx;
        agents[i].y -= fy;
        agents[j].x += fx;
        agents[j].y += fy;
      }
    }
  }
}

// ── Robustness stressor helpers (Experiment 3) ────────────────────────────────

/**
 * Box-Muller Gaussian random number.
 * Used to model sensor noise with a known standard deviation.
 * @param {number} mean
 * @param {number} stdev  Standard deviation
 * @returns {number}
 */
function _gaussianRandom(mean = 0, stdev = 1) {
  const u = Math.max(1e-10, 1 - Math.random()); // avoid log(0)
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * stdev + mean;
}

/**
 * Apply radiation damage to every pattern in an LTM instance.
 * Each pattern has `radiationRate` probability of taking a 5–15 % reliability hit.
 *
 * @param {import('./memory/LTM.js').LongTermMemory} ltm
 * @param {number} radiationRate  Per-frame corruption probability per pattern
 */
function _applyRadiationDamage(ltm, radiationRate) {
  if (!ltm || radiationRate <= 0) return;
  for (const ctx of ['exploration', 'foraging', 'avoidance']) {
    for (const pattern of ltm.patterns[ctx].values()) {
      if (Math.random() < radiationRate) {
        const hit = 0.05 + Math.random() * 0.10; // 5–15 % hit
        pattern.reliability          = Math.max(0.01, pattern.reliability * (1 - hit));
        pattern.consolidationStrength = Math.max(0.01, pattern.consolidationStrength * (1 - hit * 0.5));
      }
    }
  }
}

/**
 * Map a profile + level config object to the four stressor scalars consumed by runTrial().
 *
 * @param {'warehouse'|'physics'|'space'} profile
 * @param {object} cfg  Output of getEXP3Config()
 * @returns {{ noiseSigma, gravityMultiplier, radiationRate, driftRate }}
 */
function _buildStressorParams(profile, cfg) {
  switch (profile) {
    case 'warehouse':
      return { noiseSigma: cfg.sigma, gravityMultiplier: 1.0,         radiationRate: 0,                  driftRate: 0              };
    case 'physics':
      return { noiseSigma: 0,         gravityMultiplier: cfg.multiplier, radiationRate: 0,                  driftRate: 0              };
    case 'space':
      // Radiation degrades sensors (adds bit-flip noise) AND corrupts LTM patterns.
      // Sensor noise sigma tracks radiation_rate loosely (radiation ×5 → noise %).
      return { noiseSigma: cfg.radiation_rate * 5, gravityMultiplier: cfg.gravity, radiationRate: cfg.radiation_rate, driftRate: cfg.drift_rate };
    default:
      return { noiseSigma: 0, gravityMultiplier: 1.0, radiationRate: 0, driftRate: 0 };
  }
}

// ── Environment generators ────────────────────────────────────────────────────

function _generateObstacles(count, worldW = SIM_W, worldH = SIM_H) {
  const obstacles = [];
  const MIN_R = 12, MAX_R = 22;
  let attempts = 0;
  while (obstacles.length < count && attempts < count * 20) {
    attempts++;
    const r  = MIN_R + Math.random() * (MAX_R - MIN_R);
    const ox = r + 50 + Math.random() * (worldW - 2 * r - 100);
    const oy = r + 50 + Math.random() * (worldH - 2 * r - 100);
    const cdx = ox - worldW / 2, cdy = oy - worldH / 2;
    if (Math.sqrt(cdx * cdx + cdy * cdy) < 75) continue;
    let ok = true;
    for (const o of obstacles) {
      const dx = ox - o.x, dy = oy - o.y;
      if (Math.sqrt(dx * dx + dy * dy) < r + o.r + 10) { ok = false; break; }
    }
    if (ok) obstacles.push({ x: ox, y: oy, r });
  }
  if (obstacles.length < count)
    console.warn(`[Env] Placed ${obstacles.length}/${count} obstacles (space exhausted).`);
  return obstacles;
}

function _generateFood(count, obstacles, worldW = SIM_W, worldH = SIM_H) {
  const foods = [];
  let attempts = 0;
  while (foods.length < count && attempts < count * 20) {
    attempts++;
    const fx = 50 + Math.random() * (worldW - 100);
    const fy = 50 + Math.random() * (worldH - 100);
    let ok = true;
    for (const o of obstacles) {
      const dx = fx - o.x, dy = fy - o.y;
      if (Math.sqrt(dx * dx + dy * dy) < o.r + SIM_FOOD_PX + 5) { ok = false; break; }
    }
    if (ok) foods.push({ x: fx, y: fy });
  }
  return foods;
}

function _seedLTM(ltm) {
  for (const pat of PERFECT_PATS) {
    const triggerCondition = pat.data.map((v, i) => (i < 10 ? v : -1));
    let action = 'F';
    if (pat.data[15] === 1) action = 'L';
    else if (pat.data[17] === 1) action = 'R';
    const hasFood = triggerCondition.slice(5, 10).some(v => v === 1);
    const hasObs  = triggerCondition.slice(0, 5).some(v => v === 1);
    const context = hasFood ? 'foraging' : hasObs ? 'avoidance' : 'exploration';
    const patId   = `seed_${pat.name.replace(/[^a-zA-Z0-9_]/g, '_')}`;
    ltm.storePattern(new LTMPattern({
      patternId: patId, context, triggerCondition,
      actionSequence:        Array(5).fill(action),
      reliability:           0.85,
      consolidationStrength: 0.75,
      utility:               0.8,
      abstractDescription:   `seeded:${pat.name}`,
      usageCount:            10,
      successCount:          9,
    }));
  }
}

/**
 * Create a fresh agent wired for the given memory condition.
 * Delegates to the centralized factory in EXPERIMENT_CONFIG.js; seeds LTM for
 * Condition C after the factory returns so this module stays the sole keeper of
 * PERFECT_PATS / _seedLTM logic.
 */
function _createAgentForCondition(condition) {
  const agent = createAgentForCondition(
    condition,
    ShortTermMemory,
    LongTermMemory,
    ConsolidationEngine,
    DualMemoryController,
  );

  // Condition C: caller (factory) already set threshold=0.20 and engine=null;
  // we still own the seeding step because _seedLTM uses this module's PERFECT_PATS.
  if (condition === 'C') {
    _seedLTM(agent.ltm);
  }

  return agent;
}

// ── ExperimentRunner class ────────────────────────────────────────────────────

export class ExperimentRunner {
  constructor() {
    this.sharedParams = {
      TRIAL_DURATION:         EXPERIMENT_CONFIG.TRIAL_DURATION_FRAMES,
      NUM_TRIALS:             EXPERIMENT_CONFIG.NUM_TRIALS_PER_CONDITION,
      NUM_AGENTS:             EXPERIMENT_CONFIG.NUM_AGENTS_PER_TRIAL,
      STM_SIZE:               MEMORY_CONFIG.STM_SIZE,
      STM_DECAY_TAU:          MEMORY_CONFIG.STM_DECAY_TAU,
      HOPFIELD_NEURONS:       SIMULATION_CONFIG.HOPFIELD_NEURONS,
      REWARD_THRESHOLD:       MEMORY_CONFIG.CONSOLIDATION_REWARD_THRESHOLD,
      SURPRISE_THRESHOLD:     MEMORY_CONFIG.CONSOLIDATION_SURPRISE_THRESHOLD,
      CONSOLIDATION_INTERVAL: MEMORY_CONFIG.CONSOLIDATION_PERIODIC_INTERVAL,
      SIMILARITY_THRESHOLD:   MEMORY_CONFIG.CONSOLIDATION_SIMILARITY_THRESHOLD,
      CONFIDENCE_THRESHOLD:   CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
    };

    this.expParams = {
      1:   { name: 'Dual-Memory Validation'    },
      2:   { name: 'Environmental Complexity'  },
      3:   { name: 'Sensor Noise Robustness'   },
      4:   { name: 'Multi-Agent Coordination'  },
      4.5: { name: 'Shared LTM Consolidation'  },
      5:   { name: 'Reward Structure Variation'},
      5.5:   { name: 'Multi-Objective Learning'  },
      '5.5.5': { name: 'Weight Optimisation'       },
      6:     { name: 'Generalization & Transfer' },
      8:     { name: 'Weight Optimization (10 configs)' },
      9:     { name: 'Learning Dynamics & Curves'      },
    };

    this.progress = {
      currentExperiment: null,
      totalTrials:       0,
      completedTrials:   0,
      startTime:         null,
      isRunning:         false,
    };

    this.results  = { 1: [], 2: [], 3: [], 4: [], 4.5: [], 5: [], 5.5: [], '5.5.5': [], 6: [], 8: [], 9: [] };
    this._stopped = false;

    /**
     * Set this before calling runExperiment() to receive live progress updates.
     * Signature: fn({ completedTrials, totalTrials, percentComplete,
     *                  currentExperiment, condition, level, lastResult })
     * @type {function|null}
     */
    this.onProgressUpdate = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  stop() {
    this._stopped = true;
    this.progress.isRunning = false;
  }

  async run(experimentNumber = 'all') {
    if (experimentNumber === 'all') await this.runAllExperiments();
    else await this.runExperiment(experimentNumber);
  }

  async runAllExperiments() {
    console.log('[V2] Starting all experiments…');
    for (let exp = 1; exp <= 6; exp++) {
      if (this._stopped) break;
      await this.runExperiment(exp);
    }
    this.generateSummary();
  }

  /**
   * Run one experiment by number, then save results as a JSON download.
   * This is the correct entry point — calling runExp2() directly skips the save.
   */
  async runExperiment(expNum) {
    this._stopped = false;
    this.progress.currentExperiment = expNum;
    this.progress.startTime         = Date.now();
    this.progress.isRunning         = true;

    const dispatch = {
      1:   () => this.runExp1(),
      2:   () => this.runExp2(),
      3:   () => this.runExp3(),
      4:   () => this.runExp4(),
      4.5: () => this.runExp4_5(),
      5:   () => this.runExp5(),
      5.5:     () => this.runExp5_5(),
      '5.5.5': () => this.runExp5_5_5(),
      6:       () => this.runExp6(),
      8:       () => this.runExp8(),
      9:       () => this.runExp9(),
    };

    if (dispatch[expNum]) await dispatch[expNum]();
    else console.error(`[V2] Unknown experiment: ${expNum}`);

    this.progress.isRunning = false;
    // Save whatever results were collected (even partial if aborted)
    this.saveExperimentResults(expNum);
  }

  // ── EXPERIMENT 1: Dual-Memory Validation ─────────────────────────────────

  async runExp1() {
    const conditions = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;
    this.progress.totalTrials = conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const condition of conditions) {
      for (let trial = 0; trial < trials; trial++) {
        for (let agent = 0; agent < agents; agent++) {
          if (this._stopped) return;
          const result = await this.runTrial({ experiment: 1, condition, trial, agent, duration: dur, params: {} });
          this.results[1].push(result);
          this.progress.completedTrials++;
          this._emitProgress({ lastResult: result });
        }
      }
    }
  }

  // ── EXPERIMENT 2: Environmental Complexity Scaling ────────────────────────
  //
  //   Level 1 →   5 obstacles, 10 food   (sparse / easy)
  //   Level 2 →  15 obstacles, 20 food
  //   Level 3 →  30 obstacles, 30 food
  //   Level 4 →  50 obstacles, 40 food
  //   Level 5 → 100 obstacles, 50 food   (dense / hard)
  //
  //   4 conditions × 5 trials × 3 agents × 5 levels = 300 agent-runs total

  async runExp2() {
    const complexityLevels = [
      { level: 1, obstacles:   5, food: 10 },
      { level: 2, obstacles:  15, food: 20 },
      { level: 3, obstacles:  30, food: 30 },
      { level: 4, obstacles:  50, food: 40 },
      { level: 5, obstacles: 100, food: 50 },
    ];
    const conditions = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;

    this.progress.totalTrials     = complexityLevels.length * conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const { level, obstacles, food } of complexityLevels) {
      for (const condition of conditions) {
        for (let trial = 0; trial < trials; trial++) {
          for (let agent = 0; agent < agents; agent++) {
            if (this._stopped) return;
            const result = await this.runTrial({
              experiment: 2, condition, trial, agent, duration: dur,
              params: { complexityLevel: level, obstacleCount: obstacles, foodCount: food },
            });
            this.results[2].push(result);
            this.progress.completedTrials++;
            this._emitProgress({ level, condition, lastResult: result });
          }
        }
      }
    }
  }

  // ── EXPERIMENT 3: Robustness Across Real-World & Space Environments ──────────
  //
  //   3 profiles × 5 levels × 4 conditions × 15 trials = 900 total agent-runs
  //
  //   Warehouse profile : sensor noise (σ = 0 → 0.50)
  //   Physics profile   : gravity variation (1g → 0g)
  //   Space profile     : combined radiation + drift + gravity

  async runExp3() {
    const cfg = EXP3_ROBUSTNESS_CONFIG;

    const profiles = [];
    if (cfg.ENABLE_WAREHOUSE_PROFILE) profiles.push('warehouse');
    if (cfg.ENABLE_PHYSICS_PROFILE)   profiles.push('physics');
    if (cfg.ENABLE_SPACE_PROFILE)     profiles.push('space');

    const conditions     = ['A', 'B', 'C', 'D'];
    const trialsPerLevel = cfg.TRIALS_PER_LEVEL;
    const { TRIAL_DURATION: dur } = this.sharedParams;
    const LEVELS = 5;

    this.progress.totalTrials     = profiles.length * LEVELS * conditions.length * trialsPerLevel;
    this.progress.completedTrials = 0;

    for (const profile of profiles) {
      for (let level = 0; level < LEVELS; level++) {
        const levelCfg      = getEXP3Config(profile, level);
        const stressorParams = _buildStressorParams(profile, levelCfg);

        for (const condition of conditions) {
          for (let trial = 0; trial < trialsPerLevel; trial++) {
            if (this._stopped) return;

            const result = await this.runTrial({
              experiment: 3,
              condition,
              trial,
              agent: trial % 3,   // cycle 0/1/2 for personality tracking
              duration: dur,
              params: {
                profile,
                level,
                stressorLabel: levelCfg.label,
                ...stressorParams,
              },
            });

            this.results[3].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              currentProfile:   profile,
              currentLevel:     level,
              currentCondition: condition,
              stressorLabel:    levelCfg.label,
              lastResult:       result,
            });
          }
        }
      }
    }
  }

  // ── EXPERIMENT 4: Multi-Agent Coordination ────────────────────────────────
  //
  //   Phase 1 — Scaling test
  //     4 agent counts × 2 complexity levels × 4 conditions × 5 trials = 160 trials
  //
  //   Phase 2 — Interference test
  //     3 environment sizes × 2 complexity levels × 2 conditions × 5 trials = 60 trials
  //
  //   Total: 220 trials

  async runExp4() {
    const cfg = EXP4_MULTI_AGENT_CONFIG;

    // Compute total trials up-front for accurate progress bar
    let totalTrials = 0;
    if (cfg.SCALING_TEST.ENABLED) {
      totalTrials +=
        cfg.SCALING_TEST.AGENT_COUNTS.length *
        cfg.SCALING_TEST.COMPLEXITY_LEVELS.length *
        4 * // conditions A/B/C/D
        cfg.SCALING_TEST.TRIALS_PER_CONFIG;
    }
    if (cfg.INTERFERENCE_TEST.ENABLED) {
      totalTrials +=
        cfg.INTERFERENCE_TEST.ENVIRONMENT_CONFIGS.length *
        cfg.INTERFERENCE_TEST.COMPLEXITY_LEVELS.length *
        cfg.INTERFERENCE_TEST.TEST_CONDITIONS.length *
        cfg.INTERFERENCE_TEST.TRIALS_PER_CONFIG;
    }
    this.progress.totalTrials     = totalTrials;
    this.progress.completedTrials = 0;

    if (cfg.SCALING_TEST.ENABLED)     await this._runScalingTest();
    if (cfg.INTERFERENCE_TEST.ENABLED) await this._runInterferenceTest();
  }

  // ── Phase 1: Scaling ──────────────────────────────────────────────────────

  async _runScalingTest() {
    const cfg        = EXP4_MULTI_AGENT_CONFIG.SCALING_TEST;
    const conditions = ['A', 'B', 'C', 'D'];

    for (const numAgents of cfg.AGENT_COUNTS) {
      for (const level of cfg.COMPLEXITY_LEVELS) {
        for (const condition of conditions) {
          for (let trial = 0; trial < cfg.TRIALS_PER_CONFIG; trial++) {
            if (this._stopped) return;

            const agents = Array.from({ length: numAgents }, () =>
              _createAgentForCondition(condition),
            );

            const result = await this._runMultiAgentTrial(agents, {
              condition,
              level,
              testType:  'scaling',
              envWidth:  SIM_W,
              envHeight: SIM_H,
              envLabel:  `${numAgents} agent${numAgents > 1 ? 's' : ''}, L${level}`,
            });

            this.results[4].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              testType:          'scaling',
              currentAgentCount: numAgents,
              currentLevel:      level,
              currentCondition:  condition,
              lastResult:        result,
            });
          }
        }
      }
    }
  }

  // ── Phase 2: Interference ─────────────────────────────────────────────────

  async _runInterferenceTest() {
    const cfg = EXP4_MULTI_AGENT_CONFIG.INTERFERENCE_TEST;

    for (const envCfg of cfg.ENVIRONMENT_CONFIGS) {
      for (const level of cfg.COMPLEXITY_LEVELS) {
        for (const condition of cfg.TEST_CONDITIONS) {
          for (let trial = 0; trial < cfg.TRIALS_PER_CONFIG; trial++) {
            if (this._stopped) return;

            const agents = Array.from({ length: envCfg.agents }, () =>
              _createAgentForCondition(condition),
            );

            const result = await this._runMultiAgentTrial(agents, {
              condition,
              level,
              testType:  'interference',
              envWidth:  envCfg.width,
              envHeight: envCfg.height,
              envLabel:  envCfg.label,
            });

            this.results[4].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              testType:         'interference',
              currentEnvLabel:  envCfg.label,
              currentLevel:     level,
              currentCondition: condition,
              lastResult:       result,
            });
          }
        }
      }
    }
  }

  // ── Core multi-agent simulation ───────────────────────────────────────────

  /**
   * Run a single trial where all agents share the same physics world.
   * Agents use world-size-aware versions of all physics helpers so both
   * the 400×300 and 1200×900 interference environments work correctly.
   *
   * @param {Array} agents    Agent objects from _createAgentForCondition()
   * @param {object} trialCfg
   * @param {string} trialCfg.condition  'A'|'B'|'C'|'D'
   * @param {number} trialCfg.level      Complexity level index (e.g. 2 or 3)
   * @param {string} trialCfg.testType   'scaling' | 'interference'
   * @param {number} [trialCfg.envWidth]
   * @param {number} [trialCfg.envHeight]
   * @param {string} [trialCfg.envLabel]
   * @returns {object}
   */
  async _runMultiAgentTrial(agents, trialCfg) {
    const {
      condition, level, testType,
      envWidth  = SIM_W,
      envHeight = SIM_H,
      envLabel  = '',
    } = trialCfg;

    // Scale obstacle + food density to world area relative to standard 800×600
    const areaRatio      = (envWidth * envHeight) / (SIM_W * SIM_H);
    const BASE_OBSTACLES = { 2: 15, 3: 30 };  // matches Exp 2 levels
    const BASE_FOODS     = { 2: 20, 3: 30 };
    const obstacleCount  = Math.max(0, Math.round((BASE_OBSTACLES[level] ?? 15) * areaRatio));
    const foodCount      = Math.max(4, Math.round((BASE_FOODS[level]     ?? 20) * areaRatio));

    const obstacles = _generateObstacles(obstacleCount, envWidth, envHeight);
    const foods     = _generateFood(foodCount, obstacles, envWidth, envHeight);
    const duration  = this.sharedParams.TRIAL_DURATION;

    // Place each agent at its designated spread-out start position
    for (let i = 0; i < agents.length; i++) {
      const pos         = getAgentStartPosition(i, envWidth, envHeight);
      agents[i].x            = pos.x;
      agents[i].y            = pos.y;
      agents[i].angle        = Math.random() * Math.PI * 2;
      agents[i].score        = 0;
      agents[i].collisions   = 0;
      agents[i].distanceTraveled = 0;
    }

    const learningCurves = agents.map(() => []);

    for (let frame = 0; frame < duration; frame++) {
      if (this._stopped) break;

      if (frame % YIELD_EVERY === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      // ── Per-agent update ──────────────────────────────────────────────
      for (let i = 0; i < agents.length; i++) {
        const agent = agents[i];

        // 1. Sense (world-size-aware wall detection)
        const sensors      = _getSensors(agent.x, agent.y, agent.angle, foods, obstacles, envWidth, envHeight);
        const sensoryState = _encodeSensors(sensors.obs, sensors.food);

        // 2. Hopfield forward pass
        const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, SIM_NOISE);
        const hopfieldAction     = _decodeMotor(newState);

        // 3. Memory-conditioned action selection
        let action = hopfieldAction;
        if (agent.controller) {
          const context = _determineContext(sensors);
          const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
          action        = res.action;
        }

        // 4. Move (world-size-aware wall bouncing)
        const prevX = agent.x, prevY = agent.y;
        const moved = _moveAgent(agent.x, agent.y, agent.angle, action, obstacles,
          SIM_SPD, SIM_TURN, envWidth, envHeight);
        agent.x     = moved.x;
        agent.y     = moved.y;
        agent.angle = moved.angle;

        // Track distance traveled
        const ddx = agent.x - prevX, ddy = agent.y - prevY;
        agent.distanceTraveled += Math.sqrt(ddx * ddx + ddy * ddy);

        // 5. Food + reward (world-size-aware respawn)
        const ate    = _checkFoodCollision(agent.x, agent.y, foods, envWidth, envHeight);
        agent.score += ate;
        const reward = ate > 0 ? 1.0 : -0.01;

        // 6. Memory feedback
        if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);
        if (agent.stm)        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
        if (agent.engine)     agent.engine.update(frame, frame);

        // 7. Learning curve snapshot (10 points across trial)
        if ((frame + 1) % 240 === 0) learningCurves[i].push(agent.score);
      }

      // ── Inter-agent collision resolution ──────────────────────────────
      _resolveAgentCollisions(agents);
    }

    // ── Aggregate results ─────────────────────────────────────────────────
    const totalFood       = agents.reduce((s, a) => s + a.score,      0);
    const totalCollisions = agents.reduce((s, a) => s + a.collisions, 0);
    const meanPerAgent    = totalFood / agents.length;

    return {
      experiment:       4,
      condition,
      complexity_level: level,
      test_type:        testType,
      num_agents:       agents.length,
      environment:      { width: envWidth, height: envHeight, label: envLabel },
      results: {
        total_food_collected:      totalFood,
        mean_per_agent:            +meanPerAgent.toFixed(3),
        total_collisions:          totalCollisions,
        mean_collisions_per_agent: +(totalCollisions / agents.length).toFixed(3),
        agents: agents.map((agent, idx) => ({
          agent_id:             idx,
          food_collected:       agent.score,
          distance_traveled:    +agent.distanceTraveled.toFixed(1),
          collisions:           agent.collisions,
          patterns_consolidated: agent.ltm?.stats()?.totalPatterns  ?? 0,
          avg_reliability:      +(agent.ltm?.stats()?.avgReliability ?? 0).toFixed(3),
          ltm_usage_rate:       +(agent.controller?.stats()?.ltmUsageRate ?? 0).toFixed(3),
          learning_curve:       learningCurves[idx],
        })),
        timestamp: new Date().toISOString(),
      },
    };
  }

  // ── EXPERIMENT 4.5: Shared LTM Consolidation ─────────────────────────────
  //
  //   Two variants run back-to-back on the same scaling grid as Exp 4 Phase 1:
  //
  //     independent — each agent has a private LTM  (Exp 4 baseline)
  //     shared      — all agents share one LTM pool; each agent has its own
  //                   STM and ConsolidationEngine but writes patterns into the
  //                   same store and reads from it via its DualMemoryController
  //
  //   Hypothesis: shared LTM restores or amplifies the D-vs-A learning advantage
  //   that degrades with agent count when memories are private.
  //
  //   2 variants × 4 counts × 2 levels × 2 conditions × 5 trials = 160 trials

  async runExp4_5() {
    const cfg        = EXP4_5_SHARED_LTM_CONFIG;
    const conditions = cfg.TEST_CONDITIONS; // ['A', 'D']

    this.progress.totalTrials =
      cfg.VARIANTS.length *
      cfg.AGENT_COUNTS.length *
      cfg.COMPLEXITY_LEVELS.length *
      conditions.length *
      cfg.TRIALS_PER_CONFIG;                // 160
    this.progress.completedTrials = 0;

    for (const variant of cfg.VARIANTS) {
      for (const numAgents of cfg.AGENT_COUNTS) {
        for (const level of cfg.COMPLEXITY_LEVELS) {
          for (const condition of conditions) {
            for (let trial = 0; trial < cfg.TRIALS_PER_CONFIG; trial++) {
              if (this._stopped) return;

              // Build agent array — shared or private LTM
              const agents = variant.shared
                ? this._createSharedLTMAgents(condition, numAgents)
                : Array.from({ length: numAgents }, () => _createAgentForCondition(condition));

              const result = await this._runMultiAgentTrial(agents, {
                condition,
                level,
                testType:  'shared_ltm',
                envWidth:  SIM_W,
                envHeight: SIM_H,
                envLabel:  `${variant.name} LTM, ${numAgents} agent${numAgents > 1 ? 's' : ''}, L${level}`,
              });

              // Overwrite experiment number and annotate with shared-LTM metadata
              result.experiment     = 4.5;
              result.variant        = variant.name;
              result.variant_label  = variant.label;
              result.use_shared_ltm = variant.shared;
              // How many patterns ended up in the shared pool after this trial?
              result.shared_pool_size = variant.shared
                ? (agents[0]?.ltm?.stats()?.totalPatterns ?? 0)
                : 0;

              this.results[4.5].push(result);
              this.progress.completedTrials++;
              this._emitProgress({
                currentVariant:    variant.name,
                currentAgentCount: numAgents,
                currentLevel:      level,
                currentCondition:  condition,
                sharedPoolSize:    result.shared_pool_size,
                lastResult:        result,
              });
            }
          }
        }
      }
    }
  }

  /**
   * Build `numAgents` agent objects that all share one LongTermMemory instance.
   *
   * Architecture:
   *   Each agent has:
   *     • its own ShortTermMemory  (private experience buffer)
   *     • its own ConsolidationEngine  (fires on its own STM triggers)
   *     • its own DualMemoryController  (reads / updates the shared LTM)
   *   All agents share:
   *     • one LongTermMemory  (pattern pool grows from all agents' experiences)
   *
   * This means patterns discovered by one agent are immediately available to
   * every other agent on the next controller query — without any explicit
   * communication protocol.
   *
   * Condition A:  controller = null, engine = null  (no memory at all)
   * Condition D:  full dual — shared LTM active
   *
   * @param {'A'|'B'|'C'|'D'} condition
   * @param {number}           numAgents
   * @returns {Array<object>}
   */
  _createSharedLTMAgents(condition, numAgents) {
    // One shared LTM for the entire swarm this trial
    const sharedLtm = new LongTermMemory(1000);

    // Pre-seed the shared pool for condition C (perfect patterns, one-time)
    if (condition === 'C') _seedLTM(sharedLtm);

    return Array.from({ length: numAgents }, () => {
      const stm = new ShortTermMemory(MEMORY_CONFIG.STM_SIZE);

      // Each agent gets its own engine so its own STM drives consolidation
      // into the shared pool independently.
      const engine = new ConsolidationEngine(stm, sharedLtm, {
        windowSize:        30,
        rewardThreshold:   MEMORY_CONFIG.CONSOLIDATION_REWARD_THRESHOLD,
        surpriseThreshold: MEMORY_CONFIG.CONSOLIDATION_SURPRISE_THRESHOLD,
        periodicInterval:  MEMORY_CONFIG.CONSOLIDATION_PERIODIC_INTERVAL,
      });

      // Controller reads from / updates the shared LTM
      const controller = new DualMemoryController(sharedLtm, {
        ltmConfidenceThreshold: CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
        explorationRate:         CONTROLLER_CONFIG.EXPLORATION_RATE,
        actionWeightSTM:         CONTROLLER_CONFIG.ACTION_WEIGHT_STM,
      });

      const agent = { stm, ltm: sharedLtm, engine, controller, score: 0 };

      switch (condition) {
        case 'A':
          // No memory — pure reactive Hopfield; shared pool stays empty
          agent.controller = null;
          agent.engine     = null;
          break;
        case 'B':
          // STM records but LTM never fires; threshold effectively disabled
          agent.controller.ltmConfidenceThreshold = 9999;
          agent.engine = null;
          break;
        case 'C':
          // Pre-seeded shared LTM; lower threshold so seeded patterns activate
          agent.engine = null;
          agent.controller.ltmConfidenceThreshold = 0.20;
          break;
        case 'D':
          // Full dual memory with shared pool — default thresholds apply
          break;
        default:
          console.warn(`[SharedLTM] Unknown condition "${condition}" — defaulting to D.`);
      }

      return agent;
    });
  }

  // ── EXPERIMENT 5: Reward Structure Variation ──────────────────────────────

  // ── EXPERIMENT 5: Reward Structure Variation ──────────────────────────────
  //
  //   5 reward variants × 2 conditions (A, D) × 5 trials = 50 agent-runs
  //
  //   Each trial uses the SAME agent (single-agent), same environment density,
  //   but a different perFrameRewardFn and finalScoreFn so we can measure how
  //   well learned patterns (from Exp 1–4) generalise across customer objectives.
  //
  //   Variants:  baseline | efficiency | accuracy | speed | balance
  //   Conditions: A (no memory) vs D (full dual memory)
  //
  //   Key metrics: finalScore (variant-aware), foodEaten, wallBounces, ltmUsageRate

  async runExp5() {
    const cfg = EXP5_REWARD_VARIATION_CONFIG;

    // 5 variants × 2 conditions × 5 trials = 50
    this.progress.totalTrials =
      cfg.REWARD_VARIANTS.length *
      cfg.TEST_CONDITIONS.length *
      cfg.TRIALS_PER_CONFIG;
    this.progress.completedTrials = 0;

    for (const variant of cfg.REWARD_VARIANTS) {
      for (const condition of cfg.TEST_CONDITIONS) {
        for (let trial = 0; trial < cfg.TRIALS_PER_CONFIG; trial++) {
          if (this._stopped) return;

          const result = await this.runTrial({
            experiment: 5,
            condition,
            trial,
            agent: trial % 3,          // cycle agent index 0/1/2 for personality variation
            duration: this.sharedParams.TRIAL_DURATION,
            params: {
              rewardVariantName: variant.name,
              obstacleCount:     cfg.OBSTACLE_COUNT,
              foodCount:         cfg.FOOD_COUNT,
            },
          });

          this.results[5].push(result);
          this.progress.completedTrials++;
          this._emitProgress({
            currentVariant:    variant.name,
            variantLabel:      variant.label,
            variantEmoji:      variant.emoji,
            currentCondition:  condition,
            lastResult:        result,
          });
        }
      }
    }
  }

  // ── EXPERIMENT 5.5: Multi-Objective Learning ──────────────────────────────
  //
  //   Two-phase design per training variant:
  //
  //   Phase 1 — Training (10 trials):
  //     A single LTM pool accumulates patterns across all training trials.
  //     The per-frame reward fed to evaluateAction() and the consolidation engine
  //     is a weighted sum of all five EXP5 perFrameReward functions — so patterns
  //     that generalise across objectives get reinforced.
  //
  //   Phase 2 — Testing (5 objectives × 2 conditions × 5 trials = 50 per variant):
  //     Condition A — fresh agent, no LTM (reactive-Hopfield baseline)
  //     Condition D — fresh agent whose LTM is pre-seeded with patterns from Phase 1;
  //                   learning continues during the test trial.
  //
  //   2 variants × 10 training + 2 variants × 50 test = 20 + 100 = 120 total trials
  //   Est. runtime: 15–20 min

  async runExp5_5() {
    const cfg = EXP5_5_MULTI_OBJECTIVE_CONFIG;
    const testObjectives = Object.keys(cfg.REWARD_FUNCTIONS); // 5 keys

    // Build a name→EXP5-variant lookup so we can retrieve perFrameReward by key
    const exp5VariantByName = Object.fromEntries(
      EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
    );

    // Total progress: 2×10 training + 2×5×2×5 testing = 120
    this.progress.totalTrials =
      cfg.TRAINING_VARIANTS.length * cfg.TRAINING_TRIALS_PER_VARIANT +
      cfg.TRAINING_VARIANTS.length * testObjectives.length *
        cfg.TEST_CONDITIONS.length * cfg.TESTING_TRIALS_PER_OBJECTIVE;
    this.progress.completedTrials = 0;

    for (const variant of cfg.TRAINING_VARIANTS) {
      // ── Phase 1: Training ─────────────────────────────────────────────────
      // One shared LTM accumulates patterns across ALL training trials.
      const trainingLtm = new LongTermMemory(1000);

      // Combined per-frame reward: Σ weight[key] × perFrameReward_key(ate, frame, dur)
      const combinedPerFrameReward = (ate, frame, duration) =>
        Object.entries(variant.weights).reduce((sum, [key, w]) => {
          const fn = exp5VariantByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
          return sum + fn(ate, frame, duration) * w;
        }, 0);

      for (let trainTrial = 0; trainTrial < cfg.TRAINING_TRIALS_PER_VARIANT; trainTrial++) {
        if (this._stopped) return;

        const trainedPatterns = await this._runMultiObjTrainingTrial({
          trainingLtm,
          combinedPerFrameReward,
          obstacleCount: cfg.OBSTACLE_COUNT,
          foodCount:     cfg.FOOD_COUNT,
          duration:      this.sharedParams.TRIAL_DURATION,
        });

        this.progress.completedTrials++;
        this._emitProgress({
          phase:           'training',
          currentVariant:  variant.name,
          variantLabel:    variant.label,
          trainTrial:      trainTrial + 1,
          trainingTrials:  cfg.TRAINING_TRIALS_PER_VARIANT,
          trainedPatterns,
        });
      }

      const totalTrainedPatterns = trainingLtm.stats()?.totalPatterns ?? 0;

      // ── Phase 2: Testing ──────────────────────────────────────────────────
      for (const objectiveKey of testObjectives) {
        const objFinalFn   = cfg.REWARD_FUNCTIONS[objectiveKey];
        // Per-frame reward for the test trial uses the matching EXP5 variant signal
        const objPerFrameFn = exp5VariantByName[objectiveKey]?.perFrameReward
          ?? ((ate) => ate > 0 ? 1.0 : -0.01);

        for (const condition of cfg.TEST_CONDITIONS) {
          for (let trial = 0; trial < cfg.TESTING_TRIALS_PER_OBJECTIVE; trial++) {
            if (this._stopped) return;

            const result = await this._runMultiObjTestTrial({
              variant,
              trainingLtm:  condition === 'D' ? trainingLtm : null,
              condition,
              objectiveKey,
              objFinalFn,
              objPerFrameFn,
              obstacleCount: cfg.OBSTACLE_COUNT,
              foodCount:     cfg.FOOD_COUNT,
              duration:      this.sharedParams.TRIAL_DURATION,
              trial,
              totalTrainedPatterns,
            });

            this.results[5.5].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              phase:            'testing',
              currentVariant:   variant.name,
              variantLabel:     variant.label,
              currentObjective: objectiveKey,
              currentCondition: condition,
              totalTrainedPatterns,
              lastResult:       result,
            });
          }
        }
      }
    }
  }

  /**
   * Run one training trial that writes into a persistent shared LTM.
   * Agent is created fresh each trial but always writes to `trainingLtm`.
   * The combined multi-objective per-frame reward is used for all consolidation
   * decisions, so only patterns that generalise across objectives get reinforced.
   *
   * @returns {number} Total patterns in trainingLtm after this trial
   */
  async _runMultiObjTrainingTrial({ trainingLtm, combinedPerFrameReward, obstacleCount, foodCount, duration }) {
    const obstacles = _generateObstacles(obstacleCount);
    const foods     = _generateFood(foodCount, obstacles);

    // Fresh STM + Engine each trial but SHARED LTM carries over
    const stm    = new ShortTermMemory(MEMORY_CONFIG.STM_SIZE);
    const engine = new ConsolidationEngine(stm, trainingLtm, {
      windowSize:        30,
      rewardThreshold:   MEMORY_CONFIG.CONSOLIDATION_REWARD_THRESHOLD,
      surpriseThreshold: MEMORY_CONFIG.CONSOLIDATION_SURPRISE_THRESHOLD,
      periodicInterval:  MEMORY_CONFIG.CONSOLIDATION_PERIODIC_INTERVAL,
    });
    const controller = new DualMemoryController(trainingLtm, {
      ltmConfidenceThreshold: CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
      explorationRate:         CONTROLLER_CONFIG.EXPLORATION_RATE,
      actionWeightSTM:         CONTROLLER_CONFIG.ACTION_WEIGHT_STM,
    });
    const agent = { stm, ltm: trainingLtm, engine, controller, score: 0 };

    let ax     = SIM_W / 2 + (Math.random() - 0.5) * 40;
    let ay     = SIM_H / 2 + (Math.random() - 0.5) * 40;
    let aAngle = Math.random() * Math.PI * 2;
    let foodEaten = 0;

    for (let frame = 0; frame < duration; frame++) {
      if (this._stopped) break;
      if (frame % YIELD_EVERY === 0) await new Promise(r => setTimeout(r, 0));

      const sensors      = _getSensors(ax, ay, aAngle, foods, obstacles);
      const sensoryState = _encodeSensors(sensors.obs, sensors.food);
      const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, SIM_NOISE);
      const hopfieldAction     = _decodeMotor(newState);

      let action = hopfieldAction;
      if (agent.controller) {
        const context = _determineContext(sensors);
        const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
        action        = res.action;
      }

      const moved = _moveAgent(ax, ay, aAngle, action, obstacles);
      ax = moved.x; ay = moved.y; aAngle = moved.angle;

      const ate    = _checkFoodCollision(ax, ay, foods);
      foodEaten   += ate;
      agent.score  = foodEaten;

      // Combined multi-objective signal drives consolidation
      const reward = combinedPerFrameReward(ate, frame, duration);
      if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);
      if (agent.stm)        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
      if (agent.engine)     agent.engine.update(frame, frame);
    }

    return trainingLtm.stats()?.totalPatterns ?? 0;
  }

  /**
   * Run one test trial for a specific objective.
   *   Condition A — pure reactive Hopfield (no LTM)
   *   Condition D — starts with trained LTM patterns copied in; continues adapting
   *
   * @returns {object} Result record pushed to this.results[5.5]
   */
  async _runMultiObjTestTrial({
    variant, trainingLtm, condition, objectiveKey,
    objFinalFn, objPerFrameFn, obstacleCount, foodCount, duration, trial, totalTrainedPatterns,
  }) {
    const obstacles = _generateObstacles(obstacleCount);
    const foods     = _generateFood(foodCount, obstacles);

    let agent;
    if (condition === 'A') {
      // Reactive baseline — no memory at all
      agent = _createAgentForCondition('A');
    } else {
      // D: full dual memory, pre-seeded with trained patterns
      const testLtm    = new LongTermMemory(1000);
      const copiedCount = this._copyTrainedPatterns(trainingLtm, testLtm);

      const stm        = new ShortTermMemory(MEMORY_CONFIG.STM_SIZE);
      const engine     = new ConsolidationEngine(stm, testLtm, {
        windowSize:        30,
        rewardThreshold:   MEMORY_CONFIG.CONSOLIDATION_REWARD_THRESHOLD,
        surpriseThreshold: MEMORY_CONFIG.CONSOLIDATION_SURPRISE_THRESHOLD,
        periodicInterval:  MEMORY_CONFIG.CONSOLIDATION_PERIODIC_INTERVAL,
      });
      const controller = new DualMemoryController(testLtm, {
        ltmConfidenceThreshold: CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
        explorationRate:         CONTROLLER_CONFIG.EXPLORATION_RATE,
        actionWeightSTM:         CONTROLLER_CONFIG.ACTION_WEIGHT_STM,
      });
      agent = { stm, ltm: testLtm, engine, controller, score: 0, _copiedPatterns: copiedCount };
    }

    let ax     = SIM_W / 2 + (Math.random() - 0.5) * 40;
    let ay     = SIM_H / 2 + (Math.random() - 0.5) * 40;
    let aAngle = Math.random() * Math.PI * 2;
    let foodEaten  = 0;
    let wallBounces = 0;

    for (let frame = 0; frame < duration; frame++) {
      if (this._stopped) break;
      if (frame % YIELD_EVERY === 0) await new Promise(r => setTimeout(r, 0));

      const sensors      = _getSensors(ax, ay, aAngle, foods, obstacles);
      const sensoryState = _encodeSensors(sensors.obs, sensors.food);
      const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, SIM_NOISE);
      const hopfieldAction     = _decodeMotor(newState);

      let action = hopfieldAction;
      if (agent.controller) {
        const context = _determineContext(sensors);
        const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
        action        = res.action;
      }

      const moved = _moveAgent(ax, ay, aAngle, action, obstacles);
      ax = moved.x; ay = moved.y; aAngle = moved.angle;
      if (moved.bounced) wallBounces++;

      const ate    = _checkFoodCollision(ax, ay, foods);
      foodEaten   += ate;
      agent.score  = foodEaten;

      const reward = objPerFrameFn(ate, frame, duration);
      if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);
      if (agent.stm)        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
      if (agent.engine)     agent.engine.update(frame, frame);
    }

    // Terminal objective score (steps = duration, collisions = wallBounces)
    const objectiveScore = objFinalFn(foodEaten, duration, wallBounces, duration);

    const ltmS  = agent.ltm?.stats()        ?? {};
    const ctrlS = agent.controller?.stats() ?? {};

    return {
      experiment:          5.5,
      variant:             variant.name,
      variantLabel:        variant.label,
      condition,
      testObjective:       objectiveKey,
      trial,
      results: {
        foodCollected:           foodEaten,
        wallBounces,
        objectiveScore,
        ltmUsageRate:            +(ctrlS.ltmUsageRate   ?? 0).toFixed(3),
        ltmPatternCount:         ltmS.totalPatterns      ?? 0,
        trainedPatternsAvailable: totalTrainedPatterns,
        timestamp:               new Date().toISOString(),
      },
    };
  }

  // ── EXPERIMENT 5.5.5: Weight Optimisation ────────────────────────────────
  //
  //   Grid search over five weight combinations to find the training mix that
  //   maximises the mean D-vs-A advantage across all five test objectives.
  //
  //   Reuses _runMultiObjTrainingTrial and _runMultiObjTestTrial from Exp 5.5
  //   verbatim, so the only variable is the per-frame combined reward signal.
  //   Terminal scoring uses EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS to
  //   keep results directly comparable with Exp 5.5.
  //
  //   5 combos × (10 training + 5 obj × 2 cond × 5 testing) = 300 trials
  //   NOTE: '5.5.5' is a string key — not a valid JS float literal.
  //   Est. runtime: 25–35 min

  async runExp5_5_5() {
    const cfg          = EXP5_5_5_WEIGHT_OPTIMIZATION_CONFIG;
    const testObjectives = Object.keys(EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS);

    // EXP5 per-frame reward lookup (same source as runExp5_5)
    const exp5VariantByName = Object.fromEntries(
      EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
    );

    // 5 × 10 training + 5 × 5 obj × 2 cond × 5 testing = 300
    this.progress.totalTrials =
      cfg.WEIGHT_COMBINATIONS.length * cfg.TRAINING_TRIALS_PER_COMBO +
      cfg.WEIGHT_COMBINATIONS.length * testObjectives.length *
        cfg.TEST_CONDITIONS.length * cfg.TESTING_TRIALS_PER_OBJECTIVE;
    this.progress.completedTrials = 0;

    for (const combo of cfg.WEIGHT_COMBINATIONS) {
      // ── Phase 1: Training ───────────────────────────────────────────────
      const trainingLtm = new LongTermMemory(1000);

      const combinedPerFrameReward = (ate, frame, duration) =>
        Object.entries(combo.weights).reduce((sum, [key, w]) => {
          const fn = exp5VariantByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
          return sum + fn(ate, frame, duration) * w;
        }, 0);

      for (let trainTrial = 0; trainTrial < cfg.TRAINING_TRIALS_PER_COMBO; trainTrial++) {
        if (this._stopped) return;

        const trainedPatterns = await this._runMultiObjTrainingTrial({
          trainingLtm,
          combinedPerFrameReward,
          obstacleCount: cfg.OBSTACLE_COUNT,
          foodCount:     cfg.FOOD_COUNT,
          duration:      this.sharedParams.TRIAL_DURATION,
        });

        this.progress.completedTrials++;
        this._emitProgress({
          phase:          'training',
          currentCombo:   combo.name,
          comboLabel:     combo.label,
          trainTrial:     trainTrial + 1,
          trainingTrials: cfg.TRAINING_TRIALS_PER_COMBO,
          trainedPatterns,
        });
      }

      const totalTrainedPatterns = trainingLtm.stats()?.totalPatterns ?? 0;

      // ── Phase 2: Testing ────────────────────────────────────────────────
      for (const objectiveKey of testObjectives) {
        const objFinalFn   = EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS[objectiveKey];
        const objPerFrameFn = exp5VariantByName[objectiveKey]?.perFrameReward
          ?? ((ate) => ate > 0 ? 1.0 : -0.01);

        for (const condition of cfg.TEST_CONDITIONS) {
          for (let trial = 0; trial < cfg.TESTING_TRIALS_PER_OBJECTIVE; trial++) {
            if (this._stopped) return;

            // Delegate to the shared internal method, then rewrite metadata
            const raw = await this._runMultiObjTestTrial({
              variant:       { name: combo.name, label: combo.label, weights: combo.weights },
              trainingLtm:   condition === 'D' ? trainingLtm : null,
              condition,
              objectiveKey,
              objFinalFn,
              objPerFrameFn,
              obstacleCount: cfg.OBSTACLE_COUNT,
              foodCount:     cfg.FOOD_COUNT,
              duration:      this.sharedParams.TRIAL_DURATION,
              trial,
              totalTrainedPatterns,
            });

            const result = {
              experiment:       '5.5.5',
              weightCombo:      combo.name,
              comboLabel:       combo.label,
              comboDescription: combo.description,
              condition:        raw.condition,
              testObjective:    raw.testObjective,
              trial:            raw.trial,
              results:          raw.results,
            };

            this.results['5.5.5'].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              phase:               'testing',
              currentCombo:        combo.name,
              comboLabel:          combo.label,
              currentObjective:    objectiveKey,
              currentCondition:    condition,
              totalTrainedPatterns,
              lastResult:          result,
            });
          }
        }
      }
    }
  }

  /**
   * Deep-copy every LTMPattern from sourceLtm into destLtm using the
   * built-in fromJSON / toJSON round-trip — the only safe way to clone
   * a pattern without sharing mutable references.
   *
   * @param {LongTermMemory} sourceLtm
   * @param {LongTermMemory} destLtm
   * @returns {number} Number of patterns successfully copied
   */
  _copyTrainedPatterns(sourceLtm, destLtm) {
    if (!sourceLtm?.patterns) return 0;
    let copied = 0;
    for (const ctx of ['exploration', 'foraging', 'avoidance']) {
      const srcMap = sourceLtm.patterns[ctx];
      if (!srcMap) continue;
      for (const pat of srcMap.values()) {
        try {
          destLtm.storePattern(LTMPattern.fromJSON(pat.toJSON()));
          copied++;
        } catch {
          // skip malformed patterns (should never happen in normal operation)
        }
      }
    }
    return copied;
  }

  // ── EXPERIMENT 6: Transfer Learning ─────────────────────────────────────
  //
  //   Scientific question: Can patterns trained in the standard warehouse domain
  //   transfer to completely different domains when consolidation is frozen?
  //
  //   Phase 0 — Source training (10 trials):
  //     Train a single LTM using the smart_balance weight combination from
  //     Exp 5.5.5 (SOURCE_TRAINING_WEIGHTS). Consolidation is live during training.
  //
  //   Phase 1 — Transfer testing (5 domains × 2 conditions × 5 trials = 50 trials):
  //     Condition 'A'        — no memory (reactive Hopfield baseline)
  //     Condition 'frozen_D' — pre-trained LTM patterns loaded; engine = null
  //                            (consolidation frozen — no new patterns form)
  //
  //   10 training + 50 transfer = 60 total trials
  //   Est. runtime: 25–35 min

  async runExp6() {
    const cfg = EXP6_TRANSFER_LEARNING_CONFIG;

    // EXP5 per-frame reward lookup for the combined training signal
    const exp5VariantByName = Object.fromEntries(
      EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
    );

    // ── Phase 0: Source training ──────────────────────────────────────────
    const sourceLtm = new LongTermMemory(1000);

    const combinedPerFrameReward = (ate, frame, duration) =>
      Object.entries(cfg.SOURCE_TRAINING_WEIGHTS).reduce((sum, [key, w]) => {
        const fn = exp5VariantByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
        return sum + fn(ate, frame, duration) * w;
      }, 0);

    // 10 training + 5 domains × 2 conditions × 5 trials = 60 total
    this.progress.totalTrials =
      cfg.SOURCE_TRAINING_TRIALS +
      cfg.TRANSFER_DOMAINS.length * cfg.TEST_CONDITIONS.length * cfg.TRIALS_PER_DOMAIN;
    this.progress.completedTrials = 0;

    for (let trainTrial = 0; trainTrial < cfg.SOURCE_TRAINING_TRIALS; trainTrial++) {
      if (this._stopped) return;

      const trainedPatterns = await this._runMultiObjTrainingTrial({
        trainingLtm:          sourceLtm,
        combinedPerFrameReward,
        obstacleCount:        cfg.OBSTACLE_COUNT,
        foodCount:            cfg.FOOD_COUNT,
        duration:             this.sharedParams.TRIAL_DURATION,
      });

      this.progress.completedTrials++;
      this._emitProgress({
        phase:          'training',
        trainTrial:     trainTrial + 1,
        trainingTrials: cfg.SOURCE_TRAINING_TRIALS,
        trainedPatterns,
      });
    }

    const totalTrainedPatterns = sourceLtm.stats()?.totalPatterns ?? 0;

    // ── Phase 1: Transfer testing ─────────────────────────────────────────
    for (const domain of cfg.TRANSFER_DOMAINS) {
      for (const condition of cfg.TEST_CONDITIONS) {
        for (let trial = 0; trial < cfg.TRIALS_PER_DOMAIN; trial++) {
          if (this._stopped) return;

          const result = await this._runTransferTrial({
            domain,
            condition,
            trial,
            sourceLtm,
            obstacleCount:        cfg.OBSTACLE_COUNT,
            foodCount:            cfg.FOOD_COUNT,
            duration:             this.sharedParams.TRIAL_DURATION,
            totalTrainedPatterns,
          });

          this.results[6].push(result);
          this.progress.completedTrials++;
          this._emitProgress({
            phase:               'transfer',
            currentDomain:       domain.name,
            domainLabel:         domain.label,
            domainEmoji:         domain.emoji,
            currentCondition:    condition,
            totalTrainedPatterns,
            lastResult:          result,
          });
        }
      }
    }
  }

  /**
   * Run one transfer trial in the target domain.
   *   Condition 'A'        — reactive Hopfield, no memory
   *   Condition 'frozen_D' — pre-trained LTM; engine = null (frozen consolidation)
   *
   * Domain-specific stressors applied:
   *   gravityMultiplier — scales agent speed + turn rate
   *   noiseSigma        — adds to Hopfield noise floor (sensor degradation)
   *
   * @returns {object} Result record pushed to this.results[6]
   */
  async _runTransferTrial({
    domain, condition, trial, sourceLtm,
    obstacleCount, foodCount, duration, totalTrainedPatterns,
  }) {
    const obstacles = _generateObstacles(obstacleCount);
    const foods     = _generateFood(foodCount, obstacles);

    // Domain physics
    const effSpeed = SIM_SPD  * Math.max(0.02, domain.gravityMultiplier);
    const effTurn  = SIM_TURN * Math.max(0.02, domain.gravityMultiplier);

    let agent;
    if (condition === 'A') {
      // Reactive baseline — no memory
      agent = _createAgentForCondition('A');
    } else {
      // frozen_D: pre-trained LTM patterns loaded; consolidation engine = null
      const testLtm    = new LongTermMemory(1000);
      const copiedCount = this._copyTrainedPatterns(sourceLtm, testLtm);

      const stm        = new ShortTermMemory(MEMORY_CONFIG.STM_SIZE);
      const controller = new DualMemoryController(testLtm, {
        ltmConfidenceThreshold: CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
        explorationRate:         CONTROLLER_CONFIG.EXPLORATION_RATE,
        actionWeightSTM:         CONTROLLER_CONFIG.ACTION_WEIGHT_STM,
      });
      // engine = null — no new patterns can form (deployment mode)
      agent = { stm, ltm: testLtm, engine: null, controller, score: 0, _copiedPatterns: copiedCount };
    }

    let ax     = SIM_W / 2 + (Math.random() - 0.5) * 40;
    let ay     = SIM_H / 2 + (Math.random() - 0.5) * 40;
    let aAngle = Math.random() * Math.PI * 2;
    let foodEaten   = 0;
    let wallBounces = 0;

    for (let frame = 0; frame < duration; frame++) {
      if (this._stopped) break;
      if (frame % YIELD_EVERY === 0) await new Promise(r => setTimeout(r, 0));

      // Sensor noise: base Hopfield noise + domain noiseSigma
      const effectiveNoise = Math.min(0.95, SIM_NOISE + domain.noiseSigma);

      const sensors      = _getSensors(ax, ay, aAngle, foods, obstacles);
      const sensoryState = _encodeSensors(sensors.obs, sensors.food);
      const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, effectiveNoise);
      const hopfieldAction     = _decodeMotor(newState);

      let action = hopfieldAction;
      if (agent.controller) {
        const context = _determineContext(sensors);
        const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
        action        = res.action;
      }

      // Domain-scaled physics
      const moved = _moveAgent(ax, ay, aAngle, action, obstacles, effSpeed, effTurn);
      ax = moved.x; ay = moved.y; aAngle = moved.angle;
      if (moved.bounced) wallBounces++;

      const ate    = _checkFoodCollision(ax, ay, foods);
      foodEaten   += ate;
      agent.score  = foodEaten;

      const reward = domain.perFrameReward(ate, frame, duration);
      if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);
      if (agent.stm)        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
      // agent.engine is null for frozen_D — intentionally no consolidation
      if (agent.engine)     agent.engine.update(frame, frame);
    }

    const finalScore = domain.finalScore(foodEaten, duration, wallBounces);
    const ltmS  = agent.ltm?.stats()        ?? {};
    const ctrlS = agent.controller?.stats() ?? {};

    return {
      experiment:  6,
      domain:      domain.name,
      domainLabel: domain.label,
      domainEmoji: domain.emoji,
      condition,
      trial,
      results: {
        foodEaten,
        wallBounces,
        finalScore,
        ltmUsageRate:             +(ctrlS.ltmUsageRate   ?? 0).toFixed(3),
        ltmPatternCount:          ltmS.totalPatterns      ?? 0,
        trainedPatternsAvailable: totalTrainedPatterns,
        domainStressors: {
          gravityMultiplier: domain.gravityMultiplier,
          noiseSigma:        domain.noiseSigma,
        },
        timestamp: new Date().toISOString(),
      },
    };
  }

  // ── EXPERIMENT 8: Weight Optimization Framework ──────────────────────────
  //
  //   Extends Exp 5.5.5 from 5 to 10 weight configurations and reduces test
  //   trials per objective from 5 to 3, keeping runtime under 45 min.
  //
  //   Scientific question: Is 20/20/20/20/20 truly near-optimal, or do
  //   asymmetric weights produce a statistically significant D-vs-A improvement
  //   over the +11.27 % reference from prior experiments?
  //
  //   Hypotheses: speed emphasis | efficiency emphasis | robustness | compound | baseline
  //   10 configs × (10 training + 5 obj × 2 cond × 3 testing) = 400 trials
  //   Est. runtime: 35–45 min

  async runExp8() {
    const cfg          = EXP8_WEIGHT_OPTIMIZATION_CONFIG;
    const testObjectives = Object.keys(EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS); // 5 keys

    const exp5VariantByName = Object.fromEntries(
      EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
    );

    // 10 × 10 training + 10 × 5 × 2 × 3 testing = 100 + 300 = 400
    this.progress.totalTrials =
      cfg.WEIGHT_CONFIGURATIONS.length * cfg.TRAINING_TRIALS_PER_CONFIG +
      cfg.WEIGHT_CONFIGURATIONS.length * testObjectives.length *
        cfg.TEST_CONDITIONS.length * cfg.TESTING_TRIALS_PER_OBJECTIVE;
    this.progress.completedTrials = 0;

    for (const config of cfg.WEIGHT_CONFIGURATIONS) {
      // ── Phase 1: Training ───────────────────────────────────────────────
      const trainingLtm = new LongTermMemory(1000);

      const combinedPerFrameReward = (ate, frame, duration) =>
        Object.entries(config.weights).reduce((sum, [key, w]) => {
          const fn = exp5VariantByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
          return sum + fn(ate, frame, duration) * w;
        }, 0);

      for (let trainTrial = 0; trainTrial < cfg.TRAINING_TRIALS_PER_CONFIG; trainTrial++) {
        if (this._stopped) return;

        const trainedPatterns = await this._runMultiObjTrainingTrial({
          trainingLtm,
          combinedPerFrameReward,
          obstacleCount: cfg.OBSTACLE_COUNT,
          foodCount:     cfg.FOOD_COUNT,
          duration:      this.sharedParams.TRIAL_DURATION,
        });

        this.progress.completedTrials++;
        this._emitProgress({
          phase:          'training',
          currentConfig:  config.name,
          configLabel:    config.label,
          hypothesis:     config.hypothesis,
          trainTrial:     trainTrial + 1,
          trainingTrials: cfg.TRAINING_TRIALS_PER_CONFIG,
          trainedPatterns,
        });
      }

      const totalTrainedPatterns = trainingLtm.stats()?.totalPatterns ?? 0;

      // ── Phase 2: Testing ────────────────────────────────────────────────
      for (const objectiveKey of testObjectives) {
        const objFinalFn   = EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS[objectiveKey];
        const objPerFrameFn = exp5VariantByName[objectiveKey]?.perFrameReward
          ?? ((ate) => ate > 0 ? 1.0 : -0.01);

        for (const condition of cfg.TEST_CONDITIONS) {
          for (let trial = 0; trial < cfg.TESTING_TRIALS_PER_OBJECTIVE; trial++) {
            if (this._stopped) return;

            const raw = await this._runMultiObjTestTrial({
              variant:       { name: config.name, label: config.label, weights: config.weights },
              trainingLtm:   condition === 'D' ? trainingLtm : null,
              condition,
              objectiveKey,
              objFinalFn,
              objPerFrameFn,
              obstacleCount: cfg.OBSTACLE_COUNT,
              foodCount:     cfg.FOOD_COUNT,
              duration:      this.sharedParams.TRIAL_DURATION,
              trial,
              totalTrainedPatterns,
            });

            const result = {
              experiment:   8,
              weightConfig: config.name,
              configLabel:  config.label,
              hypothesis:   config.hypothesis,
              weightsArray: config.weightsArray, // [F, S, A, B, E] as integers
              condition:    raw.condition,
              testObjective: raw.testObjective,
              trial:        raw.trial,
              results:      raw.results,
            };

            this.results[8].push(result);
            this.progress.completedTrials++;
            this._emitProgress({
              phase:               'testing',
              currentConfig:       config.name,
              configLabel:         config.label,
              hypothesis:          config.hypothesis,
              currentObjective:    objectiveKey,
              currentCondition:    condition,
              totalTrainedPatterns,
              lastResult:          result,
            });
          }
        }
      }
    }
  }

  // ── EXPERIMENT 9: Learning Dynamics & Curves ─────────────────────────────
  //
  //   Scientific question: How does D-vs-A generalisation advantage grow with
  //   the number of training trials, and where does it plateau?
  //
  //   For each of 6 checkpoints [0, 2, 5, 10, 20, 40] training trials:
  //     - 2 independent fresh-LTM repetitions
  //     - Each rep: train for exactly N trials, then test across 5 objectives
  //                 × 2 conditions × 2 trials = 20 test trials
  //
  //   Curve shape classified (exponential / sigmoid / linear).
  //   Convergence = first checkpoint reaching 95 % of peak advantage.
  //   Minimum viable = first checkpoint reaching 90 % of peak advantage.
  //   Overfitting = last checkpoint drops > 0.3 % below second-to-last.
  //
  //   Σ training = 154 trials   Σ testing = 240 trials   Total = 394 trials
  //   Est. runtime: 30–35 min

  async runExp9() {
    const cfg          = EXP9_LEARNING_DYNAMICS_CONFIG;
    const testObjectives = Object.keys(EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS); // 5

    const exp5VariantByName = Object.fromEntries(
      EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
    );

    // Combined per-frame reward using equal weights
    const combinedPerFrameReward = (ate, frame, duration) =>
      Object.entries(cfg.TRAINING_WEIGHTS).reduce((sum, [key, w]) => {
        const fn = exp5VariantByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
        return sum + fn(ate, frame, duration) * w;
      }, 0);

    // Total: Σ(N × reps) training + checkpoints × reps × 5 obj × 2 cond × 2 test
    //      = (0+2+5+10+20+40) × 2 + 6×2×5×2×2 = 154 + 240 = 394
    this.progress.totalTrials =
      cfg.CHECKPOINTS.reduce((s, n) => s + n, 0) * cfg.REPS_PER_CHECKPOINT +
      cfg.CHECKPOINTS.length * cfg.REPS_PER_CHECKPOINT * testObjectives.length *
        cfg.TEST_CONDITIONS.length * cfg.TESTING_TRIALS_PER_OBJECTIVE;
    this.progress.completedTrials = 0;

    for (const checkpoint of cfg.CHECKPOINTS) {
      for (let rep = 0; rep < cfg.REPS_PER_CHECKPOINT; rep++) {
        if (this._stopped) return;

        // ── Fresh LTM per (checkpoint, rep) — each is fully independent ──
        const trainingLtm = new LongTermMemory(1000);

        // ── Phase 1: Train for exactly `checkpoint` full physics trials ───
        for (let trainTrial = 0; trainTrial < checkpoint; trainTrial++) {
          if (this._stopped) return;

          const trainedPatterns = await this._runMultiObjTrainingTrial({
            trainingLtm,
            combinedPerFrameReward,
            obstacleCount: cfg.OBSTACLE_COUNT,
            foodCount:     cfg.FOOD_COUNT,
            duration:      this.sharedParams.TRIAL_DURATION,
          });

          this.progress.completedTrials++;
          this._emitProgress({
            phase:           'training',
            checkpoint,
            rep,
            trainTrial:      trainTrial + 1,
            totalTrainTrials: checkpoint,
            trainedPatterns,
          });
        }

        const totalTrainedPatterns = trainingLtm.stats()?.totalPatterns ?? 0;

        // ── Phase 2: Test across objectives × conditions × trials ─────────
        for (const objectiveKey of testObjectives) {
          const objFinalFn   = EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS[objectiveKey];
          const objPerFrameFn = exp5VariantByName[objectiveKey]?.perFrameReward
            ?? ((ate) => ate > 0 ? 1.0 : -0.01);

          for (const condition of cfg.TEST_CONDITIONS) {
            for (let trial = 0; trial < cfg.TESTING_TRIALS_PER_OBJECTIVE; trial++) {
              if (this._stopped) return;

              const raw = await this._runMultiObjTestTrial({
                variant: {
                  name:    `checkpoint_${checkpoint}`,
                  label:   `${checkpoint} training trials`,
                  weights: cfg.TRAINING_WEIGHTS,
                },
                trainingLtm:   condition === 'D' ? trainingLtm : null,
                condition,
                objectiveKey,
                objFinalFn,
                objPerFrameFn,
                obstacleCount: cfg.OBSTACLE_COUNT,
                foodCount:     cfg.FOOD_COUNT,
                duration:      this.sharedParams.TRIAL_DURATION,
                trial,
                totalTrainedPatterns,
              });

              const result = {
                experiment:    9,
                checkpoint,               // training trials for this rep
                rep,                      // repetition index (0 or 1)
                condition:     raw.condition,
                testObjective: raw.testObjective,
                trial:         raw.trial,
                results:       raw.results,
              };

              this.results[9].push(result);
              this.progress.completedTrials++;
              this._emitProgress({
                phase:               'testing',
                checkpoint,
                rep,
                currentObjective:    objectiveKey,
                currentCondition:    condition,
                totalTrainedPatterns,
                lastResult:          result,
              });
            }
          }
        }
      }
    }
  }

  /**
   * Welch-pooled two-sample t-test.
   * Tests whether two arrays of numbers have significantly different means.
   * |t| > 2 approximates p < 0.05 for the degrees of freedom seen here.
   *
   * @param {number[]} arr1
   * @param {number[]} arr2
   * @returns {{ tStatistic, degreesOfFreedom, isSignificant, pValueApprox }}
   */
  _tTest(arr1, arr2) {
    if (arr1.length < 2 || arr2.length < 2) {
      return { tStatistic: null, degreesOfFreedom: 0, isSignificant: false, pValueApprox: 'n/a' };
    }
    const mean1 = arr1.reduce((a, b) => a + b, 0) / arr1.length;
    const mean2 = arr2.reduce((a, b) => a + b, 0) / arr2.length;
    const n1    = arr1.length, n2 = arr2.length;
    const var1  = arr1.reduce((s, x) => s + (x - mean1) ** 2, 0) / (n1 - 1);
    const var2  = arr2.reduce((s, x) => s + (x - mean2) ** 2, 0) / (n2 - 1);
    const pooled = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
    if (pooled === 0) return { tStatistic: 0, degreesOfFreedom: n1 + n2 - 2, isSignificant: false, pValueApprox: '1.0' };
    const t    = (mean1 - mean2) / (pooled * Math.sqrt(1 / n1 + 1 / n2));
    const df   = n1 + n2 - 2;
    const isSig = Math.abs(t) > 2;
    return {
      tStatistic:      +t.toFixed(3),
      degreesOfFreedom: df,
      isSignificant:   isSig,
      pValueApprox:    isSig ? '< 0.05' : '> 0.05',
    };
  }

  // ── Core simulation ───────────────────────────────────────────────────────

  /**
   * Run one agent-trial with self-contained physics.
   * Yields to the JS event loop every YIELD_EVERY frames so React stays responsive.
   */
  async runTrial(config) {
    const {
      experiment, condition, trial, agent: agentIdx,
      duration = this.sharedParams.TRIAL_DURATION,
      params   = {},
    } = config;
    const {
      obstacleCount    = 0,
      foodCount        = 12,
      noiseLevel,
      // ── Exp 3 robustness stressors ──────────────────────────
      noiseSigma        = 0,    // Gaussian sensor noise σ (warehouse)
      gravityMultiplier = 1.0,  // scales speed + turn rate (physics / space)
      radiationRate     = 0,    // per-frame LTM pattern corruption prob (space)
      driftRate         = 0,    // sensor offset growth rate per frame (space)
      // ── Exp 5 reward variant ─────────────────────────────────
      rewardVariantName = null, // name key into EXP5_REWARD_VARIATION_CONFIG
    } = params;

    // Resolve reward functions — fall back to baseline when not set (Exp 1–4)
    const _rewardVariant = rewardVariantName
      ? EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.find(v => v.name === rewardVariantName)
      : null;
    const perFrameRewardFn = _rewardVariant?.perFrameReward ?? ((ate) => ate > 0 ? 1.0 : -0.01);
    const finalScoreFn     = _rewardVariant?.finalScore     ?? ((food) => food);

    const noise    = noiseLevel !== undefined ? noiseLevel : SIM_NOISE;
    // Physics scaled by gravity (microgravity → very slow / imprecise movement)
    const effSpeed = SIM_SPD  * Math.max(0.02, gravityMultiplier); // floor 2 % to prevent total freeze
    const effTurn  = SIM_TURN * Math.max(0.02, gravityMultiplier);

    // Build environment
    const obstacles = _generateObstacles(obstacleCount);
    const foods     = _generateFood(foodCount, obstacles);

    // Create agent wired for this condition
    const agent = _createAgentForCondition(condition);

    // Initial placement near world centre
    let ax     = SIM_W / 2 + (Math.random() - 0.5) * 40;
    let ay     = SIM_H / 2 + (Math.random() - 0.5) * 40;
    let aAngle = Math.random() * Math.PI * 2;

    let foodEaten       = 0;
    let wallBounces     = 0;   // wall + obstacle collision counter (used by Exp 5 accuracy/balance)
    const learningCurve = [];

    for (let frame = 0; frame < duration; frame++) {
      if (this._stopped) break;

      // Yield every YIELD_EVERY frames
      if (frame % YIELD_EVERY === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      // 1. Sense
      const sensors      = _getSensors(ax, ay, aAngle, foods, obstacles);
      const sensoryState = _encodeSensors(sensors.obs, sensors.food);

      // 2. Hopfield forward pass
      // Effective noise = base + Gaussian sigma + cumulative sensor drift
      const driftFlip      = Math.min(0.45, driftRate * frame / 100);
      const effectiveNoise = Math.min(0.95, noise + noiseSigma + driftFlip);
      const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, effectiveNoise);
      const hopfieldAction     = _decodeMotor(newState);

      // 3. Memory-conditioned action selection
      let action = hopfieldAction;
      if (agent.controller) {
        const context = _determineContext(sensors);
        const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
        action        = res.action;
      }

      // 4. Move (gravity-scaled speed and turn rate)
      const moved = _moveAgent(ax, ay, aAngle, action, obstacles, effSpeed, effTurn);
      ax = moved.x; ay = moved.y; aAngle = moved.angle;
      if (moved.bounced) wallBounces++;   // Exp 5 accuracy / balance penalty

      // 5. Food + reward (variant-aware)
      const ate    = _checkFoodCollision(ax, ay, foods);
      foodEaten   += ate;
      agent.score  = foodEaten;
      const reward = perFrameRewardFn(ate, frame, duration);

      // 6. LTM feedback
      if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);

      // 7. STM recording
      if (agent.stm) {
        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
      }

      // 8. Consolidation + radiation damage (space stressor)
      if (agent.engine) agent.engine.update(frame, frame);
      if (radiationRate > 0) _applyRadiationDamage(agent.ltm, radiationRate);

      // 9. Learning curve (10 points across 2400 frames)
      if ((frame + 1) % 240 === 0) learningCurve.push(foodEaten);
    }

    const ltmS  = agent.ltm?.stats()        ?? {};
    const engS  = agent.engine?.getStats()  ?? {};
    const ctrlS = agent.controller?.stats() ?? {};

    const finalScore = finalScoreFn(agent.score, duration, wallBounces);

    return {
      experiment, condition, trial, agent: agentIdx, params,
      results: {
        foodEaten:               agent.score,
        wallBounces,
        finalScore,
        rewardVariant:           rewardVariantName ?? 'baseline',
        variantLabel:            _rewardVariant?.label ?? 'Baseline: Maximize Food',
        patternsConsolidated:    engS.newPatterns          ?? 0,
        consolidationsTriggered: engS.totalConsolidations  ?? 0,
        avgPatternReliability:   ltmS.avgReliability       ?? 0,
        ltmUsageRate:            ctrlS.ltmUsageRate        ?? 0,
        ltmPatternCount:         ltmS.totalPatterns        ?? 0,
        learningCurve,
        // Robustness stressor metadata (non-zero only in Exp 3)
        stressors: { noiseSigma, gravityMultiplier, radiationRate, driftRate },
        timestamp: new Date().toISOString(),
      },
    };
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _emitProgress(extra = {}) {
    const pct = this.progress.totalTrials > 0
      ? ((this.progress.completedTrials / this.progress.totalTrials) * 100).toFixed(1)
      : '0.0';
    if (typeof this.onProgressUpdate === 'function') {
      this.onProgressUpdate({
        completedTrials:   this.progress.completedTrials,
        totalTrials:       this.progress.totalTrials,
        percentComplete:   pct,
        currentExperiment: this.progress.currentExperiment,
        ...extra,
      });
    }
  }

  /** Trigger a browser file download. Must be called from a user-gesture context or async. */
  _downloadJSON(data, filename) {
    try {
      const json = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = filename;
      // Append to DOM — required by Firefox and some Chromium builds
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Delay revoke so the browser has time to start the download
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      console.log(`[V2] ✓ Saved: ${filename}`);
    } catch (err) {
      console.error(`[V2] Export failed for "${filename}":`, err);
    }
  }

  computeStatistics(results) {
    if (!results.length) return null;
    const vals = results.map(r => r.results.foodEaten);
    const n    = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
    return {
      n,
      mean:  +mean.toFixed(3),
      std:   +std.toFixed(3),
      sem:   +(std / Math.sqrt(n)).toFixed(3),
      min:   +Math.min(...vals).toFixed(3),
      max:   +Math.max(...vals).toFixed(3),
    };
  }

  computeExperimentSummary(expNum) {
    const results = this.results[expNum];
    if (!results.length) return {};

    // ── Experiment 5: group by variant × condition, use finalScore ────────────
    if (expNum === 5) {
      const summary = {};
      const variants   = [...new Set(results.map(r => r.results.rewardVariant))];
      const conditions = [...new Set(results.map(r => r.condition))];

      for (const variant of variants) {
        summary[variant] = {};
        for (const cond of conditions) {
          const filtered = results.filter(
            r => r.results.rewardVariant === variant && r.condition === cond
          );
          const vals = filtered
            .map(r => r.results.finalScore ?? r.results.foodEaten)
            .filter(v => typeof v === 'number' && !isNaN(v));
          if (!vals.length) { summary[variant][cond] = null; continue; }
          const n    = vals.length;
          const mean = vals.reduce((a, b) => a + b, 0) / n;
          const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
          const variantMeta = EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.find(v => v.name === variant);
          summary[variant][cond] = {
            n,
            variantLabel: variantMeta?.label ?? variant,
            variantEmoji: variantMeta?.emoji ?? '',
            mean:  +mean.toFixed(3),
            std:   +std.toFixed(3),
            sem:   +(std / Math.sqrt(n)).toFixed(3),
            min:   +Math.min(...vals).toFixed(3),
            max:   +Math.max(...vals).toFixed(3),
            // Also include raw food counts for cross-variant comparison
            meanFoodEaten: +(filtered.reduce((s, r) => s + (r.results.foodEaten ?? 0), 0) / n).toFixed(3),
            meanWallBounces: +(filtered.reduce((s, r) => s + (r.results.wallBounces ?? 0), 0) / n).toFixed(1),
          };
        }
        // Generalisation index: D-vs-A advantage (higher = memory helps more)
        if (summary[variant].D && summary[variant].A) {
          const dMean = summary[variant].D.mean;
          const aMean = summary[variant].A.mean;
          summary[variant]._generalizationIndex = aMean !== 0
            ? +((dMean - aMean) / Math.abs(aMean) * 100).toFixed(1) // % improvement
            : null;
        }
      }
      return summary;
    }

    // ── Experiment 5.5.5: group by weightCombo × objective × condition ───────
    if (expNum === '5.5.5') {
      const summary  = {};
      const combos   = [...new Set(results.map(r => r.weightCombo))];
      const objectives = [...new Set(results.map(r => r.testObjective))];
      const conditions = [...new Set(results.map(r => r.condition))];

      for (const combo of combos) {
        const comboMeta = results.find(r => r.weightCombo === combo);
        summary[combo] = {
          _comboLabel:       comboMeta?.comboLabel       ?? combo,
          _comboDescription: comboMeta?.comboDescription ?? '',
        };

        const genIndices = [];

        for (const obj of objectives) {
          summary[combo][obj] = {};
          const condStats = {};
          for (const cond of conditions) {
            const filtered = results.filter(
              r => r.weightCombo === combo && r.testObjective === obj && r.condition === cond
            );
            const vals = filtered
              .map(r => r.results.objectiveScore)
              .filter(v => typeof v === 'number' && !isNaN(v));
            if (!vals.length) { condStats[cond] = null; continue; }
            const n    = vals.length;
            const mean = vals.reduce((a, b) => a + b, 0) / n;
            const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
            condStats[cond] = {
              n,
              mean:           +mean.toFixed(3),
              std:            +std.toFixed(3),
              sem:            +(std / Math.sqrt(n)).toFixed(3),
              min:            +Math.min(...vals).toFixed(3),
              max:            +Math.max(...vals).toFixed(3),
              meanFoodEaten:  +(filtered.reduce((s, r) => s + (r.results.foodCollected ?? 0), 0) / n).toFixed(3),
              meanWallBounces: +(filtered.reduce((s, r) => s + (r.results.wallBounces ?? 0), 0) / n).toFixed(1),
            };
          }
          summary[combo][obj] = condStats;
          if (condStats.D && condStats.A && condStats.A.mean !== 0) {
            const gi = +((condStats.D.mean - condStats.A.mean) / Math.abs(condStats.A.mean) * 100).toFixed(1);
            summary[combo][obj]._generalizationIndex = gi;
            genIndices.push(gi);
          }
        }

        // Average generalisation index across all objectives — primary ranking metric
        summary[combo]._avgGeneralizationIndex = genIndices.length
          ? +(genIndices.reduce((a, b) => a + b, 0) / genIndices.length).toFixed(1)
          : null;
      }

      // Rank combos by _avgGeneralizationIndex descending
      const ranked = combos
        .filter(c => summary[c]._avgGeneralizationIndex !== null)
        .sort((a, b) => summary[b]._avgGeneralizationIndex - summary[a]._avgGeneralizationIndex);
      summary._ranking = ranked;
      summary._winner  = ranked[0] ?? null;

      return summary;
    }

    // ── Experiment 5.5: group by variant × objective × condition ─────────────
    if (expNum === 5.5) {
      const summary    = {};
      const variants   = [...new Set(results.map(r => r.variant))];
      const objectives = [...new Set(results.map(r => r.testObjective))];
      const conditions = [...new Set(results.map(r => r.condition))];

      for (const vName of variants) {
        summary[vName] = { _variantLabel: results.find(r => r.variant === vName)?.variantLabel ?? vName };
        for (const obj of objectives) {
          summary[vName][obj] = {};
          const condStats = {};
          for (const cond of conditions) {
            const filtered = results.filter(
              r => r.variant === vName && r.testObjective === obj && r.condition === cond
            );
            const vals = filtered
              .map(r => r.results.objectiveScore)
              .filter(v => typeof v === 'number' && !isNaN(v));
            if (!vals.length) { condStats[cond] = null; continue; }
            const n    = vals.length;
            const mean = vals.reduce((a, b) => a + b, 0) / n;
            const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
            condStats[cond] = {
              n,
              mean:           +mean.toFixed(3),
              std:            +std.toFixed(3),
              sem:            +(std / Math.sqrt(n)).toFixed(3),
              min:            +Math.min(...vals).toFixed(3),
              max:            +Math.max(...vals).toFixed(3),
              meanFoodEaten:  +(filtered.reduce((s, r) => s + (r.results.foodCollected ?? 0), 0) / n).toFixed(3),
              meanWallBounces: +(filtered.reduce((s, r) => s + (r.results.wallBounces ?? 0), 0) / n).toFixed(1),
            };
          }
          summary[vName][obj] = condStats;
          // Generalisation index: D-vs-A advantage for this objective+variant
          if (condStats.D && condStats.A && condStats.A.mean !== 0) {
            summary[vName][obj]._generalizationIndex =
              +((condStats.D.mean - condStats.A.mean) / Math.abs(condStats.A.mean) * 100).toFixed(1);
          }
        }
        // Average generalisation index across all objectives for this variant
        const genIndices = objectives
          .map(obj => summary[vName][obj]?._generalizationIndex)
          .filter(v => typeof v === 'number' && !isNaN(v));
        summary[vName]._avgGeneralizationIndex = genIndices.length
          ? +(genIndices.reduce((a, b) => a + b, 0) / genIndices.length).toFixed(1)
          : null;
      }
      return summary;
    }

    // ── Experiment 9: learning curve across training checkpoints ─────────────
    if (expNum === 9) {
      const cfg        = EXP9_LEARNING_DYNAMICS_CONFIG;
      const summary    = {};
      const checkpoints = [...new Set(results.map(r => r.checkpoint))].sort((a, b) => a - b);
      const objectives  = [...new Set(results.map(r => r.testObjective))];
      const conditions  = [...new Set(results.map(r => r.condition))];

      const advantageCurve = []; // [{trials, advantage, std, ltmPatterns}]

      for (const cp of checkpoints) {
        summary[cp] = {};
        const genIndices = []; // one per objective (averaged across reps+trials)

        for (const obj of objectives) {
          const condStats = {};
          for (const cond of conditions) {
            const filtered = results.filter(
              r => r.checkpoint === cp && r.testObjective === obj && r.condition === cond
            );
            const vals = filtered
              .map(r => r.results.objectiveScore)
              .filter(v => typeof v === 'number' && !isNaN(v));
            if (!vals.length) { condStats[cond] = null; continue; }
            const n    = vals.length;
            const mean = vals.reduce((a, b) => a + b, 0) / n;
            const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
            condStats[cond] = {
              n,
              mean:            +mean.toFixed(3),
              std:             +std.toFixed(3),
              sem:             +(std / Math.sqrt(n)).toFixed(3),
              min:             +Math.min(...vals).toFixed(3),
              max:             +Math.max(...vals).toFixed(3),
              meanFoodEaten:   +(filtered.reduce((s, r) => s + (r.results.foodCollected ?? 0), 0) / n).toFixed(3),
              meanWallBounces: +(filtered.reduce((s, r) => s + (r.results.wallBounces    ?? 0), 0) / n).toFixed(1),
            };
          }
          summary[cp][obj] = condStats;

          if (condStats.D && condStats.A && condStats.A.mean !== 0) {
            const gi = +((condStats.D.mean - condStats.A.mean) / Math.abs(condStats.A.mean) * 100).toFixed(1);
            summary[cp][obj]._generalizationIndex = gi;
            genIndices.push(gi);
          }
        }

        // Mean gen-index for this checkpoint (across objectives)
        const meanAdv = genIndices.length
          ? +(genIndices.reduce((a, b) => a + b, 0) / genIndices.length).toFixed(2)
          : null;
        const stdAdv = genIndices.length > 1
          ? +(Math.sqrt(
              genIndices.reduce((s, x) => s + (x - genIndices.reduce((a, b) => a + b, 0) / genIndices.length) ** 2, 0)
              / genIndices.length
            )).toFixed(2)
          : 0;

        // Mean LTM pattern count for D-condition trials at this checkpoint
        const dTrials = results.filter(r => r.checkpoint === cp && r.condition === 'D');
        const ltmPatternsMean = dTrials.length
          ? +(dTrials.reduce((s, r) => s + (r.results.ltmPatternCount ?? 0), 0) / dTrials.length).toFixed(1)
          : 0;

        summary[cp]._meanAdvantage   = meanAdv;
        summary[cp]._stdAdvantage    = stdAdv;
        summary[cp]._ltmPatternsMean = ltmPatternsMean;

        advantageCurve.push({ trials: cp, advantage: meanAdv, std: stdAdv, ltmPatterns: ltmPatternsMean });
      }

      // ── Curve shape ──────────────────────────────────────────────────────
      const validCurve = advantageCurve.filter(p => typeof p.advantage === 'number');
      const gains = [];
      for (let i = 1; i < validCurve.length; i++) {
        gains.push(validCurve[i].advantage - validCurve[i - 1].advantage);
      }

      let curveType = 'unknown';
      if (gains.length >= 3) {
        const early = (gains[0] ?? 0) + (gains[1] ?? 0);
        const late  = (gains[gains.length - 1] ?? 0) + (gains[gains.length - 2] ?? 0);
        curveType = early > late ? 'exponential' : early < late ? 'sigmoid' : 'linear';
      }

      // ── Convergence & minimum viable points ─────────────────────────────
      const maxAdvantage = Math.max(...validCurve.map(p => p.advantage ?? -Infinity));
      const convergenceThr = cfg.CONVERGENCE_THRESHOLD * maxAdvantage;
      const minViableThr   = cfg.MIN_VIABLE_THRESHOLD  * maxAdvantage;

      const convergencePoint = validCurve.find(p => (p.advantage ?? 0) >= convergenceThr);
      const minViablePoint   = validCurve.find(p => (p.advantage ?? 0) >= minViableThr);

      // ── Overfitting detection (last two checkpoints) ─────────────────────
      const lastTwo = validCurve.slice(-2);
      const overfitting = lastTwo.length === 2
        ? {
            detected:       (lastTwo[0].advantage - lastTwo[1].advantage) > cfg.OVERFITTING_THRESHOLD,
            degradation:    +(lastTwo[0].advantage - lastTwo[1].advantage).toFixed(2),
            fromCheckpoint: lastTwo[0].trials,
            toCheckpoint:   lastTwo[1].trials,
          }
        : null;

      // ── Deployment guidance ──────────────────────────────────────────────
      const minViableAdv = minViablePoint
        ? summary[minViablePoint.trials]?._meanAdvantage
        : null;
      const convergenceAdv = convergencePoint
        ? summary[convergencePoint.trials]?._meanAdvantage
        : null;
      const maxCp = checkpoints[checkpoints.length - 1];
      const maxAdv = summary[maxCp]?._meanAdvantage;

      const deploymentGuidance = {
        quickDeployment: {
          trials:      minViablePoint?.trials ?? 'beyond range',
          advantage:   minViableAdv !== null ? `+${minViableAdv}%` : 'n/a',
          explanation: minViablePoint ? `90% of peak advantage at ${minViablePoint.trials} trials` : 'performance still rising',
          useCase:     'Time-critical deployments',
        },
        standardDeployment: {
          trials:      convergencePoint?.trials ?? 'beyond range',
          advantage:   convergenceAdv !== null ? `+${convergenceAdv}%` : 'n/a',
          explanation: convergencePoint ? `Converges (95% of peak) at ${convergencePoint.trials} trials` : 'not yet converged',
          useCase:     'Standard production deployment',
        },
        maximumPerformance: {
          trials:      maxCp,
          advantage:   maxAdv !== null ? `+${maxAdv}%` : 'n/a',
          caveat:      overfitting?.detected ? `Possible degradation from ${overfitting.fromCheckpoint} to ${overfitting.toCheckpoint} trials` : 'No overfitting detected',
          useCase:     'Performance-critical applications',
        },
      };

      summary._advantageCurve     = advantageCurve;
      summary._checkpoints        = checkpoints;
      summary._maxAdvantage       = maxAdvantage;
      summary._convergencePoint   = convergencePoint?.trials ?? null;
      summary._minViablePoint     = minViablePoint?.trials   ?? null;
      summary._curveType          = curveType;
      summary._overfitting        = overfitting;
      summary._deploymentGuidance = deploymentGuidance;
      summary._baselineAdvantage  = cfg.BASELINE_ADVANTAGE;
      summary._interpretation     = convergencePoint
        ? `${curveType} curve — reaches 95% of peak at ~${convergencePoint.trials} training trials`
        : `${curveType} curve — does not converge within tested range`;
      summary._recommendation     = minViablePoint
        ? `Min viable: ${minViablePoint.trials} trials (90% of peak). ` +
          `Full convergence: ${convergencePoint?.trials ?? '>40'} trials. ` +
          (overfitting?.detected ? `⚠ Overfitting at ${overfitting.fromCheckpoint}→${overfitting.toCheckpoint} trials.` : 'No overfitting.')
        : 'Increase checkpoint range — plateau not yet reached.';

      return summary;
    }

    // ── Experiment 8: group by config × objective × condition; t-test vs baseline ─
    if (expNum === 8) {
      const summary    = {};
      const configs    = [...new Set(results.map(r => r.weightConfig))];
      const objectives = [...new Set(results.map(r => r.testObjective))];
      const conditions = [...new Set(results.map(r => r.condition))];

      // Per-config gen-index arrays (one per objective) for the t-test
      const allGenIndices = {};

      for (const cfgName of configs) {
        const cfgMeta = results.find(r => r.weightConfig === cfgName);
        summary[cfgName] = {
          _configLabel:  cfgMeta?.configLabel  ?? cfgName,
          _hypothesis:   cfgMeta?.hypothesis   ?? '',
          _weightsArray: cfgMeta?.weightsArray ?? [],
        };

        const genIndices = [];
        allGenIndices[cfgName] = [];

        for (const obj of objectives) {
          const condStats = {};
          for (const cond of conditions) {
            const filtered = results.filter(
              r => r.weightConfig === cfgName && r.testObjective === obj && r.condition === cond
            );
            const vals = filtered
              .map(r => r.results.objectiveScore)
              .filter(v => typeof v === 'number' && !isNaN(v));
            if (!vals.length) { condStats[cond] = null; continue; }
            const n    = vals.length;
            const mean = vals.reduce((a, b) => a + b, 0) / n;
            const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
            condStats[cond] = {
              n,
              mean:            +mean.toFixed(3),
              std:             +std.toFixed(3),
              sem:             +(std / Math.sqrt(n)).toFixed(3),
              min:             +Math.min(...vals).toFixed(3),
              max:             +Math.max(...vals).toFixed(3),
              meanFoodEaten:   +(filtered.reduce((s, r) => s + (r.results.foodCollected ?? 0), 0) / n).toFixed(3),
              meanWallBounces: +(filtered.reduce((s, r) => s + (r.results.wallBounces    ?? 0), 0) / n).toFixed(1),
              _rawVals:        vals, // kept for t-test input
            };
          }
          summary[cfgName][obj] = condStats;

          if (condStats.D && condStats.A && condStats.A.mean !== 0) {
            const gi = +((condStats.D.mean - condStats.A.mean) / Math.abs(condStats.A.mean) * 100).toFixed(1);
            summary[cfgName][obj]._generalizationIndex = gi;
            genIndices.push(gi);
            allGenIndices[cfgName].push(gi);
          }
        }

        summary[cfgName]._avgGeneralizationIndex = genIndices.length
          ? +(genIndices.reduce((a, b) => a + b, 0) / genIndices.length).toFixed(1)
          : null;
      }

      // Ranking
      const ranked = configs
        .filter(c => summary[c]._avgGeneralizationIndex !== null)
        .sort((a, b) => summary[b]._avgGeneralizationIndex - summary[a]._avgGeneralizationIndex);
      summary._ranking = ranked;
      summary._winner  = ranked[0] ?? null;

      // T-tests: every config vs baseline_equal
      summary._tTests = {};
      const baselineIdx = allGenIndices['baseline_equal'];
      if (baselineIdx?.length) {
        for (const cfgName of configs) {
          if (cfgName === 'baseline_equal' || !allGenIndices[cfgName]?.length) continue;
          summary._tTests[cfgName] = this._tTest(allGenIndices[cfgName], baselineIdx);
        }
      }

      // High-level interpretation
      const baselineGI = summary['baseline_equal']?._avgGeneralizationIndex;
      const winnerGI   = summary[summary._winner]?._avgGeneralizationIndex;
      const improvement = (typeof winnerGI === 'number' && typeof baselineGI === 'number')
        ? +(winnerGI - baselineGI).toFixed(2)
        : null;
      const winnerTTest = summary._tTests[summary._winner] ?? null;

      summary._baselineAdvantage  = EXP8_WEIGHT_OPTIMIZATION_CONFIG.BASELINE_ADVANTAGE;
      summary._baselineGI         = baselineGI;
      summary._improvement        = improvement;
      summary._winnerSignificant  = winnerTTest?.isSignificant ?? false;
      summary._conclusion =
        improvement === null           ? 'Insufficient data' :
        improvement > 0.5 && winnerTTest?.isSignificant
          ? `Significant improvement: ${summary._winner} (+${improvement}% vs baseline_equal, p < 0.05)` :
        improvement > 0.5
          ? `Marginal improvement: ${summary._winner} (+${improvement}% vs baseline_equal, NOT significant)` :
        `No improvement: equal weighting near-optimal (max improvement ${improvement}%, not significant)`;
      summary._recommendation =
        (improvement ?? 0) > 0.5 && (winnerTTest?.isSignificant ?? false)
          ? `Use ${summary._winner} weights [${(summary[summary._winner]?._weightsArray ?? []).join('/')}] — proven statistically superior`
          : 'Keep equal weighting (20/20/20/20/20) — simpler to explain and statistically equivalent';

      return summary;
    }

    // ── Experiment 6: group by domain × condition; compute transfer efficiency ─
    if (expNum === 6) {
      const summary    = {};
      const domains    = [...new Set(results.map(r => r.domain))];
      const conditions = [...new Set(results.map(r => r.condition))];

      for (const domainName of domains) {
        const domainMeta = results.find(r => r.domain === domainName);
        summary[domainName] = {
          _domainLabel: domainMeta?.domainLabel ?? domainName,
          _domainEmoji: domainMeta?.domainEmoji ?? '',
        };

        for (const cond of conditions) {
          const filtered = results.filter(
            r => r.domain === domainName && r.condition === cond
          );
          const vals = filtered
            .map(r => r.results.finalScore)
            .filter(v => typeof v === 'number' && !isNaN(v));
          if (!vals.length) { summary[domainName][cond] = null; continue; }
          const n    = vals.length;
          const mean = vals.reduce((a, b) => a + b, 0) / n;
          const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
          summary[domainName][cond] = {
            n,
            mean:             +mean.toFixed(3),
            std:              +std.toFixed(3),
            sem:              +(std / Math.sqrt(n)).toFixed(3),
            min:              +Math.min(...vals).toFixed(3),
            max:              +Math.max(...vals).toFixed(3),
            meanFoodEaten:    +(filtered.reduce((s, r) => s + (r.results.foodEaten    ?? 0), 0) / n).toFixed(3),
            meanWallBounces:  +(filtered.reduce((s, r) => s + (r.results.wallBounces  ?? 0), 0) / n).toFixed(1),
            meanLtmUsageRate: +(filtered.reduce((s, r) => s + (r.results.ltmUsageRate ?? 0), 0) / n).toFixed(3),
          };
        }

        // Domain advantage: frozen_D mean − A mean (absolute and %)
        const frozenD = summary[domainName]['frozen_D'];
        const noMem   = summary[domainName]['A'];
        if (frozenD && noMem) {
          summary[domainName]._domainAdvantage = +(frozenD.mean - noMem.mean).toFixed(3);
          summary[domainName]._domainAdvantagePercent = noMem.mean !== 0
            ? +((frozenD.mean - noMem.mean) / Math.abs(noMem.mean) * 100).toFixed(1)
            : null;
        }
      }

      // Transfer efficiency = (target advantage) / |source advantage| × 100
      // Source domain is 'warehouse' (control — trained in same domain as test)
      const srcAdv = summary['warehouse']?._domainAdvantage;
      if (typeof srcAdv === 'number' && srcAdv !== 0) {
        for (const domainName of domains) {
          const adv = summary[domainName]?._domainAdvantage;
          if (typeof adv === 'number') {
            summary[domainName]._transferEfficiency =
              +(adv / Math.abs(srcAdv) * 100).toFixed(1);
          }
        }
      }

      return summary;
    }

    // ── Experiments 4 / 4.5: use mean_per_agent (multi-agent aggregate) ───────
    const conditions = [...new Set(results.map(r => r.condition))];
    const summary    = {};

    const getValue = (expNum === 4 || expNum === 4.5)
      ? r => r.results.mean_per_agent
      : r => r.results.finalScore ?? r.results.foodEaten;

    for (const cond of conditions) {
      const filtered = results.filter(r => r.condition === cond);
      const vals     = filtered.map(getValue).filter(v => typeof v === 'number' && !isNaN(v));
      if (!vals.length) { summary[cond] = null; continue; }
      const n    = vals.length;
      const mean = vals.reduce((a, b) => a + b, 0) / n;
      const std  = Math.sqrt(vals.reduce((sq, x) => sq + (x - mean) ** 2, 0) / n);
      summary[cond] = {
        n,
        mean: +mean.toFixed(3),
        std:  +std.toFixed(3),
        sem:  +(std / Math.sqrt(n)).toFixed(3),
        min:  +Math.min(...vals).toFixed(3),
        max:  +Math.max(...vals).toFixed(3),
      };
    }
    return summary;
  }

  /** Build JSON payload and trigger browser download. Returns data object. */
  saveExperimentResults(expNum) {
    if (!this.results[expNum].length) {
      console.warn(`[V2] No results for experiment ${expNum} — nothing to save.`);
      return null;
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename  = `exp${expNum}_results_${timestamp}.json`;
    const data = {
      experiment: expNum,
      name:       this.expParams[expNum]?.name ?? `Experiment ${expNum}`,
      timestamp:  new Date().toISOString(),
      summary:    this.computeExperimentSummary(expNum),
      trials:     this.results[expNum],
    };
    this._downloadJSON(data, filename);
    return data;
  }

  generateSummary() {
    console.log('\n' + '═'.repeat(60));
    console.log('EXPERIMENT SUMMARY (ExperimentRunnerV2)');
    console.log('═'.repeat(60));
    for (let exp = 1; exp <= 6; exp++) {
      const results = this.results[exp];
      if (!results.length) continue;
      console.log(`\n📊 Experiment ${exp}: ${this.expParams[exp].name}`);
      const conditions = [...new Set(results.map(r => r.condition))];
      for (const cond of conditions) {
        const st = this.computeStatistics(results.filter(r => r.condition === cond));
        if (st) console.log(`   ${cond}: ${st.mean} ± ${st.std} food (n=${st.n})`);
      }
    }
    console.log('\n' + '═'.repeat(60));
  }

  getProgress() {
    return {
      ...this.progress,
      percentComplete: (
        (this.progress.completedTrials / (this.progress.totalTrials || 1)) * 100
      ).toFixed(1),
    };
  }

  exportResults() {
    return { timestamp: new Date().toISOString(), sharedParams: this.sharedParams, experiments: this.results };
  }
}

export default ExperimentRunner;
