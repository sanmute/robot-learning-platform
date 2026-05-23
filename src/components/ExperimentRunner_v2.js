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

// ── Simulation constants (mirror App.jsx) ────────────────────────────────────

const SIM_N       = 25;
const SIM_W       = 800;
const SIM_H       = 600;
const SIM_OBS_R   = 85;    // wall/obstacle detection range
const SIM_FOOD_R  = 150;   // food detection range
const SIM_AGENT_R = 9;     // agent collision radius
const SIM_FOOD_PX = 7;     // food item collision radius
const SIM_SPD     = 2.5;   // forward speed (px/frame)
const SIM_TURN    = 0.068; // turn rate (rad/frame)
const SIM_BETA    = 5;     // Hopfield softmax temperature
const SIM_NOISE   = 0.05;  // default sensory noise level
const YIELD_EVERY = 50;    // frames between event-loop yields

/** Sensor angles relative to heading: forward, FR, R, FL, L */
const SIM_SA = [0, Math.PI / 4, Math.PI / 2, -Math.PI / 4, -Math.PI / 2];

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

function _wallDist(ox, oy, angle, range) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let t = range;
  if (dx < 0) t = Math.min(t, -ox / dx);
  if (dx > 0) t = Math.min(t, (SIM_W - ox) / dx);
  if (dy < 0) t = Math.min(t, -oy / dy);
  if (dy > 0) t = Math.min(t, (SIM_H - oy) / dy);
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

function _getSensors(ax, ay, angle, foods, obstacles) {
  const obs  = Array(5).fill(false);
  const food = Array(5).fill(false);
  const DETECT = SIM_OBS_R * 0.88;

  for (let s = 0; s < 5; s++) {
    const ra = angle + SIM_SA[s];
    const dx = Math.cos(ra), dy = Math.sin(ra);

    if (_wallDist(ax, ay, ra, SIM_OBS_R) < DETECT) {
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

function _moveAgent(x, y, angle, action, obstacles) {
  let na = angle;
  if (action === 'L') na -= SIM_TURN;
  if (action === 'R') na += SIM_TURN;

  let nx = x + Math.cos(na) * SIM_SPD;
  let ny = y + Math.sin(na) * SIM_SPD;

  const mg = SIM_AGENT_R + 3;
  if (nx < mg)         { nx = mg;         na = Math.PI - na; }
  if (nx > SIM_W - mg) { nx = SIM_W - mg; na = Math.PI - na; }
  if (ny < mg)         { ny = mg;         na = -na; }
  if (ny > SIM_H - mg) { ny = SIM_H - mg; na = -na; }

  for (const o of obstacles) {
    const odx = nx - o.x, ody = ny - o.y;
    const dist = Math.sqrt(odx * odx + ody * ody);
    const minD = SIM_AGENT_R + o.r;
    if (dist < minD && dist > 0) {
      const nx_n = odx / dist, ny_n = ody / dist;
      nx = o.x + nx_n * (minD + 1);
      ny = o.y + ny_n * (minD + 1);
      const dot = Math.cos(na) * nx_n + Math.sin(na) * ny_n;
      na = Math.atan2(Math.sin(na) - 2 * dot * ny_n, Math.cos(na) - 2 * dot * nx_n);
    }
  }
  return { x: nx, y: ny, angle: na };
}

function _checkFoodCollision(ax, ay, foods) {
  let eaten = 0;
  for (const f of foods) {
    const dx = f.x - ax, dy = f.y - ay;
    if (Math.sqrt(dx * dx + dy * dy) < SIM_AGENT_R + SIM_FOOD_PX + 2) {
      f.x = 50 + Math.random() * (SIM_W - 100);
      f.y = 50 + Math.random() * (SIM_H - 100);
      eaten++;
    }
  }
  return eaten;
}

// ── Environment generators ────────────────────────────────────────────────────

function _generateObstacles(count) {
  const obstacles = [];
  const MIN_R = 12, MAX_R = 22;
  let attempts = 0;
  while (obstacles.length < count && attempts < count * 20) {
    attempts++;
    const r  = MIN_R + Math.random() * (MAX_R - MIN_R);
    const ox = r + 50 + Math.random() * (SIM_W - 2 * r - 100);
    const oy = r + 50 + Math.random() * (SIM_H - 2 * r - 100);
    const cdx = ox - SIM_W / 2, cdy = oy - SIM_H / 2;
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

function _generateFood(count, obstacles) {
  const foods = [];
  let attempts = 0;
  while (foods.length < count && attempts < count * 20) {
    attempts++;
    const fx = 50 + Math.random() * (SIM_W - 100);
    const fy = 50 + Math.random() * (SIM_H - 100);
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
 *   A — No memory  : controller=null, engine=null
 *   B — STM only   : STM records, LTM threshold=9999, engine=null
 *   C — LTM only   : pre-seeded, threshold=0.20, engine=null
 *   D — Full dual  : all systems active
 */
function _createAgentForCondition(condition) {
  const stm    = new ShortTermMemory(60);
  const ltm    = new LongTermMemory(1000);
  const engine = new ConsolidationEngine(stm, ltm, {
    windowSize: 30, rewardThreshold: 25, surpriseThreshold: 0.5, periodicInterval: 300,
  });
  const controller = new DualMemoryController(ltm, {
    ltmConfidenceThreshold: 0.25, explorationRate: 0.2, actionWeightSTM: 0.6,
  });
  const agent = { stm, ltm, engine, controller, score: 0 };

  switch (condition) {
    case 'A': agent.controller = null; agent.engine = null; break;
    case 'B': agent.controller.ltmConfidenceThreshold = 9999; agent.engine = null; break;
    case 'C': _seedLTM(agent.ltm); agent.engine = null; agent.controller.ltmConfidenceThreshold = 0.20; break;
    case 'D': break; // full dual — already wired
    default: console.warn(`[V2] Unknown condition "${condition}", using D`);
  }
  return agent;
}

// ── ExperimentRunner class ────────────────────────────────────────────────────

export class ExperimentRunner {
  constructor() {
    this.sharedParams = {
      TRIAL_DURATION:         2400,
      NUM_TRIALS:             5,
      NUM_AGENTS:             3,
      STM_SIZE:               60,
      STM_DECAY_TAU:          0.5,
      HOPFIELD_NEURONS:       25,
      REWARD_THRESHOLD:       25,
      SURPRISE_THRESHOLD:     0.5,
      CONSOLIDATION_INTERVAL: 300,
      SIMILARITY_THRESHOLD:   0.60,
      CONFIDENCE_THRESHOLD:   0.25,
    };

    this.expParams = {
      1: { name: 'Dual-Memory Validation'    },
      2: { name: 'Environmental Complexity'  },
      3: { name: 'Sensor Noise Robustness'   },
      4: { name: 'Multi-Agent Coordination'  },
      5: { name: 'Reward Structure Variation'},
      6: { name: 'Generalization & Transfer' },
    };

    this.progress = {
      currentExperiment: null,
      totalTrials:       0,
      completedTrials:   0,
      startTime:         null,
      isRunning:         false,
    };

    this.results  = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };
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
      1: () => this.runExp1(),
      2: () => this.runExp2(),
      3: () => this.runExp3(),
      4: () => this.runExp4(),
      5: () => this.runExp5(),
      6: () => this.runExp6(),
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

  // ── EXPERIMENT 3: Sensor Noise Robustness ─────────────────────────────────

  async runExp3() {
    const noiseLevels = [0, 0.05, 0.10, 0.20, 0.50];
    const conditions  = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;
    this.progress.totalTrials     = noiseLevels.length * conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const noiseLevel of noiseLevels) {
      for (const condition of conditions) {
        for (let trial = 0; trial < trials; trial++) {
          for (let agent = 0; agent < agents; agent++) {
            if (this._stopped) return;
            const result = await this.runTrial({
              experiment: 3, condition, trial, agent, duration: dur,
              params: { noiseLevel },
            });
            this.results[3].push(result);
            this.progress.completedTrials++;
            this._emitProgress({ lastResult: result });
          }
        }
      }
    }
  }

  // ── EXPERIMENT 4: Multi-Agent Coordination ────────────────────────────────

  async runExp4() {
    const configs = [
      { id: 1, name: 'Isolated',         sharedReward: false, sharedPatterns: false },
      { id: 2, name: 'SharedReward',      sharedReward: true,  sharedPatterns: false },
      { id: 3, name: 'FullCollaboration', sharedReward: true,  sharedPatterns: true  },
    ];
    const conditions = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;
    this.progress.totalTrials     = configs.length * conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const config of configs) {
      for (const condition of conditions) {
        for (let trial = 0; trial < trials; trial++) {
          if (this._stopped) return;
          const result = await this.runTrial({
            experiment: 4, condition, trial, agent: -1, duration: dur,
            params: { config: config.id, configName: config.name,
                      sharedReward: config.sharedReward, sharedPatterns: config.sharedPatterns },
          });
          this.results[4].push(result);
          this.progress.completedTrials += agents;
          this._emitProgress({ lastResult: result });
        }
      }
    }
  }

  // ── EXPERIMENT 5: Reward Structure Variation ──────────────────────────────

  async runExp5() {
    const rewardTypes = [
      { id: 'A', name: 'Linear' }, { id: 'B', name: 'Binary' },
      { id: 'C', name: 'Squared' }, { id: 'D', name: 'Inverse' },
      { id: 'E', name: 'Exponential' }, { id: 'F', name: 'PenaltyHeavy' },
    ];
    const conditions = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;
    this.progress.totalTrials     = rewardTypes.length * conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const rt of rewardTypes) {
      for (const condition of conditions) {
        for (let trial = 0; trial < trials; trial++) {
          for (let agent = 0; agent < agents; agent++) {
            if (this._stopped) return;
            const result = await this.runTrial({
              experiment: 5, condition, trial, agent, duration: dur,
              params: { rewardType: rt.id, rewardTypeName: rt.name },
            });
            this.results[5].push(result);
            this.progress.completedTrials++;
            this._emitProgress({ lastResult: result });
          }
        }
      }
    }
  }

  // ── EXPERIMENT 6: Generalization & Transfer ───────────────────────────────

  async runExp6() {
    const switches = [
      { id: 'A', name: 'Obstacles-Disappear', changeType: 'complexity-down' },
      { id: 'B', name: 'Obstacles-Double',    changeType: 'complexity-up'   },
      { id: 'C', name: 'Food-Density-High',   changeType: 'reward-up'       },
      { id: 'D', name: 'No-Change',           changeType: 'control'         },
      { id: 'E', name: 'Enemies-Appear',      changeType: 'new-threat'      },
      { id: 'F', name: 'Sensors-Inverted',    changeType: 'hard-transfer'   },
    ];
    const conditions = ['A', 'B', 'C', 'D'];
    const { NUM_TRIALS: trials, NUM_AGENTS: agents, TRIAL_DURATION: dur } = this.sharedParams;
    this.progress.totalTrials     = switches.length * conditions.length * trials * agents;
    this.progress.completedTrials = 0;

    for (const sw of switches) {
      for (const condition of conditions) {
        for (let trial = 0; trial < trials; trial++) {
          for (let agent = 0; agent < agents; agent++) {
            if (this._stopped) return;
            const result = await this.runTrial({
              experiment: 6, condition, trial, agent, duration: dur,
              params: { switchType: sw.id, switchTypeName: sw.name,
                        changeType: sw.changeType, phaseSwitch: true },
            });
            this.results[6].push(result);
            this.progress.completedTrials++;
            this._emitProgress({ lastResult: result });
          }
        }
      }
    }
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
    const { obstacleCount = 0, foodCount = 12, noiseLevel } = params;
    const noise = noiseLevel !== undefined ? noiseLevel : SIM_NOISE;

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
      const { newState, attn } = _modernStep(sensoryState, PERFECT_PATS, SIM_BETA, noise);
      const hopfieldAction     = _decodeMotor(newState);

      // 3. Memory-conditioned action selection
      let action = hopfieldAction;
      if (agent.controller) {
        const context = _determineContext(sensors);
        const res     = agent.controller.selectAction(sensoryState, hopfieldAction, context);
        action        = res.action;
      }

      // 4. Move
      const moved = _moveAgent(ax, ay, aAngle, action, obstacles);
      ax = moved.x; ay = moved.y; aAngle = moved.angle;

      // 5. Food + reward
      const ate   = _checkFoodCollision(ax, ay, foods);
      foodEaten  += ate;
      agent.score = foodEaten;
      const reward = ate > 0 ? 1.0 : -0.01;

      // 6. LTM feedback
      if (agent.controller) agent.controller.evaluateAction(sensoryState, action, reward);

      // 7. STM recording
      if (agent.stm) {
        agent.stm.add(new STMFrame(frame, sensoryState, newState, action, reward, attn));
      }

      // 8. Consolidation
      if (agent.engine) agent.engine.update(frame, frame);

      // 9. Learning curve (10 points across 2400 frames)
      if ((frame + 1) % 240 === 0) learningCurve.push(foodEaten);
    }

    const ltmS  = agent.ltm?.stats()        ?? {};
    const engS  = agent.engine?.getStats()  ?? {};
    const ctrlS = agent.controller?.stats() ?? {};

    return {
      experiment, condition, trial, agent: agentIdx, params,
      results: {
        foodEaten:               agent.score,
        patternsConsolidated:    engS.newPatterns          ?? 0,
        consolidationsTriggered: engS.totalConsolidations  ?? 0,
        avgPatternReliability:   ltmS.avgReliability       ?? 0,
        ltmUsageRate:            ctrlS.ltmUsageRate        ?? 0,
        ltmPatternCount:         ltmS.totalPatterns        ?? 0,
        learningCurve,
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
    const results    = this.results[expNum];
    const conditions = [...new Set(results.map(r => r.condition))];
    const summary    = {};
    for (const cond of conditions)
      summary[cond] = this.computeStatistics(results.filter(r => r.condition === cond));
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
