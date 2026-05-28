/**
 * EXPERIMENT_CONFIG.js — Single source of truth for all experiment parameters.
 *
 * Centralises every magic number used across the simulation, memory system,
 * controller, and experiment design so they can be tuned in one place and
 * referenced consistently by every module.
 *
 * Import paths (relative to each consumer):
 *   ConsolidationEngine.js, DualMemoryController.js, stm.js:
 *     import { ... } from './EXPERIMENT_CONFIG.js';
 *
 *   ExperimentRunner_v2.js, other files in src/components/:
 *     import { ... } from './memory/EXPERIMENT_CONFIG.js';
 *     — OR via the re-export at src/components/EXPERIMENT_CONFIG.js —
 *     import { ... } from './EXPERIMENT_CONFIG.js';
 *
 *   App.jsx (src/):
 *     import { ... } from './components/EXPERIMENT_CONFIG.js';
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

// ── Simulation physics ────────────────────────────────────────────────────────

export const SIMULATION_CONFIG = {
  /** Hopfield network size — must match sensory vector length (10 sensor + 15 motor bits). */
  HOPFIELD_NEURONS:          25,

  /** Simulation canvas width in px. */
  WORLD_WIDTH:               800,

  /** Simulation canvas height in px. */
  WORLD_HEIGHT:              600,

  /** Sensor ray range for obstacle / wall detection. */
  OBSTACLE_DETECTION_RANGE:  85,

  /** Sensor range for food detection. */
  FOOD_DETECTION_RANGE:      150,

  /** Agent collision radius in px. */
  AGENT_RADIUS:              9,

  /** Food item collision radius in px. */
  FOOD_RADIUS:               7,

  /** Forward movement speed (px / frame). */
  AGENT_SPEED:               2.5,

  /** Turn increment (rad / frame). */
  AGENT_TURN_RATE:           0.068,

  /** Hopfield softmax inverse temperature (higher → more peaked distribution). */
  HOPFIELD_BETA:             5,

  /** Default sensory noise probability (fraction of bits randomly flipped). */
  HOPFIELD_NOISE:            0.05,

  /** Frames between event-loop yields (keeps UI responsive during batch runs). */
  YIELD_EVERY_N_FRAMES:      50,

  /** Five sensor ray angles relative to agent heading: forward, FR, R, FL, L. */
  SENSOR_ANGLES:             [0, Math.PI / 4, Math.PI / 2, -Math.PI / 4, -Math.PI / 2],
};

// ── Memory system parameters ──────────────────────────────────────────────────

export const MEMORY_CONFIG = {
  // ── Short-term memory ─────────────────────────────────────────────────────
  /** Circular buffer capacity (frames). */
  STM_SIZE:     60,

  /** Exponential-decay time constant for STM frame weights. */
  STM_DECAY_TAU: 0.5,

  // ── Consolidation triggers ────────────────────────────────────────────────
  /** Cumulative reward in window that fires the reward trigger. */
  CONSOLIDATION_REWARD_THRESHOLD:     50,

  /** KL-divergence between consecutive attention frames that fires the surprise trigger. */
  CONSOLIDATION_SURPRISE_THRESHOLD:   0.8,

  /** Frames between mandatory periodic consolidation sweeps. */
  CONSOLIDATION_PERIODIC_INTERVAL:    300,

  /** Minimum sequence overlap to consider two patterns the same. */
  CONSOLIDATION_SIMILARITY_THRESHOLD: 0.60,

  // ── Initial pattern quality ───────────────────────────────────────────────
  /**
   * Starting reliability for newly consolidated patterns.
   *
   * CRITICAL: must be set so that
   *   NEW_PATTERN_INITIAL_RELIABILITY × NEW_PATTERN_INITIAL_CONSOLIDATION_STRENGTH
   *   < CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD
   *
   * At 0.3 × 0.3 = 0.09 the pattern CANNOT override the Hopfield network until
   * it has been reinforced enough times to push the product above 0.30.
   * The previous hardcoded value of 0.5 combined with a threshold of 0.01 caused
   * every freshly created pattern (even noise) to immediately override the Hopfield
   * network — this was the source of the Condition-D −33 % regression.
   */
  NEW_PATTERN_INITIAL_RELIABILITY:            0.80,
  NEW_PATTERN_INITIAL_CONSOLIDATION_STRENGTH: 0.70,
  NEW_PATTERN_INITIAL_UTILITY:                0.50,

  /**
   * Minimum reliability value enforced by recordUsage().
   * Prevents a few early failures from permanently zeroing out a pattern before
   * it has had a chance to accumulate positive evidence.
   */
  RELIABILITY_FLOOR: 0.10,
};

// ── Controller thresholds ─────────────────────────────────────────────────────

export const CONTROLLER_CONFIG = {
  /**
   * Minimum reliability × consolidationStrength score for an LTM pattern to
   * override the Hopfield action.
   *
   * THE KEY FIX — was previously overridden to 0.01 in _createAgentForCondition,
   * which let any freshly created pattern (score ≈ 0.05–0.25) bypass the Hopfield
   * network and caused Condition D to perform 33 % below the no-memory baseline.
   *
   * At 0.30 a new pattern (0.3 × 0.3 = 0.09) is rejected; the pattern must be
   * reinforced until its product clears 0.30 before it influences behaviour.
   */
  LTM_CONFIDENCE_THRESHOLD: 0.30,

  /** ε-greedy exploration rate: probability of using Hopfield even when LTM qualifies. */
  EXPLORATION_RATE: 0.20,

  /** STM weight in blend mode (fraction of action coming from Hopfield / STM). */
  ACTION_WEIGHT_STM: 0.60,
};

// ── Experiment design ─────────────────────────────────────────────────────────

export const EXPERIMENT_CONFIG = {
  /** Simulation frames per agent-trial in all experiments (2400 ≈ 40 s at 60 fps). */
  TRIAL_DURATION_FRAMES:    2400,

  /** Independent trials per memory condition per experiment factor level. */
  NUM_TRIALS_PER_CONDITION: 5,

  /** Agents run in parallel within each trial. */
  NUM_AGENTS_PER_TRIAL:     3,
};

// ── Experiment 3: Robustness across real-world & space environments ───────────

export const EXP3_ROBUSTNESS_CONFIG = {
  ENABLE_WAREHOUSE_PROFILE: true,
  ENABLE_PHYSICS_PROFILE:   true,
  ENABLE_SPACE_PROFILE:     true,

  /**
   * Warehouse profile — sensor noise caused by dust, vibration, and wear.
   * sigma maps directly to bit-flip probability passed into the Hopfield step.
   */
  WAREHOUSE_NOISE_LEVELS: [
    { level: 0, sigma: 0.00, label: 'Clean sensors'        },
    { level: 1, sigma: 0.05, label: 'Fresh deployment'     },
    { level: 2, sigma: 0.10, label: '6 months operation'   },
    { level: 3, sigma: 0.20, label: '1-2 years operation'  },
    { level: 4, sigma: 0.50, label: 'Old/dusty sensors'    },
  ],

  /**
   * Physics profile — gravity variation changes locomotion physics.
   * multiplier scales both AGENT_SPEED and AGENT_TURN_RATE.
   */
  PHYSICS_GRAVITY_LEVELS: [
    { level: 0, multiplier: 1.00, label: 'Earth (1g)'   },
    { level: 1, multiplier: 0.50, label: 'Moon (0.5g)'  },
    { level: 2, multiplier: 0.38, label: 'Mars (0.38g)' },
    { level: 3, multiplier: 0.16, label: 'Asteroid'     },
    { level: 4, multiplier: 0.00, label: 'Microgravity' },
  ],

  /**
   * Space profile — combined stressors: radiation degrades LTM patterns,
   * sensor drift accumulates over time, and gravity changes locomotion.
   */
  SPACE_MISSION_LEVELS: [
    {
      level: 0,
      label:          'Day 0: Earth sim',
      radiation_rate: 0.000,
      drift_rate:     0.000,
      gravity:        1.00,
    },
    {
      level: 1,
      label:          'Month 1: In transit',
      radiation_rate: 0.001,
      drift_rate:     0.001,
      gravity:        0.50,
    },
    {
      level: 2,
      label:          'Month 6: On Mars',
      radiation_rate: 0.003,
      drift_rate:     0.003,
      gravity:        0.38,
    },
    {
      level: 3,
      label:          'Month 12: Degradation',
      radiation_rate: 0.005,
      drift_rate:     0.005,
      gravity:        0.38,
    },
    {
      level: 4,
      label:          'Month 24: End of mission',
      radiation_rate: 0.010,
      drift_rate:     0.010,
      gravity:        0.38,
    },
  ],

  /** Independent trials per (profile × level × condition) cell. */
  TRIALS_PER_LEVEL: 15,

  // 3 profiles × 5 levels × 4 conditions × 15 trials = 900 total agent-runs
};

/**
 * Look up the stressor configuration for a profile + level index.
 *
 * @param {'warehouse'|'physics'|'space'} profile
 * @param {number} level  0–4
 * @returns {object} Level-specific config object
 */
export function getEXP3Config(profile, level) {
  const map = {
    warehouse: EXP3_ROBUSTNESS_CONFIG.WAREHOUSE_NOISE_LEVELS,
    physics:   EXP3_ROBUSTNESS_CONFIG.PHYSICS_GRAVITY_LEVELS,
    space:     EXP3_ROBUSTNESS_CONFIG.SPACE_MISSION_LEVELS,
  };
  return map[profile]?.[level] ?? map.warehouse[0];
}

// ── Agent factory (dependency-injected) ──────────────────────────────────────

/**
 * Build a fresh agent object wired for the given memory condition.
 *
 * Memory classes are injected as parameters to avoid circular imports — this
 * config module must not import from the memory modules that themselves import
 * this config.
 *
 * Conditions:
 *   A — No memory    : controller = null, engine = null
 *   B — STM only     : STM records experience; LTM threshold = 9999 (never fires); engine = null
 *   C — LTM only     : caller must seed LTM after this returns; threshold = 0.20; engine = null
 *   D — Full dual    : all systems active at LTM_CONFIDENCE_THRESHOLD
 *
 * @param {string} condition                'A' | 'B' | 'C' | 'D'
 * @param {class}  ShortTermMemory
 * @param {class}  LongTermMemory
 * @param {class}  ConsolidationEngine
 * @param {class}  DualMemoryController
 * @returns {{ stm, ltm, engine, controller, score: number }}
 */
export function createAgentForCondition(
  condition,
  ShortTermMemory,
  LongTermMemory,
  ConsolidationEngine,
  DualMemoryController,
) {
  const stm = new ShortTermMemory(MEMORY_CONFIG.STM_SIZE);
  const ltm = new LongTermMemory(1000);

  const engine = new ConsolidationEngine(stm, ltm, {
    windowSize:        30,
    rewardThreshold:   MEMORY_CONFIG.CONSOLIDATION_REWARD_THRESHOLD,
    surpriseThreshold: MEMORY_CONFIG.CONSOLIDATION_SURPRISE_THRESHOLD,
    periodicInterval:  MEMORY_CONFIG.CONSOLIDATION_PERIODIC_INTERVAL,
  });

  const controller = new DualMemoryController(ltm, {
    ltmConfidenceThreshold: CONTROLLER_CONFIG.LTM_CONFIDENCE_THRESHOLD,
    explorationRate:         CONTROLLER_CONFIG.EXPLORATION_RATE,
    actionWeightSTM:         CONTROLLER_CONFIG.ACTION_WEIGHT_STM,
  });

  const agent = { stm, ltm, engine, controller, score: 0 };

  switch (condition) {
    case 'A':
      // No memory at all — pure reactive Hopfield
      agent.controller = null;
      agent.engine     = null;
      break;

    case 'B':
      // STM records experience but LTM never overrides Hopfield
      agent.controller.ltmConfidenceThreshold = 9999;
      agent.engine = null;
      break;

    case 'C':
      // LTM only — caller must call _seedLTM(agent.ltm) after receiving this object.
      // Lower threshold (0.20) because seeded patterns have rel=0.85, cs=0.75 → score=0.64.
      agent.engine = null;
      agent.controller.ltmConfidenceThreshold = 0.20;
      break;

    case 'D':
      // Full dual memory — threshold already set to LTM_CONFIDENCE_THRESHOLD above
      break;

    default:
      console.warn(
        `[createAgentForCondition] Unknown condition "${condition}" — defaulting to D (full dual).`
      );
  }

  return agent;
}

// ── Experiment 4: Multi-Agent Coordination ────────────────────────────────────

export const EXP4_MULTI_AGENT_CONFIG = {
  // ── Scaling test ──────────────────────────────────────────────────────────
  /** Test how per-agent performance changes as the swarm grows. */
  SCALING_TEST: {
    ENABLED:           true,
    /** Number of concurrent agents per trial. */
    AGENT_COUNTS:      [1, 2, 3, 5],
    /** Complexity levels from Exp 2 (obstacles / food). */
    COMPLEXITY_LEVELS: [2, 3],
    TRIALS_PER_CONFIG: 5,
    // 4 counts × 2 levels × 4 conditions × 5 trials = 160 trials
  },

  // ── Interference test ─────────────────────────────────────────────────────
  /** Test how environment density amplifies or dampens agent-agent interference. */
  INTERFERENCE_TEST: {
    ENABLED: true,
    ENVIRONMENT_CONFIGS: [
      { width:  400, height: 300, agents: 2, label: 'Small (high interference)' },
      { width:  800, height: 600, agents: 2, label: 'Medium (light interference)' },
      { width: 1200, height: 900, agents: 2, label: 'Large (no interference)' },
    ],
    COMPLEXITY_LEVELS: [2, 3],
    /** Only baseline vs. full learning — keeps runtime reasonable. */
    TEST_CONDITIONS:   ['A', 'D'],
    TRIALS_PER_CONFIG: 5,
    // 3 envs × 2 levels × 2 conditions × 5 trials = 60 trials
  },

  /**
   * Fractional (0–1) starting positions for agents, expressed as fractions of world
   * width and height. Agents are spread across the world to minimise initial crowding.
   */
  AGENT_START_POSITIONS: [
    { x: 0.2, y: 0.2 }, // top-left
    { x: 0.8, y: 0.2 }, // top-right
    { x: 0.5, y: 0.5 }, // centre
    { x: 0.2, y: 0.8 }, // bottom-left
    { x: 0.8, y: 0.8 }, // bottom-right
  ],

  /** Pixels of proximity below which two agents are considered colliding. */
  COLLISION_DISTANCE_THRESHOLD: 20,

  /** Magnitude of the separation impulse applied to each colliding agent (px). */
  COLLISION_SEPARATION_FORCE: 1.5,
};

/**
 * Return the world-space start position for the agent at `agentIndex`.
 * Falls back to a random position if the index exceeds the table.
 *
 * @param {number} agentIndex
 * @param {number} worldWidth
 * @param {number} worldHeight
 * @returns {{ x: number, y: number }}
 */
export function getAgentStartPosition(agentIndex, worldWidth, worldHeight) {
  const entry = EXP4_MULTI_AGENT_CONFIG.AGENT_START_POSITIONS[agentIndex]
    ?? { x: Math.random(), y: Math.random() };
  return {
    x: entry.x * worldWidth,
    y: entry.y * worldHeight,
  };
}

// ── Experiment 4.5: Shared Long-Term Memory ───────────────────────────────────

/**
 * Configuration for the shared-LTM experiment.
 * Two variants run head-to-head:
 *   • independent — each agent owns its private LTM (Exp 4 behaviour)
 *   • shared      — all agents point their controller at one LTM pool;
 *                   each agent still has its own STM and ConsolidationEngine,
 *                   but consolidated patterns accumulate in a common store.
 *
 * NOTE: LongTermMemory / ConsolidationEngine instances are NOT created here
 * to avoid circular imports.  Instance creation lives in ExperimentRunner_v2.js
 * inside _createSharedLTMAgents().
 */
export const EXP4_5_SHARED_LTM_CONFIG = {
  VARIANTS: [
    { name: 'independent', shared: false, label: 'Independent LTM (Exp 4 baseline)' },
    { name: 'shared',      shared: true,  label: 'Shared LTM pool' },
  ],

  /** Same scaling grid as the Exp 4 scaling test for direct comparison. */
  AGENT_COUNTS:      [1, 2, 3, 5],
  COMPLEXITY_LEVELS: [2, 3],
  TRIALS_PER_CONFIG: 5,

  /** Only A (no memory) and D (full dual) — keeps runtime manageable. */
  TEST_CONDITIONS:   ['A', 'D'],

  // 2 variants × 4 counts × 2 levels × 2 conditions × 5 trials = 160 trials
};

// ── Experiment 5: Reward Structure Variation ──────────────────────────────────

/**
 * Five reward structures that test whether learned Hopfield patterns generalise
 * across different optimisation objectives — a proxy for "can one model serve
 * multiple customers?".
 *
 * Each variant supplies:
 *   perFrameReward(ate, frame, duration)  → float fed to evaluateAction() and
 *                                           the STM consolidation trigger each frame.
 *   finalScore(foodEaten, frames, wallBounces) → end-of-trial summary metric
 *                                                stored in results.finalScore.
 *
 * Reward functions are intentionally kept simple so differences in the D-vs-A
 * learning advantage are attributable to the reward signal, not to environment
 * complexity.
 *
 * NOTE: functions are NOT JSON-serialisable — only the variant name is stored
 * inside trial results; the runner looks up the function at runtime.
 */
export const EXP5_REWARD_VARIATION_CONFIG = {
  REWARD_VARIANTS: [
    {
      name:        'baseline',
      label:       'Baseline: Maximize Food',
      description: 'Standard warehouse task — collect as many items as possible',
      emoji:        '📦',
      /** Standard +1 / −0.01 per frame. */
      perFrameReward: (ate) => ate > 0 ? 1.0  : -0.01,
      /** Final score = raw food count. */
      finalScore:     (food) => food,
    },
    {
      name:        'efficiency',
      label:       'Efficiency: Food per Energy',
      description: 'Battery-constrained robots — moving without finding food is costly',
      emoji:        '🔋',
      /** Higher idle penalty pushes agent to find food quickly. */
      perFrameReward: (ate) => ate > 0 ? 1.0  : -0.05,
      /** Subtract small energy cost proportional to frames elapsed. */
      finalScore:     (food, frames) => food - frames * 0.001,
    },
    {
      name:        'accuracy',
      label:       'Safety: Minimal Wall Contact',
      description: 'Careful / delicate handling — wall collisions are expensive',
      emoji:        '🛡️',
      /** Slightly higher idle penalty reinforces avoidance patterns. */
      perFrameReward: (ate) => ate > 0 ? 1.0  : -0.02,
      /** Penalise wall + obstacle bounces recorded by runTrial(). */
      finalScore:     (food, _frames, wallBounces) => food - wallBounces * 0.05,
    },
    {
      name:        'speed',
      label:       'Speed: Time Pressure',
      description: 'Time-sensitive deliveries — food earns double but every frame costs',
      emoji:        '⚡',
      /** Double food reward; small per-frame penalty creates urgency. */
      perFrameReward: (ate) => ate > 0 ? 2.0  : -0.02,
      /** Summary: food × 2 minus elapsed-time cost. */
      finalScore:     (food, frames) => food * 2 - frames * 0.01,
    },
    {
      name:        'balance',
      label:       'Balance: Multi-Objective',
      description: 'General-purpose deployment — moderate food + efficiency + safety',
      emoji:        '⚖️',
      /** Balanced penalty between efficiency and accuracy. */
      perFrameReward: (ate) => ate > 0 ? 1.5  : -0.03,
      /** Weighted combination of all three objectives. */
      finalScore:     (food, frames, wallBounces) =>
        food * 1.5 - frames * 0.0005 - wallBounces * 0.05,
    },
  ],

  /** Moderate complexity throughout (matches Exp 2 Level 2: 15 obstacles, 20 food). */
  COMPLEXITY_LEVEL:  2,
  OBSTACLE_COUNT:    15,
  FOOD_COUNT:        20,

  TRIALS_PER_CONFIG: 5,
  TEST_CONDITIONS:   ['A', 'D'],

  // 5 variants × 2 conditions × 5 trials = 50 agent-runs
  // Est. runtime: 8–12 min
};

// ── Experiment 5.5: Multi-Objective Learning ──────────────────────────────────

/**
 * Train one LTM on all five reward signals simultaneously, then test whether
 * the accumulated patterns transfer to each individual objective better than
 * single-objective training from Exp 5.
 *
 * Architecture:
 *   Phase 1 (Training) — TRAINING_TRIALS_PER_VARIANT trials, all writing into
 *     one shared LTM.  The per-frame reward fed to the consolidation engine is a
 *     weighted sum of the five EXP5 perFrameReward functions.  Patterns that work
 *     well across objectives get reinforced; objective-specific patterns fade.
 *
 *   Phase 2 (Testing) — for each of the 5 test objectives, test_conditions × trials
 *     runs with the trained LTM copied in (condition D) or empty (condition A).
 *     Agents may continue adapting during the test trial (engine still active).
 *
 * Two training variants exercise different weighting strategies:
 *   average  — all five objectives matter equally
 *   weighted — commercial deployment priority (speed > baseline > balance > efficiency > accuracy)
 *
 * NOTE: weight dicts use the same keys as EXP5_REWARD_VARIATION_CONFIG variant names.
 * NOTE: REWARD_FUNCTIONS are final-scoring functions (called once after the trial).
 *       Per-frame training reward is computed in the runner from EXP5's perFrameReward.
 * NOTE: steps ≈ total trial frames (2400); collisions = wall+obstacle bounces.
 */
export const EXP5_5_MULTI_OBJECTIVE_CONFIG = {
  TRAINING_VARIANTS: [
    {
      name:        'average',
      label:       'Equal Weight (Average)',
      description: 'All five objectives matter equally during training',
      weights: {
        baseline:   0.20,
        efficiency: 0.20,
        accuracy:   0.20,
        speed:      0.20,
        balance:    0.20,
      },
    },
    {
      name:        'weighted',
      label:       'Weighted (Commercial Priority)',
      description: 'Weights reflect real-world deployment importance',
      weights: {
        baseline:   0.30,   // primary task: collect items
        speed:      0.25,   // time-sensitive logistics
        balance:    0.20,   // multi-customer deployment
        efficiency: 0.15,   // battery-aware fleets
        accuracy:   0.10,   // delicate-handling robots
      },
    },
  ],

  /**
   * Terminal (post-trial) scoring functions for each individual test objective.
   *   food      — items collected (foodEaten)
   *   steps     — total frames elapsed (duration)
   *   collisions — wall+obstacle bounces (wallBounces)
   *   frames    — same as steps; kept for signature parity
   *
   * These differ from Exp 5's finalScore deliberately to exercise a harder
   * generalisation test (learned patterns must transfer to a different metric).
   */
  REWARD_FUNCTIONS: {
    baseline:   (food, steps, collisions, frames) => food,
    efficiency: (food, steps, collisions, frames) => food * 10 - steps * 0.01,
    accuracy:   (food, steps, collisions, frames) => food * 10 - collisions * 2,
    speed:      (food, steps, collisions, frames) => food * 20 - frames * 0.05,
    balance:    (food, steps, collisions, frames) =>
      food * 10 - steps * 0.005 - collisions * 1.5,
  },

  OBSTACLE_COUNT: 15,
  FOOD_COUNT:     20,

  /** Training trials per variant (all write into one shared LTM pool). */
  TRAINING_TRIALS_PER_VARIANT:  10,

  /** Test-phase trials per (objective × condition) cell. */
  TESTING_TRIALS_PER_OBJECTIVE:  5,

  /** Only baseline vs. full-dual (same as Exp 5 for direct comparison). */
  TEST_CONDITIONS: ['A', 'D'],

  // 2 variants × 10 training + 2 × 5 obj × 2 cond × 5 testing = 20 + 100 = 120 trials
  // Est. runtime: 15–20 min
};

// ── Experiment 5.5.5: Weight Optimisation ─────────────────────────────────────

/**
 * Grid search over five weight combinations to find the optimal multi-objective
 * training mix.  Uses exactly the same training/testing pipeline as Exp 5.5 so
 * results are directly comparable:
 *   • same EXP5 perFrameReward signals for per-frame consolidation feedback
 *   • same EXP5_5 REWARD_FUNCTIONS for terminal scoring
 *   • same obstacle / food density (L2: 15 obs, 20 food)
 *
 * NOTE: The key '5.5.5' is used as a string throughout the runner because
 * 5.5.5 is not a valid JavaScript number literal.
 *
 * NOTE: REWARD_FUNCTIONS are intentionally NOT duplicated here — the runner
 * reads them from EXP5_5_MULTI_OBJECTIVE_CONFIG so 5.5 and 5.5.5 share the
 * same terminal scoring.
 *
 * Total: 5 combos × (10 training + 5 obj × 2 cond × 5 testing)
 *      = 5 × 60 = 300 trials
 * Est. runtime: 25–35 min
 */
export const EXP5_5_5_WEIGHT_OPTIMIZATION_CONFIG = {
  WEIGHT_COMBINATIONS: [
    {
      name:        'current',
      label:       'Current (Exp 5.5 weighted)',
      description: 'Control — proven baseline from Exp 5.5',
      weights: { baseline: 0.30, speed: 0.25, balance: 0.20, efficiency: 0.15, accuracy: 0.10 },
    },
    {
      name:        'pro_food',
      label:       'Pro-Food Baseline Recovery',
      description: 'Boost food weight to recover baseline regression',
      weights: { baseline: 0.40, speed: 0.20, balance: 0.20, efficiency: 0.10, accuracy: 0.10 },
    },
    {
      name:        'balanced',
      label:       'Equal Five-Way Balance',
      description: 'Same as Exp 5.5 "average" — perfect 20 % on all objectives',
      weights: { baseline: 0.20, speed: 0.20, balance: 0.20, efficiency: 0.20, accuracy: 0.20 },
    },
    {
      name:        'efficiency_first',
      label:       'Efficiency Priority',
      description: 'Maximum efficiency weight — upper bound for that objective',
      weights: { baseline: 0.20, speed: 0.20, balance: 0.15, efficiency: 0.30, accuracy: 0.15 },
    },
    {
      name:        'smart_balance',
      label:       'Smart Optimized Balance',
      description: 'Hypothesis winner: heavy food + speed with moderate efficiency boost',
      weights: { baseline: 0.35, speed: 0.25, balance: 0.15, efficiency: 0.20, accuracy: 0.05 },
    },
  ],

  OBSTACLE_COUNT: 15,
  FOOD_COUNT:     20,

  /** Training trials per weight combo (all writing into one shared LTM pool). */
  TRAINING_TRIALS_PER_COMBO:   10,

  /** Test-phase trials per (objective × condition) cell. */
  TESTING_TRIALS_PER_OBJECTIVE: 5,

  /** Same as Exp 5 / 5.5 for direct comparison. */
  TEST_CONDITIONS: ['A', 'D'],
};

// ── Experiment 6: Transfer Learning ──────────────────────────────────────────

/**
 * Scientific question: Can patterns learned in one domain transfer to
 * completely different domains when consolidation is FROZEN?
 *
 * Architecture:
 *   Phase 0 — Training (10 trials):
 *     A shared LTM accumulates patterns using the smart_balance weights from
 *     Exp 5.5.5 (the best-performing weight combination).  This is the
 *     "source model" — trained exclusively on the standard warehouse task.
 *
 *   Phase 1-5 — Transfer (5 domains × 2 conditions × 5 trials = 50 trials):
 *     Condition A        — no memory; pure Hopfield baseline for each domain
 *     Condition frozen_D — trained LTM from Phase 0 copied in; consolidation
 *                          engine is NULL (no new patterns can form); the
 *                          agent uses what it learned in the source domain.
 *
 * Transfer efficiency per domain =
 *   (frozen_D advantage in target) / |frozen_D advantage in source| × 100
 *
 * The warehouse domain itself is included as the source control — we expect
 * near-100% transfer efficiency there since it is the training domain.
 *
 * Total: 10 training + 50 transfer = 60 trials
 * Est. runtime: 25–35 min
 */
export const EXP6_TRANSFER_LEARNING_CONFIG = {
  /**
   * Training weights — smart_balance from Exp 5.5.5 (hypothesis best).
   * Keys must match EXP5_REWARD_VARIATION_CONFIG variant names.
   */
  SOURCE_TRAINING_WEIGHTS: {
    baseline:   0.35,
    speed:      0.25,
    balance:    0.15,
    efficiency: 0.20,
    accuracy:   0.05,
  },

  SOURCE_TRAINING_TRIALS: 10,

  /**
   * Transfer domains.  The first domain (warehouse) is the source control.
   * Each domain has:
   *   gravityMultiplier — scales effSpeed and effTurn (Exp 3 mechanic)
   *   noiseSigma        — additive sensor bit-flip probability (Exp 3 mechanic)
   *   perFrameReward    — per-frame signal used for controller feedback only
   *                       (engine is null for frozen_D, so it does NOT drive
   *                        new consolidation — only updates controller stats)
   *   finalScore        — terminal scoring function for analysis
   */
  TRANSFER_DOMAINS: [
    {
      name:        'warehouse',
      label:       'Source: Standard Warehouse',
      emoji:       '🏭',
      description: 'Training domain — control; expect near-100% transfer efficiency',
      gravityMultiplier: 1.0,
      noiseSigma:        0,
      perFrameReward: (ate) => ate > 0 ? 1.0 : -0.01,
      finalScore:     (food, _frames, _bounces) => food,
    },
    {
      name:        'physics',
      label:       'Mars Gravity (0.38g)',
      emoji:       '🚀',
      description: 'Mars-level gravity — movement is slower and harder to control',
      gravityMultiplier: 0.38,
      noiseSigma:        0,
      perFrameReward: (ate) => ate > 0 ? 1.0 : -0.01,
      finalScore:     (food, _frames, _bounces) => food,
    },
    {
      name:        'noise',
      label:       'High-Noise Sensors (20%)',
      emoji:       '📡',
      description: 'Standard warehouse with 20 % additive sensor bit-flip noise',
      gravityMultiplier: 1.0,
      noiseSigma:        0.20,
      perFrameReward: (ate) => ate > 0 ? 1.0 : -0.01,
      finalScore:     (food, _frames, _bounces) => food,
    },
    {
      name:        'speed',
      label:       'Speed Objective',
      emoji:       '⚡',
      description: 'Double food reward + per-frame time penalty — different task objective',
      gravityMultiplier: 1.0,
      noiseSigma:        0,
      perFrameReward: (ate) => ate > 0 ? 2.0 : -0.02,
      finalScore:     (food, frames, _bounces) => food * 2 - frames * 0.01,
    },
    {
      name:        'safety',
      label:       'Safety-Critical Task',
      emoji:       '🛡️',
      description: 'Standard physics but wall contacts are heavily penalised',
      gravityMultiplier: 1.0,
      noiseSigma:        0,
      perFrameReward: (ate) => ate > 0 ? 1.0 : -0.02,
      finalScore:     (food, _frames, bounces) => food - bounces * 0.5,
    },
  ],

  OBSTACLE_COUNT: 15,
  FOOD_COUNT:     20,
  TRIALS_PER_DOMAIN: 5,

  /**
   * 'A'        — no LTM at all (reactive Hopfield baseline for each domain)
   * 'frozen_D' — trained LTM from Phase 0; consolidation engine = null
   */
  TEST_CONDITIONS: ['A', 'frozen_D'],

  // 10 training + 5 domains × 2 conditions × 5 trials = 60 total
};

// ── Experiment 8: Weight Optimization ────────────────────────────────────────
//
//   Grid search over 10 asymmetric weight combinations to find if any outperforms
//   the balanced equal-weight baseline (20/20/20/20/20, +11.27 % D-vs-A advantage).
//
//   Weight vector order matches EXP5 variant names:
//     [F, S, A, B, E] = [baseline (food), speed, accuracy, balance, efficiency]
//
//   Five hypotheses tested across ten configurations:
//     H1 Speed emphasis        — food:30→40, reduce efficiency
//     H2 Efficiency emphasis   — balance:30→40, reduce efficiency
//     H3 Robustness            — accuracy + balance up, speed down
//     H4 Compound              — food + balance up, mixed reductions
//     H5 Conservative baseline — 20/20/20/20/20 (control)
//
//   10 configs × (10 training + 5 obj × 2 cond × 3 testing) = 400 trials
//   Est. runtime: ~35–45 min

export const EXP8_WEIGHT_OPTIMIZATION_CONFIG = {

  /**
   * All weights are fractional (sum to 1.0 exactly).
   * weightsArray stores [F, S, A, B, E] as integer percentages for the JSON report.
   */
  WEIGHT_CONFIGURATIONS: [
    {
      name: 'baseline_equal', label: 'Equal (Baseline 20/20/20/20/20)', hypothesis: 'baseline',
      weightsArray: [20, 20, 20, 20, 20],
      weights: { baseline: 0.20, speed: 0.20, accuracy: 0.20, balance: 0.20, efficiency: 0.20 },
    },
    {
      name: 'speed_moderate', label: 'Speed Moderate (30/20/20/20/10)', hypothesis: 'speed',
      weightsArray: [30, 20, 20, 20, 10],
      weights: { baseline: 0.30, speed: 0.20, accuracy: 0.20, balance: 0.20, efficiency: 0.10 },
    },
    {
      name: 'speed_strong', label: 'Speed Strong (40/10/10/20/20)', hypothesis: 'speed',
      weightsArray: [40, 10, 10, 20, 20],
      weights: { baseline: 0.40, speed: 0.10, accuracy: 0.10, balance: 0.20, efficiency: 0.20 },
    },
    {
      name: 'efficiency_moderate', label: 'Efficiency Moderate (20/20/20/30/10)', hypothesis: 'efficiency',
      weightsArray: [20, 20, 20, 30, 10],
      weights: { baseline: 0.20, speed: 0.20, accuracy: 0.20, balance: 0.30, efficiency: 0.10 },
    },
    {
      name: 'efficiency_strong', label: 'Efficiency Strong (20/20/20/40/5)', hypothesis: 'efficiency',
      weightsArray: [20, 20, 20, 40, 5],
      weights: { baseline: 0.20, speed: 0.20, accuracy: 0.20, balance: 0.40, efficiency: 0.05 },
    },
    {
      name: 'robust_moderate', label: 'Robust Moderate (20/10/20/30/20)', hypothesis: 'robustness',
      weightsArray: [20, 10, 20, 30, 20],
      weights: { baseline: 0.20, speed: 0.10, accuracy: 0.20, balance: 0.30, efficiency: 0.20 },
    },
    {
      name: 'robust_strong', label: 'Robust Strong (30/10/10/30/20)', hypothesis: 'robustness',
      weightsArray: [30, 10, 10, 30, 20],
      weights: { baseline: 0.30, speed: 0.10, accuracy: 0.10, balance: 0.30, efficiency: 0.20 },
    },
    {
      name: 'compound_1', label: 'Compound 1 (30/15/15/25/15)', hypothesis: 'compound',
      weightsArray: [30, 15, 15, 25, 15],
      weights: { baseline: 0.30, speed: 0.15, accuracy: 0.15, balance: 0.25, efficiency: 0.15 },
    },
    {
      name: 'compound_2', label: 'Compound 2 (25/20/20/20/15)', hypothesis: 'compound',
      weightsArray: [25, 20, 20, 20, 15],
      weights: { baseline: 0.25, speed: 0.20, accuracy: 0.20, balance: 0.20, efficiency: 0.15 },
    },
    {
      name: 'food_emphasis', label: 'Food Emphasis (35/15/15/15/20)', hypothesis: 'food',
      weightsArray: [35, 15, 15, 15, 20],
      weights: { baseline: 0.35, speed: 0.15, accuracy: 0.15, balance: 0.15, efficiency: 0.20 },
    },
  ],

  OBSTACLE_COUNT: 15,
  FOOD_COUNT:     20,

  /** Training trials per config (pattern accumulation phase). */
  TRAINING_TRIALS_PER_CONFIG: 10,

  /** Test trials per objective per condition (fewer than 5.5.5 to keep runtime ≈ 45 min). */
  TESTING_TRIALS_PER_OBJECTIVE: 3,

  TEST_CONDITIONS: ['A', 'D'],

  /**
   * D-vs-A generalization index reference from the balanced config in prior experiments.
   * Used to calibrate the _improvement output in the summary report.
   */
  BASELINE_ADVANTAGE: 11.27,

  // 10 × (10 training + 5 obj × 2 cond × 3 testing) = 400 trials
};

// ── Experiment 9: Learning Dynamics & Curves ──────────────────────────────────
//
//   Maps the spec's conceptual checkpoints to full physics training trials.
//   Each trial = 2400 frames of real physics — much heavier than the spec's
//   step-level simulation, so the spec's [0,10,25,50,100,200,300,500] steps
//   are scaled to [0,2,5,10,20,40] full trials while preserving curve shape.
//
//   For each checkpoint value N the runner:
//     1. Creates a fresh LTM and trains for exactly N trials
//     2. Tests with 5 objectives × 2 conditions × TESTING_TRIALS trials
//     3. Computes the D-vs-A generalisation index
//
//   REPS_PER_CHECKPOINT independent fresh-LTM repetitions per checkpoint let
//   us measure variance (are early checkpoints noisy? does overfitting show
//   consistent degradation?).
//
//   Curve-shape heuristic labels (from spec):
//     exponential — fast early gains, diminishing returns
//     sigmoid     — slow start → acceleration → plateau
//     linear      — steady gains throughout
//
//   Trial budget:
//     Training:  Σ(checkpoint × REPS) = (0+2+5+10+20+40) × 2 = 154
//     Testing:   6 checkpoints × 2 reps × 5 obj × 2 cond × 2 trials = 240
//     Total:     394 trials   Est. runtime: ~30–35 min

export const EXP9_LEARNING_DYNAMICS_CONFIG = {

  /**
   * Full-physics training trial counts to test as checkpoints.
   * Checkpoint 0 = no training (pure reactive baseline).
   * The learning curve is expected to rise steeply from 0→10 and plateau by 40.
   */
  CHECKPOINTS: [0, 2, 5, 10, 20, 40],

  /** Independent fresh-LTM repetitions per checkpoint (for variance estimation). */
  REPS_PER_CHECKPOINT: 2,

  /** Test trials per objective per condition per rep. */
  TESTING_TRIALS_PER_OBJECTIVE: 2,

  TEST_CONDITIONS: ['A', 'D'],
  OBSTACLE_COUNT: 15,
  FOOD_COUNT:     20,

  /**
   * Multi-objective training weights — equal weighting (spec baseline).
   * All keys must match EXP5 REWARD_VARIANTS names.
   */
  TRAINING_WEIGHTS: {
    baseline: 0.20, speed: 0.20, accuracy: 0.20, balance: 0.20, efficiency: 0.20,
  },

  /** D-vs-A gen-index reference for the report's absolute comparison column. */
  BASELINE_ADVANTAGE: 11.27,

  /** Fraction of peak advantage required to declare convergence (95%). */
  CONVERGENCE_THRESHOLD: 0.95,

  /** Fraction of peak advantage that constitutes "minimum viable deployment" (90%). */
  MIN_VIABLE_THRESHOLD: 0.90,

  /**
   * If the last checkpoint's mean advantage drops more than this amount below
   * the previous checkpoint, overfitting is flagged.
   */
  OVERFITTING_THRESHOLD: 0.3,

  // (0+2+5+10+20+40) × 2 training + 6 × 2 × 5 × 2 × 2 testing = 154 + 240 = 394 total
};
