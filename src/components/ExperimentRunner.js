/**
 * ExperimentRunner.js — Structured 4-condition experiment for v0.10 dual-memory system
 *
 * Manages a fully randomised 20-trial sequence comparing four memory configurations:
 *
 *   A — No memory     : Pure Hopfield network, no STM/LTM, reactive baseline
 *   B — STM only      : Short-term recording active, no LTM consolidation
 *   C — LTM only      : LTM pre-seeded from PERFECT_PATS, no real-time STM capture
 *   D — Full dual     : STM + LTM + ConsolidationEngine, complete v0.10 system
 *
 * ── Integration contract ──────────────────────────────────────────────────────
 *
 * 1. Instantiate once (before the first trial):
 *      const runner = new ExperimentRunner({ perfectPats: PERFECT_PATS, N });
 *
 * 2. Start the sequence:
 *      const { condition } = runner.start();
 *      // re-create world for this condition
 *      runner.prepareAgentsForCondition(world.agents, condition);
 *
 * 3. Gate memory operations each frame:
 *      const flags = runner.conditionFlags;
 *      // flags.useController  → whether to call agent.controller.*
 *      // flags.useSTM         → whether to call agent.stm.add()
 *      // flags.useConsolidation → whether to call agent.engine.update()
 *
 * 4. Tick once per frame (after all agents have been updated):
 *      const { trialComplete, nextCondition, isExperimentComplete }
 *        = runner.tick(frameNum, world.agents, agentStates);
 *
 *    agentStates must contain a `reward` field for metrics:
 *      agentStates[id] = { action, reward, attn, dopamine, ... }
 *
 * 5. On trial completion:
 *      if (trialComplete && !isExperimentComplete) {
 *        // re-create world, reset frame counter, then:
 *        runner.prepareAgentsForCondition(world.agents, nextCondition);
 *      }
 *
 * 6. Read runner.uiState for live display (safe to read every render).
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

import { LTMPattern } from './memory/LTM.js';

// ── Constants ─────────────────────────────────────────────────────────────────

/** Frames per experimental trial. */
const TRIAL_DURATION    = 2400;

/** Number of repetitions per condition. */
const TRIALS_PER_COND   = 5;

/** Frames per learning-curve bucket (600 / 50 = 12 data points per trial). */
const METRIC_WINDOW     = 50;

/** Ordered list of all condition labels. */
const CONDITIONS        = ['A', 'B', 'C', 'D'];

// Human-readable labels and descriptions for each condition
const CONDITION_META = {
  A: { label: 'A — No Memory',       description: 'Pure Hopfield, no STM/LTM' },
  B: { label: 'B — STM Only',        description: 'Short-term recording, no LTM' },
  C: { label: 'C — LTM Only',        description: 'Pre-seeded patterns, no real-time STM' },
  D: { label: 'D — Full Dual',       description: 'STM + LTM + Consolidation' },
};

// ── ExperimentRunner ──────────────────────────────────────────────────────────

class ExperimentRunner {
  /**
   * @param {object} config
   * @param {Array}  config.perfectPats          PERFECT_PATS from App.jsx (for Condition C seeding)
   * @param {number} [config.N=25]               Neural state dimension
   * @param {number} [config.trialDuration=2400]  Frames per trial
   * @param {number} [config.trialsPerCondition=5]
   */
  constructor(config = {}) {
    this.perfectPats   = config.perfectPats          ?? [];
    this.N             = config.N                    ?? 25;
    this.trialDuration = config.trialDuration        ?? TRIAL_DURATION;
    this.trialsPerCond = config.trialsPerCondition   ?? TRIALS_PER_COND;

    // ── Experiment sequence ───────────────────────────────────────────────
    /** Randomised list of condition labels, length = CONDITIONS × trialsPerCond */
    this.trialSequence  = this._generateTrialSequence();
    this.totalTrials    = this.trialSequence.length;

    // ── State machine ─────────────────────────────────────────────────────
    this.currentTrialIndex = 0;
    this.isRunning  = false;
    this.isComplete = false;

    // ── Per-trial accumulators ────────────────────────────────────────────
    this._trialFrameCount  = 0;
    this._trialStartTime   = null;
    this._trialResults     = [];     // completed result objects

    // Per-agent reward / action buffers (keyed by agent.id)
    this._windowReward   = {};
    this._totalReward    = {};
    this._actionCounts   = {};
    this._learningCurves = {};

    // ── Flags read by App.jsx each frame ──────────────────────────────────
    /**
     * Memory-gate flags for the current condition.
     * App.jsx checks these before calling controller, stm, engine methods.
     *
     * @type {{
     *   useController:    boolean,
     *   useSTM:           boolean,
     *   useConsolidation: boolean,
     *   label:            string,
     *   description:      string,
     * }}
     */
    this.conditionFlags = this._flagsForCondition('D');  // safe default until start()

    // ── UI state (read-only snapshot for React) ───────────────────────────
    /** @type {object} Safe to read every render frame. */
    this.uiState = this._buildUIState();
  }

  // ── a) Public API ─────────────────────────────────────────────────────────

  /**
   * Begin the 20-trial experiment.
   *
   * Returns the first condition's info so App.jsx can set up the world before
   * calling `prepareAgentsForCondition`.
   *
   * @returns {{ condition: string, trialNumber: number, conditionFlags: object }}
   */
  start() {
    this.currentTrialIndex  = 0;
    this.isRunning          = true;
    this.isComplete         = false;
    this._trialResults      = [];
    this._trialFrameCount   = 0;
    this._trialStartTime    = Date.now();

    const condition = this.trialSequence[0];
    this.conditionFlags = this._flagsForCondition(condition);
    this._resetAccumulators();
    this.uiState = this._buildUIState();

    console.log(
      `[ExperimentRunner] Experiment started: ${this.totalTrials} trials`,
      `\nSequence: ${this.trialSequence.join(' → ')}`
    );

    return this._currentTrialInfo();
  }

  /**
   * Per-frame hook. Call once per simulation frame after all agents are updated.
   *
   * Collects per-agent metrics and drives the trial state machine.
   *
   * @param {number} frame        Frame number within the current trial (0-based)
   * @param {Array}  agents       world.agents
   * @param {object} agentStates  { [agentId]: { action, reward, attn, dopamine, … } }
   *
   * @returns {{
   *   trialComplete:        boolean,
   *   nextCondition:        string|null,
   *   isExperimentComplete: boolean,
   * }}
   */
  tick(frame, agents, agentStates) {
    if (!this.isRunning) {
      return { trialComplete: false, nextCondition: null, isExperimentComplete: false };
    }

    this._trialFrameCount = frame;

    // Accumulate per-agent metrics for this frame
    for (const agent of agents) {
      this._collectFrameMetrics(frame, agent, agentStates[agent.id] ?? {});
    }

    // Refresh live UI display every METRIC_WINDOW frames
    if (frame % METRIC_WINDOW === 0) this.uiState = this._buildUIState();

    // Trial end condition
    if (frame >= this.trialDuration - 1) {
      return this._finalizeTrial(agents);
    }

    return { trialComplete: false, nextCondition: null, isExperimentComplete: false };
  }

  /**
   * Configure freshly created agents for the given condition.
   *
   * Call after App.jsx has re-initialised world.agents (e.g. via initWorld).
   * Mutates agents in-place: nulls out disabled systems, seeds LTM for C,
   * and adjusts confidence thresholds where needed.
   *
   * @param {Array}  agents     Array of agent objects from initWorld()
   * @param {string} condition  'A' | 'B' | 'C' | 'D'
   */
  prepareAgentsForCondition(agents, condition) {
    for (const agent of agents) {
      switch (condition) {
        case 'A':
          // Pure Hopfield: disable controller and consolidation engine.
          // STM/LTM objects stay on the agent for metric access but are never written.
          agent.controller = null;
          agent.engine     = null;
          break;

        case 'B':
          // STM records experience, but LTM is never queried and nothing consolidates.
          // Null out the engine to block STM → LTM transfers.
          // Set threshold impossibly high so the controller (if present) never fires LTM.
          if (agent.controller) agent.controller.ltmConfidenceThreshold = 9999;
          agent.engine = null;
          break;

        case 'C':
          // Seed LTM from PERFECT_PATS so the agent has prior knowledge.
          // Disable ConsolidationEngine (no new patterns during trial).
          // Lower threshold slightly so seeded patterns override Hopfield readily.
          this.seedLTMFromPerfectPats(agent.ltm);
          agent.engine = null;
          if (agent.controller) agent.controller.ltmConfidenceThreshold = 0.20;
          break;

        case 'D':
          // Full dual system — initWorld already wired everything correctly.
          break;

        default:
          console.warn(`[ExperimentRunner] Unknown condition "${condition}", using D defaults.`);
      }
    }

    console.log(`[ExperimentRunner] Agents configured for condition ${condition}`);
  }

  /**
   * Seed an LTM instance with patterns derived from PERFECT_PATS.
   *
   * Each pattern is injected with high reliability and consolidationStrength so
   * the controller uses them immediately (confidence = rel × cs ≈ 0.64 > 0.20).
   * usageCount=10 / successCount=9 pre-warm the Wilson CI to be tight.
   *
   * @param {import('./memory/LTM.js').LongTermMemory} ltm
   */
  seedLTMFromPerfectPats(ltm) {
    if (!this.perfectPats.length) {
      console.warn('[ExperimentRunner] seedLTMFromPerfectPats: no perfectPats provided — skipping');
      return;
    }

    for (const pat of this.perfectPats) {
      // Sensory-only trigger condition: keep bits 0–9, zero out motor region 10–24
      const triggerCondition = pat.data.map((v, i) => i < 10 ? v : -1);

      // Decode the motor action from Hopfield bit layout: 15=L, 16=F, 17=R
      let action = 'F';
      if (pat.data[15] === 1) action = 'L';
      else if (pat.data[17] === 1) action = 'R';

      // Classify context from active sensor zones
      const hasFood = triggerCondition.slice(5, 10).some(v => v === 1);
      const hasObs  = triggerCondition.slice(0, 5).some(v => v === 1);
      const context = hasFood ? 'foraging' : hasObs ? 'avoidance' : 'exploration';

      // Sanitize pattern name for ID (strip → ← arrows, spaces, etc.)
      const patternId = `seed_${pat.name.replace(/[^a-zA-Z0-9_]/g, '_')}`;

      const pattern = new LTMPattern({
        patternId,
        context,
        triggerCondition,
        // Store 5 repetitions; controller uses actionSequence[0] for the current frame
        actionSequence:        Array(5).fill(action),
        reliability:           0.85,  // high prior confidence
        consolidationStrength: 0.75,  // strong trace
        utility:               0.8,
        abstractDescription:   `seeded:${pat.name}`,
        usageCount:            10,    // pre-warmed so Wilson CI is tight
        successCount:          9,
      });

      ltm.storePattern(pattern);
    }

    console.log(`[ExperimentRunner] Seeded LTM with ${this.perfectPats.length} patterns`);
  }

  // ── b) Trial lifecycle ────────────────────────────────────────────────────

  /**
   * Finalize the current trial, export data, and advance to the next.
   * @private
   */
  _finalizeTrial(agents) {
    const condition   = this.trialSequence[this.currentTrialIndex];
    const trialNumber = this._countCompletedForCondition(condition) + 1;
    const endTime     = Date.now();

    // Compile per-agent final metrics
    const agentMetrics = {};
    for (const agent of agents) {
      agentMetrics[agent.id] = this._compileAgentMetrics(agent);
    }

    const result = {
      condition,
      trialNumber,
      trialIndex:     this.currentTrialIndex,
      startTime:      this._trialStartTime,
      endTime,
      durationMs:     endTime - this._trialStartTime,
      durationFrames: this.trialDuration,
      agents:         agentMetrics,
    };

    this._trialResults.push(result);

    console.log(
      `[ExperimentRunner] Trial ${this.currentTrialIndex + 1}/${this.totalTrials}`,
      `[Condition ${condition}] complete —`,
      Object.entries(agentMetrics)
        .map(([id, m]) =>
          `agent${id}: food=${m.foodEaten} ltm=${(m.ltmUsageRate * 100).toFixed(1)}%`
        )
        .join(', ')
    );

    this._exportTrialJSON(result);

    this.currentTrialIndex++;

    if (this.currentTrialIndex >= this.totalTrials) {
      this.isRunning  = false;
      this.isComplete = true;
      this._exportSummaryJSON();
      this.uiState = this._buildUIState();
      console.log('[ExperimentRunner] All trials complete. Experiment done.');
      return { trialComplete: true, nextCondition: null, isExperimentComplete: true };
    }

    // Prepare next trial
    const nextCondition = this.trialSequence[this.currentTrialIndex];
    this.conditionFlags   = this._flagsForCondition(nextCondition);
    this._trialFrameCount = 0;
    this._trialStartTime  = Date.now();
    this._resetAccumulators();
    this.uiState = this._buildUIState();

    console.log(
      `[ExperimentRunner] Next: Trial ${this.currentTrialIndex + 1}/${this.totalTrials}`,
      `Condition ${nextCondition} — ${CONDITION_META[nextCondition].description}`
    );

    return { trialComplete: true, nextCondition, isExperimentComplete: false };
  }

  // ── c) Metrics collection ─────────────────────────────────────────────────

  /** @private */
  _resetAccumulators() {
    for (let id = 0; id < 3; id++) {
      this._windowReward[id]   = 0;
      this._totalReward[id]    = 0;
      this._actionCounts[id]   = { L: 0, F: 0, R: 0 };
      this._learningCurves[id] = [];
    }
  }

  /** @private */
  _collectFrameMetrics(frame, agent, state) {
    const id     = agent.id;
    const reward = state.reward ?? 0;
    const action = state.action ?? 'F';

    this._totalReward[id]  += reward;
    this._windowReward[id] += reward;

    if (action in (this._actionCounts[id] ?? {})) {
      this._actionCounts[id][action]++;
    }

    // Close a learning-curve window every METRIC_WINDOW frames
    if ((frame + 1) % METRIC_WINDOW === 0) {
      this._learningCurves[id].push(
        +this._windowReward[id].toFixed(4)
      );
      this._windowReward[id] = 0;
    }
  }

  /**
   * Build the full metrics object for one agent at trial end.
   * @private
   */
  _compileAgentMetrics(agent) {
    const id     = agent.id;
    const counts = this._actionCounts[id] ?? { L: 0, F: 0, R: 0 };
    const total  = counts.L + counts.F + counts.R || 1;

    // Shannon entropy of the action distribution
    const entropy = -Object.values(counts)
      .map(c => c / total)
      .reduce((h, p) => (p > 0 ? h + p * Math.log(p) : h), 0);

    const curve       = this._learningCurves[id] ?? [];
    const finalReward = curve.length > 0 ? curve[curve.length - 1] : 0;
    const firstReward = curve.length > 0 ? curve[0]                : 0;

    // Pull live stats from subsystems (graceful fallback when null)
    const ctrlStats = agent.controller?.stats()   ?? {};
    const engStats  = agent.engine?.getStats()    ?? {};
    const ltmStats  = agent.ltm?.stats()          ?? {};

    return {
      personality:           agent.personality?.name ?? 'unknown',
      foodEaten:             agent.score,
      totalReward:           +this._totalReward[id].toFixed(4),
      learningCurve:         [...curve],
      finalReward:           +finalReward.toFixed(4),
      learningGain:          +(finalReward - firstReward).toFixed(4),
      actionCounts:          { ...counts },
      actionEntropy:         +entropy.toFixed(4),
      ltmUsageRate:          +(ctrlStats.ltmUsageRate          ?? 0),
      avgLTMConfidence:      +(ctrlStats.avgLTMConfidence      ?? 0),
      consolidationsTotal:   engStats.totalConsolidations      ?? 0,
      patternsCreated:       engStats.newPatterns               ?? 0,
      patternsStrengthened:  engStats.strengthened              ?? 0,
      avgPatternReliability: +(ltmStats.avgReliability         ?? 0),
      ltmPatternCount:       ltmStats.totalPatterns             ?? 0,
      ltmByContext:          ltmStats.byContext                 ?? {},
    };
  }

  // ── d) Condition flags ────────────────────────────────────────────────────

  /**
   * Return the memory-gate flags for a given condition.
   *
   * Flags are read by App.jsx each frame to decide which subsystem calls to make:
   *   - useController    → call agent.controller.selectAction / evaluateAction
   *   - useSTM           → call agent.stm.add(stmFrame)
   *   - useConsolidation → call agent.engine.update(...)
   *
   * @param {'A'|'B'|'C'|'D'} condition
   * @returns {object}
   * @private
   */
  _flagsForCondition(condition) {
    const FLAG_TABLE = {
      A: { useController: false, useSTM: false, useConsolidation: false },
      B: { useController: false, useSTM: true,  useConsolidation: false },
      C: { useController: true,  useSTM: false, useConsolidation: false },
      D: { useController: true,  useSTM: true,  useConsolidation: true  },
    };
    const base = FLAG_TABLE[condition] ?? FLAG_TABLE['D'];
    return { ...base, ...CONDITION_META[condition] };
  }

  // ── e) Export ─────────────────────────────────────────────────────────────

  /** @private */
  _exportTrialJSON(result) {
    const padded   = String(result.trialIndex + 1).padStart(2, '0');
    const filename = `exp_trial_${padded}_cond${result.condition}_${Date.now()}.json`;
    this._downloadJSON(JSON.stringify(result, null, 2), filename);
  }

  /** @private */
  _exportSummaryJSON() {
    const summary  = this.generateExperimentSummary();
    const filename = `exp_summary_${Date.now()}.json`;
    this._downloadJSON(JSON.stringify(summary, null, 2), filename);

    // Human-readable console digest
    console.log('\n══════════════════════════════════════════════════════');
    console.log('  EXPERIMENT COMPLETE — v0.10 Dual-Memory Learning');
    console.log('══════════════════════════════════════════════════════');
    console.log(`Trials: ${summary.totalTrials}   Duration/trial: ${summary.trialDuration} frames`);
    console.log('\nFood Eaten Rankings:');
    summary.comparisons.foodEatenRanking.forEach(r =>
      console.log(`  ${r.condition}  mean=${r.mean}  (${r.improvement_vs_A} vs A)`)
    );
    console.log('\nKey Findings:');
    summary.keyFindings.forEach((f, i) => console.log(`  ${i+1}. ${f}`));
    console.log('\nEffect Sizes (D vs A):');
    console.log(`  foodEaten   d=${summary.effectSizes.D_vs_A_foodEaten}`);
    console.log(`  totalReward d=${summary.effectSizes.D_vs_A_totalReward}`);
    console.log('\nRecommendations:');
    summary.recommendations.forEach((r, i) => console.log(`  ${i+1}. ${r}`));
    console.log('══════════════════════════════════════════════════════\n');
  }

  // ── Statistical analysis ──────────────────────────────────────────────────

  /**
   * Generate the full experiment summary with statistics, comparisons,
   * key findings, effect sizes, and auto-generated recommendations.
   *
   * Called automatically after all trials complete, and available as a public
   * method for on-demand re-analysis.
   *
   * @returns {object} Summary object matching the EXPERIMENT_1_PROTOCOL.md schema
   */
  generateExperimentSummary() {
    // ── Per-condition aggregation ─────────────────────────────────────────
    const conditions = {};

    for (const cond of CONDITIONS) {
      const trials = this._trialResults.filter(r => r.condition === cond);
      if (!trials.length) continue;

      // Pool all agent values across all trials (n = 5 trials × 3 agents = 15 per condition)
      const METRIC_KEYS = [
        'foodEaten', 'totalReward', 'actionEntropy', 'learningGain',
        'ltmUsageRate', 'consolidationsTotal', 'patternsCreated', 'avgPatternReliability',
      ];
      const pooled = Object.fromEntries(METRIC_KEYS.map(k => [k, []]));

      for (const trial of trials) {
        for (const agent of Object.values(trial.agents)) {
          for (const k of METRIC_KEYS) {
            if (agent[k] !== undefined) pooled[k].push(agent[k]);
          }
        }
      }

      // Per-agent detail within the condition
      const agentDetail = {};
      for (let id = 0; id < 3; id++) {
        const agentTrials = trials.map(t => t.agents[id]).filter(Boolean);
        if (!agentTrials.length) continue;

        agentDetail[id] = {
          personality: agentTrials[0].personality,
          metrics: {
            foodEaten:             this._calcStats(agentTrials.map(a => a.foodEaten)),
            totalReward:           this._calcStats(agentTrials.map(a => a.totalReward)),
            learningGain:          this._calcStats(agentTrials.map(a => a.learningGain)),
            ltmUsageRate:          this._calcStats(agentTrials.map(a => a.ltmUsageRate)),
            actionEntropy:         this._calcStats(agentTrials.map(a => a.actionEntropy)),
            avgPatternReliability: this._calcStats(agentTrials.map(a => a.avgPatternReliability)),
            consolidationsTotal:   this._calcStats(agentTrials.map(a => a.consolidationsTotal)),
            patternsCreated:       this._calcStats(agentTrials.map(a => a.patternsCreated)),
          },
          // Averaged learning curve across trials for this agent
          avgLearningCurve: this._meanCurves(agentTrials.map(a => a.learningCurve)),
          // Raw arrays for external statistical tests
          allFoodEaten:    agentTrials.map(a => a.foodEaten),
          allTotalReward:  agentTrials.map(a => a.totalReward),
          allLearningGain: agentTrials.map(a => a.learningGain),
        };
      }

      // Condition-level averaged learning curve (across agents and trials)
      const allCurves = trials.flatMap(t =>
        Object.values(t.agents).map(a => a.learningCurve)
      );

      conditions[cond] = {
        name:             CONDITION_META[cond].label,
        description:      CONDITION_META[cond].description,
        trials:           trials.length,
        metrics:          Object.fromEntries(METRIC_KEYS.map(k => [k, this._calcStats(pooled[k])])),
        avgLearningCurve: this._meanCurves(allCurves),
        agents:           agentDetail,
      };
    }

    // ── Comparisons ──────────────────────────────────────────────────────
    const comparisons = {
      bestCondition:        this._rankConditions(conditions, 'foodEaten')[0]?.condition ?? '?',
      foodEatenRanking:     this._rankConditions(conditions, 'foodEaten'),
      learningGainRanking:  this._rankConditions(conditions, 'learningGain'),
      ltmUsageRanking:      this._rankConditions(conditions, 'ltmUsageRate'),
      consolidationRanking: this._rankConditions(conditions, 'consolidationsTotal'),
      reliabilityRanking:   this._rankConditions(conditions, 'avgPatternReliability'),
    };

    // ── Effect sizes ─────────────────────────────────────────────────────
    // Helper: pool all agents for a condition × metric
    const pool = (cond, metric) =>
      this._trialResults
        .filter(r => r.condition === cond)
        .flatMap(t => Object.values(t.agents).map(a => a[metric] ?? 0));

    const effectSizes = {
      D_vs_A_foodEaten:    this._calcCohensD(pool('D','foodEaten'),    pool('A','foodEaten')),
      D_vs_A_totalReward:  this._calcCohensD(pool('D','totalReward'),  pool('A','totalReward')),
      D_vs_A_learningGain: this._calcCohensD(pool('D','learningGain'), pool('A','learningGain')),
      D_vs_C_foodEaten:    this._calcCohensD(pool('D','foodEaten'),    pool('C','foodEaten')),
      D_vs_C_totalReward:  this._calcCohensD(pool('D','totalReward'),  pool('C','totalReward')),
      C_vs_A_foodEaten:    this._calcCohensD(pool('C','foodEaten'),    pool('A','foodEaten')),
      B_vs_A_foodEaten:    this._calcCohensD(pool('B','foodEaten'),    pool('A','foodEaten')),
    };

    const keyFindings     = this._generateKeyFindings(conditions, comparisons, effectSizes);
    const recommendations = this._generateRecommendations(conditions, effectSizes);

    return {
      experimentName: 'v0.10 Dual-Memory Learning — Experiment 1',
      totalTrials:    this._trialResults.length,
      trialDuration:  this.trialDuration,
      timestamp:      new Date().toISOString(),
      trialSequence:  this.trialSequence,
      conditions,
      comparisons,
      keyFindings,
      effectSizes,
      recommendations,
      // Raw trial data kept for external analysis (Python, R, etc.)
      rawTrials: this._trialResults,
    };
  }

  /**
   * Descriptive statistics for a numeric array.
   *
   * @param {number[]} values
   * @returns {{ mean, stdev, min, max, median, n }}
   */
  _calcStats(values) {
    if (!values.length) {
      return { mean: 0, stdev: 0, min: 0, max: 0, median: 0, n: 0 };
    }
    const n      = values.length;
    const mean   = values.reduce((s, v) => s + v, 0) / n;
    // Sample variance (Bessel's correction: n-1)
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, n - 1);
    const stdev    = Math.sqrt(variance);
    const sorted   = [...values].sort((a, b) => a - b);
    const median   = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];

    return {
      mean:   +mean.toFixed(4),
      stdev:  +stdev.toFixed(4),
      min:    +Math.min(...values).toFixed(4),
      max:    +Math.max(...values).toFixed(4),
      median: +median.toFixed(4),
      n,
    };
  }

  /**
   * Cohen's d effect size between two independent groups.
   *
   * Uses pooled standard deviation (unbiased, with Bessel's correction).
   * Interpretation guidelines: |d| < 0.2 = negligible, 0.2–0.5 = small,
   *   0.5–0.8 = medium, > 0.8 = large (Cohen 1988).
   *
   * @param {number[]} g1 - Group 1 values (e.g. condition D)
   * @param {number[]} g2 - Group 2 values (e.g. condition A)
   * @returns {number} Signed Cohen's d (positive = g1 > g2)
   */
  _calcCohensD(g1, g2) {
    if (!g1.length || !g2.length) return 0;
    const s1 = this._calcStats(g1);
    const s2 = this._calcStats(g2);
    const n1 = s1.n, n2 = s2.n;
    // Pooled variance
    const pooledVar = ((n1 - 1) * s1.stdev ** 2 + (n2 - 1) * s2.stdev ** 2)
                      / Math.max(1, n1 + n2 - 2);
    const pooledSD = Math.sqrt(pooledVar);
    if (pooledSD === 0) return 0;
    return +((s1.mean - s2.mean) / pooledSD).toFixed(4);
  }

  /**
   * Rank all conditions by a metric in descending order.
   *
   * @param {object} condMetrics  { cond: { metrics: { [key]: { mean } } } }
   * @param {string} metric       Key from condMetrics[cond].metrics
   * @returns {Array<{ condition, mean, improvement_vs_A }>}
   */
  _rankConditions(condMetrics, metric) {
    const baselineMean = condMetrics['A']?.metrics?.[metric]?.mean ?? 0;

    return CONDITIONS
      .filter(c => condMetrics[c])
      .map(c => {
        const mean = condMetrics[c].metrics?.[metric]?.mean ?? 0;
        const vsA  = baselineMean !== 0
          ? +(((mean - baselineMean) / Math.abs(baselineMean)) * 100).toFixed(1)
          : 0;
        return {
          condition:        c,
          mean,
          improvement_vs_A: `${vsA >= 0 ? '+' : ''}${vsA}%`,
        };
      })
      .sort((a, b) => b.mean - a.mean);
  }

  /**
   * Auto-generate human-readable key findings from the aggregated data.
   * @private
   */
  _generateKeyFindings(conditions, comparisons, effectSizes) {
    const findings = [];
    const m = (cond, metric) => conditions[cond]?.metrics?.[metric]?.mean ?? 0;

    // ── 1. Best-performing condition ──────────────────────────────────────
    const best    = comparisons.bestCondition;
    const bestVal = m(best, 'foodEaten');
    const aFood   = m('A', 'foodEaten');
    const bestPct = aFood > 0 ? ((bestVal - aFood) / aFood * 100).toFixed(1) : 'N/A';
    findings.push(
      `Condition ${best} (${CONDITION_META[best]?.description}) achieved the highest mean food eaten ` +
      `(${bestVal}), a ${bestPct}% improvement over the no-memory baseline (A = ${aFood}).`
    );

    // ── 2. Full dual memory vs baseline ──────────────────────────────────
    const dFood   = m('D', 'foodEaten');
    const dVsAPct = aFood > 0 ? ((dFood - aFood) / aFood * 100).toFixed(1) : '0';
    const dEffect = effectSizes.D_vs_A_foodEaten;
    const dLabel  = Math.abs(dEffect) < 0.2 ? 'negligible'
                  : Math.abs(dEffect) < 0.5 ? 'small'
                  : Math.abs(dEffect) < 0.8 ? 'medium' : 'large';
    findings.push(
      `Full dual memory (D) vs baseline (A) food eaten: Δ=${dVsAPct}%, Cohen's d=${dEffect} (${dLabel} effect). ` +
      (Math.abs(dEffect) < 0.2
        ? 'Memory system has not yet produced measurable performance gains — longer trials needed.'
        : 'Dual memory is providing a measurable performance benefit.')
    );

    // ── 3. LTM usage rate in D ────────────────────────────────────────────
    const dLTM = m('D', 'ltmUsageRate');
    findings.push(
      `Condition D LTM usage: ${(dLTM * 100).toFixed(1)}% of actions sourced from LTM recall. ` +
      (dLTM > 0.15
        ? 'LTM patterns are being actively used for action selection.'
        : dLTM > 0.05
          ? 'LTM recall is low but present — patterns need more maturation time.'
          : 'LTM is almost never recalled — confidence threshold or trial length may need adjustment.')
    );

    // ── 4. Pre-seeded (C) vs learned (D) ─────────────────────────────────
    const cFood = m('C', 'foodEaten');
    const cvsDPct = dFood > 0 ? ((cFood - dFood) / dFood * 100).toFixed(1) : '0';
    const cdEffect = effectSizes.D_vs_C_foodEaten;
    findings.push(
      `Pre-seeded LTM (C = ${cFood}) vs learned LTM (D = ${dFood}): Δ=${cvsDPct}%, Cohen's d=${cdEffect}. ` +
      (cFood > dFood
        ? `Prior knowledge outperforms real-time learning at ${this.trialDuration}-frame trial length — ` +
          'D needs more frames to build competitive patterns.'
        : 'Real-time dual-memory learning matches or exceeds pre-seeded performance, ' +
          'suggesting successful within-trial consolidation.')
    );

    // ── 5. STM-only vs baseline ───────────────────────────────────────────
    const bFood = m('B', 'foodEaten');
    const bVsAPct = aFood > 0 ? ((bFood - aFood) / aFood * 100).toFixed(1) : '0';
    findings.push(
      `STM-only (B = ${bFood}) vs no memory (A = ${aFood}): Δ=${bVsAPct}%. ` +
      'STM recording alone does not change action selection (expected — STM feeds consolidation, not direct recall). ' +
      'Any difference reflects stochasticity, not learning.'
    );

    // ── 6. Action entropy across conditions ──────────────────────────────
    const aEnt = m('A', 'actionEntropy');
    const dEnt = m('D', 'actionEntropy');
    const entDelta = (dEnt - aEnt).toFixed(3);
    findings.push(
      `Action entropy — A: ${aEnt.toFixed(3)}, D: ${dEnt.toFixed(3)} (Δ=${entDelta}). ` +
      (Math.abs(dEnt - aEnt) < 0.05
        ? 'Conditions produce similar action diversity.'
        : dEnt > aEnt
          ? 'Dual memory increases action diversity, suggesting exploratory LTM influence.'
          : 'Dual memory reduces action diversity, suggesting the controller is converging on preferred patterns.')
    );

    // ── 7. Pattern reliability in D ───────────────────────────────────────
    const dRel = m('D', 'avgPatternReliability');
    const dPats = m('D', 'patternsCreated');
    findings.push(
      `Condition D created ${dPats.toFixed(1)} patterns on average with mean reliability ` +
      `${(dRel * 100).toFixed(1)}%. ` +
      (dRel > 0.6 ? 'Patterns are achieving trustworthy reliability scores.'
        : dRel > 0.35 ? 'Patterns are partially reliable — additional usage cycles will improve scores.'
        : 'Low pattern reliability — Laplace smoothing initialises at 0.5 but patterns may not be strengthened enough.')
    );

    // ── 8. Consolidation activity ─────────────────────────────────────────
    const dConsol = m('D', 'consolidationsTotal');
    findings.push(
      `Average consolidation events per trial in D: ${dConsol.toFixed(1)}. ` +
      (dConsol > 10
        ? 'Active consolidation pipeline — STM events are frequently triggering LTM writes.'
        : dConsol > 3
          ? 'Moderate consolidation activity. Increasing food density could raise reward triggers.'
          : 'Very few consolidations — consider reducing periodicInterval or rewardThreshold.')
    );

    return findings;
  }

  /**
   * Auto-generate actionable recommendations based on experiment results.
   * @private
   */
  _generateRecommendations(conditions, effectSizes) {
    const recs = [];
    const m = (cond, metric) => conditions[cond]?.metrics?.[metric]?.mean ?? 0;

    const dLTM    = m('D', 'ltmUsageRate');
    const dRel    = m('D', 'avgPatternReliability');
    const dFood   = m('D', 'foodEaten');
    const cFood   = m('C', 'foodEaten');
    const aFood   = m('A', 'foodEaten');
    const dConsol = m('D', 'consolidationsTotal');
    const dEffect = Math.abs(effectSizes.D_vs_A_foodEaten);

    // ── Threshold / duration recommendations ─────────────────────────────
    if (dLTM < 0.05) {
      recs.push(
        `LTM is almost unused (${(dLTM*100).toFixed(1)}%). ` +
        `Reduce ltmConfidenceThreshold from 0.25 to 0.15, or extend trial duration to ≥2400 frames ` +
        `so patterns can accumulate enough usageCount to build confidence.`
      );
    } else if (dLTM < 0.15) {
      recs.push(
        `LTM usage is low (${(dLTM*100).toFixed(1)}%). ` +
        `Consider extending trial duration to 2400 frames (Experiment 2) ` +
        `to give patterns more maturation time.`
      );
    }

    // ── Reliability / consolidation recommendations ───────────────────────
    if (dRel < 0.4) {
      recs.push(
        `Pattern reliability is low (${(dRel*100).toFixed(1)}%). ` +
        `The success threshold in strengthenPattern() uses rewardThreshold * 0.5 = ${(15 * 0.5).toFixed(1)}. ` +
        `Consider lowering to rewardThreshold * 0.3, or increasing agent reward density (more food items).`
      );
    }

    // ── Trial duration / C vs D comparison ───────────────────────────────
    if (cFood > dFood) {
      const shortfall = ((cFood - dFood) / cFood * 100).toFixed(1);
      recs.push(
        `Pre-seeded knowledge (C) outperforms real-time learning (D) by ${shortfall}%. ` +
        `${this.trialDuration}-frame trials are too short for D to build competitive LTM. ` +
        `Run Experiment 2 with trialDuration = 3600 frames to test whether D eventually surpasses C.`
      );
    }

    // ── Consolidation frequency ───────────────────────────────────────────
    if (dConsol < 4) {
      recs.push(
        `Very few consolidation events (mean=${dConsol.toFixed(1)}/trial). ` +
        `Reduce periodicInterval (currently 300 frames) to 150 frames, ` +
        `or lower rewardThreshold (currently 15) to trigger more frequent pattern creation.`
      );
    }

    // ── Overall effect size ───────────────────────────────────────────────
    if (dEffect < 0.2) {
      recs.push(
        `Negligible effect size for D vs A (Cohen's d=${effectSizes.D_vs_A_foodEaten}). ` +
        `The memory system has not yet produced a measurable performance benefit. ` +
        `Priority: verify the consolidation pipeline is firing (check console for [findSimilarPattern] logs) ` +
        `and that evaluateAction is updating patterns when source = LTM.`
      );
    } else if (dEffect >= 0.5) {
      recs.push(
        `Medium-to-large effect size (Cohen's d=${effectSizes.D_vs_A_foodEaten}) for D vs A — ` +
        `proceed to Experiment 2 with 3600-frame trials and personality-stratified analysis ` +
        `(AGGRESSIVE / CURIOUS / CAUTIOUS separately).`
      );
    }

    // ── Personality analysis ──────────────────────────────────────────────
    recs.push(
      `Run per-personality analysis: inspect conditions[cond].agents[0/1/2] in the summary JSON ` +
      `to determine whether AGGRESSIVE, CURIOUS, or CAUTIOUS agents benefit most from dual memory.`
    );

    // ── Statistical significance ──────────────────────────────────────────
    recs.push(
      `Run a paired t-test on allFoodEaten arrays in conditions.D vs conditions.A ` +
      `(n=15 per condition). The rawTrials array in the summary JSON contains ` +
      `all agent-level observations for import into Python/R.`
    );

    // ── Baseline sanity check ─────────────────────────────────────────────
    if (Math.abs(m('B','foodEaten') - aFood) / Math.max(aFood, 1) > 0.1) {
      recs.push(
        `WARNING: Condition B (STM only) differs from A by >10% on food eaten. ` +
        `B should behave identically to A in action selection — ` +
        `check that flags.useController = false is correctly applied for condition B.`
      );
    }

    return recs;
  }

  /** @private */
  _downloadJSON(json, filename) {
    try {
      const blob = new Blob([json], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      console.log(`[ExperimentRunner] ✓ Exported: ${filename}`);
    } catch (err) {
      console.error(`[ExperimentRunner] Export failed for "${filename}":`, err);
    }
  }

  // ── f) UI state ───────────────────────────────────────────────────────────

  /** @private */
  _buildUIState() {
    const condition = this.trialSequence[this.currentTrialIndex] ?? null;
    const flags     = condition ? this._flagsForCondition(condition) : null;

    const completedByCondition = {};
    for (const c of CONDITIONS) {
      completedByCondition[c] = this._trialResults.filter(r => r.condition === c).length;
    }

    return {
      phase:           this.isComplete ? 'complete' : this.isRunning ? 'running' : 'idle',
      currentTrial:    this.currentTrialIndex + 1,
      totalTrials:     this.totalTrials,
      condition,
      conditionLabel:  flags?.label       ?? '—',
      conditionDesc:   flags?.description ?? '—',
      progress:        this.currentTrialIndex / this.totalTrials,
      trialProgress:   this._trialFrameCount / this.trialDuration,
      trialFrame:      this._trialFrameCount,
      completedCount:  this._trialResults.length,
      completedByCondition,
      trialSequence:   [...this.trialSequence],
    };
  }

  // ── g) Helpers ────────────────────────────────────────────────────────────

  /** Randomise trial order: 5 × each of A, B, C, D. @private */
  _generateTrialSequence() {
    const pool = CONDITIONS.flatMap(c => Array(this.trialsPerCond).fill(c));
    // Fisher-Yates in-place shuffle
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool;
  }

  /** @private */
  _countCompletedForCondition(condition) {
    return this._trialResults.filter(r => r.condition === condition).length;
  }

  /** @private */
  _currentTrialInfo() {
    const condition = this.trialSequence[this.currentTrialIndex] ?? 'D';
    return { condition, trialNumber: this._countCompletedForCondition(condition) + 1, conditionFlags: this.conditionFlags };
  }

  /** Arithmetic mean, rounded to 4 d.p. @private */
  _mean(arr) {
    if (!arr.length) return 0;
    return +(arr.reduce((s, v) => s + v, 0) / arr.length).toFixed(4);
  }

  /** Element-wise mean across variable-length curves. @private */
  _meanCurves(curves) {
    if (!curves.length) return [];
    const maxLen = Math.max(...curves.map(c => c.length));
    return Array.from({ length: maxLen }, (_, i) => {
      const vals = curves.map(c => c[i] ?? 0);
      return this._mean(vals);
    });
  }
}

// ── Default free-play flags (used by App.jsx when no experiment is running) ──
const FREE_PLAY_FLAGS = {
  useController:    true,
  useSTM:           true,
  useConsolidation: true,
  label:       'Free Play',
  description: 'Full dual-memory system',
};

// ── Exports ───────────────────────────────────────────────────────────────────

export { ExperimentRunner, FREE_PLAY_FLAGS };
