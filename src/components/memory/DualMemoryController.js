/**
 * DualMemoryController.js — Integrated action selection across STM and LTM
 *
 * The behavioral layer that sits between the Hopfield network's reflexive
 * action (short-term / reactive) and the LTM's stored patterns (long-term /
 * learned). Each frame it decides which memory source should drive the agent:
 *
 *   • STM  (default): use the action the Hopfield network produced this frame
 *   • LTM           : recall a stored pattern whose trigger matches current senses
 *   • blend         : weighted probabilistic mix of both (optional mode)
 *
 * Codebase adaptations vs. spec:
 *   - Actions are strings ('L' | 'F' | 'R'), not {left, forward, right} booleans.
 *   - There is no HopfieldNetwork object; the Hopfield result is the already-
 *     decoded string action produced by decodeMotor(modernStep(…)).
 *     selectAction() therefore accepts it as an explicit parameter.
 *   - LTMPattern.recordUsage() is used for stat updates (it includes Laplace
 *     smoothing, consolidation strength adjustments, and CI recomputation in
 *     one call) instead of the manual field mutations described in the spec.
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

// ── Constants ─────────────────────────────────────────────────────────────────

/** Rolling window sizes used for running averages and recent-action logs. */
const CONFIDENCE_WINDOW = 200;  // frames of LTM confidence tracked for avg
const ACTION_LOG_CAP    = 100;  // entries in actionHistory
const SOURCE_LOG_CAP    = 2000; // entries in memorySourceLog (trimmed in bulk)
const SOURCE_LOG_TRIM   = 200;  // entries removed per trim cycle

/** Safe fallback action used when LTM returns nothing and no STM action given. */
const FALLBACK_ACTION = 'F';

// ── DualMemoryController ──────────────────────────────────────────────────────

/**
 * DualMemoryController — per-agent memory integration for action selection.
 *
 * One instance per agent. Instantiate after the agent's ltm is ready.
 * Call selectAction() each frame in place of the raw decodeMotor() result,
 * then call evaluateAction() after stepWorld() returns the reward.
 *
 * @example
 * // In initWorld (after stm/ltm/engine are set):
 * agent.controller = new DualMemoryController(agent.ltm);
 *
 * // In simLoop, replacing the raw `action` variable:
 * const hopfieldAction = decodeMotor(newState);
 * const { action, source } = agent.controller.selectAction(
 *   partial, hopfieldAction, determineContext(sensors)
 * );
 * const { reward } = stepWorld(world, aid, action);
 * agent.controller.evaluateAction(partial, action, reward);
 */
class DualMemoryController {
  /**
   * @param {import('./ltm.js').LongTermMemory} ltm - LTM instance for this agent
   * @param {object}  [config={}]
   * @param {number}  [config.ltmConfidenceThreshold=0.65]
   *   Minimum `reliability × consolidationStrength` score for a pattern to
   *   override Hopfield output. Below this the Hopfield action is kept.
   * @param {number}  [config.explorationRate=0.2]
   *   Probability [0, 1] of falling back to the Hopfield action even when an
   *   LTM pattern exceeds the confidence threshold. Preserves novelty-seeking.
   * @param {number}  [config.actionWeightSTM=0.6]
   *   STM selection probability used in blend mode.
   *   See blendActions() for the exact weighting formula.
   *
   * @throws {Error} If ltm is not provided
   */
  constructor(ltm, config = {}) {
    if (!ltm) throw new Error('DualMemoryController: ltm is required.');

    /** @type {import('./ltm.js').LongTermMemory} */
    this.ltm = ltm;

    // ── Config ────────────────────────────────────────────────────────────
    /** @type {number} Confidence floor for LTM override [0, 1] */
    this.ltmConfidenceThreshold = config.ltmConfidenceThreshold ?? 0.30;

    /** @type {number} ε-greedy exploration probability [0, 1] */
    this.explorationRate = config.explorationRate ?? 0.2;

    /**
     * @type {number} STM blend weight [0, 1].
     * At default 0.6: blended selections are 60 % Hopfield, 40 % LTM on average.
     */
    this.actionWeightSTM = config.actionWeightSTM ?? 0.6;

    // ── State ─────────────────────────────────────────────────────────────
    /**
     * Memory source that drove the most recent action.
     * @type {'STM'|'LTM'|'blend'}
     */
    this.lastMemorySource = 'STM';

    /**
     * Rolling log of recent action selections.
     * Each entry: { action, source, confidence, patternId, timestamp }
     * @type {object[]}
     */
    this.actionHistory = [];

    /**
     * Compact source-only log for getMemoryBalance() / stats().
     * Stores 'STM' | 'LTM' | 'blend' strings.
     * @type {string[]}
     */
    this.memorySourceLog = [];

    // ── Internal tracking ─────────────────────────────────────────────────
    /** @type {number} Total STM-sourced selections */
    this._stmCount = 0;

    /** @type {number} Total LTM-sourced selections */
    this._ltmCount = 0;

    /** @type {number} Total blend-sourced selections */
    this._blendCount = 0;

    /**
     * The LTM pattern that drove the most recent LTM or blend action.
     * Stored so evaluateAction() can update it without needing a re-lookup.
     * @type {import('./ltm.js').LTMPattern|null}
     */
    this._lastPattern = null;

    /** @type {string|null} Context of the most recent LTM selection */
    this._lastContext = null;

    /**
     * Sliding window of raw confidence values for avgLTMConfidence.
     * @type {number[]}
     */
    this._confidenceWindow = [];

    /** @type {number} Total selectAction() calls */
    this._totalSelections = 0;
  }

  // ── a) selectAction ───────────────────────────────────────────────────────

  /**
   * Choose the action to execute this frame.
   *
   * Decision flow:
   *   1. Search LTM for a pattern matching the current sensory state.
   *   2. Compute confidence = reliability × consolidationStrength.
   *   3. If confidence ≥ threshold AND random draw > explorationRate → use LTM.
   *   4. Otherwise → use the Hopfield/STM action unchanged.
   *
   * Note: `hopfieldAction` replaces the spec's internal `hopfieldNetwork.getAction()`
   * call because in this codebase the Hopfield step is a free function whose
   * result is already decoded to a string before the controller is invoked.
   *
   * @param {number[]}  sensoryState    25-element binary sensory vector (the `partial` array)
   * @param {string}    hopfieldAction  Pre-computed Hopfield action: 'L' | 'F' | 'R'
   * @param {string}   [currentContext='exploration'] Behavioral context for LTM search
   * @returns {{
   *   action:     string,
   *   source:     'STM'|'LTM'|'blend',
   *   confidence: number,
   *   patternId:  string|null
   * }}
   */
  selectAction(sensoryState, hopfieldAction = FALLBACK_ACTION, currentContext = 'exploration') {
    this._totalSelections++;

    const { bestPattern, confidence } = this.retrievePatterns(sensoryState, currentContext);

    // Track confidence for running average regardless of which branch fires
    this._recordConfidence(confidence);

    let action;
    let source;

    if (bestPattern && confidence >= this.ltmConfidenceThreshold && Math.random() > this.explorationRate) {
      // ── LTM branch ───────────────────────────────────────────────────────
      action = this.executePattern(bestPattern);
      source = 'LTM';
      this._ltmCount++;
      this._lastPattern = bestPattern;
      this._lastContext  = currentContext;
    } else {
      // ── STM / Hopfield branch ────────────────────────────────────────────
      action = hopfieldAction ?? FALLBACK_ACTION;
      source = 'STM';
      this._stmCount++;
      this._lastPattern = null;
      this._lastContext  = null;
    }

    this.lastMemorySource = source;

    // Append to rolling logs
    this.actionHistory.push({
      action,
      source,
      confidence,
      patternId: this._lastPattern?.patternId ?? null,
      timestamp: Date.now(),
    });
    this.memorySourceLog.push(source);

    // Trim logs to their caps
    if (this.actionHistory.length   > ACTION_LOG_CAP) this.actionHistory.shift();
    if (this.memorySourceLog.length > SOURCE_LOG_CAP) {
      this.memorySourceLog.splice(0, SOURCE_LOG_TRIM);
    }

    return {
      action,
      source,
      confidence,
      patternId: this._lastPattern?.patternId ?? null,
    };
  }

  // ── b) retrievePatterns ───────────────────────────────────────────────────

  /**
   * Query LTM for patterns that match the current sensory state.
   *
   * Confidence is the geometric product of reliability and consolidation strength —
   * a pattern must score well on both dimensions to override Hopfield output.
   *
   * @param {number[]} sensoryState   25-element binary sensory vector
   * @param {string}   context        LTM search context
   * @param {number}  [topK=5]        Maximum candidates to retrieve
   * @returns {{
   *   bestPattern:  import('./ltm.js').LTMPattern|null,
   *   confidence:   number,
   *   candidates:   Array<{pattern, score, matchScore}>
   * }}
   */
  retrievePatterns(sensoryState, context, topK = 5) {
    if (!sensoryState?.length) {
      return { bestPattern: null, confidence: 0, candidates: [] };
    }

    const candidates = this.ltm.searchPatterns(sensoryState, context, topK);
    if (!candidates.length) {
      return { bestPattern: null, confidence: 0, candidates: [] };
    }

    const { pattern: bestPattern } = candidates[0];
    const confidence = bestPattern.reliability * bestPattern.consolidationStrength;

    return { bestPattern, confidence, candidates };
  }

  // ── c) executePattern ─────────────────────────────────────────────────────

  /**
   * Extract the immediate action from a stored pattern.
   *
   * Returns the first action in the pattern's sequence — this is the one
   * relevant to the current frame. Future frames will re-query LTM, so
   * there is no need to step through the sequence here.
   *
   * @param {import('./ltm.js').LTMPattern} pattern
   * @returns {string} 'L' | 'F' | 'R'
   */
  executePattern(pattern) {
    if (!pattern?.actionSequence?.length) return FALLBACK_ACTION;
    return pattern.actionSequence[0];
  }

  // ── d) evaluateAction ─────────────────────────────────────────────────────

  /**
   * Provide outcome feedback after an action has been executed and a reward
   * received. Must be called once per frame after stepWorld().
   *
   * When the most recent action came from LTM, the pattern is updated via
   * LTMPattern.recordUsage() which applies Laplace-smoothed reliability,
   * EMA utility, consolidation-strength adjustment, and a fresh Wilson CI.
   *
   * @param {number[]}      sensoryState    Sensory state that prompted the action
   * @param {string}        action          Action that was executed
   * @param {number}        reward          Scalar reward returned by stepWorld()
   * @param {number[]|null} [newSensoryState=null] Post-action sensory state (reserved)
   * @returns {{
   *   action:         string,
   *   reward:         number,
   *   source:         'STM'|'LTM'|'blend',
   *   patternId:      string|null,
   *   patternUpdated: boolean,
   *   wasSuccess:     boolean,
   * }}
   */
  evaluateAction(sensoryState, action, reward, newSensoryState = null) {
    const wasSuccess = reward > 0;

    const feedback = {
      action,
      reward,
      source:         this.lastMemorySource,
      patternId:      this._lastPattern?.patternId ?? null,
      patternUpdated: false,
      wasSuccess,
    };

    // Update the LTM pattern that was used (if any)
    if (this.lastMemorySource === 'LTM' && this._lastPattern) {
      // recordUsage() handles: usageCount++, successCount, reliability (Laplace),
      // utility (EMA), consolidationStrength (±), confidenceInterval (Wilson)
      this._lastPattern.recordUsage(wasSuccess, reward);
      feedback.patternUpdated = true;
      console.log(
        `[evaluateAction] pattern=${this._lastPattern.patternId}`,
        `wasSuccess=${wasSuccess}, reward=${reward.toFixed(2)},`,
        `rel=${this._lastPattern.reliability.toFixed(3)},`,
        `cs=${this._lastPattern.consolidationStrength.toFixed(3)},`,
        `uses=${this._lastPattern.usageCount}`
      );
    }

    return feedback;
  }

  // ── e) blendActions ───────────────────────────────────────────────────────

  /**
   * Probabilistic blend of two string actions weighted by LTM confidence.
   *
   * Because actions are discrete strings (not continuous vectors), blending
   * is implemented as a weighted random draw rather than vector interpolation.
   *
   * Weight formula (per spec):
   *   weight = ltmConfidence × actionWeightSTM + (1 − ltmConfidence) × (1 − actionWeightSTM)
   *
   * `weight` is the probability of selecting the STM/Hopfield action:
   *   - ltmConfidence = 0.0 → P(STM) = 1 − actionWeightSTM  (e.g. 0.40)
   *   - ltmConfidence = 0.5 → P(STM) = 0.50 (neutral)
   *   - ltmConfidence = 1.0 → P(STM) = actionWeightSTM       (e.g. 0.60)
   *
   * To make LTM dominate at high confidence, set actionWeightSTM < 0.5.
   * Default (0.6) keeps STM slightly dominant at all confidence levels, which
   * is appropriate for early learning when LTM patterns are not yet reliable.
   *
   * @param {string} hopfieldAction  Hopfield/STM action: 'L' | 'F' | 'R'
   * @param {string} ltmAction       LTM pattern action: 'L' | 'F' | 'R'
   * @param {number} ltmConfidence   Confidence from retrievePatterns() [0, 1]
   * @returns {{ action: string, source: 'blend', stmWeight: number }}
   */
  blendActions(hopfieldAction, ltmAction, ltmConfidence) {
    const stmWeight =
      ltmConfidence * this.actionWeightSTM +
      (1 - ltmConfidence) * (1 - this.actionWeightSTM);

    const action = Math.random() < stmWeight ? hopfieldAction : ltmAction;

    this.lastMemorySource = 'blend';
    this._blendCount++;

    this.actionHistory.push({
      action,
      source:     'blend',
      confidence: ltmConfidence,
      patternId:  this._lastPattern?.patternId ?? null,
      timestamp:  Date.now(),
    });
    this.memorySourceLog.push('blend');

    if (this.actionHistory.length   > ACTION_LOG_CAP) this.actionHistory.shift();
    if (this.memorySourceLog.length > SOURCE_LOG_CAP) {
      this.memorySourceLog.splice(0, SOURCE_LOG_TRIM);
    }

    return { action, source: 'blend', stmWeight };
  }

  // ── f) getMemoryBalance ───────────────────────────────────────────────────

  /**
   * Return the lifetime ratio of STM vs LTM vs blend action selections.
   *
   * A rising ltmRatio over a trial indicates the agent is learning to trust
   * its stored patterns — a key metric for dual-memory effectiveness.
   *
   * @returns {{
   *   stmRatio:        number,
   *   ltmRatio:        number,
   *   blendRatio:      number,
   *   totalSelections: number,
   * }}
   */
  getMemoryBalance() {
    const total = this._stmCount + this._ltmCount + this._blendCount;
    if (total === 0) {
      return { stmRatio: 1, ltmRatio: 0, blendRatio: 0, totalSelections: 0 };
    }
    return {
      stmRatio:        this._stmCount   / total,
      ltmRatio:        this._ltmCount   / total,
      blendRatio:      this._blendCount / total,
      totalSelections: total,
    };
  }

  // ── g) stats ──────────────────────────────────────────────────────────────

  /**
   * Comprehensive snapshot of the controller's current state.
   *
   * @returns {{
   *   lastMemorySource:        'STM'|'LTM'|'blend',
   *   ltmUsageRate:            number,
   *   stmUsageRate:            number,
   *   blendUsageRate:          number,
   *   avgLTMConfidence:        number,
   *   patternUsageFrequency:   Record<string, number>,
   *   recentActions:           string[],
   *   totalSelections:         number,
   *   currentThreshold:        number,
   *   currentExplorationRate:  number,
   * }}
   */
  stats() {
    const balance          = this.getMemoryBalance();
    const avgLTMConfidence = this._confidenceWindow.length > 0
      ? this._confidenceWindow.reduce((s, c) => s + c, 0) / this._confidenceWindow.length
      : 0;

    // Pattern usage frequency from the rolling action history
    const patternUsageFrequency = {};
    for (const entry of this.actionHistory) {
      if (entry.patternId) {
        patternUsageFrequency[entry.patternId] =
          (patternUsageFrequency[entry.patternId] ?? 0) + 1;
      }
    }

    return {
      lastMemorySource:       this.lastMemorySource,
      ltmUsageRate:           +balance.ltmRatio.toFixed(3),
      stmUsageRate:           +balance.stmRatio.toFixed(3),
      blendUsageRate:         +balance.blendRatio.toFixed(3),
      avgLTMConfidence:       +avgLTMConfidence.toFixed(3),
      patternUsageFrequency,
      recentActions:          this.memorySourceLog.slice(-10),
      totalSelections:        this._totalSelections,
      currentThreshold:       this.ltmConfidenceThreshold,
      currentExplorationRate: this.explorationRate,
    };
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  /**
   * Push a confidence value onto the sliding window, trimming as needed.
   *
   * @param {number} confidence
   */
  _recordConfidence(confidence) {
    this._confidenceWindow.push(confidence);
    if (this._confidenceWindow.length > CONFIDENCE_WINDOW) {
      this._confidenceWindow.shift();
    }
  }
}

// ── Export ────────────────────────────────────────────────────────────────────

export { DualMemoryController };
