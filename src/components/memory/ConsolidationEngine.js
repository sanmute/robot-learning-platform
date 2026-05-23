/**
 * ConsolidationEngine.js — STM → LTM Pattern Consolidation
 *
 * The core learning pipeline for v0.10. Monitors the agent's short-term
 * memory and decides when and what to write into long-term memory via
 * three complementary triggers:
 *
 *   • Reward    — high cumulative reward in the recent STM window
 *   • Surprise  — sudden shift in attention (KL divergence of weights)
 *   • Periodic  — scheduled sweep every N frames to catch low-salience learning
 *
 * For each trigger a behavioural sequence is extracted from STM, matched
 * against existing LTM patterns, and either stored as a new pattern or used
 * to strengthen an existing one.
 *
 * Sensor vector layout (must match encodeSensors in App.jsx):
 *   indices 0–4   obstacle / nearby-agent sensors  (active = 1, inactive = −1)
 *   indices 5–9   food sensors                      (active = 1, inactive = −1)
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

import { LTMPattern, VALID_CONTEXTS } from './LTM.js';

// ── Constants ─────────────────────────────────────────────────────────────────

/** Numerical floor added to KL inputs to avoid log(0). */
const EPSILON = 1e-10;

/** Index range of obstacle/agent sensor bits in the sensory vector. */
const OBS_START  = 0;
const OBS_END    = 5;

/** Index range of food sensor bits in the sensory vector. */
const FOOD_START = 5;
const FOOD_END   = 10;

/** Minimum overlap required to consider two sequences the same pattern. */
const SIMILARITY_THRESHOLD = 0.60;

/** Maximum entries kept in consolidationHistory before trimming. */
const HISTORY_CAP = 5_000;

/** Number of entries removed when HISTORY_CAP is reached (10 %). */
const HISTORY_TRIM = 500;

// ── Local helpers (no external dependencies) ─────────────────────────────────

/**
 * Wilson score 95 % confidence interval for a proportion.
 * Re-declared here because ltm.js does not export it.
 *
 * @param {number} successes
 * @param {number} trials
 * @returns {{ low: number, high: number }}
 */
function wilsonInterval(successes, trials) {
  if (trials === 0) return { low: 0, high: 1 };
  const p      = successes / trials;
  const z2     = 1.96 * 1.96;
  const center = (p + z2 / (2 * trials)) / (1 + z2 / trials);
  const margin = (1.96 * Math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)))
                 / (1 + z2 / trials);
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}

// ── ConsolidationEngine ───────────────────────────────────────────────────────

/**
 * ConsolidationEngine — orchestrates memory consolidation for one agent.
 *
 * One instance per agent. Create it once after the agent's STM and LTM are
 * initialised, then call `engine.update(currentFrame, currentFrame)` inside
 * the simulation loop each frame.
 *
 * @example
 * // In initWorld / agent factory:
 * agent.consolidation = new ConsolidationEngine(agent.stm, agent.ltm);
 *
 * // In simLoop, after agent.stm.add(stmFrame):
 * agent.consolidation.update(currentFrame, currentFrame);
 */
class ConsolidationEngine {
  /**
   * @param {import('./stm.js').ShortTermMemory} stm - Source of raw experience frames
   * @param {import('./ltm.js').LongTermMemory}  ltm - Destination for consolidated patterns
   * @param {object}  [config={}]                    - Optional overrides
   * @param {number}  [config.windowSize=30]          Frames pulled per consolidation event
   * @param {number}  [config.rewardThreshold=15]     Cumulative-reward level that triggers reward consolidation
   * @param {number}  [config.surpriseThreshold=0.3]  KL-divergence level that triggers surprise consolidation
   * @param {number}  [config.periodicInterval=300]   Frames between periodic consolidation sweeps
   *
   * @throws {Error} If stm or ltm are not provided
   */
  constructor(stm, ltm, config = {}) {
    if (!stm) throw new Error('ConsolidationEngine: stm is required.');
    if (!ltm) throw new Error('ConsolidationEngine: ltm is required.');

    /** @type {import('./stm.js').ShortTermMemory} */
    this.stm = stm;

    /** @type {import('./ltm.js').LongTermMemory} */
    this.ltm = ltm;

    // ── Config ────────────────────────────────────────────────────────────
    /** @type {number} Frames to include in each consolidated sequence */
    this.windowSize        = config.windowSize        ?? 30;

    /** @type {number} Total STM-window reward that fires the reward trigger */
    this.rewardThreshold   = config.rewardThreshold   ?? 15;

    /** @type {number} KL divergence between consecutive attention vectors that fires the surprise trigger */
    this.surpriseThreshold = config.surpriseThreshold ?? 0.3;

    /** @type {number} Frames between mandatory periodic consolidations */
    this.periodicInterval  = config.periodicInterval  ?? 300;

    // ── State ─────────────────────────────────────────────────────────────
    /**
     * Frame number of the most recent periodic consolidation.
     * Initialised to 0 so the first sweep fires after periodicInterval frames.
     * @type {number}
     */
    this.lastPeriodicConsolidation = 0;

    /**
     * Audit log of every consolidation event.
     * Each entry: { timestamp, trigger, triggerStrength, context, patternId, eventType, totalReward, frameCount }
     * @type {object[]}
     */
    this.consolidationHistory = [];

    // ── Internal metrics ──────────────────────────────────────────────────
    /** @type {number} Running total of consolidation events (all triggers) */
    this._totalConsolidations = 0;

    /** @type {number} Patterns created as new LTM entries */
    this._newPatternCount = 0;

    /** @type {number} Existing LTM patterns that were strengthened */
    this._strengthenCount = 0;

    /**
     * Per-context counters for the "ctx_###" ID format.
     * Instance-level so multi-agent runs don't collide.
     * @type {{ exploration: number, foraging: number, avoidance: number }}
     */
    this._patternCounters = { exploration: 0, foraging: 0, avoidance: 0 };
  }

  // ── a) update ─────────────────────────────────────────────────────────────

  /**
   * Main entry point — call once per simulation frame after stm.add().
   *
   * Evaluates all three triggers, extracts the relevant STM window for each
   * that fired, and consolidates each sequence independently.
   *
   * @param {number} currentFrame - Monotonically increasing simulation frame counter
   * @param {number} currentTime  - Same as currentFrame in this implementation;
   *                                reserved for wall-clock timestamp support
   */
  update(currentFrame, currentTime) {
    const triggers = this.checkTriggers(currentFrame, currentTime);
    if (!triggers.length) return;

    for (const trigger of triggers) {
      const sequence = this.extractSequence(trigger.frameCount);
      if (sequence.frames.length < 2) continue;
      this.consolidate(sequence, trigger);
    }

    // Advance periodic checkpoint after processing so the next window is fresh
    if (triggers.some(t => t.reason === 'periodic')) {
      this.lastPeriodicConsolidation = currentFrame;
    }
  }

  // ── b) checkTriggers ──────────────────────────────────────────────────────

  /**
   * Evaluate all three consolidation conditions and return those that fired.
   *
   * Triggers are independent — more than one can fire on the same frame,
   * each producing its own consolidation event with an appropriate window size
   * and strength.
   *
   * @param {number} currentFrame
   * @param {number} currentTime
   * @returns {Array<{ reason: string, frameCount: number, strength: number }>}
   *          May be empty if no trigger conditions are met
   */
  checkTriggers(currentFrame, currentTime) {
    const triggers = [];

    // Require at least 2 frames for any meaningful comparison
    if (this.stm.buffer.length < 2) return triggers;

    const available    = Math.min(this.windowSize, this.stm.buffer.length);
    const recentFrames = this.stm.getWindow(available);
    if (!recentFrames.length) return triggers;

    // ── Reward trigger ────────────────────────────────────────────────────
    // Fires when cumulative reward in the window exceeds the dopamine threshold.
    // Strength scales with how far the reward exceeds the threshold (soft cap at 2×).
    const totalReward = recentFrames.reduce((s, f) => s + f.reward, 0);
    if (totalReward > this.rewardThreshold) {
      triggers.push({
        reason:     'reward',
        frameCount: this.windowSize,
        strength:   Math.min(1, totalReward / (this.rewardThreshold * 2)),
      });
    }

    // ── Surprise trigger ──────────────────────────────────────────────────
    // Fires when attention shifts sharply between consecutive frames.
    // Uses a shorter window — surprise is a local, moment-specific signal.
    const surprise = this.calculateSurprise(currentFrame);
    if (surprise > this.surpriseThreshold) {
      triggers.push({
        reason:     'surprise',
        frameCount: Math.min(this.windowSize, 15),
        strength:   Math.min(1, surprise / (this.surpriseThreshold * 2)),
      });
    }

    // ── Periodic trigger ──────────────────────────────────────────────────
    // Scheduled sweep to consolidate low-salience sequences that never crossed
    // the reward or surprise thresholds but still contain useful information.
    if ((currentFrame - this.lastPeriodicConsolidation) > this.periodicInterval) {
      triggers.push({
        reason:     'periodic',
        frameCount: this.windowSize,
        strength:   0.5, // neutral — not driven by an exceptional event
      });
    }

    return triggers;
  }

  // ── c) calculateSurprise ──────────────────────────────────────────────────

  /**
   * Measure how unexpected the current moment is by comparing the attention
   * weight distributions of the two most recent STM frames via KL divergence.
   *
   * A high value means the agent's Hopfield network suddenly shifted which
   * stored pattern it is matching — a reliable marker of a novel situation.
   *
   * @param {number} _currentFrame - Reserved for future multi-step lookback
   * @returns {number} Normalised surprise in [0, 1]; 0 if data unavailable
   */
  calculateSurprise(_currentFrame) {
    const current = this.stm.getFrame(0);   // most recent
    const prev    = this.stm.getFrame(-1);  // one frame earlier
    if (!current || !prev) return 0;

    const p = current.attentionWeights;
    const q = prev.attentionWeights;
    if (!p?.length || !q?.length) return 0;

    return this.klDivergence(p, q);
  }

  /**
   * KL divergence: KL(P ∥ Q) = Σᵢ pᵢ · log(pᵢ / qᵢ)
   *
   * Both arrays should be valid probability distributions (non-negative, sum ≈ 1).
   * EPSILON is added before each term to handle zeros without throwing.
   * The raw KL value (which is unbounded) is mapped to [0, 1] via
   * 1 − e^(−kl), a smooth monotone function.
   *
   * @param {number[]} p - Distribution P (e.g. current attention weights)
   * @param {number[]} q - Distribution Q (e.g. previous attention weights)
   * @returns {number} Normalised divergence in [0, 1)
   */
  klDivergence(p, q) {
    if (!p?.length || !q?.length || p.length !== q.length) return 0;

    let kl = 0;
    for (let i = 0; i < p.length; i++) {
      const pi = Math.max(0, p[i]) + EPSILON;
      const qi = Math.max(0, q[i]) + EPSILON;
      kl += pi * Math.log(pi / qi);
    }

    // Soft mapping from [0, ∞) → [0, 1): kl=0 → 0, kl→∞ → 1
    return 1 - Math.exp(-kl);
  }

  // ── d) extractSequence ────────────────────────────────────────────────────

  /**
   * Pull the last `lookback` frames from the STM circular buffer and build a
   * structured sequence object ready for the consolidation pipeline.
   *
   * @param {number} lookback - Requested number of frames (clamped to buffer size)
   * @returns {{
   *   frames:            import('./stm.js').STMFrame[],
   *   actions:           string[],
   *   totalReward:       number,
   *   triggerCondition:  number[]|null,
   *   keyFrameIndices:   number[]
   * }}
   */
  extractSequence(lookback) {
    const frames = this.stm.getWindow(Math.min(lookback, this.stm.buffer.length));

    const actions          = frames.map(f => f.action);
    const totalReward      = frames.reduce((s, f) => s + f.reward, 0);

    // triggerCondition = sensory state of the most recent frame (the "cue" for recall)
    const triggerCondition = frames.length > 0
      ? frames[frames.length - 1].sensoryState
      : null;

    // Key frames: positions where reward changes abruptly (|Δreward| > 5)
    // plus mandatory start and end anchors for structural coverage.
    const keyFrameIndices = [0];
    for (let i = 1; i < frames.length - 1; i++) {
      if (Math.abs(frames[i].reward - frames[i - 1].reward) > 5) {
        keyFrameIndices.push(i);
      }
    }
    if (frames.length > 1) keyFrameIndices.push(frames.length - 1);

    return { frames, actions, totalReward, triggerCondition, keyFrameIndices };
  }

  // ── e) consolidate ────────────────────────────────────────────────────────

  /**
   * Core consolidation decision: strengthen an existing pattern or create a new one.
   * Appends a record to consolidationHistory regardless of which branch is taken.
   *
   * @param {{ frames, actions, totalReward, triggerCondition, keyFrameIndices }} sequence
   * @param {{ reason: string, frameCount: number, strength: number }}            trigger
   */
  consolidate(sequence, trigger) {
    const context = this.inferContext(sequence);
    const similar = this.findSimilarPattern(sequence, context);

    let patternId;
    let eventType;

    if (similar) {
      this.strengthenPattern(similar, sequence, trigger);
      patternId = similar.patternId;
      eventType = 'strengthen';
      this._strengthenCount++;
    } else {
      patternId = this.generatePatternId(context);
      this.createNewPattern(sequence, context, patternId, trigger);
      eventType = 'create';
      this._newPatternCount++;
    }

    this._totalConsolidations++;

    this.consolidationHistory.push({
      timestamp:       Date.now(),
      trigger:         trigger.reason,
      triggerStrength: trigger.strength,
      context,
      patternId,
      eventType,
      totalReward:     sequence.totalReward,
      frameCount:      sequence.frames.length,
    });

    // Bounded history — trim oldest 10 % when cap is reached
    if (this.consolidationHistory.length > HISTORY_CAP) {
      this.consolidationHistory.splice(0, HISTORY_TRIM);
    }
  }

  // ── f) inferContext ───────────────────────────────────────────────────────

  /**
   * Determine the dominant behavioral context of a sequence by scanning every
   * frame's sensory state with a single pass.
   *
   * Precedence rule (same as determineContext in App.jsx):
   *   foraging  ≻  avoidance  ≻  exploration
   *
   * "Dominant" means the context whose frame count is highest.
   * In a tie between food and obstacle, foraging wins (food-seeking takes priority).
   *
   * @param {{ frames: import('./stm.js').STMFrame[] }} sequence
   * @returns {'exploration'|'foraging'|'avoidance'}
   */
  inferContext(sequence) {
    let foodFrames     = 0;
    let obstacleFrames = 0;

    for (const frame of sequence.frames) {
      const s = frame.sensoryState;
      if (!s) continue;

      let hasFood = false, hasObs = false;

      for (let i = FOOD_START; i < FOOD_END && !hasFood; i++) {
        if (s[i] === 1) hasFood = true;
      }
      for (let i = OBS_START; i < OBS_END && !hasObs; i++) {
        if (s[i] === 1) hasObs = true;
      }

      if (hasFood) foodFrames++;
      if (hasObs)  obstacleFrames++;
    }

    if (foodFrames >= obstacleFrames && foodFrames > 0) return 'foraging';
    if (obstacleFrames > 0) return 'avoidance';
    return 'exploration';
  }

  // ── g) findSimilarPattern ─────────────────────────────────────────────────

  /**
   * Search the LTM for an existing pattern similar enough to the current sequence.
   *
   * Uses ltm.searchPatterns() for a fast sensory-similarity pre-filter, then
   * refines each candidate with sequenceSimilarity() (action + trigger).
   * Returns the first candidate that clears SIMILARITY_THRESHOLD (0.7).
   *
   * @param {{ actions: string[], triggerCondition: number[] }} sequence
   * @param {string} context
   * @returns {import('./ltm.js').LTMPattern|null} Matching pattern or null
   */
  findSimilarPattern(sequence, context) {
    if (!sequence.triggerCondition) return null;

    // Pre-filter: up to 5 candidates ranked by sensory match × reliability × cs
    const candidates = this.ltm.searchPatterns(sequence.triggerCondition, context, 5);

    let bestSim = 0;
    let bestId  = null;

    for (const { pattern } of candidates) {
      const sim = this.sequenceSimilarity(sequence, pattern);
      if (sim > bestSim) { bestSim = sim; bestId = pattern.patternId; }
      if (sim > SIMILARITY_THRESHOLD) return pattern;
    }

    // Log when candidates were found but none cleared the threshold.
    // This surfaces matching failures (low similarity) separately from the case
    // where searchPatterns returned nothing (reliability or cs = 0).
    if (candidates.length > 0) {
      console.log(
        `[findSimilarPattern] ctx=${context}, candidates=${candidates.length},`,
        `bestSim=${bestSim.toFixed(3)} (${bestId}), threshold=${SIMILARITY_THRESHOLD} → no match`
      );
    }

    return null;
  }

  // ── h) sequenceSimilarity ─────────────────────────────────────────────────

  /**
   * Combined similarity score between a live sequence and a stored LTM pattern.
   *
   * Averaged from two orthogonal signals:
   *   1. Action similarity  — position-aligned match ratio over the longer sequence
   *   2. Trigger similarity — Hamming similarity of the two sensory trigger vectors
   *
   * @param {{ actions: string[], triggerCondition: number[] }} sequence
   * @param {import('./ltm.js').LTMPattern}                     pattern
   * @returns {number} Combined similarity in [0, 1]
   */
  sequenceSimilarity(sequence, pattern) {
    // ── Action similarity ─────────────────────────────────────────────────
    const a      = sequence.actions;
    const b      = pattern.actionSequence ?? [];
    const minLen = Math.min(a.length, b.length);
    const maxLen = Math.max(a.length, b.length);

    let actionMatches = 0;
    for (let i = 0; i < minLen; i++) {
      if (a[i] === b[i]) actionMatches++;
    }
    const actionSimilarity = maxLen > 0 ? actionMatches / maxLen : 0;

    // ── Trigger similarity ────────────────────────────────────────────────
    const triggerSimilarity = this.bitSimilarity(
      sequence.triggerCondition,
      pattern.triggerCondition
    );

    return (actionSimilarity + triggerSimilarity) / 2;
  }

  // ── i) bitSimilarity ─────────────────────────────────────────────────────

  /**
   * Hamming similarity: fraction of element-wise matches between two equal-length
   * binary (−1 / 1) vectors.
   *
   * @param {number[]|null} bits1
   * @param {number[]|null} bits2
   * @returns {number} Similarity in [0, 1]; 0 on null or length mismatch
   */
  bitSimilarity(bits1, bits2) {
    if (!bits1 || !bits2 || bits1.length !== bits2.length || !bits1.length) return 0;

    let matches = 0;
    for (let i = 0; i < bits1.length; i++) {
      if (bits1[i] === bits2[i]) matches++;
    }
    return matches / bits1.length;
  }

  // ── j) createNewPattern ───────────────────────────────────────────────────

  /**
   * Instantiate a new LTMPattern from a consolidated sequence and write it to LTM.
   *
   * Initial parameters:
   *   reliability          = 0.5   (neutral Bayesian prior; updates with each recall)
   *   consolidationStrength = 0.3 + trigger.strength × 0.2  (stronger trigger → stronger trace)
   *   utility              = clamped totalReward / rewardThreshold
   *
   * @param {{ frames, actions, totalReward, triggerCondition, keyFrameIndices }} sequence
   * @param {'exploration'|'foraging'|'avoidance'} context
   * @param {string} patternId
   * @param {{ strength: number }} trigger
   * @returns {import('./ltm.js').LTMPattern} The newly stored pattern
   */
  createNewPattern(sequence, context, patternId, trigger) {
    // Materialise key frames as plain objects (no live STMFrame references)
    const keyFrames = sequence.keyFrameIndices
      .map(i => sequence.frames[i])
      .filter(Boolean)
      .map(({ timestamp, sensoryState, neuralState, action, reward }) =>
        ({ timestamp, sensoryState, neuralState, action, reward })
      );

    const pattern = new LTMPattern({
      patternId,
      context,
      triggerCondition:      sequence.triggerCondition,
      actionSequence:        sequence.actions,
      keyFrames,
      utility:               Math.min(1, sequence.totalReward / Math.max(1, this.rewardThreshold)),
      reliability:           0.5,
      consolidationStrength: Math.min(1, 0.3 + trigger.strength * 0.2),
      abstractDescription:   this.abstractPattern(sequence),
    });

    this.ltm.storePattern(pattern);
    return pattern;
  }

  // ── k) strengthenPattern ──────────────────────────────────────────────────

  /**
   * Reinforce an existing LTM pattern with evidence from the current sequence.
   *
   * Updates are applied directly to the pattern object (which is held by
   * reference inside the LTM Map, so no separate updatePattern() call is needed):
   *
   *   usageCount++
   *   utility              ← EMA: 0.7 × old + 0.3 × new
   *   consolidationStrength ← min(1, old + trigger.strength × 0.1)
   *   reliability          ← successCount / usageCount  (raw ratio per spec)
   *   confidenceInterval   ← Wilson 95 % CI (recomputed)
   *
   * @param {import('./ltm.js').LTMPattern}                     pattern
   * @param {{ totalReward: number }}                           sequence
   * @param {{ strength: number }}                              trigger
   */
  strengthenPattern(pattern, sequence, trigger) {
    pattern.usageCount++;

    // Exponential moving average on utility — recent reward weighted 30 %
    pattern.utility = 0.7 * pattern.utility + 0.3 * sequence.totalReward;

    // Consolidation strength grows with each reinforced recall, plateaus at 1.0
    pattern.consolidationStrength = Math.min(
      1.0,
      pattern.consolidationStrength + trigger.strength * 0.1
    );

    pattern.lastUsed = Date.now();

    // Count a success when reward exceeds half the trigger threshold.
    // Using a dynamic value (rather than hard-coded 10) keeps the bar proportional
    // to whatever rewardThreshold was configured for this engine instance.
    if (sequence.totalReward > this.rewardThreshold * 0.5) pattern.successCount++;

    // Laplace-smoothed reliability — avoids crashing to 0 on early failures and
    // stays consistent with LTMPattern.recordUsage() which uses the same formula.
    // Raw ratio (successCount / usageCount) would hit 0 the first time a periodic
    // trigger fires with no food eaten, permanently disabling the pattern.
    pattern.reliability        = (pattern.successCount + 1) / (pattern.usageCount + 2);
    pattern.confidenceInterval = wilsonInterval(pattern.successCount, pattern.usageCount);
  }

  // ── l) abstractPattern ────────────────────────────────────────────────────

  /**
   * Produce a compact human-readable tag string for a sequence.
   *
   * Format: "<action-run><_food?><_obstacle?>"
   * Examples: "LFFR_food",  "RRFF_obstacle",  "FFF",  "LFR_food_obstacle"
   *
   * The action run is the raw concatenation of every action in the window.
   * Sensor tags are appended if any frame in the window saw food / an obstacle.
   *
   * @param {{ actions: string[], frames: import('./stm.js').STMFrame[] }} sequence
   * @returns {string}
   */
  abstractPattern(sequence) {
    const actionStr = sequence.actions.join('');

    let hasFood = false;
    let hasObs  = false;

    for (const frame of sequence.frames) {
      const s = frame.sensoryState;
      if (!s) continue;
      for (let i = FOOD_START; i < FOOD_END && !hasFood; i++) if (s[i] === 1) hasFood = true;
      for (let i = OBS_START;  i < OBS_END  && !hasObs;  i++) if (s[i] === 1) hasObs  = true;
      if (hasFood && hasObs) break; // short-circuit once both found
    }

    return actionStr + (hasFood ? '_food' : '') + (hasObs ? '_obstacle' : '');
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  /**
   * Generate a unique pattern ID in "ctx_###" format.
   *
   * Counter is instance-level (not module-level) so three agents running in
   * parallel produce non-colliding IDs within each agent's LTM.
   *
   * @param {'exploration'|'foraging'|'avoidance'} context
   * @returns {string} e.g. "exp_001", "for_042", "avo_007"
   * @throws {TypeError} If context is not a valid VALID_CONTEXTS member
   */
  generatePatternId(context) {
    if (!VALID_CONTEXTS.includes(context)) {
      throw new TypeError(`ConsolidationEngine.generatePatternId: invalid context "${context}".`);
    }
    this._patternCounters[context]++;
    const prefix = context.slice(0, 3);                                  // exp / for / avo
    const num    = String(this._patternCounters[context]).padStart(3, '0');
    return `${prefix}_${num}`;
  }

  /**
   * Snapshot of consolidation activity and the current LTM state.
   *
   * @returns {{
   *   totalConsolidations:  number,
   *   newPatterns:          number,
   *   strengthened:         number,
   *   consolidationRate:    number,
   *   historyEntries:       number,
   *   ltmPatterns:          number,
   *   byContext:            Record<string, number>,
   *   avgReliability:       number,
   *   avgConsolidationStr:  number,
   *   triggerBreakdown:     { reward: number, surprise: number, periodic: number },
   * }}
   */
  getStats() {
    const ltmStats = this.ltm.stats();

    // Consolidation rate: events per second based on wall-clock history timestamps
    let consolidationRate = 0;
    if (this.consolidationHistory.length >= 2) {
      const first     = this.consolidationHistory[0].timestamp;
      const last      = this.consolidationHistory[this.consolidationHistory.length - 1].timestamp;
      const elapsedMs = last - first;
      if (elapsedMs > 0) {
        consolidationRate = (this.consolidationHistory.length / elapsedMs) * 1000;
      }
    }

    // Count how often each trigger type fired
    const triggerBreakdown = { reward: 0, surprise: 0, periodic: 0 };
    for (const ev of this.consolidationHistory) {
      if (ev.trigger in triggerBreakdown) triggerBreakdown[ev.trigger]++;
    }

    return {
      totalConsolidations:  this._totalConsolidations,
      newPatterns:          this._newPatternCount,
      strengthened:         this._strengthenCount,
      consolidationRate:    +consolidationRate.toFixed(4), // events / second
      historyEntries:       this.consolidationHistory.length,
      ltmPatterns:          ltmStats.totalPatterns,
      byContext:            ltmStats.byContext,
      avgReliability:       +ltmStats.avgReliability.toFixed(3),
      avgConsolidationStr:  +ltmStats.avgConsolidationStrength.toFixed(3),
      triggerBreakdown,
    };
  }
}

// ── Export ────────────────────────────────────────────────────────────────────

export { ConsolidationEngine };
