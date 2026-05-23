/**
 * LTM.js - Long-Term Memory for Embodied Agents
 *
 * Implements hierarchical pattern storage that persists successful behavioral
 * sequences beyond the STM window. Patterns are organised by behavioral context
 * and scored by reliability, utility, and consolidation strength so the weakest
 * patterns are evicted first when capacity is reached.
 *
 * Contexts: "exploration" | "foraging" | "avoidance"
 *
 * Integration with STM:
 *   Consolidation pipelines read STMFrame windows and call ltm.storePattern()
 *   with the summarised experience. The LTM is queried during action selection
 *   via ltm.searchPatterns(agent.sensoryState, context).
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

// ── Constants ─────────────────────────────────────────────────────────────────

/** All valid behavioral contexts. Checked on every write. */
const VALID_CONTEXTS = ['exploration', 'foraging', 'avoidance'];

/** z-score for 95 % Wilson confidence interval. */
const Z_95 = 1.96;

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Generate a unique pattern ID string.
 * Format: ltm_<timestamp>_<random-7-chars>
 * Collision probability negligible for < 1 M patterns / second.
 *
 * @returns {string}
 */
function generatePatternId() {
  return `ltm_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Compute the Wilson score confidence interval for a proportion.
 *
 * Returns a tight interval when n is small and gracefully degrades to
 * [0, 1] with zero observations.
 *
 * @param {number} successes - Number of successes (≥ 0)
 * @param {number} trials    - Total observations (≥ 0)
 * @returns {{ low: number, high: number }} 95 % confidence interval
 */
function wilsonInterval(successes, trials) {
  if (trials === 0) return { low: 0, high: 1 };

  const p  = successes / trials;
  const z2 = Z_95 * Z_95;
  const n  = trials;

  const center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const margin = (Z_95 * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n);

  return {
    low:  Math.max(0, center - margin),
    high: Math.min(1, center + margin),
  };
}

// ── LTMPattern ────────────────────────────────────────────────────────────────

/**
 * LTMPattern — A single consolidated behavioral pattern stored in LTM.
 *
 * Represents a reusable sensory→action mapping that the agent has learned
 * is effective in a particular context. Patterns evolve over time: each
 * time the agent reuses a pattern its usageCount / successCount are updated
 * and reliability is recalculated, causing consolidationStrength to grow or
 * decay accordingly.
 *
 * Coordinate conventions match STM:
 *   triggerCondition  — same 25-element binary (-1/1) vector as STMFrame.sensoryState
 *   actionSequence    — ordered array of 'L' | 'F' | 'R' strings
 *   keyFrames         — array of { timestamp, sensoryState, neuralState, action, reward }
 *                       snapshot objects (plain-object mirror of STMFrame fields)
 */
class LTMPattern {
  /**
   * @param {object} cfg
   * @param {string}   [cfg.patternId]             Auto-generated if omitted
   * @param {string}    cfg.context                 'exploration' | 'foraging' | 'avoidance'
   * @param {number[]}  cfg.triggerCondition        25-element sensory vector that activates this pattern
   * @param {string[]}  cfg.actionSequence          Ordered actions ['F','R','F',…]
   * @param {object[]} [cfg.keyFrames=[]]           Representative STM snapshots
   * @param {number}   [cfg.reliability=0.5]        Prior reliability estimate [0,1]
   * @param {number}   [cfg.utility=0.5]            Expected reward utility [0,1]
   * @param {number}   [cfg.consolidationStrength=0.1] Memory strength from repetition [0,1]
   * @param {string}   [cfg.abstractDescription=''] Human-readable summary
   * @param {number}   [cfg.usageCount=0]           Times this pattern has been recalled
   * @param {number}   [cfg.successCount=0]         Times recall led to positive reward
   */
  constructor({
    patternId            = generatePatternId(),
    context,
    triggerCondition,
    actionSequence,
    keyFrames            = [],
    reliability          = 0.5,
    utility              = 0.5,
    consolidationStrength = 0.1,
    abstractDescription  = '',
    usageCount           = 0,
    successCount         = 0,
  } = {}) {
    // ── Identity ──────────────────────────────────────────────
    /** @type {string} Unique identifier for this pattern */
    this.patternId = patternId;

    /** @type {string} Behavioral context: 'exploration' | 'foraging' | 'avoidance' */
    this.context = context;

    // ── Sensory / action content ──────────────────────────────
    /** @type {number[]} 25-element binary sensory vector that triggers recall */
    this.triggerCondition = triggerCondition;

    /** @type {string[]} Ordered sequence of actions ['L'|'F'|'R', …] */
    this.actionSequence = actionSequence;

    /** @type {object[]} Key STM snapshots that characterise this pattern */
    this.keyFrames = keyFrames;

    // ── Quality metrics ───────────────────────────────────────
    /**
     * @type {number} Probability this pattern leads to positive outcome [0,1].
     * Updated as: reliability = successCount / usageCount (with Laplace smoothing).
     */
    this.reliability = reliability;

    /**
     * @type {number} Expected reward magnitude when pattern is applied [0,1].
     * Scaled to [0,1] from raw cumulative reward.
     */
    this.utility = utility;

    /**
     * @type {number} Consolidation strength: grows with each successful recall,
     * decays via pruneLowestValue() when memory is under pressure [0,1].
     */
    this.consolidationStrength = consolidationStrength;

    // ── Statistics ────────────────────────────────────────────
    /** @type {number} Total number of times this pattern has been recalled */
    this.usageCount = usageCount;

    /** @type {number} Recalls that resulted in positive reward */
    this.successCount = successCount;

    // ── Metadata ──────────────────────────────────────────────
    /** @type {number} Unix ms timestamp at creation */
    this.createdAt = Date.now();

    /** @type {number} Unix ms timestamp of most recent recall */
    this.lastUsed = this.createdAt;

    /** @type {string} Short natural-language description (set externally) */
    this.abstractDescription = abstractDescription;

    /**
     * @type {{ low: number, high: number }}
     * Wilson 95 % confidence interval on the success rate.
     * Recomputed on every updateStats() call.
     */
    this.confidenceInterval = wilsonInterval(successCount, usageCount);
  }

  // ── Getters ───────────────────────────────────────────────────────────────

  /**
   * Current success rate: successCount / usageCount.
   * Returns 0 when never used.
   *
   * @returns {number} [0, 1]
   */
  get successRate() {
    return this.usageCount > 0 ? this.successCount / this.usageCount : 0;
  }

  /**
   * Age of this pattern in fractional days since creation.
   *
   * @returns {number} days (float)
   */
  get age() {
    return (Date.now() - this.createdAt) / 86_400_000;
  }

  /**
   * Fractional days since this pattern was last recalled.
   *
   * @returns {number} days (float)
   */
  get daysSinceUse() {
    return (Date.now() - this.lastUsed) / 86_400_000;
  }

  /**
   * Composite value used by the pruner. Higher = more worth keeping.
   *   value = reliability × consolidationStrength / (1 + daysSinceUse)
   * Recent, reliable, well-consolidated patterns are hardest to evict.
   *
   * @returns {number}
   */
  get value() {
    return (this.reliability * this.consolidationStrength) / (1 + this.daysSinceUse);
  }

  /**
   * True when the pattern has been used enough times to be trusted.
   * Threshold: reliability > 0.6 with at least 5 trials.
   *
   * @returns {boolean}
   */
  get isReliable() {
    return this.reliability > 0.6 && this.usageCount >= 5;
  }

  // ── Methods ───────────────────────────────────────────────────────────────

  /**
   * Record a recall event and update all derived statistics.
   * Call this every time the pattern is retrieved and acted upon.
   *
   * @param {boolean} wasSuccessful - Whether the recall led to positive reward
   * @param {number}  [rewardDelta=0] - Raw reward increment (used to update utility)
   */
  recordUsage(wasSuccessful, rewardDelta = 0) {
    this.usageCount++;
    if (wasSuccessful) this.successCount++;
    this.lastUsed = Date.now();

    // Reliability: Laplace-smoothed frequency (avoids 0 and 1 extremes early on)
    this.reliability = (this.successCount + 1) / (this.usageCount + 2);

    // Utility: exponential moving average of reward signal
    const alpha = 0.1;
    this.utility = (1 - alpha) * this.utility + alpha * Math.max(0, Math.min(1, rewardDelta));

    // Consolidation grows with successful use, plateau at 1
    if (wasSuccessful) {
      this.consolidationStrength = Math.min(1, this.consolidationStrength + 0.05);
    } else {
      this.consolidationStrength = Math.max(0, this.consolidationStrength - 0.02);
    }

    // Refresh confidence interval
    this.confidenceInterval = wilsonInterval(this.successCount, this.usageCount);
  }

  /**
   * Human-readable summary for console debugging.
   *
   * @returns {string}
   */
  toString() {
    const ci = this.confidenceInterval;
    return (
      `LTMPattern(id=${this.patternId}, ctx=${this.context}, ` +
      `rel=${this.reliability.toFixed(2)}, cs=${this.consolidationStrength.toFixed(2)}, ` +
      `uses=${this.usageCount}, succ=${this.successCount}, ` +
      `CI=[${ci.low.toFixed(2)}, ${ci.high.toFixed(2)}], ` +
      `age=${this.age.toFixed(3)}d, val=${this.value.toFixed(4)}, ` +
      `desc="${this.abstractDescription}")`
    );
  }

  /**
   * Plain-object snapshot suitable for JSON.stringify.
   *
   * @returns {object}
   */
  toJSON() {
    return {
      patternId:            this.patternId,
      context:              this.context,
      triggerCondition:     this.triggerCondition,
      actionSequence:       this.actionSequence,
      keyFrames:            this.keyFrames,
      reliability:          this.reliability,
      utility:              this.utility,
      consolidationStrength: this.consolidationStrength,
      usageCount:           this.usageCount,
      successCount:         this.successCount,
      createdAt:            this.createdAt,
      lastUsed:             this.lastUsed,
      abstractDescription:  this.abstractDescription,
      confidenceInterval:   this.confidenceInterval,
    };
  }

  /**
   * Reconstruct an LTMPattern from a toJSON() snapshot.
   *
   * @param {object} data - Output of toJSON()
   * @returns {LTMPattern}
   */
  static fromJSON(data) {
    const p = new LTMPattern({
      patternId:            data.patternId,
      context:              data.context,
      triggerCondition:     data.triggerCondition,
      actionSequence:       data.actionSequence,
      keyFrames:            data.keyFrames || [],
      reliability:          data.reliability,
      utility:              data.utility,
      consolidationStrength: data.consolidationStrength,
      abstractDescription:  data.abstractDescription || '',
      usageCount:           data.usageCount,
      successCount:         data.successCount,
    });
    // Restore timestamps exactly (constructor sets them to Date.now())
    p.createdAt  = data.createdAt;
    p.lastUsed   = data.lastUsed;
    p.confidenceInterval = data.confidenceInterval ?? wilsonInterval(data.successCount, data.usageCount);
    return p;
  }
}

// ── LongTermMemory ────────────────────────────────────────────────────────────

/**
 * LongTermMemory — Hierarchical pattern store for an embodied agent.
 *
 * Patterns are bucketed by behavioral context so searches only scan the
 * relevant subset (O(k) where k ≤ maxCapacity / numContexts on average).
 * Within each context a Map gives O(1) lookup by patternId.
 *
 * Capacity management: when patternCount reaches maxCapacity, the single
 * lowest-value pattern across all contexts is evicted before the new one
 * is inserted, keeping total storage bounded.
 *
 * Usage sketch:
 * ```js
 * const ltm = new LongTermMemory();
 *
 * // Store a pattern after consolidation from STM
 * const p = new LTMPattern({ context: 'foraging', triggerCondition: sensory, actionSequence: ['F','F','R'] });
 * ltm.storePattern(p);
 *
 * // Recall during action selection
 * const candidates = ltm.searchPatterns(agent.sensoryState, 'foraging', 3);
 * if (candidates.length) console.log('Best match:', candidates[0].pattern.toString());
 * ```
 */
class LongTermMemory {
  /**
   * @param {number} [maxCapacity=1000] Hard upper bound on total stored patterns
   */
  constructor(maxCapacity = 1000) {
    /**
     * Hierarchical pattern store.
     * patterns[context] → Map<patternId, LTMPattern>
     *
     * @type {{ exploration: Map<string,LTMPattern>, foraging: Map<string,LTMPattern>, avoidance: Map<string,LTMPattern> }}
     */
    this.patterns = {};
    for (const ctx of VALID_CONTEXTS) {
      this.patterns[ctx] = new Map();
    }

    /** @type {number} Hard capacity ceiling across all contexts */
    this.maxCapacity = maxCapacity;

    /** @type {number} Running total of stored patterns (faster than summing Map sizes) */
    this.patternCount = 0;

    /**
     * Audit log of consolidation events.
     * Each entry: { timestamp, event, patternId, context, detail }
     *
     * @type {Array<object>}
     */
    this.consolidationHistory = [];

    /** @type {number} Unix ms timestamp at construction */
    this.createdAt = Date.now();
  }

  // ── Context validation ────────────────────────────────────────────────────

  /**
   * Assert that a context string is one of the three valid values.
   * Throws a descriptive TypeError so callers get clear feedback.
   *
   * @param {string} context
   * @throws {TypeError}
   */
  _assertValidContext(context) {
    if (!VALID_CONTEXTS.includes(context)) {
      throw new TypeError(
        `LongTermMemory: invalid context "${context}". ` +
        `Expected one of: ${VALID_CONTEXTS.join(', ')}.`
      );
    }
  }

  // ── Core CRUD ─────────────────────────────────────────────────────────────

  /**
   * Store a pattern in LTM.
   *
   * If the patternId already exists in the given context the existing entry is
   * silently replaced (idempotent upsert). If total capacity would be exceeded,
   * the lowest-value pattern is pruned first.
   *
   * @param {LTMPattern} pattern - Must have a valid .context property
   * @returns {LTMPattern} The stored pattern (same object)
   * @throws {TypeError} If pattern.context is invalid
   * @throws {Error}     If pattern is not an LTMPattern instance
   */
  storePattern(pattern) {
    if (!(pattern instanceof LTMPattern)) {
      throw new Error('LongTermMemory.storePattern: argument must be an LTMPattern instance.');
    }
    this._assertValidContext(pattern.context);

    const bucket = this.patterns[pattern.context];
    const isUpdate = bucket.has(pattern.patternId);

    // Prune before inserting a genuinely new pattern
    if (!isUpdate && this.patternCount >= this.maxCapacity) {
      this.pruneLowestValue();
    }

    bucket.set(pattern.patternId, pattern);
    if (!isUpdate) this.patternCount++;

    this._logConsolidation(
      isUpdate ? 'update' : 'store',
      pattern.patternId,
      pattern.context,
      `reliability=${pattern.reliability.toFixed(2)}`
    );

    return pattern;
  }

  /**
   * Retrieve a pattern by ID.
   *
   * If context is provided the lookup is O(1). If omitted all three context
   * buckets are searched (still O(1) per bucket, at most 3 checks).
   *
   * @param {string}  patternId
   * @param {string} [context]  Narrow the search to one context for speed
   * @returns {LTMPattern|null} The pattern or null if not found
   */
  getPattern(patternId, context = null) {
    if (context !== null) {
      this._assertValidContext(context);
      return this.patterns[context].get(patternId) ?? null;
    }
    for (const ctx of VALID_CONTEXTS) {
      const found = this.patterns[ctx].get(patternId);
      if (found) return found;
    }
    return null;
  }

  /**
   * Apply a partial update to an existing pattern.
   *
   * Only the keys present in `updates` are written; all others are left intact.
   * Automatically refreshes `confidenceInterval` if usageCount or successCount
   * is changed.
   *
   * @param {string} patternId
   * @param {string} context      Required to keep the O(1) bucket lookup
   * @param {object} updates      Key/value pairs to merge into the pattern
   * @returns {LTMPattern|null}   Updated pattern or null if not found
   * @throws {TypeError}          If context is invalid
   */
  updatePattern(patternId, context, updates) {
    this._assertValidContext(context);

    const pattern = this.patterns[context].get(patternId);
    if (!pattern) return null;

    const statsChanged = 'usageCount' in updates || 'successCount' in updates;

    Object.assign(pattern, updates);
    pattern.lastUsed = Date.now();

    if (statsChanged) {
      pattern.confidenceInterval = wilsonInterval(pattern.successCount, pattern.usageCount);
    }

    this._logConsolidation('update', patternId, context, JSON.stringify(Object.keys(updates)));
    return pattern;
  }

  // ── Search ────────────────────────────────────────────────────────────────

  /**
   * Find the best-matching patterns for a given sensory state.
   *
   * Each candidate is ranked by a composite score:
   *   score = matchScore(sensoryState, pattern) × reliability × consolidationStrength
   *
   * Patterns with score = 0 are excluded from results.
   *
   * @param {number[]}  triggerCondition  25-element sensory vector to match against
   * @param {string}   [context=null]     Restrict to one context; null searches all
   * @param {number}   [topK=5]           Maximum results to return
   * @returns {Array<{ pattern: LTMPattern, score: number, matchScore: number }>}
   *          Sorted descending by composite score
   * @throws {TypeError} If context is specified but invalid
   */
  searchPatterns(triggerCondition, context = null, topK = 5) {
    if (context !== null) this._assertValidContext(context);

    const candidates = [];
    const searchContexts = context ? [context] : VALID_CONTEXTS;

    for (const ctx of searchContexts) {
      for (const pattern of this.patterns[ctx].values()) {
        const ms    = this.matchScore(triggerCondition, pattern);
        const score = ms * pattern.reliability * pattern.consolidationStrength;
        if (score > 0) {
          candidates.push({ pattern, score, matchScore: ms });
        }
      }
    }

    // Sort descending by composite score, slice to topK
    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, topK);
  }

  /**
   * Compute similarity between a live sensory state and a stored pattern's
   * trigger condition.
   *
   * Metric: fraction of element-wise matches (Hamming similarity).
   * Both vectors are expected to be the same length; mismatched lengths
   * return 0 rather than throwing.
   *
   * @param {number[]}  sensoryState       25-element binary (-1/1) vector
   * @param {LTMPattern} pattern
   * @returns {number} Similarity in [0, 1]; 1 = perfect match
   */
  matchScore(sensoryState, pattern) {
    const trigger = pattern.triggerCondition;
    if (!sensoryState || !trigger || sensoryState.length !== trigger.length) return 0;

    let matches = 0;
    for (let i = 0; i < sensoryState.length; i++) {
      if (sensoryState[i] === trigger[i]) matches++;
    }
    return matches / sensoryState.length;
  }

  // ── Capacity management ───────────────────────────────────────────────────

  /**
   * Remove the single lowest-value pattern across all contexts.
   *
   * Value = reliability × consolidationStrength / (1 + daysSinceUse)
   *
   * Ties are broken by oldest createdAt (evict older first).
   * No-op if LTM is empty.
   *
   * @returns {LTMPattern|null} The evicted pattern or null if nothing to prune
   */
  pruneLowestValue() {
    let lowestValue = Infinity;
    let victim      = null;
    let victimCtx   = null;

    for (const ctx of VALID_CONTEXTS) {
      for (const pattern of this.patterns[ctx].values()) {
        const v = pattern.value;
        if (v < lowestValue || (v === lowestValue && pattern.createdAt < victim?.createdAt)) {
          lowestValue = v;
          victim      = pattern;
          victimCtx   = ctx;
        }
      }
    }

    if (!victim) return null;

    this.patterns[victimCtx].delete(victim.patternId);
    this.patternCount--;

    this._logConsolidation(
      'prune',
      victim.patternId,
      victimCtx,
      `value=${lowestValue.toFixed(6)}, capacity=${this.patternCount}/${this.maxCapacity}`
    );

    return victim;
  }

  // ── Bulk access ───────────────────────────────────────────────────────────

  /**
   * Return all stored patterns, optionally filtered to one context.
   *
   * @param {string|null} [context=null] Restrict to one context or null for all
   * @returns {LTMPattern[]} Flat array; order within a context is insertion order (Map)
   * @throws {TypeError} If context is specified but invalid
   */
  getAllPatterns(context = null) {
    if (context !== null) {
      this._assertValidContext(context);
      return Array.from(this.patterns[context].values());
    }
    return VALID_CONTEXTS.flatMap(ctx => Array.from(this.patterns[ctx].values()));
  }

  // ── Analytics ─────────────────────────────────────────────────────────────

  /**
   * Summary statistics for the entire LTM store.
   *
   * @returns {{
   *   totalPatterns:          number,
   *   byContext:              Record<string, number>,
   *   avgReliability:         number,
   *   avgConsolidationStrength: number,
   *   avgSuccessRate:         number,
   *   avgUtility:             number,
   *   mostUsedPattern:        LTMPattern|null,
   *   strongestPattern:       LTMPattern|null,
   *   consolidationEvents:    number,
   *   capacityUsed:           number,
   * }}
   */
  stats() {
    const all = this.getAllPatterns();

    const byContext = {};
    for (const ctx of VALID_CONTEXTS) {
      byContext[ctx] = this.patterns[ctx].size;
    }

    const avg = (field) =>
      all.length > 0 ? all.reduce((s, p) => s + p[field], 0) / all.length : 0;

    const mostUsedPattern = all.length > 0
      ? all.reduce((best, p) => p.usageCount > best.usageCount ? p : best, all[0])
      : null;

    const strongestPattern = all.length > 0
      ? all.reduce((best, p) => p.consolidationStrength > best.consolidationStrength ? p : best, all[0])
      : null;

    return {
      totalPatterns:           this.patternCount,
      byContext,
      avgReliability:          avg('reliability'),
      avgConsolidationStrength: avg('consolidationStrength'),
      avgSuccessRate:          avg('successRate'),   // uses getter
      avgUtility:              avg('utility'),
      mostUsedPattern,
      strongestPattern,
      consolidationEvents:     this.consolidationHistory.length,
      capacityUsed:            this.patternCount / this.maxCapacity,
    };
  }

  /**
   * Retrieve consolidation history, optionally filtered by event type.
   *
   * @param {'store'|'update'|'prune'|null} [eventType=null]
   * @returns {object[]}
   */
  getHistory(eventType = null) {
    if (eventType === null) return [...this.consolidationHistory];
    return this.consolidationHistory.filter(e => e.event === eventType);
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  /**
   * Serialise the full LTM to a plain JSON-safe object.
   * Used for analysis exports and save/load between trials.
   *
   * @returns {object}
   */
  toJSON() {
    const patternsData = {};
    for (const ctx of VALID_CONTEXTS) {
      patternsData[ctx] = Array.from(this.patterns[ctx].values()).map(p => p.toJSON());
    }
    return {
      maxCapacity:          this.maxCapacity,
      patternCount:         this.patternCount,
      createdAt:            this.createdAt,
      consolidationHistory: this.consolidationHistory,
      patterns:             patternsData,
    };
  }

  /**
   * Reconstruct a LongTermMemory from a toJSON() snapshot.
   *
   * @param {object} data - Output of toJSON()
   * @returns {LongTermMemory}
   */
  static fromJSON(data) {
    const ltm = new LongTermMemory(data.maxCapacity);
    ltm.createdAt            = data.createdAt;
    ltm.consolidationHistory = data.consolidationHistory ?? [];

    for (const ctx of VALID_CONTEXTS) {
      const list = data.patterns?.[ctx] ?? [];
      for (const patternData of list) {
        const p = LTMPattern.fromJSON(patternData);
        ltm.patterns[ctx].set(p.patternId, p);
      }
    }
    // Recount from Maps (source of truth)
    ltm.patternCount = VALID_CONTEXTS.reduce((n, ctx) => n + ltm.patterns[ctx].size, 0);
    return ltm;
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  /**
   * Append an entry to consolidationHistory.
   * Keeps the log bounded at 10 000 entries to avoid unbounded growth.
   *
   * @param {string} event     'store' | 'update' | 'prune'
   * @param {string} patternId
   * @param {string} context
   * @param {string} detail    Free-text detail string for diagnostics
   */
  _logConsolidation(event, patternId, context, detail = '') {
    this.consolidationHistory.push({
      timestamp: Date.now(),
      event,
      patternId,
      context,
      detail,
    });
    // Trim oldest entries if log grows too large
    if (this.consolidationHistory.length > 10_000) {
      this.consolidationHistory.splice(0, 1000);
    }
  }
}

// ── Exports ───────────────────────────────────────────────────────────────────

export { LTMPattern, LongTermMemory, VALID_CONTEXTS };
