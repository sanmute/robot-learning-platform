/**
 * WeightOptimizationTest.js
 *
 * Standalone entry-point for Experiment 8 (Weight Optimization).
 * Wraps ExperimentRunnerV2 and exposes the interface described in the spec
 * (runAllConfigurations, generateFinalReport) while adapting it to the
 * browser ESM environment — no Node.js `fs`, no `module.exports`, no
 * `Agent` class with spec-style methods.
 *
 * Usage (from any component or test harness):
 *
 *   import WeightOptimizationTest from './experiment/WeightOptimizationTest.js';
 *
 *   const test = new WeightOptimizationTest({ advantage: 11.27 });
 *   test.onProgress = (info) => console.log(info);          // optional
 *   const report   = await test.runAllConfigurations(3);    // 3 trials/obj
 *   // report is the full JSON described in the spec
 *   // Results are also auto-downloaded by the runner (exp8_results_<ts>.json)
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

import { ExperimentRunner as ExperimentRunnerV2 } from '../components/ExperimentRunner_v2.js';
import { EXP8_WEIGHT_OPTIMIZATION_CONFIG } from '../components/EXPERIMENT_CONFIG.js';

export default class WeightOptimizationTest {

  /**
   * @param {object} [baselineMetrics]
   * @param {number[]} [baselineMetrics.weights]   Integer percentages [F, S, A, B, E]
   * @param {number}   [baselineMetrics.advantage] D-vs-A gen. index reference (default 11.27)
   * @param {number}   [baselineMetrics.reward]    Absolute reward reference  (default 31.4)
   */
  constructor(baselineMetrics = null) {
    this.baselineMetrics = {
      weights:   [20, 20, 20, 20, 20],
      advantage: EXP8_WEIGHT_OPTIMIZATION_CONFIG.BASELINE_ADVANTAGE,
      reward:    31.4,
      ...(baselineMetrics ?? {}),
    };

    /** Set to a function before calling runAllConfigurations() to receive live updates. */
    this.onProgress = null;

    this._runner = null;
    this._report = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Run all 10 weight configurations and return the final comparison report.
   * Internally calls ExperimentRunnerV2.runExperiment(8); the runner handles
   * trial dispatch, LTM management, and auto-downloading the JSON.
   *
   * @param {number} [_trialsPerConfig]  Accepted for API compatibility; actual
   *   trial count is set in EXP8_WEIGHT_OPTIMIZATION_CONFIG.TESTING_TRIALS_PER_OBJECTIVE.
   * @returns {Promise<object>} Full report in the spec output format
   */
  async runAllConfigurations(_trialsPerConfig = 3) {
    this._runner = new ExperimentRunnerV2();
    if (typeof this.onProgress === 'function') {
      this._runner.onProgressUpdate = this.onProgress;
    }

    await this._runner.runExperiment(8);
    this._report = this.generateFinalReport();
    return this._report;
  }

  /** Abort a running experiment. */
  stop() {
    this._runner?.stop();
  }

  /**
   * Build the spec-compliant JSON report from the runner's computed summary.
   * Can be called independently after runAllConfigurations() completes.
   *
   * @returns {object}
   */
  generateFinalReport() {
    if (!this._runner) throw new Error('[WeightOptimizationTest] Call runAllConfigurations() first');

    const summary = this._runner.computeExperimentSummary(8);
    const cfgs    = EXP8_WEIGHT_OPTIMIZATION_CONFIG.WEIGHT_CONFIGURATIONS;

    // Build allResults array sorted by avg gen index descending
    const allResults = (summary._ranking ?? []).map(cfgName => {
      const s = summary[cfgName];
      return {
        configuration:    cfgName,
        weights:          s._weightsArray ?? [],
        hypothesis:       s._hypothesis   ?? '',
        meanAdvantage:    s._avgGeneralizationIndex,
        advantageStd:     this._std(
          EXP8_WEIGHT_OPTIMIZATION_CONFIG.WEIGHT_CONFIGURATIONS
            .find(c => c.name === cfgName)
            ? Object.values(summary[cfgName] ?? {})
                .filter(v => v && typeof v._generalizationIndex === 'number')
                .map(v => v._generalizationIndex)
            : []
        ),
        tTestVsBaseline:  summary._tTests?.[cfgName] ?? null,
        metricsBreakdown: this._buildBreakdown(summary[cfgName]),
      };
    });

    const best     = cfgs.find(c => c.name === summary._winner) ?? cfgs[0];
    const baseline = cfgs.find(c => c.name === 'baseline_equal');

    const report = {
      timestamp:        new Date().toISOString(),
      experimentName:   'Experiment 8: Multi-Objective Weight Optimization',
      baselineWeights:  this.baselineMetrics.weights,
      baselineAdvantage: this.baselineMetrics.advantage,
      baselineGI:        summary._baselineGI ?? null,

      allResults,

      bestPerformer: {
        configuration:     summary._winner,
        weights:           best?.weightsArray ?? [],
        advantage:         summary[summary._winner]?._avgGeneralizationIndex ?? null,
        improvement:       summary._improvement,
        improvementVsRef:  typeof summary._improvement === 'number'
          ? +(summary._improvement + (summary._baselineGI ?? 0) - this.baselineMetrics.advantage).toFixed(2)
          : null,
        isSignificant:     summary._winnerSignificant ?? false,
      },

      ranking:        summary._ranking ?? [],
      tTests:         summary._tTests  ?? {},
      conclusion:     summary._conclusion     ?? '',
      recommendation: this._buildRecommendation(summary, cfgs),
    };

    return report;
  }

  // ── Spec-compatible helpers ───────────────────────────────────────────────

  /**
   * Validate that an integer weight array sums to 100.
   * @param {number[]} weights
   * @returns {true}
   */
  validateWeights(weights) {
    const sum = weights.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 100) > 0.01) throw new Error(`Weights must sum to 100. Got ${sum}`);
    return true;
  }

  /**
   * Normalise an integer weight array to fractional (sum = 1).
   * @param {number[]} weights
   * @returns {number[]}
   */
  normalizeWeights(weights) {
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => w / sum);
  }

  /** Arithmetic mean. */
  mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  /** Population standard deviation. */
  std(arr) { return this._std(arr); }

  /**
   * Pooled two-sample t-test (spec API).
   * @param {number[]} config1Results
   * @param {number[]} config2Results
   */
  tTest(config1Results, config2Results) {
    return this._runner?._tTest(config1Results, config2Results)
      ?? { tStatistic: null, isSignificant: false, pValueApprox: 'n/a' };
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  _std(arr) {
    if (!arr.length) return 0;
    const m = this.mean(arr);
    return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length);
  }

  _buildBreakdown(cfgSummary) {
    const breakdown = {};
    for (const key of ['baseline', 'speed', 'accuracy', 'balance', 'efficiency']) {
      const cond = cfgSummary?.[key];
      if (cond?.D) breakdown[key] = cond.D.meanFoodEaten ?? cond.D.mean;
    }
    return breakdown;
  }

  _buildRecommendation(summary, cfgs) {
    const improvement = summary._improvement ?? 0;
    const isSignificant = summary._winnerSignificant ?? false;
    const winnerCfg = cfgs.find(c => c.name === summary._winner);

    if (improvement > 0.5 && isSignificant) {
      return {
        recommendation: 'Use optimized weights',
        weights:        winnerCfg?.weightsArray ?? [],
        rationale:      `${summary._winner} shows statistically significant improvement (p < 0.05)`,
        commercialImplication: 'Can advertise higher performance with optimized weights',
      };
    }
    return {
      recommendation: 'Keep equal weighting',
      weights:        [20, 20, 20, 20, 20],
      rationale:      'Equal weighting is near-optimal and simpler to explain',
      commercialImplication: "Easier product messaging: 'balanced approach proven optimal'",
    };
  }
}
