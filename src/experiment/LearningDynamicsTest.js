/**
 * LearningDynamicsTest.js
 *
 * Standalone entry-point for Experiment 9 (Learning Dynamics & Curves).
 * Wraps ExperimentRunnerV2 and exposes the interface from the spec
 * (runLearningDynamicsTest, generateFinalReport) adapted for the browser
 * ESM environment — no Node.js `fs`, no `module.exports`, no abstract `Agent`.
 *
 * Usage:
 *
 *   import LearningDynamicsTest from './experiment/LearningDynamicsTest.js';
 *
 *   const test = new LearningDynamicsTest({ advantage: 11.27 });
 *   test.onProgress = (info) => setProgress(info);     // optional live UI updates
 *   const report   = await test.runLearningDynamicsTest(3);
 *   // Results also auto-downloaded as exp9_results_<ts>.json
 *
 * Checkpoint scaling note:
 *   The spec's checkpoints [0,10,25,50,100,200,300,500] are calibrated for a
 *   step-level simulation. This codebase runs full 2400-frame physics trials;
 *   each trial ≈ 2400 steps. The adapted checkpoints are [0,2,5,10,20,40]
 *   full trials, which covers the same learning-curve shape at the correct
 *   timescale (~30–35 min total vs the spec's 30 min estimate).
 *
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

import { ExperimentRunner as ExperimentRunnerV2 } from '../components/ExperimentRunner_v2.js';
import { EXP9_LEARNING_DYNAMICS_CONFIG } from '../components/EXPERIMENT_CONFIG.js';

export default class LearningDynamicsTest {

  /**
   * @param {object} [baselineMetrics]
   * @param {number} [baselineMetrics.fullTrainingTrials]  Reference training depth (default 300 spec / 40 system)
   * @param {number} [baselineMetrics.advantage]           Reference D-vs-A gen index (default 11.27)
   * @param {number} [baselineMetrics.reward]              Reference absolute reward (default 31.4)
   */
  constructor(baselineMetrics = null) {
    this.baselineMetrics = {
      fullTrainingTrials: 40,     // system equivalent of spec's 300
      advantage:          EXP9_LEARNING_DYNAMICS_CONFIG.BASELINE_ADVANTAGE,
      reward:             31.4,
      ...(baselineMetrics ?? {}),
    };

    /** Spec-compatible checkpoint list (for external consumers / logging). */
    this.checkpoints = EXP9_LEARNING_DYNAMICS_CONFIG.CHECKPOINTS;

    /** Set to a function before calling runLearningDynamicsTest() for live updates. */
    this.onProgress = null;

    this._runner = null;
    this._report = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Run the full learning-dynamics experiment and return the report.
   *
   * @param {number} [_trialsPerCheckpoint]  Accepted for API compatibility;
   *   actual rep count is set in EXP9_LEARNING_DYNAMICS_CONFIG.REPS_PER_CHECKPOINT.
   * @returns {Promise<object>} Spec-compatible JSON report
   */
  async runLearningDynamicsTest(_trialsPerCheckpoint = 3) {
    this._runner = new ExperimentRunnerV2();
    if (typeof this.onProgress === 'function') {
      this._runner.onProgressUpdate = this.onProgress;
    }

    await this._runner.runExperiment(9);
    this._report = this.generateFinalReport();
    return this._report;
  }

  /** Abort a running experiment. */
  stop() {
    this._runner?.stop();
  }

  /**
   * Build the spec-compatible JSON report from the runner's computed summary.
   * Can be called independently after runLearningDynamicsTest() completes.
   *
   * @returns {object}
   */
  generateFinalReport() {
    if (!this._runner) throw new Error('[LearningDynamicsTest] Call runLearningDynamicsTest() first');

    const summary = this._runner.computeExperimentSummary(9);

    const report = {
      timestamp:       new Date().toISOString(),
      experimentName:  'Experiment 9: Learning Dynamics & Deployment Curves',
      baselineMetrics: this.baselineMetrics,

      learningCurves: {
        advantage:       summary._advantageCurve ?? [],
        detectedPattern: summary._curveType      ?? 'unknown',
        analysis: {
          maxAdvantage:        summary._maxAdvantage,
          convergenceThreshold: EXP9_LEARNING_DYNAMICS_CONFIG.CONVERGENCE_THRESHOLD * (summary._maxAdvantage ?? 0),
          convergenceTrials:    summary._convergencePoint ?? 'not reached',
          interpretation:       summary._interpretation   ?? '',
        },
      },

      overfittingAnalysis: summary._overfitting ? {
        detected:          summary._overfitting.detected,
        degradationAmount: summary._overfitting.degradation,
        fromCheckpoint:    summary._overfitting.fromCheckpoint,
        toCheckpoint:      summary._overfitting.toCheckpoint,
        recommendation:    summary._overfitting.detected
          ? `Stop training at ~${summary._overfitting.fromCheckpoint} trials to avoid degradation`
          : `No overfitting — can safely train to ${summary._overfitting.toCheckpoint}+ trials`,
      } : null,

      recommendations: {
        minimalTraining:   this._buildMinimalGuidance(summary),
        optimalTraining:   this._buildOptimalGuidance(summary),
        safeMaximum:       this._buildSafeMaxGuidance(summary),
        deploymentGuidance: summary._deploymentGuidance ?? {},
      },

      conclusions: {
        curveShape:          summary._curveType,
        convergenceBehavior: summary._interpretation,
        overallMessage:      this._buildOverallMessage(summary),
      },

      rawData: {
        checkpoints:    summary._checkpoints ?? [],
        advantageCurve: summary._advantageCurve ?? [],
      },
    };

    return report;
  }

  // ── Spec-compatible helpers ───────────────────────────────────────────────

  /**
   * Estimate convergence phase label for a given training trial count.
   * Mirrors the spec's `estimateConvergence()` but scaled to our system.
   * @param {number} trainingTrials
   * @returns {'baseline'|'early'|'learning'|'converging'|'converged'}
   */
  estimateConvergence(trainingTrials) {
    if (trainingTrials === 0) return 'baseline';
    if (trainingTrials < 5)  return 'early';
    if (trainingTrials < 10) return 'learning';
    if (trainingTrials < 25) return 'converging';
    return 'converged';
  }

  /** Detect overfitting from the last two checkpoints. Spec-compatible API. */
  detectOverfitting() {
    if (!this._runner) return null;
    const summary = this._runner.computeExperimentSummary(9);
    return summary._overfitting ?? null;
  }

  /** Arithmetic mean. */
  mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  /** Population standard deviation. */
  std(arr) {
    if (!arr.length) return 0;
    const m = this.mean(arr);
    return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length);
  }

  // ── Internal report helpers ───────────────────────────────────────────────

  _buildMinimalGuidance(summary) {
    const pt = summary._minViablePoint;
    const s  = pt !== null ? summary[pt] : null;
    return {
      trials:      pt ?? 'beyond range',
      advantage:   s?._meanAdvantage !== null ? `+${s._meanAdvantage}%` : 'n/a',
      explanation: pt !== null
        ? `Reaches 90% of max advantage (${summary._maxAdvantage?.toFixed(2)}%) at ${pt} trials`
        : 'Performance still improving at highest checkpoint',
      useCase: 'Time-critical deployments',
    };
  }

  _buildOptimalGuidance(summary) {
    const pt = summary._convergencePoint;
    const s  = pt !== null ? summary[pt] : null;
    const lastCp = (summary._checkpoints ?? [])[summary._checkpoints?.length - 1];
    return {
      trials:      pt ?? lastCp ?? 'beyond range',
      advantage:   s?._meanAdvantage !== null ? `+${s._meanAdvantage}%` : 'n/a',
      explanation: pt !== null
        ? `Reaches plateau (95% of max) at ${pt} trials`
        : 'Continues improving through final checkpoint',
      useCase: 'Standard production deployment',
    };
  }

  _buildSafeMaxGuidance(summary) {
    const ov     = summary._overfitting;
    const lastCp = (summary._checkpoints ?? [])[summary._checkpoints?.length - 1];
    if (ov?.detected) {
      return {
        trials:      ov.fromCheckpoint,
        explanation: `Stop before degradation (${ov.fromCheckpoint}→${ov.toCheckpoint} trials)`,
        warning:     `Overfitting detected: +${ov.degradation}% performance loss at final checkpoint`,
        safety:      'Moderate — stop at recommended point',
      };
    }
    return {
      trials:      lastCp,
      explanation: 'No overfitting detected — can safely train full duration',
      safety:      'High — no degradation observed',
    };
  }

  _buildOverallMessage(summary) {
    const curve   = summary._curveType    ?? 'unknown';
    const minVia  = summary._minViablePoint;
    const conv    = summary._convergencePoint;
    return `Learning follows ${curve} pattern. ` +
      (minVia !== null ? `Achieves 90% performance in ~${minVia} trials. ` : '') +
      (conv   !== null ? `Full convergence at ~${conv} trials. ` : '') +
      (summary._overfitting?.detected
        ? `⚠ Overfitting detected at ${summary._overfitting.fromCheckpoint}→${summary._overfitting.toCheckpoint} trials.`
        : 'No overfitting observed.');
  }
}
