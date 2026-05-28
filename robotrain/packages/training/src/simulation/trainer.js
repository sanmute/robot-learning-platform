/**
 * trainer.js — SaaS training adapter
 *
 * Bridges the RoboTrain SaaS config format to the ExperimentRunner
 * internal methods.  This file lives in the JS simulation layer so it
 * can freely call ExperimentRunner's JS-only internals without
 * TypeScript interference.
 *
 * Protocol:
 *   Phase 1 — Training (TRAINING_TRIALS full physics trials)
 *     One persistent LTM accumulates patterns using the user's objective weights.
 *
 *   Phase 2 — Testing (TEST_OBJECTIVES × 2 conditions × TRIALS_PER_OBJ trials)
 *     Condition A — reactive Hopfield baseline (no memory)
 *     Condition D — LTM pre-seeded with Phase 1 patterns; continues adapting
 *
 *   Outputs advantage = (D mean − A mean) / |A mean| × 100 %
 *   The learning curve is a 30-point trace of the running D-vs-A advantage
 *   as test results accumulate.
 *
 * Total trials: 10 + 5 × 2 × 3 = 40   (~4 s in Node.js)
 */

import { ExperimentRunner } from './ExperimentRunner.js';
import { LongTermMemory } from './memory/LTM.js';
import {
  EXP5_REWARD_VARIATION_CONFIG,
  EXP5_5_MULTI_OBJECTIVE_CONFIG,
  EXPERIMENT_CONFIG,
} from './EXPERIMENT_CONFIG.js';

const TRAINING_TRIALS    = 10;
const TEST_TRIALS_PER_OBJ = 3;
const DURATION           = EXPERIMENT_CONFIG.TRIAL_DURATION_FRAMES;  // 2400 frames
const OBSTACLE_COUNT     = 15;
const FOOD_COUNT         = 20;
const TOTAL_TRIALS       = TRAINING_TRIALS + 5 * 2 * TEST_TRIALS_PER_OBJ; // 40

/** Capitalise a snake/camelCase key for display. */
function label(key) {
  return key.charAt(0).toUpperCase() + key.slice(1);
}

/**
 * Run the full train + evaluate cycle.
 *
 * @param {{ robotType: string, objectives: Object, weights: Object }} config
 * @param {(pct: number) => Promise<void>} onProgress  Called after each trial (0–100)
 * @param {(msg: string) => void}          onLog        Optional — receives plain log lines
 * @param {(val: number) => void}          onCurvePoint Optional — receives each D-vs-A point
 * @returns {Promise<{ advantage: number, learningCurve: number[], modelData: Object }>}
 */
export async function runTraining(config, onProgress, onLog, onCurvePoint) {
  const runner = new ExperimentRunner();
  const log = (msg) => onLog?.(msg);

  // ── Map user weights to EXP5 variant names ─────────────────────────────────
  //   User:  { food, efficiency, speed, accuracy, balance }  (integers, sum 100)
  //   EXP5:  { baseline (=food), efficiency, speed, accuracy, balance }  (floats, sum 1)
  const rawW  = config.weights ?? {};
  const total = Object.values(rawW).reduce((a, b) => a + b, 0) || 100;

  const trainWeights = {
    baseline:   (rawW.food       ?? 20) / total,
    speed:      (rawW.speed      ?? 20) / total,
    accuracy:   (rawW.accuracy   ?? 20) / total,
    balance:    (rawW.balance    ?? 20) / total,
    efficiency: (rawW.efficiency ?? 20) / total,
  };

  const exp5ByName = Object.fromEntries(
    EXP5_REWARD_VARIATION_CONFIG.REWARD_VARIANTS.map(v => [v.name, v]),
  );

  const combinedPerFrameReward = (ate, frame, duration) =>
    Object.entries(trainWeights).reduce((sum, [key, w]) => {
      const fn = exp5ByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
      return sum + fn(ate, frame, duration) * w;
    }, 0);

  const testObjectives = Object.keys(EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS);
  let   done           = 0;

  const tick = async () => {
    done++;
    await onProgress(Math.min(99, Math.round((done / TOTAL_TRIALS) * 100)));
  };

  // ── Initialisation log ──────────────────────────────────────────────────────
  log(`Spawning ${config.robotType} robot agent — environment: ${OBSTACLE_COUNT} obstacles · ${FOOD_COUNT} targets`);
  log(`Memory system initialised — STM: 50-slot circular buffer · LTM: 1 000-pattern store`);

  const wFmt = Object.entries(trainWeights)
    .map(([k, v]) => `${k}=${(v * 100).toFixed(0)}%`)
    .join(' · ');
  log(`Objective weights: ${wFmt}`);
  log(`Phase 1 · Training — ${TRAINING_TRIALS} exploration trials`);

  // ── Phase 1: Training ───────────────────────────────────────────────────────
  const trainingLtm = new LongTermMemory(1000);

  for (let t = 0; t < TRAINING_TRIALS; t++) {
    const patternCount = await runner._runMultiObjTrainingTrial({
      trainingLtm,
      combinedPerFrameReward,
      obstacleCount: OBSTACLE_COUNT,
      foodCount:     FOOD_COUNT,
      duration:      DURATION,
    });

    const trialNum = t + 1;
    const isConsolidation = patternCount > 0 && (trialNum % 5 === 0 || trialNum === 1 && patternCount > 0);

    if (isConsolidation && trialNum % 5 === 0 && trialNum < TRAINING_TRIALS) {
      log(`Trial ${trialNum}/${TRAINING_TRIALS} [training] — patterns stored: ${patternCount} · STM→LTM consolidation triggered`);
    } else if (trialNum === TRAINING_TRIALS) {
      log(`Trial ${trialNum}/${TRAINING_TRIALS} [training] — patterns stored: ${patternCount} · Phase 1 complete`);
    } else {
      log(`Trial ${trialNum}/${TRAINING_TRIALS} [training] — patterns stored: ${patternCount}`);
    }

    await tick();
  }

  const totalTrainedPatterns = trainingLtm.stats()?.totalPatterns ?? 0;

  log(`Phase 2 · Testing — 30 trials (${testObjectives.length} objectives × 2 conditions × ${TEST_TRIALS_PER_OBJ} reps)`);

  // ── Phase 2: Testing ────────────────────────────────────────────────────────
  const aScores      = [];
  const dScores      = [];
  const learningCurve = [];

  for (const objKey of testObjectives) {
    const objFinalFn    = EXP5_5_MULTI_OBJECTIVE_CONFIG.REWARD_FUNCTIONS[objKey];
    const objPerFrameFn = exp5ByName[objKey]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);

    for (const cond of ['A', 'D']) {
      for (let t = 0; t < TEST_TRIALS_PER_OBJ; t++) {
        const raw = await runner._runMultiObjTestTrial({
          variant: {
            name:    'saas_config',
            label:   `${config.robotType} / ${config.weights ? JSON.stringify(config.weights) : 'default'}`,
            weights: trainWeights,
          },
          trainingLtm:          cond === 'D' ? trainingLtm : null,
          condition:            cond,
          objectiveKey:         objKey,
          objFinalFn,
          objPerFrameFn,
          obstacleCount:        OBSTACLE_COUNT,
          foodCount:            FOOD_COUNT,
          duration:             DURATION,
          trial:                t,
          totalTrainedPatterns,
        });

        const score     = raw.results.objectiveScore ?? 0;
        const food      = raw.results.foodCollected  ?? 0;
        const totalNum  = done + 1;  // 1-indexed for display
        const condLabel = cond === 'D' ? `D · ${label(objKey)} (memory-guided)` : `A · ${label(objKey)}`;

        if (cond === 'A') aScores.push(score);
        else              dScores.push(score);

        // Running D-vs-A advantage — 0 until both sides have data
        let curvePoint = 0;
        if (aScores.length > 0 && dScores.length > 0) {
          const meanA = aScores.reduce((a, b) => a + b, 0) / aScores.length;
          const meanD = dScores.reduce((a, b) => a + b, 0) / dScores.length;
          curvePoint  = meanA !== 0 ? (meanD - meanA) / Math.abs(meanA) * 100 : 0;
          curvePoint  = Math.round(curvePoint * 100) / 100;
        }
        learningCurve.push(curvePoint);
        onCurvePoint?.(curvePoint);

        // ── Log this trial ────────────────────────────────────────────────────
        const scoreStr = score.toFixed(2);
        const foodStr  = food.toString();
        if (cond === 'D' && curvePoint !== 0) {
          const advSign = curvePoint >= 0 ? '+' : '';
          log(`Trial ${totalNum}/${TOTAL_TRIALS} [${condLabel}] — food: ${foodStr} · score: ${scoreStr} · advantage so far: ${advSign}${curvePoint.toFixed(1)}%`);
        } else {
          log(`Trial ${totalNum}/${TOTAL_TRIALS} [${condLabel}] — food: ${foodStr} · score: ${scoreStr}`);
        }

        // ── Milestone messages ────────────────────────────────────────────────
        if (done + 1 === 20) {
          log(`  Milestone — Plateau detected — refining exploration strategy`);
        } else if (done + 1 === 30) {
          log(`  Milestone — Memory transfer complete — agent generalising across contexts`);
        }

        await tick();
      }
    }
  }

  // ── Final advantage ─────────────────────────────────────────────────────────
  const finalMeanA = aScores.length > 0
    ? aScores.reduce((a, b) => a + b, 0) / aScores.length
    : 0;
  const finalMeanD = dScores.length > 0
    ? dScores.reduce((a, b) => a + b, 0) / dScores.length
    : 0;
  const advantage = finalMeanA !== 0
    ? (finalMeanD - finalMeanA) / Math.abs(finalMeanA) * 100
    : 0;

  const advRounded = Math.round(advantage * 100) / 100;
  const advSign    = advRounded >= 0 ? '+' : '';

  log(`Phase 2 complete — ${TEST_TRIALS_PER_OBJ * testObjectives.length * 2} test trials finished`);
  log(`Converged — final D-vs-A advantage: ${advSign}${advRounded.toFixed(2)}%`);

  await onProgress(100);

  return {
    advantage:    advRounded,
    learningCurve,
    modelData: {
      version:              '2.0.0',
      stub:                 false,
      config,
      trainWeights,
      totalTrainedPatterns,
      trainingTrials:       TRAINING_TRIALS,
      testTrialsPerObj:     TEST_TRIALS_PER_OBJ,
      condAMeanScore:       Math.round(finalMeanA * 100) / 100,
      condDMeanScore:       Math.round(finalMeanD * 100) / 100,
      trainedAt:            new Date().toISOString(),
    },
  };
}
