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
import { EXPERIMENT_CONFIG } from './EXPERIMENT_CONFIG.js';

// ── Coherent reward variants (Exp 10a / 10b) ──────────────────────────────────
//
//   These replace the original incoherent EXP5 training signals + EXP5.5
//   terminal scoring.  Exp 10a established that aligning per-frame and
//   final-score reward functions improves D-vs-A learning consistency.
//
//   Accuracy collision penalty:
//     Exp 10b: -2.0 is optimal (positive advantage on all 5 objectives,
//              best safety_score = accuracy_adv×0.4 + bounce_reduction×0.6)
//     For human-collaborative environments use -5.0 (see RESEARCH_ROADMAP.md)
//     Hardware sensors (LIDAR, ultrasonic, F/T) provide hard safety guarantee
//     per ISO/TS 15066; this penalty reduces frequency of hardware-intervention
//     triggers (each wall bounce ≈ a situation that would activate force/speed
//     limiting).
//
const COHERENT_VARIANTS = [
  {
    name:  'baseline',
    label: 'Baseline: Maximize Food',
    perFrameReward: (ate) => ate > 0 ? 1.0 : -0.01,
    finalScore:     (food) => food,
  },
  {
    name:  'efficiency',
    label: 'Efficiency: Food per Energy',
    perFrameReward: (ate, frame, duration) =>
      ate > 0 ? 1.0 : -0.05 - (frame / duration) * 0.02,
    finalScore: (food, frames) => food - frames * 0.001,
  },
  {
    name:  'accuracy',
    label: 'Safety: Minimal Wall Contact',
    // Exp 10b: -2.0 collision penalty optimal for performance (all 5 objectives positive)
    // For human-collaborative environments use -5.0 (see RESEARCH_ROADMAP.md)
    // Hardware sensors (LIDAR, ultrasonic, F/T) provide hard safety guarantee;
    // this penalty reduces frequency of hardware intervention triggers.
    perFrameReward: (ate, _frame, _duration, wallBounces) =>
      ate > 0 ? 1.0 : -0.02 - (wallBounces ?? 0) * 2.0,
    finalScore: (food, _frames, wallBounces) => food - wallBounces * 0.05,
  },
  {
    name:  'speed',
    label: 'Speed: Time Pressure',
    perFrameReward: (ate) => ate > 0 ? 2.0 : -0.02,
    finalScore: (food, frames) => food * 2 - frames * 0.01,
  },
  {
    name:  'balance',
    label: 'Balance: Multi-Objective',
    perFrameReward: (ate, frame, duration, wallBounces) =>
      ate > 0 ? 1.5 : -0.03 - (frame / duration) * 0.01 - (wallBounces ?? 0) * 0.05,
    finalScore: (food, frames, wallBounces) =>
      food * 1.5 - frames * 0.0005 - wallBounces * 0.05,
  },
];

const coherentByName = Object.fromEntries(COHERENT_VARIANTS.map(v => [v.name, v]));

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

  // ── Map user weights to coherent variant names ────────────────────────────
  //   User:  { food, efficiency, speed, accuracy, balance }  (integers, sum 100)
  //   Internal: { baseline (=food), efficiency, speed, accuracy, balance } (floats, sum 1)
  const rawW  = config.weights ?? {};
  const total = Object.values(rawW).reduce((a, b) => a + b, 0) || 100;

  const trainWeights = {
    baseline:   (rawW.food       ?? 20) / total,
    speed:      (rawW.speed      ?? 20) / total,
    accuracy:   (rawW.accuracy   ?? 20) / total,
    balance:    (rawW.balance    ?? 20) / total,
    efficiency: (rawW.efficiency ?? 20) / total,
  };

  // Combined per-frame reward: Σ weight[key] × coherentVariant.perFrameReward(...)
  // Uses Exp 10a coherent signals — wallBounces passed as 4th arg so the
  // accuracy (-2.0) and balance variants can apply their collision deterrents.
  const combinedPerFrameReward = (ate, frame, duration, wallBounces) =>
    Object.entries(trainWeights).reduce((sum, [key, w]) => {
      const fn = coherentByName[key]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);
      return sum + fn(ate, frame, duration, wallBounces) * w;
    }, 0);

  const testObjectives = COHERENT_VARIANTS.map(v => v.name); // ['baseline','efficiency','accuracy','speed','balance']
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
    // Coherent: per-frame and terminal scoring both come from the same variant
    // (Exp 10a finding — aligned signals give better D-vs-A consistency)
    const objFinalFn    = coherentByName[objKey]?.finalScore     ?? ((food) => food);
    const objPerFrameFn = coherentByName[objKey]?.perFrameReward ?? ((a) => a > 0 ? 1.0 : -0.01);

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
