/**
 * Training engine — real robot-learning simulation.
 *
 * Delegates to the JS simulation layer (trainer.js) which runs the full
 * ExperimentRunner pipeline:
 *
 *   Phase 1 — 10 training trials (LTM accumulates patterns)
 *   Phase 2 — 30 test trials (5 objectives × 2 conditions × 3 trials)
 *
 * Total: 40 trials (~4 s in Node.js at 2400 frames/trial).
 *
 * @param config        Robot configuration (robot type + objectives/weights)
 * @param onProgress    Called after each trial with progress % (0–100)
 * @param onLog         Optional — called with each plain log message (no timestamp)
 * @param onCurvePoint  Optional — called with each running D-vs-A advantage value
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { runTraining } = require('./simulation/trainer.js');

export interface TrainingConfig {
  robotType: string;
  objectives: Record<string, number>;
  weights: Record<string, number>;
}

export interface TrainingOutput {
  advantage: number;
  learningCurve: number[];
  modelData: Record<string, unknown>;
}

export async function trainModel(
  config: TrainingConfig,
  onProgress: (progress: number) => Promise<void>,
  onLog?: (message: string) => void,
  onCurvePoint?: (value: number) => void,
): Promise<TrainingOutput> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (runTraining as any)(config, onProgress, onLog, onCurvePoint) as Promise<TrainingOutput>;
}
