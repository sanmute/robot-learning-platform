/**
 * Training engine stub.
 *
 * Simulates 40 training iterations (~4 seconds total).
 * Replace this implementation with the real robot-learning simulation.
 *
 * @param config   Robot configuration (robot type + objectives/weights)
 * @param onProgress  Called each iteration with the current progress % (0–100).
 *                    Return a rejected promise to abort training.
 */

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

const ITERATIONS = 40;
const DELAY_MS = 100; // 40 × 100ms ≈ 4 seconds

export async function trainModel(
  config: TrainingConfig,
  onProgress: (progress: number) => Promise<void>,
): Promise<TrainingOutput> {
  const learningCurve: number[] = [];

  for (let i = 0; i < ITERATIONS; i++) {
    await new Promise<void>((resolve) => setTimeout(resolve, DELAY_MS));

    // Simulate a learning curve: fast early gains, diminishing returns
    const base = 11.27 * (1 - Math.exp(-0.15 * (i + 1)));
    const noise = (Math.random() - 0.5) * 1.2;
    learningCurve.push(Math.round((base + noise) * 100) / 100);

    const progress = Math.round(((i + 1) / ITERATIONS) * 100);
    await onProgress(progress);
  }

  const advantage = 11.27 + (Math.random() * 4 - 2);

  return {
    advantage: Math.round(advantage * 100) / 100,
    learningCurve,
    modelData: {
      version: '1.0.0',
      stub: true,
      config,
      trainedAt: new Date().toISOString(),
    },
  };
}
