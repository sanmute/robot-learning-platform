/**
 * DB-backed job queue runner.
 *
 * Polls the database every 2 seconds for pending jobs.
 * Processes one job at a time (concurrency = 1 at MVP scale).
 */

import { prisma } from '../db';
import { trainModel } from '@robotrain/training';

const POLL_INTERVAL_MS = 2_000;

let busy = false;

export function startJobRunner(): void {
  console.log('🔄 Job runner started — polling every 2 s');
  setInterval(processNextJob, POLL_INTERVAL_MS);
}

async function processNextJob(): Promise<void> {
  if (busy) return;

  const job = await prisma.trainingJob.findFirst({
    where: { status: 'pending' },
    include: { config: true },
    orderBy: { id: 'asc' }, // FIFO
  });

  if (!job) return;

  busy = true;

  console.log(`⚙️  Processing job ${job.id} (config: ${job.config.name})`);

  try {
    await prisma.trainingJob.update({
      where: { id: job.id },
      data: { status: 'running', startedAt: new Date(), progress: 0 },
    });

    const result = await trainModel(
      {
        robotType: job.config.robotType,
        objectives: job.config.objectives as unknown as Record<string, number>,
        weights: job.config.weights as Record<string, number>,
      },
      async (progress: number) => {
        await prisma.trainingJob.update({
          where: { id: job.id },
          data: { progress },
        });
      },
    );

    await prisma.trainingResult.create({
      data: {
        jobId: job.id,
        advantage: result.advantage,
        learningCurve: result.learningCurve,
        modelData: result.modelData,
      },
    });

    await prisma.trainingJob.update({
      where: { id: job.id },
      data: { status: 'done', progress: 100, finishedAt: new Date() },
    });

    console.log(`✅ Job ${job.id} complete — advantage: +${result.advantage}%`);
  } catch (err) {
    console.error(`❌ Job ${job.id} failed:`, err);
    await prisma.trainingJob.update({
      where: { id: job.id },
      data: { status: 'failed', finishedAt: new Date() },
    }).catch(() => null); // best-effort
  } finally {
    busy = false;
  }
}
