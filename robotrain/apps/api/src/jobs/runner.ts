/**
 * DB-backed job queue runner.
 *
 * Polls the database every 2 seconds for pending jobs.
 * Processes one job at a time (concurrency = 1 at MVP scale).
 *
 * Log lines and the live learning curve are persisted to the TrainingJob row
 * so that any Cloud Run instance can serve the /logs polling endpoint.
 * Writes are fire-and-forget to avoid blocking the simulation loop.
 */

import { prisma } from '../db';
import { Prisma } from '@prisma/client';
import { trainModel } from '@robotrain/training';

const POLL_INTERVAL_MS = 2_000;

let busy = false;

// Kept purely for relative-timestamp formatting — loss on restart is harmless.
const jobStartTimes = new Map<string, number>();

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
  jobStartTimes.set(job.id, Date.now());

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
      (message: string) => {
        const startedAt = jobStartTimes.get(job.id) ?? Date.now();
        const elapsed   = ((Date.now() - startedAt) / 1000).toFixed(1);
        const line      = `[${elapsed}s] ${message}`;
        // Fire-and-forget — training loop must not be blocked by log I/O
        prisma.trainingJob.update({
          where: { id: job.id },
          data:  { logs: { push: line } },
        }).catch((e) => console.error('log write failed:', e));
      },
      (value: number) => {
        prisma.trainingJob.update({
          where: { id: job.id },
          data:  { liveCurve: { push: value } },
        }).catch((e) => console.error('curve write failed:', e));
      },
    );

    await prisma.trainingResult.create({
      data: {
        jobId: job.id,
        advantage: result.advantage,
        learningCurve: result.learningCurve,
        modelData: result.modelData as unknown as Prisma.InputJsonValue,
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
    jobStartTimes.delete(job.id);
  }
}
