import { Router } from 'express';
import { prisma } from '../db';
import { AuthRequest } from '../middleware/auth';
import { CreateJobRequest } from '@robotrain/shared';

const router = Router();

// ── POST /api/jobs — start a training job ─────────────────────────────────────
router.post('/', async (req: AuthRequest, res) => {
  const { configId } = req.body as CreateJobRequest;

  if (!configId) {
    return res.status(400).json({ error: 'configId is required' });
  }

  // Verify config belongs to caller
  const config = await prisma.config.findFirst({
    where: { id: configId, userId: req.userId },
  });
  if (!config) {
    return res.status(404).json({ error: 'Config not found' });
  }

  try {
    const job = await prisma.trainingJob.create({
      data: { configId },
    });
    res.status(201).json({ id: job.id });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create job' });
  }
});

// ── GET /api/jobs — list all jobs for the current user ────────────────────────
router.get('/', async (req: AuthRequest, res) => {
  try {
    const jobs = await prisma.trainingJob.findMany({
      where: { config: { userId: req.userId } },
      include: { config: true, result: true },
      orderBy: [{ startedAt: 'desc' }, { id: 'desc' }],
    });
    res.json(jobs);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch jobs' });
  }
});

// ── GET /api/jobs/:id/status — polling endpoint ───────────────────────────────
router.get('/:id/status', async (req: AuthRequest, res) => {
  try {
    const job = await prisma.trainingJob.findFirst({
      where: { id: req.params.id, config: { userId: req.userId } },
      include: { result: true },
    });
    if (!job) return res.status(404).json({ error: 'Job not found' });

    res.json({
      id: job.id,
      status: job.status,
      progress: job.progress,
      result: job.result
        ? {
            advantage: job.result.advantage,
            learningCurve: job.result.learningCurve,
          }
        : null,
    });
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch job status' });
  }
});

// ── GET /api/jobs/:id — full job record (for results page) ────────────────────
router.get('/:id', async (req: AuthRequest, res) => {
  try {
    const job = await prisma.trainingJob.findFirst({
      where: {
        id: req.params.id,
        config: { userId: req.userId },
      },
      include: { config: true, result: true },
    });
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json(job);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch job' });
  }
});

// ── GET /api/jobs/:id/model — download trained model as JSON ──────────────────
router.get('/:id/model', async (req: AuthRequest, res) => {
  try {
    const job = await prisma.trainingJob.findFirst({
      where: {
        id: req.params.id,
        config: { userId: req.userId },
        status: 'done',
      },
      include: { result: true, config: true },
    });

    if (!job || !job.result) {
      return res.status(404).json({ error: 'Model not available' });
    }

    const model = {
      jobId: job.id,
      configName: job.config.name,
      robotType: job.config.robotType,
      advantage: job.result.advantage,
      learningCurve: job.result.learningCurve,
      modelData: job.result.modelData,
      exportedAt: new Date().toISOString(),
    };

    res.setHeader('Content-Type', 'application/json');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="model_${job.id}.json"`,
    );
    res.json(model);
  } catch (err) {
    res.status(500).json({ error: 'Failed to export model' });
  }
});

export default router;
