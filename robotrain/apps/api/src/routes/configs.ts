import { Router } from 'express';
import { prisma } from '../db';
import { AuthRequest } from '../middleware/auth';
import { CreateConfigRequest } from '@robotrain/shared';

const router = Router();

const VALID_ROBOT_TYPES = ['warehouse', 'manufacturing', 'space'];
const OBJECTIVE_KEYS = ['food', 'efficiency', 'speed', 'accuracy', 'balance'];

function validateWeights(weights: Record<string, number>): string | null {
  const keys = Object.keys(weights);
  if (!OBJECTIVE_KEYS.every((k) => keys.includes(k))) {
    return 'weights must include: food, efficiency, speed, accuracy, balance';
  }
  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  if (Math.round(total) !== 100) {
    return `weights must sum to 100 (got ${total})`;
  }
  return null;
}

// ── POST /api/configs ─────────────────────────────────────────────────────────
router.post('/', async (req: AuthRequest, res) => {
  const { name, robotType, objectives, weights } = req.body as CreateConfigRequest;

  if (!name?.trim()) {
    return res.status(400).json({ error: 'name is required' });
  }
  if (!VALID_ROBOT_TYPES.includes(robotType)) {
    return res.status(400).json({ error: `robotType must be one of: ${VALID_ROBOT_TYPES.join(', ')}` });
  }

  const weightError = validateWeights(weights);
  if (weightError) return res.status(400).json({ error: weightError });

  const objectiveError = validateWeights(objectives);
  if (objectiveError) return res.status(400).json({ error: `objectives: ${objectiveError}` });

  try {
    const config = await prisma.config.create({
      data: {
        userId: req.userId!,
        name: name.trim(),
        robotType,
        objectives,
        weights,
      },
    });
    res.status(201).json(config);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create config' });
  }
});

// ── GET /api/configs ──────────────────────────────────────────────────────────
router.get('/', async (req: AuthRequest, res) => {
  try {
    const configs = await prisma.config.findMany({
      where: { userId: req.userId },
      orderBy: { createdAt: 'desc' },
    });
    res.json(configs);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch configs' });
  }
});

// ── GET /api/configs/:id ──────────────────────────────────────────────────────
router.get('/:id', async (req: AuthRequest, res) => {
  try {
    const config = await prisma.config.findFirst({
      where: { id: req.params.id, userId: req.userId },
    });
    if (!config) return res.status(404).json({ error: 'Config not found' });
    res.json(config);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch config' });
  }
});

export default router;
