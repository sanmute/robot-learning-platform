import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import passport from 'passport';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import jwt from 'jsonwebtoken';

import { prisma } from './db';
import { authMiddleware, AuthRequest } from './middleware/auth';
import authRouter from './routes/auth';
import configsRouter from './routes/configs';
import jobsRouter from './routes/jobs';
import { startJobRunner } from './jobs/runner';

// ── Passport setup ────────────────────────────────────────────────────────────
passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      callbackURL: process.env.GOOGLE_CALLBACK_URL!,
    },
    async (_accessToken, _refreshToken, profile, done) => {
      try {
        const user = await prisma.user.upsert({
          where: { googleId: profile.id },
          update: {
            name: profile.displayName ?? '',
            email: profile.emails?.[0].value ?? '',
          },
          create: {
            googleId: profile.id,
            name: profile.displayName ?? '',
            email: profile.emails?.[0].value ?? '',
          },
        });
        done(null, user);
      } catch (err) {
        done(err as Error);
      }
    },
  ),
);

// ── Express app ───────────────────────────────────────────────────────────────
const app = express();

app.use(cors({ origin: true, credentials: true }));
app.use(express.json());
app.use(passport.initialize());

// ── Health check (Cloud Run liveness probe) ───────────────────────────────────
app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ── Public auth routes ────────────────────────────────────────────────────────
app.use('/api/auth', authRouter);

// ── Protected: current user ───────────────────────────────────────────────────
app.get('/api/me', authMiddleware, async (req: AuthRequest, res) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: req.userId } });
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
  } catch {
    res.status(500).json({ error: 'Internal server error' });
  }
});

// ── Protected: configs & jobs ─────────────────────────────────────────────────
app.use('/api/configs', authMiddleware, configsRouter);
app.use('/api/jobs', authMiddleware, jobsRouter);

// ── Serve React frontend in production ────────────────────────────────────────
if (process.env.NODE_ENV === 'production') {
  const publicDir = path.join(process.cwd(), 'public');
  app.use(express.static(publicDir));
  // SPA catch-all: any non-API path returns index.html
  app.get(/^(?!\/api).*/, (_req, res) => {
    res.sendFile(path.join(publicDir, 'index.html'));
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT ?? '3000', 10);

async function main() {
  await prisma.$connect();
  console.log('📦 Database connected');

  startJobRunner();

  app.listen(PORT, () => {
    console.log(`🚀 RoboTrain API → http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Fatal startup error:', err);
  process.exit(1);
});
