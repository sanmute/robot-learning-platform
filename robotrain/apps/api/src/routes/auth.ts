import { Router } from 'express';
import passport from 'passport';
import jwt from 'jsonwebtoken';
import { prisma } from '../db';

const router = Router();

// ── Initiate Google OAuth ─────────────────────────────────────────────────────
router.get(
  '/google',
  passport.authenticate('google', { scope: ['profile', 'email'], session: false }),
);

// ── Google OAuth callback ─────────────────────────────────────────────────────
router.get(
  '/google/callback',
  passport.authenticate('google', {
    session: false,
    failureRedirect: '/login?error=oauth_failed',
  }),
  (req: any, res) => {
    const user = req.user as { id: string };
    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET!, {
      expiresIn: '7d',
    });
    // Redirect to frontend; the Login page reads ?token= and stores it
    const frontendUrl = process.env.NODE_ENV === 'production' 
      ? '' 
      : 'http://localhost:5174';
    res.redirect(`${frontendUrl}/login?token=${token}`);
  },
);

// ── Dev-only: create a test user without OAuth ────────────────────────────────
// Remove this block before going to production, or guard with NODE_ENV check.
if (process.env.NODE_ENV !== 'production') {
  router.post('/dev-login', async (req, res) => {
    try {
      const user = await prisma.user.upsert({
        where: { email: 'dev@robotrain.local' },
        update: {},
        create: {
          email: 'dev@robotrain.local',
          name: 'Dev User',
          googleId: 'dev-local',
        },
      });
      const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET!, {
        expiresIn: '7d',
      });
      res.json({ token, user });
    } catch (err) {
      res.status(500).json({ error: 'Dev login failed' });
    }
  });
}

export default router;
