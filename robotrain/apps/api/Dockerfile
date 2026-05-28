# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

# Copy manifests first for layer caching
COPY package*.json ./
COPY apps/api/package*.json ./apps/api/
COPY apps/web/package*.json ./apps/web/
COPY packages/shared/package*.json ./packages/shared/
COPY packages/training/package*.json ./packages/training/

RUN npm ci

# Copy all source
COPY . .

# Build in dependency order
RUN npm run build -w packages/shared
RUN npm run build -w packages/training
RUN npm run build -w apps/web
# build:api runs `prisma generate && tsc`
RUN npm run build -w apps/api

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM node:20-alpine AS production
WORKDIR /app

ENV NODE_ENV=production
# Cloud Run injects PORT; default to 8080
ENV PORT=8080

# Production deps only
COPY package*.json ./
COPY apps/api/package*.json ./apps/api/
COPY packages/shared/package*.json ./packages/shared/
COPY packages/training/package*.json ./packages/training/

RUN npm ci --omit=dev --ignore-scripts

# Compiled API
COPY --from=builder /app/apps/api/dist ./apps/api/dist

# Generated Prisma client
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma

# Compiled workspace packages
COPY --from=builder /app/packages/shared/dist ./packages/shared/dist
COPY --from=builder /app/packages/training/dist ./packages/training/dist

# Prisma schema (needed for migrate:deploy on startup)
COPY apps/api/prisma ./apps/api/prisma

# React frontend served as static files from /app/public
COPY --from=builder /app/apps/web/dist ./public

EXPOSE 8080

CMD ["node", "apps/api/dist/index.js"]
