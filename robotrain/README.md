# RoboTrain 🤖

Self-serve SaaS for robot learning configuration and training. Users configure objectives, click Train, wait ~4 seconds, and download a trained model.

**Stack:** Node.js · Express · TypeScript · React · Vite · Tailwind · PostgreSQL (Prisma) · Google OAuth · Cloud Run

---

## Project structure

```
robotrain/
├── apps/
│   ├── api/              Express backend (+ serves built frontend in prod)
│   │   ├── Dockerfile    Multi-stage: builds web → compiles API → prod image
│   │   ├── prisma/       schema.prisma + migrations
│   │   └── src/
│   │       ├── index.ts         Entry point, Passport, static serving
│   │       ├── db.ts            Prisma singleton
│   │       ├── middleware/auth  JWT auth middleware
│   │       ├── routes/          auth · configs · jobs
│   │       └── jobs/runner.ts   DB-backed job queue (polls every 2 s)
│   └── web/              React + Vite + Tailwind frontend
│       └── src/
│           ├── App.tsx          Router + auth context
│           ├── api.ts           Typed API client
│           ├── pages/           Landing · Login · Dashboard · Train · Results
│           └── components/      NavBar · ProtectedRoute · JobStatusBadge · LearningCurveChart
├── packages/
│   ├── shared/           TypeScript types shared between API and web
│   └── training/         Training engine stub (replace with real simulation)
├── cloudbuild.yaml       GCP Cloud Build → Artifact Registry → Cloud Run
└── .env.example
```

---

## Local development

### Prerequisites
- Node.js 20+
- PostgreSQL (local or Docker)
- Google OAuth credentials (see [GCP setup](#gcp-setup))

### 1. Install

```bash
cd robotrain
cp .env.example .env
# Fill in DATABASE_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, JWT_SECRET
npm install
```

### 2. Database

```bash
# Start Postgres if not running:
docker run -d --name pg -e POSTGRES_DB=robotrain -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:16-alpine

# Run migrations (creates tables)
npm run migrate:dev
```

### 3. Build workspace packages (required before running API)

```bash
npm run build:packages
```

### 4. Run

```bash
# Two terminals — or use the combined dev script:
npm run dev

# API:  http://localhost:3000
# Web:  http://localhost:5173  (Vite proxies /api → :3000)
```

The **Dev login** button on the login page (visible in `DEV` mode only) lets you skip Google OAuth locally.

---

## Build for production

```bash
npm run build        # builds packages → web → api
```

The Dockerfile does this automatically in CI.

---

## GCP setup

### Enable APIs

```bash
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

### Create Artifact Registry repo

```bash
gcloud artifacts repositories create robotrain \
  --repository-format=docker \
  --location=us-central1
```

### Create Cloud SQL (PostgreSQL) instance

```bash
gcloud sql instances create robotrain-db \
  --database-version=POSTGRES_16 \
  --cpu=1 --memory=3840MB \
  --region=us-central1

gcloud sql databases create robotrain --instance=robotrain-db
gcloud sql users create robotrain --instance=robotrain-db --password=YOUR_DB_PASSWORD
```

### Store secrets in Secret Manager

```bash
# Format for Cloud SQL Unix socket:
# postgresql://robotrain:PASSWORD@localhost/robotrain?host=/cloudsql/PROJECT:REGION:INSTANCE

echo -n "postgresql://robotrain:PASSWORD@localhost/robotrain?host=/cloudsql/PROJECT_ID:us-central1:robotrain-db" \
  | gcloud secrets create DATABASE_URL --data-file=-

echo -n "$(openssl rand -base64 64)" | gcloud secrets create JWT_SECRET --data-file=-
echo -n "YOUR_CLIENT_ID"             | gcloud secrets create GOOGLE_CLIENT_ID --data-file=-
echo -n "YOUR_CLIENT_SECRET"         | gcloud secrets create GOOGLE_CLIENT_SECRET --data-file=-
echo -n "https://YOUR_RUN_URL/api/auth/google/callback" \
  | gcloud secrets create GOOGLE_CALLBACK_URL --data-file=-
```

### Grant Cloud Run service account access

```bash
PROJECT_ID=$(gcloud config get-value project)
SA="serviceAccount:${PROJECT_ID}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding DATABASE_URL     --member="$SA" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding JWT_SECRET        --member="$SA" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding GOOGLE_CLIENT_ID  --member="$SA" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding GOOGLE_CLIENT_SECRET --member="$SA" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding GOOGLE_CALLBACK_URL  --member="$SA" --role="roles/secretmanager.secretAccessor"

# Cloud SQL access
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="$SA" --role="roles/cloudsql.client"
```

### Connect Cloud Build to the repo & trigger

```bash
# In GCP console: Cloud Build → Triggers → Connect repository
# Or push to your repo — cloudbuild.yaml runs automatically.
```

### Run database migrations on first deploy

```bash
# After first deploy, SSH into Cloud Run or use a one-off job:
gcloud run jobs create migrate \
  --image=REGION-docker.pkg.dev/PROJECT/robotrain/robotrain:latest \
  --set-cloudsql-instances=PROJECT:REGION:robotrain-db \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest \
  --command="node" --args="apps/api/dist/index.js" \
  --region=us-central1

# Or just execute prisma migrate deploy manually:
DATABASE_URL="..." npx prisma migrate deploy --schema=apps/api/prisma/schema.prisma
```

---

## API reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/api/auth/google` | — | Initiate Google OAuth |
| `GET` | `/api/auth/google/callback` | — | OAuth callback → redirect with JWT |
| `POST` | `/api/auth/dev-login` | — | Dev-only quick login |
| `GET` | `/api/me` | ✓ | Current user |
| `POST` | `/api/configs` | ✓ | Create config (weights must sum to 100) |
| `GET` | `/api/configs` | ✓ | List user configs |
| `GET` | `/api/configs/:id` | ✓ | Get config |
| `POST` | `/api/jobs` | ✓ | Start training job (returns `{ id }` immediately) |
| `GET` | `/api/jobs` | ✓ | List user jobs |
| `GET` | `/api/jobs/:id` | ✓ | Full job record |
| `GET` | `/api/jobs/:id/status` | ✓ | Poll: `{ status, progress, result? }` |
| `GET` | `/api/jobs/:id/model` | ✓ | Download `model_[id].json` |

---

## Replacing the training stub

Edit [`packages/training/src/index.ts`](packages/training/src/index.ts). Your real implementation should:

1. Accept `(config: TrainingConfig, onProgress: (pct: number) => Promise<void>)`  
2. Call `onProgress` periodically so the progress bar updates  
3. Return `{ advantage: number, learningCurve: number[], modelData: {} }`

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (Unix socket in prod) |
| `GOOGLE_CLIENT_ID` | OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth 2.0 client secret |
| `GOOGLE_CALLBACK_URL` | Full URL to `/api/auth/google/callback` |
| `JWT_SECRET` | Random secret for signing JWTs (min 32 chars) |
| `NODE_ENV` | `development` \| `production` |
| `PORT` | HTTP port (Cloud Run sets this to 8080 automatically) |
