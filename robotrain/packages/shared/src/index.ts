// ── Domain types ─────────────────────────────────────────────────────────────

export type RobotType = 'warehouse' | 'manufacturing' | 'space';

export type JobStatus = 'pending' | 'running' | 'done' | 'failed';

export type Objectives = {
  food: number;
  efficiency: number;
  speed: number;
  accuracy: number;
  balance: number;
  [key: string]: number;
}

// ── API resource shapes ──────────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}

export interface Config {
  id: string;
  userId: string;
  name: string;
  robotType: RobotType;
  objectives: Objectives;
  weights: Objectives;
  createdAt: string;
}

export interface TrainingResult {
  id: string;
  jobId: string;
  advantage: number;
  learningCurve: number[];
  modelData: Record<string, unknown>;
}

export interface TrainingJob {
  id: string;
  configId: string;
  config?: Config;
  status: JobStatus;
  progress: number;
  startedAt?: string | null;
  finishedAt?: string | null;
  result?: TrainingResult | null;
}

// ── Request / response shapes ────────────────────────────────────────────────

export interface CreateConfigRequest {
  name: string;
  robotType: RobotType;
  objectives: Objectives;
  weights: Objectives;
}

export interface CreateJobRequest {
  configId: string;
}

export interface JobStatusResponse {
  id: string;
  status: JobStatus;
  progress: number;
  result?: {
    advantage: number;
    learningCurve: number[];
  } | null;
}

export interface ApiError {
  error: string;
}
