/**
 * In-memory live data store for running training jobs.
 *
 * Keyed by jobId. Holds log lines (with relative timestamps) and the
 * live learning curve as it accumulates trial by trial.
 *
 * Both runner.ts (writer) and routes/jobs.ts (reader) import from here.
 * Works correctly as long as both run in the same Node.js process (MVP).
 */

interface JobLiveData {
  logs: string[];
  learningCurve: number[];
  startedAt: number; // Date.now() at job init
}

const store = new Map<string, JobLiveData>();

/** Call when a job transitions to 'running'. Resets any stale data. */
export function initJob(jobId: string): void {
  store.set(jobId, { logs: [], learningCurve: [], startedAt: Date.now() });
}

/** Append a plain message; a relative-time prefix is added automatically. */
export function appendLog(jobId: string, message: string): void {
  const data = store.get(jobId);
  if (!data) return;
  const elapsed = ((Date.now() - data.startedAt) / 1000).toFixed(1);
  data.logs.push(`[${elapsed}s] ${message}`);
}

/** Append one learning-curve data point (a running D-vs-A advantage value). */
export function appendCurvePoint(jobId: string, value: number): void {
  const data = store.get(jobId);
  if (!data) return;
  data.learningCurve.push(value);
}

/** Returns current logs + learningCurve, or empty arrays if unknown job. */
export function getJobData(jobId: string): { logs: string[]; learningCurve: number[] } {
  const data = store.get(jobId);
  return data
    ? { logs: [...data.logs], learningCurve: [...data.learningCurve] }
    : { logs: [], learningCurve: [] };
}

/** Free memory after job completes (optional but good hygiene). */
export function clearJob(jobId: string): void {
  store.delete(jobId);
}
