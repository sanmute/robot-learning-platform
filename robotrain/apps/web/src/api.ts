import type {
  User,
  Config,
  TrainingJob,
  JobStatusResponse,
  CreateConfigRequest,
  CreateJobRequest,
} from '@robotrain/shared';

// ── Token helpers ─────────────────────────────────────────────────────────────

export const getToken = (): string | null => localStorage.getItem('token');
export const setToken = (token: string): void => { localStorage.setItem('token', token); };
export const clearToken = (): void => { localStorage.removeItem('token'); };

// ── Base request ──────────────────────────────────────────────────────────────

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(url, { ...init, headers });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}

// ── API client ────────────────────────────────────────────────────────────────

export const api = {
  // Auth
  getMe: (): Promise<User> =>
    request<User>('/api/me'),

  // Configs
  createConfig: (data: CreateConfigRequest): Promise<Config> =>
    request<Config>('/api/configs', { method: 'POST', body: JSON.stringify(data) }),

  getConfigs: (): Promise<Config[]> =>
    request<Config[]>('/api/configs'),

  getConfig: (id: string): Promise<Config> =>
    request<Config>(`/api/configs/${id}`),

  // Jobs
  createJob: (data: CreateJobRequest): Promise<{ id: string }> =>
    request<{ id: string }>('/api/jobs', { method: 'POST', body: JSON.stringify(data) }),

  getJobs: (): Promise<TrainingJob[]> =>
    request<TrainingJob[]>('/api/jobs'),

  getJob: (id: string): Promise<TrainingJob> =>
    request<TrainingJob>(`/api/jobs/${id}`),

  getJobStatus: (id: string): Promise<JobStatusResponse> =>
    request<JobStatusResponse>(`/api/jobs/${id}/status`),

  downloadModel: (id: string): Promise<Response> => {
    const token = getToken();
    return fetch(`/api/jobs/${id}/model`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  },

  // Dev only — remove before real production launch
  devLogin: (): Promise<{ token: string; user: User }> =>
    request<{ token: string; user: User }>('/api/auth/dev-login', { method: 'POST' }),
};
