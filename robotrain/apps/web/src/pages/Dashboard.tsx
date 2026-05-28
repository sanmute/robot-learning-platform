import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { TrainingJob } from '@robotrain/shared';
import { api } from '../api';
import { useAuth } from '../App';
import NavBar from '../components/NavBar';
import JobStatusBadge from '../components/JobStatusBadge';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getJobs()
      .then(setJobs)
      .catch(() => setError('Failed to load training history'))
      .finally(() => setLoading(false));
  }, []);

  // Poll running jobs every 2 s
  useEffect(() => {
    const running = jobs.some((j) => j.status === 'pending' || j.status === 'running');
    if (!running) return;
    const id = setInterval(() => {
      api.getJobs().then(setJobs).catch(() => null);
    }, 2000);
    return () => clearInterval(id);
  }, [jobs]);

  const formatDate = (s?: string | null) => {
    if (!s) return '—';
    return new Date(s).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      <main className="mx-auto max-w-5xl px-4 py-10">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Welcome back, {user?.name?.split(' ')[0]} 👋
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              {jobs.length === 0 ? 'No training runs yet — start your first one!' : `${jobs.length} training run${jobs.length !== 1 ? 's' : ''}`}
            </p>
          </div>
          <Link to="/train" className="btn-primary">
            + New Training
          </Link>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="space-y-3">
            {[1, 2, 3].map((n) => (
              <div key={n} className="card animate-pulse h-16 bg-gray-100" />
            ))}
          </div>
        )}

        {/* Empty state */}
        {!loading && jobs.length === 0 && (
          <div className="card flex flex-col items-center gap-4 py-16 text-center">
            <span className="text-5xl">🤖</span>
            <h2 className="text-lg font-semibold text-gray-900">Train your first robot</h2>
            <p className="max-w-xs text-sm text-gray-500">
              Configure objectives, hit Train, and get a trained model in ~4 seconds.
            </p>
            <Link to="/train" className="btn-primary">
              Start training
            </Link>
          </div>
        )}

        {/* Jobs table */}
        {!loading && jobs.length > 0 && (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="border-b border-gray-100 bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
                <tr>
                  <th className="px-6 py-3 text-left">Config</th>
                  <th className="px-6 py-3 text-left">Robot type</th>
                  <th className="px-6 py-3 text-left">Status</th>
                  <th className="px-6 py-3 text-right">Advantage</th>
                  <th className="px-6 py-3 text-right">Started</th>
                  <th className="px-6 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {jobs.map((job) => (
                  <tr
                    key={job.id}
                    className="hover:bg-gray-50 cursor-pointer transition"
                    onClick={() => job.status === 'done' && navigate(`/results/${job.id}`)}
                  >
                    <td className="px-6 py-4 font-medium text-gray-900">
                      {job.config?.name ?? '—'}
                    </td>
                    <td className="px-6 py-4 capitalize text-gray-600">
                      {job.config?.robotType ?? '—'}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <JobStatusBadge status={job.status} />
                        {job.status === 'running' && (
                          <span className="text-xs text-gray-400">{job.progress}%</span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-gray-700">
                      {job.result
                        ? <span className="font-semibold text-green-700">+{job.result.advantage.toFixed(2)}%</span>
                        : '—'}
                    </td>
                    <td className="px-6 py-4 text-right text-gray-500">
                      {formatDate(job.startedAt)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {job.status === 'done' && (
                        <Link
                          to={`/results/${job.id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="text-brand-600 hover:underline font-medium"
                        >
                          View →
                        </Link>
                      )}
                      {job.status === 'failed' && (
                        <Link to="/train" className="text-red-600 hover:underline font-medium">
                          Retry
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}
