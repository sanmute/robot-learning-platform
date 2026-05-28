import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import type { TrainingJob } from '@robotrain/shared';
import { api } from '../api';
import NavBar from '../components/NavBar';
import JobStatusBadge from '../components/JobStatusBadge';
import LearningCurveChart from '../components/LearningCurveChart';

export default function Results() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then(setJob)
      .catch(() => setError('Failed to load results'))
      .finally(() => setLoading(false));
  }, [jobId]);

  const handleDownload = async () => {
    if (!jobId) return;
    try {
      const res = await api.downloadModel(jobId);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${jobId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Download failed');
    }
  };

  const share = () => {
    navigator.clipboard.writeText(window.location.href).catch(() => null);
  };

  const formatDate = (s?: string | null) =>
    s ? new Date(s).toLocaleString(undefined, {
      month: 'long', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    }) : '—';

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <NavBar />
        <main className="mx-auto max-w-3xl px-4 py-10 space-y-4">
          {[1, 2, 3].map((n) => (
            <div key={n} className="card animate-pulse h-24 bg-gray-100" />
          ))}
        </main>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="min-h-screen bg-gray-50">
        <NavBar />
        <main className="mx-auto max-w-3xl px-4 py-20 text-center">
          <div className="card">
            <p className="text-red-600 mb-4">{error ?? 'Results not found'}</p>
            <Link to="/dashboard" className="btn-secondary">← Dashboard</Link>
          </div>
        </main>
      </div>
    );
  }

  const result = job.result;
  const config = job.config;

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      <main className="mx-auto max-w-3xl px-4 py-10 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-2xl font-bold text-gray-900">
                {config?.name ?? 'Training Results'}
              </h1>
              <JobStatusBadge status={job.status} />
            </div>
            <p className="text-sm text-gray-500">
              {config?.robotType && <span className="capitalize">{config.robotType} robot · </span>}
              Completed {formatDate(job.finishedAt)}
            </p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <button onClick={share} className="btn-secondary gap-2">
              🔗 Copy link
            </button>
            {result && (
              <button onClick={handleDownload} className="btn-primary gap-2">
                ⬇️ Download model
              </button>
            )}
          </div>
        </div>

        {/* Key metric */}
        {result && (
          <div className="card flex items-center gap-6">
            <div className="text-center min-w-[120px]">
              <div className="text-4xl font-extrabold text-green-600">
                +{result.advantage.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 mt-1 font-medium uppercase tracking-wide">
                Generalization advantage
              </div>
            </div>
            <div className="text-sm text-gray-600 leading-relaxed">
              The robot trained with your configuration outperformed the baseline by{' '}
              <strong className="text-gray-900">+{result.advantage.toFixed(2)}%</strong> on
              held-out test scenarios, measured as the D-vs-A generalization index.
            </div>
          </div>
        )}

        {/* Learning curve */}
        {result && (
          <div className="card">
            <h2 className="mb-4 font-semibold text-gray-900">Learning curve</h2>
            <LearningCurveChart
              data={result.learningCurve as number[]}
              advantage={result.advantage}
            />
            <p className="mt-3 text-xs text-gray-400">
              Each point shows the cumulative advantage after that training iteration.
            </p>
          </div>
        )}

        {/* Config details */}
        {config && (
          <div className="card">
            <h2 className="mb-4 font-semibold text-gray-900">Configuration</h2>
            <dl className="grid gap-3 sm:grid-cols-2 text-sm">
              <div>
                <dt className="text-gray-500">Robot type</dt>
                <dd className="font-medium capitalize text-gray-900">{config.robotType}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Job ID</dt>
                <dd className="font-mono text-xs text-gray-700 break-all">{job.id}</dd>
              </div>
              {Object.entries(config.weights as unknown as Record<string, number>).map(([key, val]) => (
                <div key={key}>
                  <dt className="text-gray-500 capitalize">{key} weight</dt>
                  <dd className="font-medium text-gray-900">{val}%</dd>
                </div>
              ))}
            </dl>
          </div>
        )}

        {/* Back link */}
        <div className="flex gap-3">
          <Link to="/dashboard" className="btn-secondary">
            ← Dashboard
          </Link>
          <Link to="/train" className="btn-secondary">
            + New training
          </Link>
        </div>
      </main>
    </div>
  );
}
