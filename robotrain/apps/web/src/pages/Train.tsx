import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { RobotType, Objectives } from '@robotrain/shared';
import { api } from '../api';
import NavBar from '../components/NavBar';
import LearningCurveChart from '../components/LearningCurveChart';

// ── Constants ─────────────────────────────────────────────────────────────────

const ROBOT_TYPES: { value: RobotType; label: string; emoji: string }[] = [
  { value: 'warehouse',      label: 'Warehouse',     emoji: '🏭' },
  { value: 'manufacturing',  label: 'Manufacturing', emoji: '⚙️' },
  { value: 'space',          label: 'Space',          emoji: '🚀' },
];

const OBJECTIVE_LABELS: Record<keyof Objectives, string> = {
  food:       'Food Collection',
  efficiency: 'Energy Efficiency',
  speed:      'Movement Speed',
  accuracy:   'Navigation Accuracy',
  balance:    'Multi-task Balance',
};

const DEFAULT_WEIGHTS: Objectives = {
  food: 20, efficiency: 20, speed: 20, accuracy: 20, balance: 20,
};

type Step = 1 | 2 | 3;

// ── Log line colour ───────────────────────────────────────────────────────────

function logLineClass(line: string): string {
  const m = line.toLowerCase();
  if (m.includes('error') || m.includes('failed')) return 'text-red-400';
  if (
    m.includes('converged') ||
    m.includes('phase 2 complete') ||
    m.includes('final d-vs-a') ||
    m.includes('phase 1 complete') ||
    m.includes('advantage so far')
  ) return 'text-green-400';
  if (
    m.includes('milestone') ||
    m.includes('consolidation triggered') ||
    m.includes('memory transfer') ||
    m.includes('plateau') ||
    m.includes('refined')
  ) return 'text-yellow-400';
  return 'text-gray-300';
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Train() {
  const navigate = useNavigate();

  // Step 1 — config form
  const [configName, setConfigName] = useState('My Robot Config');
  const [robotType, setRobotType] = useState<RobotType>('warehouse');
  const [weights, setWeights] = useState<Objectives>({ ...DEFAULT_WEIGHTS });

  // Step 2 — progress + live data
  const [step, setStep]         = useState<Step>(1);
  const [jobId, setJobId]       = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs]         = useState<string[]>([]);
  const [liveCurve, setLiveCurve] = useState<number[]>([]);

  // Step 3 — results
  const [result, setResult] = useState<{ advantage: number; learningCurve: number[] } | null>(null);

  // Errors / loading
  const [error, setError]         = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Auto-scroll ref for log panel
  const logRef = useRef<HTMLDivElement>(null);

  // ── Weight helpers ──────────────────────────────────────────────────────────

  const totalWeight  = Object.values(weights).reduce((a, b) => a + b, 0);
  const weightsValid = Math.round(totalWeight) === 100;

  const handleWeightChange = (key: keyof Objectives, value: number) => {
    setWeights((prev) => ({ ...prev, [key]: value }));
  };

  // ── Step 1 → 2: start training ──────────────────────────────────────────────

  const handleTrain = async () => {
    if (!weightsValid || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const config = await api.createConfig({
        name: configName.trim() || 'Untitled',
        robotType,
        objectives: weights,
        weights,
      });
      const job = await api.createJob({ configId: config.id });
      setJobId(job.id);
      setProgress(0);
      setLogs([]);
      setLiveCurve([]);
      setStep(2);
    } catch (err: unknown) {
      setError((err as Error).message ?? 'Failed to start training. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  // ── Step 2: poll status + logs ───────────────────────────────────────────────

  useEffect(() => {
    if (step !== 2 || !jobId) return;

    const poll = setInterval(async () => {
      try {
        // Status + logs fetched in parallel
        const [status, logData] = await Promise.all([
          api.getJobStatus(jobId),
          api.getJobLogs(jobId),
        ]);

        setProgress(status.progress);
        setLogs(logData.logs);
        setLiveCurve(logData.learningCurve);

        if (status.status === 'done') {
          clearInterval(poll);
          setResult(status.result ?? null);
          setStep(3);
        } else if (status.status === 'failed') {
          clearInterval(poll);
          setError('Training failed. Please try again.');
          setStep(1);
        }
      } catch {
        // transient network error — keep polling
      }
    }, 1000);

    return () => clearInterval(poll);
  }, [step, jobId]);

  // Auto-scroll log panel when new lines arrive
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  // ── Step 3 actions ───────────────────────────────────────────────────────────

  const handleDownload = async () => {
    if (!jobId) return;
    try {
      const res = await api.downloadModel(jobId);
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `model_${jobId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Failed to download model');
    }
  };

  const handleViewResults = () => {
    if (jobId) navigate(`/results/${jobId}`);
  };

  const handleReset = () => {
    setStep(1);
    setJobId(null);
    setProgress(0);
    setResult(null);
    setLogs([]);
    setLiveCurve([]);
    setError(null);
    setWeights({ ...DEFAULT_WEIGHTS });
  };

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      <main className="mx-auto max-w-2xl px-4 py-10">

        {/* Step indicator */}
        <div className="mb-8 flex items-center gap-2 text-sm">
          {(['Configure', 'Training', 'Results'] as const).map((lbl, i) => {
            const stepNum = (i + 1) as Step;
            const active  = step === stepNum;
            const done    = step > stepNum;
            return (
              <div key={lbl} className="flex items-center gap-2">
                {i > 0 && <div className="h-px w-8 bg-gray-200" />}
                <div className={`flex items-center gap-1.5 font-medium ${active ? 'text-brand-600' : done ? 'text-green-600' : 'text-gray-400'}`}>
                  <span className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${active ? 'bg-brand-600 text-white' : done ? 'bg-green-100 text-green-700' : 'bg-gray-200 text-gray-500'}`}>
                    {done ? '✓' : stepNum}
                  </span>
                  {lbl}
                </div>
              </div>
            );
          })}
        </div>

        {error && (
          <div className="mb-6 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* ── Step 1: Config form ─────────────────────────────────────────── */}
        {step === 1 && (
          <div className="card space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Configure your robot</h2>

            {/* Config name */}
            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-700">
                Configuration name
              </label>
              <input
                type="text"
                value={configName}
                onChange={(e) => setConfigName(e.target.value)}
                className="input"
                placeholder="My Robot Config"
              />
            </div>

            {/* Robot type */}
            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-700">
                Robot environment
              </label>
              <div className="grid grid-cols-3 gap-3">
                {ROBOT_TYPES.map(({ value, label, emoji }) => (
                  <button
                    key={value}
                    onClick={() => setRobotType(value)}
                    className={`rounded-xl border-2 p-3 text-center transition ${
                      robotType === value
                        ? 'border-brand-600 bg-brand-50 text-brand-700'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="mb-1 text-2xl">{emoji}</div>
                    <div className="text-sm font-medium">{label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Objective weights */}
            <div>
              <div className="mb-3 flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">Objective weights</label>
                <span className={`text-sm font-semibold ${weightsValid ? 'text-green-600' : 'text-red-600'}`}>
                  {totalWeight}/100
                </span>
              </div>
              <div className="space-y-4">
                {(Object.keys(DEFAULT_WEIGHTS) as (keyof Objectives)[]).map((key) => (
                  <div key={key}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span className="text-gray-700">{OBJECTIVE_LABELS[key]}</span>
                      <span className="w-8 text-right font-mono font-medium text-gray-900">
                        {weights[key]}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={weights[key]}
                      onChange={(e) => handleWeightChange(key, Number(e.target.value))}
                      className="w-full accent-brand-600"
                    />
                  </div>
                ))}
              </div>
              {!weightsValid && (
                <p className="mt-2 text-xs text-red-600">
                  Adjust sliders so the total equals 100
                </p>
              )}
            </div>

            <button
              onClick={handleTrain}
              disabled={!weightsValid || submitting}
              className="btn-primary w-full justify-center py-3 text-base"
            >
              {submitting ? 'Starting…' : '🚀 Train robot'}
            </button>
          </div>
        )}

        {/* ── Step 2: Training in progress ────────────────────────────────── */}
        {step === 2 && (
          <div className="space-y-4">

            {/* Progress header */}
            <div className="card space-y-4">
              <div className="flex items-center gap-3">
                <span className="text-3xl">⚙️</span>
                <div>
                  <h2 className="font-bold text-gray-900">Training in progress…</h2>
                  <p className="text-sm text-gray-500">Your robot is learning. This takes about 4 seconds.</p>
                </div>
                <span className="ml-auto font-mono text-sm font-semibold text-brand-600">
                  {progress}%
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-gray-200">
                <div
                  className="h-full rounded-full bg-brand-600 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Live learning curve */}
            <div className="card">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">
                Learning curve
                <span className="ml-2 text-xs font-normal text-gray-400">
                  {liveCurve.length > 0
                    ? `${liveCurve.length} / 30 points`
                    : 'waiting for first test trial…'}
                </span>
              </h3>
              {liveCurve.length > 0 ? (
                <LearningCurveChart data={liveCurve} />
              ) : (
                <div className="flex h-40 items-center justify-center rounded-lg bg-gray-50 text-sm text-gray-400">
                  Chart will appear after trial 13…
                </div>
              )}
            </div>

            {/* Terminal log */}
            <div className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900">
              <div className="flex items-center gap-2 border-b border-gray-800 px-4 py-2">
                <span className="h-3 w-3 rounded-full bg-red-500/70" />
                <span className="h-3 w-3 rounded-full bg-yellow-500/70" />
                <span className="h-3 w-3 rounded-full bg-green-500/70" />
                <span className="ml-2 font-mono text-xs text-gray-500">training.log</span>
              </div>
              <div
                ref={logRef}
                className="h-52 overflow-y-auto p-4 font-mono text-xs leading-5"
              >
                {logs.length === 0 ? (
                  <span className="text-gray-600">Waiting for job to start…</span>
                ) : (
                  logs.map((line, i) => (
                    <div key={i} className={logLineClass(line)}>
                      {line}
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>
        )}

        {/* ── Step 3: Results ─────────────────────────────────────────────── */}
        {step === 3 && result && (
          <div className="space-y-6">
            <div className="card text-center">
              <div className="mb-4 text-5xl">🏆</div>
              <h2 className="mb-1 text-xl font-bold text-gray-900">Training complete!</h2>
              <p className="text-sm text-gray-500">
                Your robot learned to perform {result.advantage.toFixed(2)}% better.
              </p>

              <div className="my-6 inline-block rounded-2xl bg-green-50 px-8 py-4">
                <div className="text-4xl font-extrabold text-green-600">
                  +{result.advantage.toFixed(2)}%
                </div>
                <div className="mt-1 text-sm font-medium text-green-700">generalization advantage</div>
              </div>

              <div className="flex flex-wrap justify-center gap-3">
                <button onClick={handleDownload} className="btn-primary gap-2">
                  ⬇️ Download model
                </button>
                <button onClick={handleViewResults} className="btn-secondary">
                  📊 Full results page
                </button>
                <button onClick={handleReset} className="btn-secondary">
                  + New training
                </button>
              </div>
            </div>

            {/* Final learning curve */}
            <div className="card">
              <h3 className="mb-4 font-semibold text-gray-900">Learning curve</h3>
              <LearningCurveChart data={result.learningCurve} advantage={result.advantage} />
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
