import type { JobStatus } from '@robotrain/shared';

const CONFIG: Record<JobStatus, { label: string; className: string }> = {
  pending:  { label: 'Pending',  className: 'bg-gray-100  text-gray-600'  },
  running:  { label: 'Training', className: 'bg-blue-100  text-blue-700 animate-pulse' },
  done:     { label: 'Done',     className: 'bg-green-100 text-green-700' },
  failed:   { label: 'Failed',   className: 'bg-red-100   text-red-700'  },
};

interface Props {
  status: JobStatus;
}

export default function JobStatusBadge({ status }: Props) {
  const { label, className } = CONFIG[status] ?? CONFIG.pending;
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${className}`}>
      {label}
    </span>
  );
}
