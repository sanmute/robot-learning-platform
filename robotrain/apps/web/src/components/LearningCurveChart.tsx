import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface Props {
  data: number[];
  advantage?: number;
}

export default function LearningCurveChart({ data, advantage }: Props) {
  const chartData = data.map((value, i) => ({
    iteration: i + 1,
    advantage: value,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis
          dataKey="iteration"
          label={{ value: 'Training iteration', position: 'insideBottom', offset: -2 }}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          label={{ value: 'Advantage (%)', angle: -90, position: 'insideLeft', offset: 10 }}
          tick={{ fontSize: 12 }}
          domain={['auto', 'auto']}
        />
        <Tooltip
          formatter={(v: number) => [`${v.toFixed(2)}%`, 'Advantage']}
          labelFormatter={(l: number) => `Iteration ${l}`}
        />
        {advantage !== undefined && (
          <ReferenceLine
            y={advantage}
            stroke="#2563eb"
            strokeDasharray="6 3"
            label={{ value: `Final: +${advantage.toFixed(2)}%`, position: 'right', fontSize: 11, fill: '#2563eb' }}
          />
        )}
        <Line
          type="monotone"
          dataKey="advantage"
          stroke="#2563eb"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
