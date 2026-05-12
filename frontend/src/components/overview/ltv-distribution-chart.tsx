"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  data: Record<string, unknown>[];
}

function buildHistogram(values: number[], bins = 20) {
  if (values.length === 0) return [];
  const min = 0;
  const max = Math.min(
    values.reduce((a, b) => Math.max(a, b), 0),
    values.sort((a, b) => a - b)[Math.floor(values.length * 0.99)] // p99
  );
  const step = (max - min) / bins;

  const counts = Array(bins).fill(0);
  values.forEach((v) => {
    const bin = Math.min(Math.floor((v - min) / step), bins - 1);
    if (bin >= 0) counts[bin]++;
  });

  return counts.map((count, i) => ({
    range: `£${Math.round(min + i * step / 1000)}K`,
    count,
    from: min + i * step,
    to:   min + (i + 1) * step,
  }));
}

export function LTVDistributionChart({ data }: Props) {
  const values = data.map(r => Number(r.ltv_36m) || 0).filter(v => v > 0);
  const histogram = buildHistogram(values, 25);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>LTV 36m Distribution</CardTitle>
        <span className="text-xs text-slate-400">
          {values.length.toLocaleString()} customers (p99 capped)
        </span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={histogram} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="range"
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            interval={4}
          />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip
            formatter={(v: number) => [v.toLocaleString(), "Customers"]}
            labelFormatter={(l) => `LTV range: ${l}`}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {histogram.map((_, i) => (
              <Cell
                key={i}
                fill={i < 5 ? "#94a3b8" : i < 15 ? "#60a5fa" : "#6366f1"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}