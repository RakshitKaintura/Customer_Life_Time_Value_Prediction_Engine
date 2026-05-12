"use client";

import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  data: Record<string, unknown>[];
}

export function RevenueConcentrationChart({ data }: Props) {
  const values = data
    .map(r => Number(r.ltv_36m) || 0)
    .filter(v => v > 0)
    .sort((a, b) => a - b);

  if (values.length === 0) return null;

  const totalRevenue = values.reduce((s, v) => s + v, 0);
  let cumRevenue = 0;

  const lorenzData = values.map((v, i) => {
    cumRevenue += v;
    return {
      pctCustomers: Math.round(((i + 1) / values.length) * 100),
      pctRevenue:   Math.round((cumRevenue / totalRevenue) * 100),
    };
  });

  // Sample every 1% to reduce data points
  const sampled = lorenzData.filter((_, i) => i % Math.ceil(values.length / 100) === 0);

  // Add equality line reference
  const equalityLine = [
    { pctCustomers: 0, equality: 0 },
    { pctCustomers: 100, equality: 100 },
  ];

  // Find top 20% stats
  const top20idx = Math.floor(values.length * 0.8);
  const top20Revenue = values.slice(top20idx).reduce((s, v) => s + v, 0);
  const top20Pct = ((top20Revenue / totalRevenue) * 100).toFixed(1);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Revenue Concentration (Lorenz Curve)</CardTitle>
        <span className="text-sm font-medium text-blue-600">
          Top 20% of customers → {top20Pct}% of revenue
        </span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={sampled} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="ltvGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="pctCustomers"
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
          />
          <Tooltip
            formatter={(v: number) => [`${v}%`]}
            labelFormatter={(l) => `Bottom ${l}% of customers`}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 100, y: 100 },
            ]}
            stroke="#e2e8f0"
            strokeDasharray="4 4"
            label={{ value: "Perfect equality", fontSize: 10, fill: "#94a3b8" }}
          />
          <Area
            type="monotone"
            dataKey="pctRevenue"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#ltvGrad)"
            name="Revenue %"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}