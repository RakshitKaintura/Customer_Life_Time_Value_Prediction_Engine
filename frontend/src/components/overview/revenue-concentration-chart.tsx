"use client";

import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { chartAxisTick, chartTooltipStyle } from "@/components/ui/chart-theme";

interface Props {
  data: Record<string, unknown>[];
}

export function RevenueConcentrationChart({ data }: Props) {
  const values = data
    .map((r) => Number(r.ltv_36m) || 0)
    .filter((v) => v > 0)
    .sort((a, b) => a - b);

  if (values.length === 0) return null;

  const totalRevenue = values.reduce((s, v) => s + v, 0);
  const lorenzData = values.reduce<{ pctCustomers: number; pctRevenue: number; cum: number }[]>((acc, v, i) => {
    const prevCum = i === 0 ? 0 : acc[i - 1].cum;
    const cum = prevCum + v;
    acc.push({
      pctCustomers: Math.round(((i + 1) / values.length) * 100),
      pctRevenue: Math.round((cum / totalRevenue) * 100),
      cum,
    });
    return acc;
  }, []);

  const sampled = lorenzData.filter((_, i) => i % Math.ceil(values.length / 100) === 0);

  const top20idx = Math.floor(values.length * 0.8);
  const top20Revenue = values.slice(top20idx).reduce((s, v) => s + v, 0);
  const top20Pct = ((top20Revenue / totalRevenue) * 100).toFixed(1);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Revenue Concentration (Lorenz Curve)</CardTitle>
        <span className="text-sm font-medium text-foreground">Top 20% of customers -&gt; {top20Pct}% of revenue</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={sampled} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="ltvGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.18} />
              <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <XAxis dataKey="pctCustomers" tickFormatter={(v) => `${v}%`} tick={chartAxisTick} />
          <YAxis tickFormatter={(v) => `${v}%`} tick={chartAxisTick} />
          <Tooltip
            formatter={(v: number) => [`${v}%`]}
            labelFormatter={(l) => `Bottom ${l}% of customers`}
            contentStyle={chartTooltipStyle}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 100, y: 100 },
            ]}
            stroke="hsl(var(--border))"
            strokeDasharray="4 4"
            label={{ value: "Perfect equality", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <Area
            type="monotone"
            dataKey="pctRevenue"
            stroke="hsl(var(--chart-2))"
            strokeWidth={2}
            fill="url(#ltvGrad)"
            name="Revenue %"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
