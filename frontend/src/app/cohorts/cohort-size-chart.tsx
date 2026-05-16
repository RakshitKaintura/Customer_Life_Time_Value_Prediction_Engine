"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";

interface Props {
  data: Record<string, unknown>[];
}

export function CohortSizeChart({ data }: Props) {
  const chartData = data.map(row => ({
    month:     String(row.cohort_month ?? ""),
    customers: Number(row.customers ?? 0),
    avgLTV:    Number(row.avg_predicted_ltv_36m ?? 0),
  }));

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Monthly Cohort Acquisition Size</CardTitle>
        <span className="text-xs text-muted-foreground">New customers per cohort month</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} />
          <XAxis dataKey="month" tick={chartAxisTick} />
          <YAxis tick={chartAxisTick} />
          <Tooltip
            formatter={(v: number) => [v.toLocaleString(), "Customers"]}
            contentStyle={chartTooltipStyle}
          />
          <Bar dataKey="customers" fill="hsl(var(--chart-2))" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
