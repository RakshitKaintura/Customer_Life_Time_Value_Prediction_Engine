"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";

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
        <span className="text-xs text-slate-400">New customers per cohort month</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip
            formatter={(v: number) => [v.toLocaleString(), "Customers"]}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="customers" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}