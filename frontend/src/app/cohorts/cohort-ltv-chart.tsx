"use client";

import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency } from "@/lib/utils";

interface CohortRow {
  cohort_month: string;
  customers?: number;
  avg_predicted_ltv_36m?: number;
  avg_actual_ltv_12m?: number;
  avg_frequency?: number;
  avg_order_value?: number;
}

interface Props {
  data: CohortRow[];
}

const COLORS = [
  "#6366f1", "#3b82f6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
];

export function CohortLTVChart({ data }: Props) {
  const chartData = data.map(row => ({
    month:          row.cohort_month,
    predictedLTV:   Number(row.avg_predicted_ltv_36m ?? 0),
    actualLTV12m:   Number(row.avg_actual_ltv_12m ?? 0),
    customers:      Number(row.customers ?? 0),
    avgOrderValue:  Number(row.avg_order_value ?? 0),
  }));

  return (
    <div className="chart-container lg:col-span-2">
      <CardHeader>
        <CardTitle>Average LTV by Acquisition Cohort</CardTitle>
        <span className="text-xs text-slate-400">Predicted 36m vs Actual 12m LTV</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <YAxis
            tickFormatter={(v) => formatCurrency(v)}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
          />
          <Tooltip
            formatter={(v: number, name: string) => [
              formatCurrency(v),
              name === "predictedLTV" ? "Predicted LTV 36m" : "Actual LTV 12m",
            ]}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Legend
            formatter={(v) =>
              v === "predictedLTV" ? "Predicted LTV 36m" : "Actual LTV 12m"
            }
          />
          <Line
            type="monotone"
            dataKey="predictedLTV"
            stroke="#6366f1"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="predictedLTV"
          />
          <Line
            type="monotone"
            dataKey="actualLTV12m"
            stroke="#10b981"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ r: 3 }}
            name="actualLTV12m"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}