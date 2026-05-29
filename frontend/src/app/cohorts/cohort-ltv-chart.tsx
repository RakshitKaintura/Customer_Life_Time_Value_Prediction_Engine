"use client";

import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency } from "@/lib/utils";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";
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

function toChartValue(value: number | null | undefined) {
  if (value == null) return null;
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

export function CohortLTVChart({ data }: Props) {
  const chartData = data.map(row => ({
    month:          row.cohort_month,
    predictedLTV:   toChartValue(row.avg_predicted_ltv_36m),
    actualLTV12m:   toChartValue(row.avg_actual_ltv_12m),
    customers:      Number(row.customers ?? 0),
    avgOrderValue:  Number(row.avg_order_value ?? 0),
  }));
  const hasPredictedLTV = chartData.some((row) => row.predictedLTV != null);
  const hasActualLTV = chartData.some((row) => row.actualLTV12m != null);

  return (
    <div className="chart-container lg:col-span-2">
      <CardHeader>
        <CardTitle>Average LTV by Acquisition Cohort</CardTitle>
        <span className="text-xs text-muted-foreground">Predicted 36m vs Actual 12m LTV where available</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} />
          <XAxis dataKey="month" tick={chartAxisTick} />
          <YAxis
            tickFormatter={(v) => formatCurrency(v)}
            tick={chartAxisTick}
          />
          <Tooltip
            formatter={(v: number | null, name: string) => [
              v == null ? "Not available" : formatCurrency(v),
              name === "predictedLTV" ? "Predicted LTV 36m" : "Actual LTV 12m",
            ]}
            contentStyle={chartTooltipStyle}
          />
          {(hasPredictedLTV || hasActualLTV) && (
            <Legend
              formatter={(v) =>
                v === "predictedLTV" ? "Predicted LTV 36m" : "Actual LTV 12m"
              }
            />
          )}
          {hasPredictedLTV && (
            <Line
              type="monotone"
              dataKey="predictedLTV"
              stroke="hsl(var(--chart-1))"
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls={false}
              name="predictedLTV"
            />
          )}
          {hasActualLTV && (
            <Line
              type="monotone"
              dataKey="actualLTV12m"
              stroke="hsl(var(--chart-3))"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={{ r: 3 }}
              connectNulls={false}
              name="actualLTV12m"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
