"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";

interface ShapRow {
  feature_name:  string;
  mean_abs_shap: number;
  rank:          number;
}

interface Props {
  data: ShapRow[];
}

function fmt(name: string): string {
  return name
    .replace(/bgnbd_|transformer_/g, "")
    .replace(/_/g, " ")
    .replace(/ltv/gi, "LTV")
    .replace(/\b\w/g, l => l.toUpperCase());
}

export function ShapBarChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Global SHAP Feature Importance</CardTitle>
        <span className="text-xs text-slate-400">Mean |SHAP value| for LTV 12m model</span>
      </CardHeader>
      {sorted.length === 0 ? (
        <p className="py-8 text-center text-sm text-slate-400">
          SHAP data not yet computed. Run fusion notebook with shap installed.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={sorted}
            layout="vertical"
            margin={{ top: 4, right: 8, left: 130, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <YAxis
              type="category"
              dataKey="feature_name"
              tickFormatter={fmt}
              tick={{ fontSize: 10, fill: "#475569" }}
              width={125}
            />
            <Tooltip
              formatter={(v: number) => [v.toFixed(4), "Mean |SHAP|"]}
              labelFormatter={fmt}
              contentStyle={{ fontSize: 12, borderRadius: 8 }}
            />
            <Bar dataKey="mean_abs_shap" fill="#6366f1" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}