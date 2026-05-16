"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";

interface ShapRow {
  feature_name: string;
  mean_abs_shap: number;
  rank: number;
}

interface Props {
  data: ShapRow[];
}

function formatFeature(name: string): string {
  return name
    .replace(/bgnbd_|transformer_/g, "")
    .replace(/_/g, " ")
    .replace(/ltv/g, "LTV")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

export function ShapImportanceChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap).slice(0, 10);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>SHAP Feature Importance</CardTitle>
        <Badge variant="default">XGBoost Fusion</Badge>
      </CardHeader>
      {sorted.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">SHAP importance data not yet computed.</p>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 16, left: 140, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} horizontal={false} />
            <XAxis type="number" tick={chartAxisTick} />
            <YAxis type="category" dataKey="feature_name" tickFormatter={formatFeature} tick={chartAxisTick} width={135} />
            <Tooltip
              formatter={(v: number) => [v.toFixed(4), "Mean |SHAP|"]}
              labelFormatter={formatFeature}
              contentStyle={chartTooltipStyle}
            />
            <Bar dataKey="mean_abs_shap" fill="hsl(var(--chart-1))" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
