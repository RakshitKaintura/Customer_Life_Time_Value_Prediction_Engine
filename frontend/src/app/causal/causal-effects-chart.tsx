"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, CartesianGrid,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";

interface CausalEffect {
  treatment_name: string;
  ate: number;
  ate_lower_ci: number;
  ate_upper_ci: number;
  is_significant: boolean;
  effect_description: string;
}

interface Props {
  data: CausalEffect[];
}

function formatTreatmentName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

export function CausalEffectsChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => b.ate - a.ate);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Average Treatment Effects on LTV (GBP)</CardTitle>
        <Badge variant="info">Double ML</Badge>
      </CardHeader>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 16, left: 140, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} horizontal={false} />
          <XAxis type="number" tickFormatter={(v) => `GBP ${v.toFixed(0)}`} tick={chartAxisTick} />
          <YAxis
            type="category"
            dataKey="treatment_name"
            tickFormatter={formatTreatmentName}
            tick={chartAxisTick}
            width={135}
          />
          <ReferenceLine x={0} stroke={chartGridStroke} />
          <Tooltip
            formatter={(v: number, _: string, p: { payload?: CausalEffect }) => [
              formatCurrency(v),
              p.payload?.is_significant ? "Significant" : "Not significant",
            ]}
            labelFormatter={formatTreatmentName}
            contentStyle={chartTooltipStyle}
          />
          <Bar dataKey="ate" radius={[0, 3, 3, 0]}>
            {sorted.map((entry) => (
              <Cell
                key={entry.treatment_name}
                fill={
                  !entry.is_significant
                    ? "hsl(var(--muted))"
                    : entry.ate > 0
                    ? "hsl(var(--chart-1))"
                    : "hsl(var(--chart-3))"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-3 text-xs text-muted-foreground">
        Causal effect estimated via Double ML. Gray bars are not statistically significant (p &gt;= 0.05).
      </p>
    </div>
  );
}
