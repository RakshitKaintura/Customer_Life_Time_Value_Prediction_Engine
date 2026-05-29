"use client";

import {
  CartesianGrid,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { chartAxisTick, chartGridStroke } from "@/components/ui/chart-theme";
import { formatCurrency } from "@/lib/utils";

interface CausalEffect {
  treatment_name: string;
  ate: number;
  ate_lower_ci: number;
  ate_upper_ci: number;
  ate_pvalue: number;
  is_significant: boolean;
}

interface Props {
  data: CausalEffect[];
}

interface ChartRow extends CausalEffect {
  label: string;
  ciRange: [number, number];
}

interface TooltipPayload {
  payload: ChartRow;
}

function formatTreatmentName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/ltv/gi, "LTV")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function IntervalTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) {
  if (!active || !payload?.length) return null;

  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-lg">
      <p className="mb-1 font-medium text-foreground">{row.label}</p>
      <p className="text-muted-foreground">
        ATE: <span className="text-foreground">{formatCurrency(row.ate)}</span>
      </p>
      <p className="text-muted-foreground">
        95% CI:{" "}
        <span className="text-foreground">
          {formatCurrency(row.ate_lower_ci)} to {formatCurrency(row.ate_upper_ci)}
        </span>
      </p>
      <p className="text-muted-foreground">
        p-value: <span className="text-foreground">{row.ate_pvalue.toFixed(4)}</span>
      </p>
    </div>
  );
}

export function CausalIntervalChart({ data }: Props) {
  const chartData: ChartRow[] = [...data]
    .sort((a, b) => Math.abs(b.ate) - Math.abs(a.ate))
    .map((effect) => ({
      ...effect,
      label: formatTreatmentName(effect.treatment_name),
      ciRange: [effect.ate - effect.ate_lower_ci, effect.ate_upper_ci - effect.ate],
    }));

  return (
    <div className="chart-container">
      <CardHeader className="mb-3">
        <div>
          <CardTitle>Lever Uplift Range</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Dot = estimated effect. Line = 95% confidence interval. Center line = no effect.
          </p>
        </div>
        <Badge variant="default">Causal ATE</Badge>
      </CardHeader>

      {chartData.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">Causal effects are not available yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 12, right: 28, left: 130, bottom: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} horizontal={false} />
            <ReferenceLine x={0} stroke={chartGridStroke} strokeWidth={2} />
            <XAxis
              type="number"
              dataKey="ate"
              tickFormatter={(value: number) => formatCurrency(value)}
              tick={chartAxisTick}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={chartAxisTick}
              width={126}
            />
            <Tooltip content={<IntervalTooltip />} cursor={{ stroke: "hsl(var(--border))" }} />
            <Scatter data={chartData} dataKey="ate" fill="hsl(var(--foreground))">
              <ErrorBar
                dataKey="ciRange"
                direction="x"
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={2}
              />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
