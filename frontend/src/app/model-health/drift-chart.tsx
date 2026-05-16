"use client";

import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";

interface DriftDataPoint {
  date: string;
  psi_score: number;
  alert_type: string;
}

interface Props {
  data: DriftDataPoint[];
}

export function DriftChart({ data }: Props) {
  const hasDrift = data.some((d) => d.psi_score > 0.15);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>LTV Distribution Drift (PSI Over Time)</CardTitle>
        <Badge variant={hasDrift ? "danger" : "success"}>{hasDrift ? "Drift detected" : "Stable"}</Badge>
      </CardHeader>

      {data.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No drift data yet. Drift checks run weekly after model deployment.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} />
            <XAxis dataKey="date" tick={chartAxisTick} />
            <YAxis domain={[0, 0.4]} tick={chartAxisTick} />
            <Tooltip formatter={(v: number) => [v.toFixed(4), "PSI Score"]} contentStyle={chartTooltipStyle} />
            <ReferenceLine
              y={0.1}
              stroke="hsl(var(--chart-3))"
              strokeDasharray="4 4"
              label={{ value: "Warn (0.10)", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <ReferenceLine
              y={0.15}
              stroke="hsl(var(--chart-2))"
              strokeDasharray="4 4"
              label={{ value: "Alert (0.15)", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <Line type="monotone" dataKey="psi_score" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={{ r: 3 }} name="PSI Score" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
