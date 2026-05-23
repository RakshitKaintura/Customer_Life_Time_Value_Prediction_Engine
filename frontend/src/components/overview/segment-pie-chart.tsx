"use client";

import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { SEGMENT_CONFIG, formatCurrency } from "@/lib/utils";

interface Props {
  data: Record<string, unknown>[];
}

const SEGMENT_ORDER = ["champions", "high_value", "medium_value", "low_value"];

export function SegmentPieChart({ data }: Props) {
  const chartData = SEGMENT_ORDER.map((seg) => {
    const row = data.find((r) => r.segment === seg) as Record<string, unknown> | undefined;
    const cfg = SEGMENT_CONFIG[seg as keyof typeof SEGMENT_CONFIG];
    return {
      name:         cfg.label,
      segment:      seg,
      n_customers:  Number(row?.n_customers ?? 0),
      pct_revenue:  Number(row?.pct_revenue ?? 0),
      avg_ltv:      Number(row?.avg_ltv_36m ?? 0),
      color:        cfg.color,
    };
  }).filter(d => d.n_customers > 0);

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Customer Count by Segment</CardTitle>
        <span className="text-xs text-muted-foreground">LTV 36m segments</span>
      </CardHeader>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={70}
            outerRadius={110}
            paddingAngle={3}
            dataKey="n_customers"
          >
            {chartData.map((entry) => (
              <Cell key={entry.segment} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            wrapperStyle={{ zIndex: 20 }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const entry = payload[0]?.payload as {
                name?: string;
                n_customers?: number;
                avg_ltv?: number;
                pct_revenue?: number;
              };
              return (
                <div
                  className="rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-[0_10px_20px_-18px_rgba(0,0,0,0.45)]"
                  style={{ color: "hsl(var(--foreground))" }}
                >
                  <div className="font-semibold text-foreground">
                    {entry?.name ?? "Segment"}
                  </div>
                  <div className="mt-1 text-muted-foreground">
                    {Number(entry?.n_customers ?? 0).toLocaleString()} customers
                  </div>
                  <div className="text-muted-foreground">
                    Avg LTV: {formatCurrency(Number(entry?.avg_ltv ?? 0))}
                  </div>
                  <div className="text-muted-foreground">
                    Revenue share: {Number(entry?.pct_revenue ?? 0).toFixed(1)}%
                  </div>
                </div>
              );
            }}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
              color: "hsl(var(--foreground))",
            }}
          />
          <Legend
            formatter={(value) => (
              <span style={{ fontSize: 12, color: "hsl(var(--muted-foreground))" }}>{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Revenue share table */}
      <div className="mt-4 space-y-2 border-t border-border pt-4">
        {chartData.map((d) => (
          <div key={d.segment} className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: d.color }}
              />
              <span className="text-muted-foreground">{d.name}</span>
            </div>
            <div className="flex gap-4 text-right">
              <span className="text-muted-foreground">{d.n_customers.toLocaleString()}</span>
              <span className="w-16 font-medium text-foreground">
                {d.pct_revenue.toFixed(1)}% rev
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}