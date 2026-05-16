"use client";

import { formatCurrency } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";

interface ColdStartRow {
  vertical: string;
  company_size: string;
  channel: string;
  plan_tier: string;
  ltv_36m_estimate: number;
  predicted_segment: string;
}

interface Props {
  data: ColdStartRow[];
}

function heatColor(value: number, min: number, max: number): string {
  const pct = max > min ? (value - min) / (max - min) : 0;
  if (pct > 0.75) return "bg-foreground text-background";
  if (pct > 0.5) return "bg-muted text-foreground";
  if (pct > 0.25) return "bg-accent text-foreground";
  return "border border-border bg-card text-muted-foreground";
}

export function ColdStartHeatmap({ data }: Props) {
  const verticals = [...new Set(data.map((d) => d.vertical))].sort();
  const sizes = [...new Set(data.map((d) => d.company_size))].sort();

  const lookup: Record<string, Record<string, number>> = {};
  data.forEach((row) => {
    if (!lookup[row.vertical]) lookup[row.vertical] = {};
    if (!lookup[row.vertical][row.company_size]) {
      lookup[row.vertical][row.company_size] = row.ltv_36m_estimate;
    } else {
      lookup[row.vertical][row.company_size] = (lookup[row.vertical][row.company_size] + row.ltv_36m_estimate) / 2;
    }
  });

  const allValues = data.map((d) => d.ltv_36m_estimate);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  return (
    <div className="chart-container overflow-x-auto">
      <CardHeader>
        <CardTitle>Cold-Start LTV Prior: Vertical x Company Size</CardTitle>
        <span className="text-xs text-muted-foreground">Average predicted 36m LTV for zero-transaction customers</span>
      </CardHeader>
      {verticals.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          Firmographic data not yet available. Run the causal ML notebook first.
        </p>
      ) : (
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="pb-3 pr-4 text-left font-medium text-muted-foreground">Vertical</th>
              {sizes.map((size) => (
                <th key={size} className="px-2 pb-3 text-center font-medium capitalize text-muted-foreground">
                  {size.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {verticals.map((vertical) => (
              <tr key={vertical} className="border-t border-border">
                <td className="py-2 pr-4 font-medium capitalize text-foreground">{vertical}</td>
                {sizes.map((size) => {
                  const val = lookup[vertical]?.[size];
                  return (
                    <td key={size} className="px-2 py-2 text-center">
                      {val != null ? (
                        <div className={`mx-auto w-24 rounded-lg px-2 py-1.5 text-xs font-medium ${heatColor(val, minVal, maxVal)}`}>
                          {formatCurrency(val)}
                        </div>
                      ) : (
                        <div className="mx-auto w-24 rounded-lg border border-border bg-card px-2 py-1.5 text-xs text-muted-foreground">--</div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
