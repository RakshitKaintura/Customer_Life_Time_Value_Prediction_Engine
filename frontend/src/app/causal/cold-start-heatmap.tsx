"use client";

import { formatCurrency } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";

interface ColdStartRow {
  vertical:          string;
  company_size:      string;
  channel:           string;
  plan_tier:         string;
  ltv_36m_estimate:  number;
  predicted_segment: string;
}

interface Props {
  data: ColdStartRow[];
}

function heatColor(value: number, min: number, max: number): string {
  const pct = max > min ? (value - min) / (max - min) : 0;
  if (pct > 0.75) return "bg-indigo-600 text-white";
  if (pct > 0.50) return "bg-blue-500 text-white";
  if (pct > 0.25) return "bg-cyan-400 text-slate-800";
  return "bg-slate-100 text-slate-600";
}

export function ColdStartHeatmap({ data }: Props) {
  const verticals   = [...new Set(data.map(d => d.vertical))].sort();
  const sizes       = [...new Set(data.map(d => d.company_size))].sort();

  const lookup: Record<string, Record<string, number>> = {};
  data.forEach(row => {
    if (!lookup[row.vertical]) lookup[row.vertical] = {};
    if (!lookup[row.vertical][row.company_size]) {
      lookup[row.vertical][row.company_size] = row.ltv_36m_estimate;
    } else {
      lookup[row.vertical][row.company_size] =
        (lookup[row.vertical][row.company_size] + row.ltv_36m_estimate) / 2;
    }
  });

  const allValues = data.map(d => d.ltv_36m_estimate);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  return (
    <div className="chart-container overflow-x-auto">
      <CardHeader>
        <CardTitle>Cold-Start LTV Prior: Vertical × Company Size</CardTitle>
        <span className="text-xs text-slate-400">
          Average predicted 36m LTV for zero-transaction customers
        </span>
      </CardHeader>
      {verticals.length === 0 ? (
        <p className="text-sm text-slate-400 py-8 text-center">
          Firmographic data not yet available. Run causal ML notebook first.
        </p>
      ) : (
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="pb-3 pr-4 text-left font-medium text-slate-500">Vertical</th>
              {sizes.map(size => (
                <th key={size} className="pb-3 px-2 text-center font-medium text-slate-500 capitalize">
                  {size.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {verticals.map(vertical => (
              <tr key={vertical} className="border-t border-slate-50">
                <td className="py-2 pr-4 capitalize text-slate-700 font-medium">
                  {vertical}
                </td>
                {sizes.map(size => {
                  const val = lookup[vertical]?.[size];
                  return (
                    <td key={size} className="py-2 px-2 text-center">
                      {val != null ? (
                        <div
                          className={`mx-auto w-24 rounded-lg px-2 py-1.5 text-xs font-medium ${heatColor(val, minVal, maxVal)}`}
                        >
                          {formatCurrency(val)}
                        </div>
                      ) : (
                        <div className="mx-auto w-24 rounded-lg px-2 py-1.5 text-xs text-slate-300 bg-slate-50">
                          —
                        </div>
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