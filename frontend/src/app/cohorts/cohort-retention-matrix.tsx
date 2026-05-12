"use client";

import { CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  data: Record<string, unknown>[];
}

function getColor(pct: number): string {
  if (pct >= 80) return "bg-green-600 text-white";
  if (pct >= 60) return "bg-green-400 text-white";
  if (pct >= 40) return "bg-yellow-300 text-slate-800";
  if (pct >= 20) return "bg-orange-300 text-slate-800";
  return "bg-red-200 text-slate-700";
}

export function CohortRetentionMatrix({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="chart-container">
        <CardHeader>
          <CardTitle>Cohort Retention Matrix</CardTitle>
        </CardHeader>
        <p className="text-sm text-slate-400">
          Retention data will appear here after RFM pipeline runs.
        </p>
      </div>
    );
  }

  // Group by cohort_month and months_since_first
  const cohorts = [...new Set(data.map(r => String(r.cohort_month ?? "")))].sort();
  const months  = [...new Set(data.map(r => Number(r.months_since_first ?? 0)))].sort((a, b) => a - b);

  const lookup: Record<string, Record<number, number>> = {};
  data.forEach(row => {
    const cohort = String(row.cohort_month ?? "");
    const month  = Number(row.months_since_first ?? 0);
    const pct    = Number(row.retention_rate_pct ?? 0);
    if (!lookup[cohort]) lookup[cohort] = {};
    lookup[cohort][month] = pct;
  });

  return (
    <div className="chart-container overflow-x-auto">
      <CardHeader>
        <CardTitle>Cohort Retention Matrix</CardTitle>
        <span className="text-xs text-slate-400">
          % of each cohort still purchasing at month N
        </span>
      </CardHeader>
      <table className="min-w-full text-xs">
        <thead>
          <tr>
            <th className="pb-2 pr-3 text-left font-medium text-slate-500 whitespace-nowrap">
              Cohort
            </th>
            {months.map(m => (
              <th key={m} className="pb-2 px-1 text-center font-medium text-slate-500 whitespace-nowrap">
                M+{m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cohorts.map(cohort => (
            <tr key={cohort}>
              <td className="py-1 pr-3 font-mono text-slate-600 whitespace-nowrap">
                {cohort}
              </td>
              {months.map(m => {
                const pct = lookup[cohort]?.[m];
                return (
                  <td key={m} className="py-1 px-0.5 text-center">
                    {pct != null ? (
                      <div
                        className={`mx-auto flex h-7 w-10 items-center justify-center rounded text-xs font-medium ${getColor(pct)}`}
                      >
                        {pct.toFixed(0)}%
                      </div>
                    ) : (
                      <div className="mx-auto flex h-7 w-10 items-center justify-center rounded bg-slate-50 text-slate-300">
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
    </div>
  );
}