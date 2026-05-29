"use client";

import { CardHeader, CardTitle } from "@/components/ui/card";

// ─── Types ────────────────────────────────────────────────────────────────────

interface RetentionRow {
  cohort_month:       string;
  months_since_first: number;
  active_customers:   number;
  cohort_n:           number;
  retention_rate_pct: number;
}

interface Props {
  data: Record<string, unknown>[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function hasRetentionFields(row: Record<string, unknown>): row is RetentionRow {
  return (
    row.cohort_month        != null &&
    row.months_since_first  != null &&
    row.retention_rate_pct  != null
  );
}

function addMonths(month: string, offset: number): string {
  const [year, monthIndex] = month.split("-").map(Number);
  const date = new Date(Date.UTC(year, monthIndex - 1 + offset, 1));
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

/**
 * Data-relative heat colour.
 * Maps pct / maxPct → 0..1 through a warm terracotta → amber/gold palette
 * that glows beautifully on dark backgrounds.
 */
function heatStyle(
  pct: number,
  maxPct: number,
): { background: string; color: string; fontWeight: string } {
  if (maxPct === 0) {
    return { background: "transparent", color: "hsl(var(--muted-foreground))", fontWeight: "400" };
  }
  const t = Math.min(pct / maxPct, 1);

  // Hue sweeps: dark terracotta (12°) → bright amber/gold (44°)
  const hue        = Math.round(12  + t * 32);   // 12° → 44°
  const saturation = Math.round(22  + t * 73);   // 22% → 95%
  const lightness  = Math.round(16  + t * 42);   // 16% → 58%
  const alpha      = 0.18 + t * 0.82;

  const textLight = lightness > 36
    ? "hsl(40 30% 98%)"
    : `hsl(${hue} 25% 65%)`;

  return {
    background: `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`,
    color:      textLight,
    fontWeight: t > 0.6 ? "600" : "400",
  };
}

// ─── Component ────────────────────────────────────────────────────────────────

export function CohortRetentionMatrix({ data }: Props) {
  const retentionRows = (data ?? []).filter(hasRetentionFields);

  // ── Empty state ──
  if (retentionRows.length === 0) {
    return (
      <div className="chart-container">
        <CardHeader>
          <CardTitle>Cohort Retention Matrix</CardTitle>
        </CardHeader>
        <p className="py-6 text-sm text-muted-foreground">
          Retention data is not available yet. Run the transaction pipeline or
          add the cohort retention RPC.
        </p>
      </div>
    );
  }

  // ── Build axis arrays ──
  const latestObservedMonth = retentionRows.reduce((latest, row) => {
    const obs = addMonths(String(row.cohort_month), Number(row.months_since_first));
    return obs > latest ? obs : latest;
  }, "");

  const cohorts = [
    ...new Set(retentionRows.map((r) => String(r.cohort_month))),
  ]
    .filter((c) => addMonths(c, 1) <= latestObservedMonth)
    .sort();

  const months = [
    ...new Set(retentionRows.map((r) => Number(r.months_since_first))),
  ]
    .filter((m) => m > 0)
    .sort((a, b) => a - b);

  // ── Lookup tables ──
  const lookup: Record<string, Record<number, number>> = {};
  const cohortSize: Record<string, number>             = {};

  retentionRows.forEach((row) => {
    const c   = String(row.cohort_month);
    const m   = Number(row.months_since_first);
    const pct = Number(row.retention_rate_pct);
    if (!lookup[c]) lookup[c] = {};
    lookup[c][m] = pct;
    if (row.cohort_n) cohortSize[c] = Number(row.cohort_n);
  });

  // ── Global max for relative colour scale ──
  const allPcts = retentionRows.map((r) => Number(r.retention_rate_pct));
  const maxPct  = Math.max(...allPcts);

  // ── Column averages (for summary row) ──
  const colAvg: Record<number, number | null> = {};
  months.forEach((m) => {
    const vals = cohorts
      .map((c) => lookup[c]?.[m])
      .filter((v): v is number => v != null);
    colAvg[m] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  });

  const hasSize = Object.keys(cohortSize).length > 0;

  return (
    <div className="chart-container">
      {/* ── Header ── */}
      <CardHeader>
        <CardTitle>Cohort Retention Matrix</CardTitle>
        <span className="text-xs text-muted-foreground">
          % of each monthly acquisition cohort returning in subsequent months
        </span>
      </CardHeader>

      {/* ── Table ── */}
      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-y-0.5 text-xs">
          {/* ── Column headers ── */}
          <thead>
            <tr>
              <th className="pb-3 pr-3 text-left font-medium text-muted-foreground whitespace-nowrap">
                Cohort
              </th>
              {hasSize && (
                <th className="pb-3 px-2 text-center font-medium text-muted-foreground whitespace-nowrap">
                  N
                </th>
              )}
              {months.map((m) => (
                <th
                  key={m}
                  className="pb-3 px-0.5 text-center font-medium text-muted-foreground whitespace-nowrap"
                >
                  M+{m}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {/* ── Data rows ── */}
            {cohorts.map((cohort) => (
              <tr key={cohort} className="group">
                {/* Cohort label */}
                <td className="py-0.5 pr-3 font-mono text-xs text-muted-foreground whitespace-nowrap group-hover:text-foreground transition-colors">
                  {cohort}
                </td>

                {/* Cohort size */}
                {hasSize && (
                  <td className="py-0.5 px-2 text-center text-xs text-muted-foreground tabular-nums">
                    {cohortSize[cohort] != null
                      ? cohortSize[cohort].toLocaleString()
                      : "—"}
                  </td>
                )}

                {/* Retention cells */}
                {months.map((m) => {
                  const isObservable =
                    addMonths(cohort, m) <= latestObservedMonth;
                  const pct =
                    lookup[cohort]?.[m] ?? (isObservable ? 0 : null);

                  if (pct == null) {
                    // Future — not enough time elapsed
                    return (
                      <td key={m} className="py-0.5 px-0.5 text-center">
                        <div
                          title="Not enough elapsed time yet"
                          className="mx-auto flex h-7 w-10 items-center justify-center text-muted-foreground/20 select-none"
                        >
                          ·
                        </div>
                      </td>
                    );
                  }

                  const style = heatStyle(pct, maxPct);
                  const isMax = pct === maxPct;

                  return (
                    <td key={m} className="py-0.5 px-0.5 text-center">
                      <div
                        title={`${cohort} → M+${m}: ${pct.toFixed(1)}%`}
                        className="mx-auto flex h-7 w-10 items-center justify-center rounded transition-transform hover:scale-110 cursor-default select-none"
                        style={{
                          background: style.background,
                          color:      style.color,
                          fontWeight: style.fontWeight,
                          outline:    isMax ? "1px solid hsl(44 95% 60% / 0.7)" : "none",
                        }}
                      >
                        {pct.toFixed(0)}%
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}

            {/* ── Average row ── */}
            <tr className="border-t border-border">
              <td className="pt-3 pr-3 text-xs font-semibold text-muted-foreground whitespace-nowrap">
                Avg
              </td>
              {hasSize && <td className="pt-3 px-2" />}
              {months.map((m) => {
                const avg = colAvg[m];
                if (avg == null) {
                  return <td key={m} className="pt-3 px-0.5" />;
                }
                const style = heatStyle(avg, maxPct);
                return (
                  <td key={m} className="pt-3 px-0.5 text-center">
                    <div
                      className="mx-auto flex h-7 w-10 items-center justify-center rounded text-xs"
                      style={{
                        background: style.background,
                        color:      style.color,
                        fontWeight: "600",
                        opacity:    0.85,
                      }}
                    >
                      {avg.toFixed(0)}%
                    </div>
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── Legend ── */}
      <div className="mt-5 flex flex-wrap items-center gap-5 text-xs text-muted-foreground">
        {/* Gradient swatch */}
        <div className="flex items-center gap-2">
          <div
            className="h-3 w-24 rounded"
            style={{
              background:
                "linear-gradient(to right, hsla(12,22%,16%,0.2), hsla(44,95%,58%,1))",
            }}
          />
          <span>Low → High return rate</span>
        </div>

        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded"
            style={{
              outline: "1px solid hsl(44 95% 60% / 0.7)",
              background: "transparent",
            }}
          />
          Peak cell
        </span>

        <span className="flex items-center gap-1.5">
          <span className="text-muted-foreground/30 leading-none">·</span>
          Not enough elapsed time
        </span>

        <span className="ml-auto">
          Scale normalised to max observed:{" "}
          <strong className="text-foreground">{maxPct.toFixed(1)}%</strong>
        </span>
      </div>
    </div>
  );
}
