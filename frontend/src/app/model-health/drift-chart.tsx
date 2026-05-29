"use client";

import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, ReferenceArea,
} from "recharts";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";
import { Activity } from "lucide-react";

// ─── Data shape ───────────────────────────────────────────────────────────────

interface DriftDataPoint {
  date:           string;
  psi_score:      number;
  alert_type:     string;
  status?:        string;
  mean_shift_pct?: number | null;
}

interface Props {
  data: DriftDataPoint[];
}

// ─── PSI severity helpers ─────────────────────────────────────────────────────

// Thresholds match backend: PSI_THRESHOLD_WARN = 0.10, PSI_THRESHOLD_ALERT = 0.15
// Industry bands: < 0.10 stable | 0.10–0.25 moderate | > 0.25 major
const WARN  = 0.10;
const ALERT = 0.15;

type Zone = { label: string; color: string };

function getPsiZone(psi: number): Zone {
  if (psi < WARN)  return { label: "Stable",  color: "hsl(142 71% 45%)" };
  if (psi < ALERT) return { label: "Warning", color: "hsl(45 93% 47%)"  };
  if (psi < 0.25)  return { label: "Alert",   color: "hsl(22 90% 55%)"  };
  return              { label: "Critical", color: "hsl(0 84% 60%)"   };
}

// ─── Custom dot — coloured by zone ───────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomDot(props: any) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const { color } = getPsiZone(payload.psi_score as number);
  return (
    <circle
      cx={cx} cy={cy} r={5}
      fill={color}
      stroke="hsl(var(--card))"
      strokeWidth={2}
    />
  );
}

// ─── Rich tooltip ─────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function DriftTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as DriftDataPoint;
  const zone = getPsiZone(d.psi_score);

  return (
    <div
      style={{
        ...chartTooltipStyle,
        padding: "10px 14px",
        minWidth: 190,
      }}
    >
      <p
        className="text-xs font-semibold mb-2"
        style={{ color: "hsl(var(--foreground))" }}
      >
        {d.date}
      </p>

      <Row label="PSI Score"  value={d.psi_score.toFixed(4)}                         color={zone.color} />
      <Row label="Zone"       value={zone.label}                                       color={zone.color} />
      {d.mean_shift_pct != null && (
        <Row label="Mean Shift" value={`${d.mean_shift_pct.toFixed(1)}%`} />
      )}
      <Row
        label="Status"
        value={d.status ?? "—"}
        color={d.status === "open" ? zone.color : undefined}
      />
    </div>
  );
}

function Row({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-6 mb-1 last:mb-0">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span
        className="text-xs font-semibold capitalize"
        style={{ color: color ?? "hsl(var(--foreground))" }}
      >
        {value}
      </span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function DriftChart({ data }: Props) {
  const hasDrift = data.some((d) => d.psi_score > ALERT);

  // Filter out corrupted / test rows entirely (real PSI is always < 10)
  const cleanData = data.filter(
    (d) => isFinite(d.psi_score) && d.psi_score >= 0 && d.psi_score < 10
  );

  const latest     = cleanData.at(-1);
  const latestZone = latest ? getPsiZone(latest.psi_score) : null;

  // Y-axis: scale to max observed PSI, padded 25%, minimum 0.35, hard cap 1.0
  const rawMax  = cleanData.length > 0
    ? Math.max(...cleanData.map((d) => d.psi_score))
    : 0;
  const yMax    = Math.min(Math.max(rawMax * 1.25, 0.35), 1.0);
  const yTicks  = [0, 0.10, 0.15, 0.25, +(yMax / 2).toFixed(2), +yMax.toFixed(2)];
  const yDomain: [number, number] = [0, yMax];

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>LTV Distribution Drift (PSI Over Time)</CardTitle>
        <Badge variant={hasDrift ? "danger" : "success"}>
          {hasDrift ? "Drift detected" : "Stable"}
        </Badge>
      </CardHeader>

      {/* ── Empty state ── */}
      {data.length === 0 ? (
        <div className="py-10 text-center space-y-2">
          <Activity className="mx-auto h-8 w-8 text-muted-foreground opacity-40" />
          <p className="text-sm font-medium text-muted-foreground">
            No drift events recorded yet
          </p>
          <p className="text-xs text-muted-foreground opacity-60 max-w-xs mx-auto">
            Drift checks run automatically after each model scoring run.
            The chart will populate once enough scoring data accumulates.
          </p>
        </div>
      ) : (
        <>
          {/* ── Chart ── */}
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={cleanData} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="psiAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="hsl(var(--chart-1))" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.02} />
                </linearGradient>
              </defs>

              {/* ── Zone background bands ── */}
              <ReferenceArea
                y1={0} y2={WARN}
                fill="rgba(34,197,94,0.07)"
                strokeOpacity={0}
              />
              <ReferenceArea
                y1={WARN} y2={ALERT}
                fill="rgba(234,179,8,0.08)"
                strokeOpacity={0}
              />
              <ReferenceArea
                y1={ALERT} y2={2}
                fill="rgba(239,68,68,0.07)"
                strokeOpacity={0}
                ifOverflow="hidden"
              />

              <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} />
              <XAxis dataKey="date" tick={chartAxisTick} />
              <YAxis
                type="number"
                domain={yDomain}
                ticks={yTicks}
                tick={chartAxisTick}
                allowDataOverflow
              />

              <Tooltip content={<DriftTooltip />} />

              {/* ── Threshold reference lines ── */}
              <ReferenceLine
                y={WARN}
                stroke="hsl(45 93% 47%)"
                strokeDasharray="4 4"
                strokeOpacity={0.8}
                label={{
                  value: "Warn (0.10)",
                  fontSize: 9,
                  fill: "hsl(45 93% 47%)",
                  position: "insideTopRight",
                }}
              />
              <ReferenceLine
                y={ALERT}
                stroke="hsl(0 84% 60%)"
                strokeDasharray="4 4"
                strokeOpacity={0.8}
                label={{
                  value: "Alert (0.15)",
                  fontSize: 9,
                  fill: "hsl(0 84% 60%)",
                  position: "insideTopRight",
                }}
              />

              {/* ── Data series ── */}
              <Area
                type="monotone"
                dataKey="psi_score"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                fill="url(#psiAreaGradient)"
                dot={<CustomDot />}
                activeDot={false}
                name="PSI Score"
              />
            </AreaChart>
          </ResponsiveContainer>

          {/* ── Summary stat row ── */}
          {latest && latestZone && (
            <div className="mt-3 grid grid-cols-3 divide-x divide-border border-t border-border pt-3">
              <StatCell
                label="Latest PSI"
                value={latest.psi_score.toFixed(4)}
                sub={latestZone.label}
                color={latestZone.color}
                mono
              />
              <StatCell
                label="Mean Shift"
                value={
                  latest.mean_shift_pct != null
                    ? `${latest.mean_shift_pct.toFixed(1)}%`
                    : "—"
                }
                sub="vs baseline"
              />
              <StatCell
                label="Last Event"
                value={latest.date}
                sub={latest.status ?? "—"}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Stat cell helper ─────────────────────────────────────────────────────────

function StatCell({
  label,
  value,
  sub,
  color,
  mono = false,
}: {
  label: string;
  value: string;
  sub:   string;
  color?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-0.5 px-3 py-1 text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={`text-sm font-bold ${mono ? "font-mono" : ""}`}
        style={{ color: color ?? "hsl(var(--foreground))" }}
      >
        {value}
      </p>
      <p className="text-xs capitalize text-muted-foreground">{sub}</p>
    </div>
  );
}
