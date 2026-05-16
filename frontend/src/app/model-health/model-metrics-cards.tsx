import { StatCard } from "@/components/ui/stat-card";
import { CheckCircle, XCircle } from "lucide-react";

interface Props {
  fusion: Record<string, unknown> | null;
  bgnbd: Record<string, unknown> | null;
  transformer: Record<string, unknown> | null;
}

function MetricTarget({
  value,
  target,
  label,
  higherIsBetter = false,
}: {
  value: number | null;
  target: number;
  label: string;
  higherIsBetter?: boolean;
}) {
  if (value == null) return null;
  const passed = higherIsBetter ? value >= target : value <= target;
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {passed ? <CheckCircle className="h-3.5 w-3.5 text-foreground" /> : <XCircle className="h-3.5 w-3.5 text-muted-foreground" />}
      <span className={passed ? "text-foreground" : "text-muted-foreground"}>
        {label}: {passed ? "pass" : "fail"} target {higherIsBetter ? ">=" : "<="} {target}
      </span>
    </div>
  );
}

export function ModelMetricsCards({ fusion, bgnbd, transformer }: Props) {
  const gini = Number(fusion?.gini_coefficient ?? 0);
  const lift = Number(fusion?.top_decile_lift ?? 0);
  const mae_pct = Number(fusion?.mae_pct_12m ?? 0);
  const calib = Number(fusion?.calibration_error ?? 0);
  const bgnbd_r2 = Number(bgnbd?.r2_frequency ?? 0);
  const _transformerGini = Number(transformer?.gini_coefficient ?? 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <StatCard title="Gini Coefficient" value={gini.toFixed(4)} format="text" subtitle="Target: > 0.65" />
        <StatCard title="Top Decile Lift" value={`${lift.toFixed(2)}x`} format="text" subtitle="Target: > 3.0x" />
        <StatCard title="MAE % of Mean LTV" value={`${(mae_pct * 100).toFixed(1)}%`} format="text" subtitle="Target: < 15%" />
        <StatCard title="Calibration Error" value={calib.toFixed(4)} format="text" subtitle="Target: < 0.10" />
        <StatCard title="BG/NBD R^2" value={bgnbd_r2.toFixed(4)} format="text" subtitle="Target: > 0.85" />
      </div>

      <div className="rounded-lg border border-border bg-card px-5 py-4">
        <p className="mb-3 text-sm font-semibold text-foreground">Model Target Checklist</p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          <MetricTarget value={gini} target={0.65} label="Gini" higherIsBetter />
          <MetricTarget value={lift} target={3.0} label="Top Decile" higherIsBetter />
          <MetricTarget value={mae_pct} target={0.15} label="MAE %" />
          <MetricTarget value={calib} target={0.10} label="Calibration" />
          <MetricTarget value={bgnbd_r2} target={0.85} label="BG/NBD R^2" higherIsBetter />
        </div>
      </div>
    </div>
  );
}
