import { Topbar } from "@/components/nav/topbar";
import { getModelPerformanceDB } from "@/lib/supabase/queries";
import {
  getDriftHistory,
  getRetrainingLog,
  getOpenDriftAlerts,
} from "@/lib/supabase/monitoring-queries";
import { ModelMetricsCards } from "./model-metrics-cards";
import { ModelVersionTable } from "./model-version-table";
import { ShapBarChart } from "./shap-bar-chart";
import { DriftChart } from "./drift-chart";
import { RetrainingLog } from "./retraining-log";
import { AlertTriangle } from "lucide-react";

export const revalidate = 60;

export default async function ModelHealthPage() {
  const [
    { fusion, bgnbd, transformer, shap },
    driftHistory,
    retrainingRuns,
    openAlerts,
  ] = await Promise.all([
    getModelPerformanceDB(),
    getDriftHistory(30),
    getRetrainingLog(10),
    getOpenDriftAlerts(),
  ]);

  return (
    <div className="page-container">
      <Topbar
        title="Model Health"
        subtitle="Live MAE, calibration, Gini, drift monitoring, and retraining history"
      />
      <div className="page-content space-y-6">
        {/* Open drift alerts banner */}
        {openAlerts.length > 0 && (
          <div className="flex items-start gap-3 rounded-lg border border-border bg-card px-4 py-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-foreground" />
            <div>
              <p className="text-sm font-medium text-foreground">
                {openAlerts.length} open drift alert{openAlerts.length > 1 ? "s" : ""}
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                PSI score exceeded threshold. Model retraining may be required.
                Monthly retraining is scheduled automatically.
              </p>
            </div>
          </div>
        )}

        {/* Model metrics KPI cards */}
        <ModelMetricsCards fusion={fusion} bgnbd={bgnbd} transformer={transformer} />

        {/* Charts row */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ShapBarChart data={shap} />
          <ModelVersionTable fusion={fusion} bgnbd={bgnbd} transformer={transformer} />
        </div>

        {/* Drift monitoring */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <DriftChart data={driftHistory} />
          <RetrainingLog data={retrainingRuns as Parameters<typeof RetrainingLog>[0]["data"]} />
        </div>
      </div>
    </div>
  );
}
