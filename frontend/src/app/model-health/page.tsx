import { Topbar } from "@/components/nav/topbar";
import { getModelPerformanceDB } from "@/lib/supabase/queries";
import { ModelMetricsCards } from "./model-metrics-cards";
import { ModelVersionTable } from "./model-version-table";
import { ShapBarChart } from "./shap-bar-chart";

export const revalidate = 120;

export default async function ModelHealthPage() {
  const { fusion, bgnbd, transformer, shap } = await getModelPerformanceDB();

  return (
    <div className="page-container">
      <Topbar
        title="Model Health"
        subtitle="Live MAE, calibration, Gini, and feature importance metrics"
      />
      <div className="page-content space-y-6">
        <ModelMetricsCards fusion={fusion} bgnbd={bgnbd} transformer={transformer} />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ShapBarChart data={shap} />
          <ModelVersionTable fusion={fusion} bgnbd={bgnbd} transformer={transformer} />
        </div>
      </div>
    </div>
  );
}
