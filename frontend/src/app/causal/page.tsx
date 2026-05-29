import { Topbar } from "@/components/nav/topbar";
import { getCausalEffects, getColdStartSlices, getShapImportance } from "@/lib/supabase/queries";
import { CausalEffectsChart } from "./causal-effects-chart";
import { CausalIntervalChart } from "./causal-interval-chart";
import { ColdStartHeatmap } from "./cold-start-heatmap";
import { ShapImportanceChart } from "./shap-importance-chart";
import { LeverRecommendations } from "./lever-recommendations";

export const revalidate = 600;

export default async function CausalPage() {
  const [effects, coldSlices, shapData] = await Promise.all([
    getCausalEffects(),
    getColdStartSlices(),
    getShapImportance(),
  ]);

  return (
    <div className="page-container">
      <Topbar
        title="Causal Insights"
        subtitle="Features that CAUSE high LTV - Double ML attribution"
      />
      <div className="page-content space-y-6">
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.08fr)_minmax(380px,0.92fr)]">
          <CausalEffectsChart data={effects} />
          <div className="grid gap-6">
            <ShapImportanceChart data={shapData} />
            <CausalIntervalChart data={effects} />
          </div>
        </div>
        <LeverRecommendations
          data={effects}
          fallbackText="Recommendation coming soon. Add a lever description in causal_treatment_effects."
        />
        <ColdStartHeatmap data={coldSlices} />
      </div>
    </div>
  );
}
