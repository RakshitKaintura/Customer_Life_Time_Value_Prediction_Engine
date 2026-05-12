import { Topbar } from "@/components/nav/topbar";
import { MarketingROISimulator } from "./marketing-roi-simulator";

export default function SimulatorPage() {
  return (
    <div className="page-container">
      <Topbar
        title="Marketing ROI Simulator"
        subtitle="Project revenue from LTV-informed bidding vs cost-per-click baseline"
      />
      <div className="page-content">
        <MarketingROISimulator />
      </div>
    </div>
  );
}
