import { Topbar } from "@/components/nav/topbar";
import { getCohortData } from "@/lib/supabase/queries";
import { CohortRetentionMatrix } from "./cohort-retention-matrix";
import { CohortLTVChart } from "./cohort-ltv-chart";
import { CohortSizeChart } from "./cohort-size-chart";

export const revalidate = 300;

export default async function CohortsPage() {
  const cohortData = await getCohortData();

  return (
    <div className="page-container">
      <Topbar
        title="Cohort Analysis"
        subtitle="Retention curves and LTV development by monthly acquisition cohort"
      />
      <div className="page-content space-y-6">
        <CohortSizeChart data={cohortData} />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <CohortLTVChart data={cohortData} />
        </div>
        <CohortRetentionMatrix data={cohortData} />
      </div>
    </div>
  );
}
