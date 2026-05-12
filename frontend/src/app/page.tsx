import { Topbar } from "@/components/nav/topbar";
import { getOverviewStats } from "@/lib/supabase/queries";
import { OverviewStats } from "@/components/overview/overview-stats";
import { LTVDistributionChart } from "@/components/overview/ltv-distribution-chart";
import { SegmentPieChart } from "@/components/overview/segment-pie-chart";
import { RevenueConcentrationChart } from "@/components/overview/revenue-concentration-chart";
import { TopCustomersTable } from "@/components/overview/top-customers-table";

export const revalidate = 300; // Revalidate every 5 minutes

export default async function OverviewPage() {
  const { segmentData, scoreData, totalCustomers } = await getOverviewStats();

  return (
    <div className="page-container">
      <Topbar
        title="LTV Overview"
        subtitle="Predicted revenue distribution and customer value concentration"
      />
      <div className="page-content space-y-6">
        {/* KPI cards */}
        <OverviewStats
          segmentData={segmentData}
          scoreData={scoreData}
          totalCustomers={totalCustomers as number}
        />

        {/* Charts row */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <LTVDistributionChart data={scoreData} />
          <SegmentPieChart data={segmentData} />
        </div>

        {/* Revenue concentration */}
        <RevenueConcentrationChart data={scoreData} />

        {/* Top customers */}
        <TopCustomersTable data={scoreData.slice(0, 10)} />
      </div>
    </div>
  );
}