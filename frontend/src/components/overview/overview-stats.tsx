import { StatCard } from "@/components/ui/stat-card";
import { formatCurrency, formatNumber } from "@/lib/utils";
import { Users, TrendingUp, DollarSign, Target } from "lucide-react";

interface Props {
  segmentData: Record<string, unknown>[];
  scoreData:   Record<string, unknown>[];
  totalCustomers: number;
}

export function OverviewStats({ segmentData, scoreData, totalCustomers }: Props) {
  const totalLTV36m = scoreData.reduce(
    (sum, r) => sum + (Number(r.ltv_36m) || 0), 0
  );
  const avgLTV36m   = scoreData.length > 0 ? totalLTV36m / scoreData.length : 0;
  const champCount  = scoreData.filter(r => r.segment === "champions").length;
  const highCount   = scoreData.filter(r => r.segment === "high_value").length;

  const champRevenue = scoreData
    .filter(r => r.segment === "champions")
    .reduce((s, r) => s + (Number(r.ltv_36m) || 0), 0);
  const champRevPct = totalLTV36m > 0 ? (champRevenue / totalLTV36m * 100) : 0;

  return (
    <div className="grid-cols-stats">
      <StatCard
        title="Total Customers Scored"
        value={scoreData.length}
        format="number"
        subtitle={`of ${formatNumber(totalCustomers)} total customers`}
        icon={<Users className="h-5 w-5" />}
      />
      <StatCard
        title="Total Predicted LTV (36m)"
        value={totalLTV36m}
        format="currency"
        subtitle="Sum of all 36-month predictions"
        icon={<DollarSign className="h-5 w-5" />}
      />
      <StatCard
        title="Average LTV per Customer (36m)"
        value={avgLTV36m}
        format="currency"
        subtitle="Across all scored customers"
        icon={<TrendingUp className="h-5 w-5" />}
      />
      <StatCard
        title="Champions + High Value"
        value={`${champCount + highCount}`}
        format="number"
        subtitle={`Generate ${champRevPct.toFixed(0)}% of predicted revenue`}
        icon={<Target className="h-5 w-5" />}
      />
    </div>
  );
}