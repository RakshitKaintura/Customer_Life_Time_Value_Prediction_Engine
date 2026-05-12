import { cn, formatCurrency, formatNumber } from "@/lib/utils";
import { Card } from "./card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: number;        // percent change
  format?: "currency" | "number" | "text" | "percent";
  icon?: React.ReactNode;
  className?: string;
  loading?: boolean;
}

export function StatCard({
  title,
  value,
  subtitle,
  trend,
  format = "text",
  icon,
  className,
  loading = false,
}: StatCardProps) {
  const formatted =
    loading
      ? "—"
      : format === "currency"
      ? formatCurrency(Number(value))
      : format === "number"
      ? formatNumber(Number(value))
      : format === "percent"
      ? `${Number(value).toFixed(1)}%`
      : String(value);

  const trendPositive = trend !== undefined && trend > 0;
  const trendNegative = trend !== undefined && trend < 0;

  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-500">{title}</p>
          {loading ? (
            <div className="mt-2 h-8 w-24 animate-pulse rounded bg-slate-200" />
          ) : (
            <p className="mt-1 text-2xl font-bold text-slate-900">{formatted}</p>
          )}
          {subtitle && (
            <p className="mt-1 text-xs text-slate-400">{subtitle}</p>
          )}
          {trend !== undefined && (
            <div
              className={cn(
                "mt-2 flex items-center gap-1 text-xs font-medium",
                trendPositive && "text-green-600",
                trendNegative && "text-red-500",
                !trendPositive && !trendNegative && "text-slate-400"
              )}
            >
              {trendPositive && <TrendingUp className="h-3 w-3" />}
              {trendNegative && <TrendingDown className="h-3 w-3" />}
              {!trendPositive && !trendNegative && <Minus className="h-3 w-3" />}
              <span>
                {trend > 0 ? "+" : ""}
                {trend.toFixed(1)}% vs last period
              </span>
            </div>
          )}
        </div>
        {icon && (
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
}