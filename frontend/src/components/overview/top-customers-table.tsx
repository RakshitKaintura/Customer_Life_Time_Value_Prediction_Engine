import { formatCurrency } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { ExternalLink } from "lucide-react";

interface Props {
  data: Record<string, unknown>[];
}

export function TopCustomersTable({ data }: Props) {
  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Top 10 Customers by Predicted LTV (36m)</CardTitle>
        <Link
          href="/customers"
          className="flex items-center gap-1 text-xs text-blue-600 hover:underline"
        >
          View all <ExternalLink className="h-3 w-3" />
        </Link>
      </CardHeader>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 text-left">
              <th className="pb-3 pr-4 font-medium text-slate-500">Customer ID</th>
              <th className="pb-3 pr-4 font-medium text-slate-500">LTV 12m</th>
              <th className="pb-3 pr-4 font-medium text-slate-500">LTV 36m</th>
              <th className="pb-3 pr-4 font-medium text-slate-500">Segment</th>
              <th className="pb-3 font-medium text-slate-500">Percentile</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={String(row.customer_id ?? i)}
                className="border-b border-slate-50 hover:bg-slate-50 transition-colors"
              >
                <td className="py-3 pr-4">
                  <Link
                    href={`/customers/${row.customer_id}`}
                    className="font-mono text-blue-600 hover:underline"
                  >
                    {String(row.customer_id ?? "—")}
                  </Link>
                </td>
                <td className="py-3 pr-4 font-medium text-slate-900">
                  {formatCurrency(Number(row.ltv_12m))}
                </td>
                <td className="py-3 pr-4 font-bold text-slate-900">
                  {formatCurrency(Number(row.ltv_36m))}
                </td>
                <td className="py-3 pr-4">
                  <SegmentBadge segment={String(row.segment ?? "low_value")} size="sm" />
                </td>
                <td className="py-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-blue-500"
                        style={{ width: `${Number(row.ltv_percentile ?? 0)}%` }}
                      />
                    </div>
                    <span className="text-slate-500">
                      {Number(row.ltv_percentile ?? 0)}th
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}