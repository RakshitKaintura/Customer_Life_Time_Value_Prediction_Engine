import { formatCurrency } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Table, THead, TBody, TH, TD } from "@/components/ui/table";
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
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          View all <ExternalLink className="h-3 w-3" />
        </Link>
      </CardHeader>
      <div className="overflow-x-auto">
        <Table>
          <THead>
            <tr>
              <TH className="pl-0">Customer ID</TH>
              <TH>LTV 12m</TH>
              <TH>LTV 36m</TH>
              <TH>Segment</TH>
              <TH>Percentile</TH>
            </tr>
          </THead>
          <TBody>
            {data.map((row, i) => (
              <tr
                key={`${String(row.customer_id ?? "unknown")}-${i}`}
                className="hover:bg-accent/40"
              >
                <TD className="pl-0">
                  <Link
                    href={`/customers/${row.customer_id}`}
                    className="font-mono text-sm text-foreground hover:underline"
                  >
                    {String(row.customer_id ?? "—")}
                  </Link>
                </TD>
                <TD className="font-medium text-foreground">
                  {formatCurrency(Number(row.ltv_12m))}
                </TD>
                <TD className="font-semibold text-foreground">
                  {formatCurrency(Number(row.ltv_36m))}
                </TD>
                <TD>
                  <SegmentBadge segment={String(row.segment ?? "low_value")} size="sm" />
                </TD>
                <TD>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-foreground"
                        style={{ width: `${Number(row.ltv_percentile ?? 0)}%` }}
                      />
                    </div>
                    <span className="text-muted-foreground">
                      {Number(row.ltv_percentile ?? 0)}th
                    </span>
                  </div>
                </TD>
              </tr>
            ))}
          </TBody>
        </Table>
      </div>
    </div>
  );
}