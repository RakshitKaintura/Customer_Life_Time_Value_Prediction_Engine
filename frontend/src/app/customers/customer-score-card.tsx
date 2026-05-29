"use client";

import { formatCurrency, formatPercent } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { LookalikePanel } from "./lookalike-panel";
import { TrendingUp, Shield, DollarSign, Target, ChevronRight, AlertCircle } from "lucide-react";

interface Props {
  data: Record<string, unknown>;
}

export function CustomerScoreCard({ data }: Props) {
  const isColdStart = data.ltv_source === "firmographic_prior";
  const ltv36m = Number(data.ltv_36m ?? 0);
  const ltv12m = Number(data.ltv_12m ?? 0);
  const ci = data.confidence_interval_36m as [number, number] | null;
  const ciText = ci
    ? ci[1] <= 1
      ? `Uncertainty: ${formatPercent(ci[0])}-${formatPercent(ci[1])}`
      : `CI: [${formatCurrency(ci[0])}, ${formatCurrency(ci[1])}]`
    : null;

  return (
    <div className="animate-fade-in space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-foreground">Customer {String(data.customer_id ?? "--")}</h2>
          <div className="mt-1 flex items-center gap-2">
            <SegmentBadge segment={String(data.segment ?? "low_value")} />
            {isColdStart && (
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                <AlertCircle className="h-3 w-3" />
                Cold-start estimate
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold text-foreground">{formatCurrency(ltv36m)}</p>
          <p className="text-sm text-muted-foreground">Predicted LTV (36m)</p>
          {ciText && <p className="text-xs text-muted-foreground">{ciText}</p>}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card className="py-4 text-center">
          <TrendingUp className="mx-auto mb-1 h-5 w-5 text-foreground" />
          <p className="text-lg font-bold text-foreground">{formatCurrency(ltv12m)}</p>
          <p className="text-xs text-muted-foreground">LTV 12m</p>
        </Card>
        <Card className="py-4 text-center">
          <Shield className="mx-auto mb-1 h-5 w-5 text-foreground" />
          <p className="text-lg font-bold text-foreground">{formatPercent(Number(data.probability_alive_12m ?? 0))}</p>
          <p className="text-xs text-muted-foreground">P(Alive)</p>
        </Card>
        <Card className="py-4 text-center">
          <DollarSign className="mx-auto mb-1 h-5 w-5 text-foreground" />
          <p className="text-lg font-bold text-foreground">{formatCurrency(Number(data.recommended_max_cac ?? 0))}</p>
          <p className="text-xs text-muted-foreground">Max CAC</p>
        </Card>
        <Card className="py-4 text-center">
          <Target className="mx-auto mb-1 h-5 w-5 text-foreground" />
          <p className="text-lg font-bold text-foreground">{Number(data.ltv_percentile ?? 0)}th</p>
          <p className="text-xs text-muted-foreground">Percentile</p>
        </Card>
      </div>

      {Array.isArray(data.top_ltv_drivers) && data.top_ltv_drivers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top LTV Drivers</CardTitle>
          </CardHeader>
          <ul className="space-y-2">
            {(data.top_ltv_drivers as string[]).map((driver, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-foreground" />
                {driver}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {Array.isArray(data.causal_levers) && data.causal_levers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Causal Levers</CardTitle>
            <span className="text-xs text-muted-foreground">Actions that increase this customer&apos;s LTV</span>
          </CardHeader>
          <ul className="space-y-2">
            {(data.causal_levers as string[]).map((lever, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-foreground" />
                <span className="text-muted-foreground">{lever}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {!isColdStart && (
        <LookalikePanel customerId={String(data.customer_id ?? "")} initialLookalikes={data.lookalike_customer_ids as string[]} />
      )}
    </div>
  );
}
