"use client";

import { formatCurrency, formatPercent, getSegmentConfig } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { LookalikePanel } from "./lookalike-panel";
import {
  TrendingUp, Shield, DollarSign, Target,
  ChevronRight, AlertCircle,
} from "lucide-react";

interface Props {
  data: Record<string, unknown>;
}

export function CustomerScoreCard({ data }: Props) {
  const isColdStart = data.ltv_source === "firmographic_prior";
  const ltv36m      = Number(data.ltv_36m ?? 0);
  const ltv12m      = Number(data.ltv_12m ?? 0);
  const ci          = data.confidence_interval_36m as [number, number] | null;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900">
            Customer {String(data.customer_id ?? "—")}
          </h2>
          <div className="mt-1 flex items-center gap-2">
            <SegmentBadge segment={String(data.segment ?? "low_value")} />
            {isColdStart && (
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-700">
                <AlertCircle className="h-3 w-3" />
                Cold-start estimate
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold text-slate-900">
            {formatCurrency(ltv36m)}
          </p>
          <p className="text-sm text-slate-500">Predicted LTV (36m)</p>
          {ci && (
            <p className="text-xs text-slate-400">
              CI: [{formatCurrency(ci[0])}, {formatCurrency(ci[1])}]
            </p>
          )}
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card className="text-center py-4">
          <TrendingUp className="mx-auto mb-1 h-5 w-5 text-blue-500" />
          <p className="text-lg font-bold text-slate-900">{formatCurrency(ltv12m)}</p>
          <p className="text-xs text-slate-500">LTV 12m</p>
        </Card>
        <Card className="text-center py-4">
          <Shield className="mx-auto mb-1 h-5 w-5 text-green-500" />
          <p className="text-lg font-bold text-slate-900">
            {formatPercent(Number(data.probability_alive_12m ?? 0))}
          </p>
          <p className="text-xs text-slate-500">P(Alive)</p>
        </Card>
        <Card className="text-center py-4">
          <DollarSign className="mx-auto mb-1 h-5 w-5 text-purple-500" />
          <p className="text-lg font-bold text-slate-900">
            {formatCurrency(Number(data.recommended_max_cac ?? 0))}
          </p>
          <p className="text-xs text-slate-500">Max CAC</p>
        </Card>
        <Card className="text-center py-4">
          <Target className="mx-auto mb-1 h-5 w-5 text-orange-500" />
          <p className="text-lg font-bold text-slate-900">
            {Number(data.ltv_percentile ?? 0)}th
          </p>
          <p className="text-xs text-slate-500">Percentile</p>
        </Card>
      </div>

      {/* LTV Drivers */}
      {Array.isArray(data.top_ltv_drivers) && data.top_ltv_drivers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top LTV Drivers</CardTitle>
          </CardHeader>
          <ul className="space-y-2">
            {(data.top_ltv_drivers as string[]).map((driver, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-blue-500" />
                {driver}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Causal Levers */}
      {Array.isArray(data.causal_levers) && data.causal_levers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Causal Levers</CardTitle>
            <span className="text-xs text-slate-400">Actions that increase this customer's LTV</span>
          </CardHeader>
          <ul className="space-y-2">
            {(data.causal_levers as string[]).map((lever, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-green-500 mt-1.5" />
                <span className="text-slate-700">{lever}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Lookalikes */}
      {!isColdStart && (
        <LookalikePanel
          customerId={String(data.customer_id ?? "")}
          initialLookalikes={data.lookalike_customer_ids as string[]}
        />
      )}
    </div>
  );
}