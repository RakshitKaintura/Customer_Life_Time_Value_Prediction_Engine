"use client";

import { CheckCircle, CircleMinus, TrendingDown, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { cn, formatCurrency } from "@/lib/utils";

interface CausalEffect {
  treatment_name: string;
  ate: number;
  ate_lower_ci: number;
  ate_upper_ci: number;
  ate_pvalue: number;
  is_significant: boolean;
  effect_description: string;
}

interface Props {
  data: CausalEffect[];
}

function formatTreatmentName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/ltv/gi, "LTV")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getRecommendation(effect: CausalEffect) {
  if (!effect.is_significant) return "Monitor only";
  if (effect.ate > 0) return "Prioritize";
  return "Avoid as lever";
}

function getRecommendationStyle(effect: CausalEffect) {
  if (!effect.is_significant) return "text-muted-foreground";
  if (effect.ate > 0) return "text-foreground";
  return "text-muted-foreground";
}

export function CausalEffectsChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => {
    if (a.is_significant !== b.is_significant) return a.is_significant ? -1 : 1;
    return Math.abs(b.ate) - Math.abs(a.ate);
  });
  const significantCount = sorted.filter((effect) => effect.is_significant).length;

  return (
    <div className="chart-container">
      <CardHeader>
        <div>
          <CardTitle>Causal Lever Readout</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Latest Double ML run. {significantCount} of {sorted.length} levers are decision-ready.
          </p>
        </div>
        <Badge variant="info">Double ML</Badge>
      </CardHeader>

      <div className="max-h-[500px] overflow-auto rounded-lg border border-border">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 z-10 border-b border-border bg-muted text-xs text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">Lever</th>
              <th className="px-3 py-2 text-right font-medium">ATE</th>
              <th className="px-3 py-2 text-right font-medium">95% CI</th>
              <th className="px-3 py-2 text-right font-medium">p-value</th>
              <th className="px-3 py-2 font-medium">Decision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {sorted.map((effect) => {
              const isPositive = effect.ate > 0;
              const Icon = effect.is_significant
                ? isPositive
                  ? TrendingUp
                  : TrendingDown
                : CircleMinus;

              return (
                <tr key={effect.treatment_name} className="bg-card/40">
                  <td className="px-3 py-3">
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <div>
                        <p className="font-medium text-foreground">{formatTreatmentName(effect.treatment_name)}</p>
                        <p className="mt-0.5 line-clamp-1 text-xs text-muted-foreground">
                          {effect.effect_description || "No lever description available"}
                        </p>
                      </div>
                    </div>
                  </td>
                  <td className={cn("px-3 py-3 text-right font-medium", isPositive ? "text-foreground" : "text-muted-foreground")}>
                    {isPositive ? "+" : ""}
                    {formatCurrency(effect.ate)}
                  </td>
                  <td className="px-3 py-3 text-right text-muted-foreground">
                    {formatCurrency(effect.ate_lower_ci)} to {formatCurrency(effect.ate_upper_ci)}
                  </td>
                  <td className="px-3 py-3 text-right text-muted-foreground">
                    {effect.ate_pvalue.toFixed(4)}
                  </td>
                  <td className={cn("px-3 py-3 font-medium", getRecommendationStyle(effect))}>
                    <span className="inline-flex items-center gap-1.5">
                      {effect.is_significant && <CheckCircle className="h-3.5 w-3.5" />}
                      {getRecommendation(effect)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        Prioritize levers are ready for campaign targeting. Monitor only means the estimated effect is not statistically reliable yet.
      </p>
    </div>
  );
}
