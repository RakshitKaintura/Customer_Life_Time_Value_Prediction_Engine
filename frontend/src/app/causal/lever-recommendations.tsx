import { formatCurrency } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, ArrowRight } from "lucide-react";

interface CausalEffect {
  treatment_name:     string;
  ate:                number;
  ate_lower_ci:       number;
  ate_upper_ci:       number;
  is_significant:     boolean;
  effect_description: string;
}

interface Props {
  data: CausalEffect[];
  fallbackText?: string;
}

export function LeverRecommendations({ data, fallbackText = "Recommendation not yet available for this lever." }: Props) {
  const positive = data
    .filter(d => d.ate > 0 && d.is_significant)
    .sort((a, b) => b.ate - a.ate);

  const formatTreatmentName = (value: string) =>
    value
      .replace(/_/g, " ")
      .replace(/ltv/gi, "LTV")
      .replace(/\b\w/g, (char) => char.toUpperCase());

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Actionable Lever Recommendations</CardTitle>
        <span className="text-xs text-muted-foreground">
          {positive.length} significant positive levers identified
        </span>
      </CardHeader>
      <div className="space-y-3">
        {positive.map((effect, index) => (
          (() => {
            const label = formatTreatmentName(effect.treatment_name);
            const actionText = effect.effect_description?.trim();
            return (
          <div
            key={`${effect.treatment_name}-${index}`}
            className="flex items-start gap-4 rounded-lg border border-border bg-card p-4"
          >
            <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-foreground" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="font-medium text-foreground text-sm">
                  {label}
                </p>
                <span className="text-sm font-bold text-foreground">
                  +{formatCurrency(effect.ate)} avg LTV
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                CI: [{formatCurrency(effect.ate_lower_ci)}, {formatCurrency(effect.ate_upper_ci)}]
              </p>
              {(actionText || !effect.effect_description) && (
                <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <ArrowRight className="h-3 w-3 text-foreground" />
                  {actionText || fallbackText}
                </div>
              )}
            </div>
          </div>
            );
          })()
        ))}
        {positive.length === 0 && (
          <p className="text-sm text-muted-foreground py-4 text-center">
            No significant positive levers found yet. Run the causal ML notebook first.
          </p>
        )}
      </div>
    </div>
  );
}
