import { formatCurrency } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, XCircle, ArrowRight } from "lucide-react";

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
}

export function LeverRecommendations({ data }: Props) {
  const positive = data
    .filter(d => d.ate > 0 && d.is_significant)
    .sort((a, b) => b.ate - a.ate);

  const actions: Record<string, string> = {
    onboarding_completed:      "Send personalised onboarding email sequence within 24h of signup",
    high_value_first_purchase: "Offer premium product bundles at checkout to new customers",
    multi_category_buyer:      "Cross-sell into adjacent categories after first purchase",
    fast_repeat_buyer:         "Trigger win-back campaigns within 14 days of first purchase",
    high_frequency:            "Enrol into VIP programme after 3rd purchase",
    international_buyer:       "Expand localised marketing to international markets",
  };

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Actionable Lever Recommendations</CardTitle>
        <span className="text-xs text-slate-400">
          {positive.length} significant positive levers identified
        </span>
      </CardHeader>
      <div className="space-y-3">
        {positive.map((effect) => (
          <div
            key={effect.treatment_name}
            className="flex items-start gap-4 rounded-lg border border-green-100 bg-green-50 p-4"
          >
            <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="font-medium text-slate-900 text-sm">
                  {effect.effect_description}
                </p>
                <span className="text-sm font-bold text-green-700">
                  +{formatCurrency(effect.ate)} avg LTV
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                CI: [{formatCurrency(effect.ate_lower_ci)}, {formatCurrency(effect.ate_upper_ci)}]
              </p>
              {actions[effect.treatment_name] && (
                <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-600">
                  <ArrowRight className="h-3 w-3 text-blue-500" />
                  {actions[effect.treatment_name]}
                </div>
              )}
            </div>
          </div>
        ))}
        {positive.length === 0 && (
          <p className="text-sm text-slate-400 py-4 text-center">
            No significant positive levers found yet. Run the causal ML notebook first.
          </p>
        )}
      </div>
    </div>
  );
}