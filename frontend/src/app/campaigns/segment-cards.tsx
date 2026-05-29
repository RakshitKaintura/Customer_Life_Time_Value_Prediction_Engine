"use client";

import { Crown, TrendingUp, Users, ArrowUpRight } from "lucide-react";
import { CardHeader, CardTitle } from "@/components/ui/card";

interface SegmentStat {
  segment:        string;
  customer_count: number;
  avg_ltv_36m:    number;
}

interface Props {
  segments:       SegmentStat[];
  brevoTemplates: Record<string, number | null>;
}

const SEGMENT_CONFIG: Record<string, {
  label:       string;
  icon:        React.ElementType;
  action:      string;
  description: string;
}> = {
  champions: {
    label:       "Champions",
    icon:        Crown,
    action:      "VIP Retention",
    description: "Highest-value customers. Reward loyalty, offer exclusive benefits, and prevent churn at all costs.",
  },
  high_value: {
    label:       "High Value",
    icon:        TrendingUp,
    action:      "Upsell & Cross-sell",
    description: "Strong buyers with headroom to grow. Target with premium offers and product upgrades.",
  },
  medium_value: {
    label:       "Medium Value",
    icon:        Users,
    action:      "Nurture & Grow",
    description: "Mid-tier buyers showing engagement. Move them up with personalised recommendations.",
  },
  low_value: {
    label:       "Low Value",
    icon:        ArrowUpRight,
    action:      "Re-engagement",
    description: "At-risk or dormant customers. Send win-back campaigns with incentives to reactivate.",
  },
};

const SEGMENT_ORDER = ["champions", "high_value", "medium_value", "low_value"];

export function SegmentCards({ segments, brevoTemplates }: Props) {
  const lookup: Record<string, SegmentStat> = {};
  segments.forEach((s) => { lookup[s.segment] = s; });

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Segment Campaign Targets</CardTitle>
        <span className="text-xs text-muted-foreground">
          Live counts from{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono">final_ltv_scores</code>
          {" "}· Brevo template ID per segment
        </span>
      </CardHeader>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {SEGMENT_ORDER.map((segKey) => {
          const cfg        = SEGMENT_CONFIG[segKey];
          const stat       = lookup[segKey];
          const Icon       = cfg.icon;
          const templateId = brevoTemplates[segKey];

          return (
            <div
              key={segKey}
              className="flex flex-col gap-4 rounded-xl border border-border bg-secondary p-4 transition-shadow hover:shadow-md"
            >
              {/* Icon + label */}
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border bg-card text-foreground">
                  <Icon className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">{cfg.label}</p>
                  <p className="text-xs text-muted-foreground">{cfg.action}</p>
                </div>
              </div>

              {/* Description */}
              <p className="text-xs text-muted-foreground leading-relaxed">
                {cfg.description}
              </p>

              {/* Live stats from DB */}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg border border-border bg-card p-2.5 text-center">
                  <p className="text-lg font-bold text-foreground tabular-nums">
                    {stat ? stat.customer_count.toLocaleString() : "—"}
                  </p>
                  <p className="text-xs text-muted-foreground">customers</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-2.5 text-center">
                  <p className="text-lg font-bold text-foreground tabular-nums">
                    {stat ? `£${Math.round(stat.avg_ltv_36m).toLocaleString()}` : "—"}
                  </p>
                  <p className="text-xs text-muted-foreground">avg LTV 36m</p>
                </div>
              </div>

              {/* Real Brevo template ID from env */}
              <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="h-1.5 w-1.5 rounded-full bg-foreground shrink-0" />
                  <span className="text-xs text-muted-foreground">Brevo template</span>
                </div>
                {templateId != null ? (
                  <span className="font-mono text-xs font-semibold text-foreground">
                    #{templateId}
                  </span>
                ) : (
                  <span className="text-xs text-muted-foreground italic">not set</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
