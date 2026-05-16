"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { ltvApi } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";

const VERTICALS     = ["healthcare","fintech","ecommerce","saas","manufacturing","retail","education","other"];
const COMPANY_SIZES = ["smb","mid_market","enterprise"];
const CHANNELS      = ["organic","paid_search","paid_social","email","referral","direct"];
const PLAN_TIERS    = ["free","starter","professional","enterprise_trial","enterprise"];

export function ColdStartForm() {
  const [form, setForm] = useState({
    vertical:            "healthcare",
    company_size:        "mid_market",
    acquisition_channel: "paid_search",
    plan_tier:           "professional",
  });
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState<Record<string, unknown> | null>(null);
  const [error,   setError]   = useState<string | null>(null);

  async function handleScore() {
    setLoading(true);
    setError(null);
    try {
      const res = await ltvApi.coldStart(form);
      setResult(res as unknown as Record<string, unknown>);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function sel(field: string) {
    return (e: React.ChangeEvent<HTMLSelectElement>) =>
      setForm(f => ({ ...f, [field]: e.target.value }));
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Industry Vertical",    field: "vertical",            opts: VERTICALS },
          { label: "Company Size",         field: "company_size",        opts: COMPANY_SIZES },
          { label: "Acquisition Channel",  field: "acquisition_channel", opts: CHANNELS },
          { label: "Plan Tier",            field: "plan_tier",           opts: PLAN_TIERS },
        ].map(({ label, field, opts }) => (
          <div key={field}>
            <label className="mb-1.5 block text-sm font-medium text-foreground">
              {label}
            </label>
            <Select
              value={form[field as keyof typeof form]}
              onChange={sel(field)}
            >
              {opts.map(o => (
                <option key={o} value={o}>{o.replace(/_/g, " ")}</option>
              ))}
            </Select>
          </div>
        ))}
      </div>

      <Button
        onClick={handleScore}
        disabled={loading}
      >
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        Estimate LTV
      </Button>

      {error && (
        <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
          {error}
        </div>
      )}

      {result && (
        <Card className="animate-fade-in">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Estimated LTV (36m)</p>
              <p className="text-3xl font-bold text-foreground">
                {formatCurrency(Number(result.ltv_36m ?? 0))}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                CI: [{formatCurrency(Number(result.ci_lower_36m ?? 0))},
                     {formatCurrency(Number(result.ci_upper_36m ?? 0))}]
              </p>
            </div>
            <div className="text-right space-y-2">
              <SegmentBadge segment={String(result.segment ?? "low_value")} />
              <p className="text-sm text-muted-foreground">
                Max CAC: <strong>{formatCurrency(Number(result.recommended_max_cac ?? 0))}</strong>
              </p>
            </div>
          </div>
          <div className="mt-3 border-t border-border pt-3 text-xs text-muted-foreground">
            Match quality: <strong>{String(result.match_quality ?? "—")}</strong>
            {" · "}
            Latency: {Number(result.scoring_latency_ms ?? 0)}ms
          </div>
        </Card>
      )}
    </div>
  );
}
