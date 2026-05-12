"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { ltvApi } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { Card } from "@/components/ui/card";

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
            <label className="mb-1.5 block text-sm font-medium text-slate-700">
              {label}
            </label>
            <select
              value={form[field as keyof typeof form]}
              onChange={sel(field)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {opts.map(o => (
                <option key={o} value={o}>{o.replace(/_/g, " ")}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <button
        onClick={handleScore}
        disabled={loading}
        className="flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
      >
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        Estimate LTV
      </button>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <Card className="animate-fade-in">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-500">Estimated LTV (36m)</p>
              <p className="text-3xl font-bold text-slate-900">
                {formatCurrency(Number(result.ltv_36m ?? 0))}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                CI: [{formatCurrency(Number(result.ci_lower_36m ?? 0))},
                     {formatCurrency(Number(result.ci_upper_36m ?? 0))}]
              </p>
            </div>
            <div className="text-right space-y-2">
              <SegmentBadge segment={String(result.segment ?? "low_value")} />
              <p className="text-sm text-slate-600">
                Max CAC: <strong>{formatCurrency(Number(result.recommended_max_cac ?? 0))}</strong>
              </p>
            </div>
          </div>
          <div className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-400">
            Match quality: <strong>{String(result.match_quality ?? "—")}</strong>
            {" · "}
            Latency: {Number(result.scoring_latency_ms ?? 0)}ms
          </div>
        </Card>
      )}
    </div>
  );
}