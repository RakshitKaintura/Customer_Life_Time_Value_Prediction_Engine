/**
 * API client for the FastAPI LTV scoring backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface LTVScore {
  customer_id: string;
  ltv_source: string;
  ltv_12m: number;
  ltv_24m: number;
  ltv_36m: number;
  ltv_percentile: number | null;
  segment: string;
  probability_alive_12m: number | null;
  recommended_max_cac: number;
  confidence_interval_36m: [number, number] | null;
  top_ltv_drivers: string[];
  causal_levers: string[];
  lookalike_customer_ids: string[];
  scoring_latency_ms: number | null;
}

export interface ColdStartScore {
  customer_id?: string;
  ltv_source: string;
  ltv_12m: number;
  ltv_36m: number;
  ci_lower_36m: number;
  ci_upper_36m: number;
  segment: string;
  recommended_max_cac: number;
  match_quality: string;
  firmographic_inputs: Record<string, string>;
  scoring_latency_ms: number | null;
}

export interface ModelPerformance {
  fusion_model_version: string | null;
  fusion_mae_ltv_12m: number | null;
  fusion_gini: number | null;
  fusion_top_decile_lift: number | null;
  fusion_calibration_error: number | null;
  bgnbd_r2_frequency: number | null;
  segment_distribution: Record<string, number>;
  n_customers_scored: number | null;
  last_scored_at: string | null;
}

export interface SegmentStat {
  segment: string;
  avg_ltv: number;
  avg_max_cac: number;
  pct_customers: number;
  max_cac_pct: number;
}

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error ${res.status}`);
  }

  return res.json() as Promise<T>;
}

export const ltvApi = {
  scoreCustomer: (customerId: string) =>
    apiFetch<LTVScore>("/score", {
      method: "POST",
      body: JSON.stringify({ customer_id: customerId }),
    }),

  coldStart: (params: {
    vertical: string;
    company_size: string;
    acquisition_channel: string;
    plan_tier: string;
  }) =>
    apiFetch<ColdStartScore>("/cold-start", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getCustomer: (customerId: string) =>
    apiFetch<Record<string, unknown>>(`/customer/${customerId}`),

  getLookalikes: (customerId: string, topN = 10) =>
    apiFetch<{ lookalikes: Record<string, unknown>[] }>(
      `/customer/${customerId}/lookalikes?top_n=${topN}`
    ),

  getSegmentCustomers: (segment: string, page = 1, pageSize = 50) =>
    apiFetch<{ customers: Record<string, unknown>[]; total: number }>(
      `/segment/${segment}?page=${page}&page_size=${pageSize}`
    ),

  getModelPerformance: () =>
    apiFetch<ModelPerformance>("/model-performance"),

  getSegmentStats: () =>
    apiFetch<{ data: SegmentStat[] }>("/segment-stats"),

  health: () => apiFetch<{ status: string }>("/health"),
};