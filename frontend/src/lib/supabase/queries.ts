/**
 * Supabase query helpers used by Server Components.
 */

import { createServerSupabaseClient } from "./server";

export async function getOverviewStats() {
  const supabase = await createServerSupabaseClient();

  const { data: segmentData } = await supabase
    .from("v_segment_revenue_concentration")
    .select("*");

  const { data: scoreData } = await supabase
    .from("final_ltv_scores")
    .select("customer_id, ltv_36m, ltv_12m, segment, ltv_percentile")
    .order("ltv_36m", { ascending: false })
    .limit(5000);

  const { count: customerCount } = await supabase
    .from("customers")
    .select("customer_id", { count: "exact", head: true });

  return {
    segmentData:    segmentData    ?? [],
    scoreData:      scoreData      ?? [],
    totalCustomers: customerCount  ?? 0,
  };
}

export async function getCohortData() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("v_cohort_ltv")
    .select("*")
    .order("cohort_month");

  return data ?? [];
}

export async function getCohortRetention() {
  const supabase = await createServerSupabaseClient();

  const { data, error } = await supabase.rpc("get_cohort_retention_matrix");
  if (error) return [];
  return data ?? [];
}

export async function getCausalEffects() {
  const supabase = await createServerSupabaseClient();

  const { data: latest } = await supabase
    .from("causal_treatment_effects")
    .select("model_version")
    .order("computed_at", { ascending: false })
    .limit(1)
    .single();

  if (!latest?.model_version) return [];

  const { data } = await supabase
    .from("causal_treatment_effects")
    .select("*")
    .eq("model_version", latest.model_version)
    .order("ate", { ascending: false });

  return (data ?? []).map((row) => ({
    ...row,
    ate:          Number(row.ate          ?? 0),
    ate_lower_ci: Number(row.ate_lower_ci ?? 0),
    ate_upper_ci: Number(row.ate_upper_ci ?? 0),
    ate_stderr:   Number(row.ate_stderr   ?? 0),
    ate_pvalue:   Number(row.ate_pvalue   ?? 0),
    cate_mean:    Number(row.cate_mean    ?? 0),
    cate_std:     Number(row.cate_std     ?? 0),
    cate_min:     Number(row.cate_min     ?? 0),
    cate_max:     Number(row.cate_max     ?? 0),
  }));
}

export async function getColdStartSlices() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("v_coldstart_segments")
    .select("*")
    .order("ltv_36m_estimate", { ascending: false })
    .limit(100);

  return data ?? [];
}

export async function getCustomerDetail(customerId: string) {
  const supabase = await createServerSupabaseClient();

  const { data: customer } = await supabase
    .from("v_final_customer_scores")
    .select("*")
    .eq("customer_id", customerId)
    .single();

  const { data: rfm } = await supabase
    .from("v_latest_rfm")
    .select("*")
    .eq("customer_id", customerId)
    .single();

  const { data: levers } = await supabase
    .from("causal_lever_recommendations")
    .select("*")
    .eq("customer_id", customerId)
    .single();

  const { data: transactions } = await supabase
    .from("transactions")
    .select("invoice_date, invoice_no, quantity, unit_price, product_category")
    .eq("customer_id", customerId)
    .order("invoice_date", { ascending: false })
    .limit(20);

  return { customer, rfm, levers, transactions: transactions ?? [] };
}

export async function getShapImportance() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("shap_global_importance")
    .select("*")
    .order("rank");

  return data ?? [];
}

export async function searchCustomers(query: string, limit = 10) {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("v_final_customer_scores")
    .select("customer_id, ltv_36m, segment, country, acquisition_channel")
    .or(`customer_id.ilike.%${query}%`)
    .limit(limit);

  return data ?? [];
}

export async function getModelPerformanceDB() {
  const supabase = await createServerSupabaseClient();

  const { data: fusion } = await supabase
    .from("fusion_model_registry")
    .select("*")
    .order("trained_at", { ascending: false })
    .limit(1)
    .single();

  const { data: bgnbd } = await supabase
    .from("bgnbd_model_params")
    .select("*")
    .order("fitted_at", { ascending: false })
    .limit(1)
    .single();

  const { data: transformer } = await supabase
    .from("transformer_model_registry")
    .select("*")
    .order("trained_at", { ascending: false })
    .limit(1)
    .single();

  const { data: shap } = await supabase
    .from("shap_global_importance")
    .select("feature_name, mean_abs_shap, rank")
    .order("rank")
    .limit(10);

  return { fusion, bgnbd, transformer, shap: shap ?? [] };
}

export async function getCampaignData() {
  const supabase = await createServerSupabaseClient();

  // Segment stats — customer count + avg LTV per segment
  const { data: segmentStats } = await supabase
    .from("final_ltv_scores")
    .select("segment, ltv_36m")
    .not("segment", "is", null);

  // Last 10 marketing sync pipeline runs
  const { data: pipelineRuns } = await supabase
    .from("pipeline_runs")
    .select("*")
    .eq("pipeline_name", "marketing_sync")
    .order("started_at", { ascending: false })
    .limit(10);

  // Aggregate segment stats client-side (avoids need for a DB view)
  const segmentMap: Record<string, { count: number; totalLtv: number }> = {};
  for (const row of segmentStats ?? []) {
    const seg = String(row.segment ?? "unknown");
    if (!segmentMap[seg]) segmentMap[seg] = { count: 0, totalLtv: 0 };
    segmentMap[seg].count    += 1;
    segmentMap[seg].totalLtv += Number(row.ltv_36m ?? 0);
  }

  const segments = Object.entries(segmentMap).map(([segment, { count, totalLtv }]) => ({
    segment,
    customer_count: count,
    avg_ltv_36m:    count > 0 ? totalLtv / count : 0,
  }));

  return {
    segments,
    pipelineRuns: pipelineRuns ?? [],
  };
}
