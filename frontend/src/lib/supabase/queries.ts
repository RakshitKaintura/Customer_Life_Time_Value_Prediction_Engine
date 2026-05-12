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
    .select("ltv_36m, ltv_12m, segment, ltv_percentile")
    .order("ltv_36m", { ascending: false })
    .limit(5000);

  const { data: customerCount } = await supabase
    .from("customers")
    .select("customer_id", { count: "exact", head: true });

  return {
    segmentData: segmentData ?? [],
    scoreData:   scoreData   ?? [],
    totalCustomers: customerCount ?? 0,
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

  // Fetch retention from rfm_features cohort
  const { data } = await supabase.rpc("get_cohort_retention_matrix");
  return data ?? [];
}

export async function getCausalEffects() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("causal_treatment_effects")
    .select("*")
    .order("ate", { ascending: false });

  return data ?? [];
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