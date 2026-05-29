/**
 * Monitoring-specific Supabase queries for Week 8 dashboard enhancements.
 */

import { createServerSupabaseClient } from "./server";

export async function getDriftHistory(days = 30) {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("ltv_drift_alerts")
    .select("alert_id, detected_at, alert_type, psi_score, mean_shift_pct, status")
    .gte("detected_at", new Date(Date.now() - days * 86400000).toISOString())
    .lt("psi_score", 10)          // exclude corrupted/test rows — real PSI is always < 2
    .order("detected_at", { ascending: true });

  return (data ?? []).map((row) => ({
    date:           new Date(row.detected_at).toLocaleDateString("en-GB", {
      month: "short", day: "numeric",
    }),
    psi_score:      Number(row.psi_score ?? 0),
    mean_shift_pct: row.mean_shift_pct != null ? Number(row.mean_shift_pct) : null,
    alert_type:     row.alert_type ?? "",
    status:         row.status ?? "open",
  }));
}

export async function getRetrainingLog(limit = 10) {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("retraining_log")
    .select("*")
    .order("triggered_at", { ascending: false })
    .limit(limit);

  return data ?? [];
}

export async function getPerformanceTrend() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("v_performance_trend")
    .select("*")
    .order("eval_date", { ascending: true });

  return data ?? [];
}

export async function getOpenDriftAlerts() {
  const supabase = await createServerSupabaseClient();

  const { data } = await supabase
    .from("ltv_drift_alerts")
    .select("*")
    .eq("status", "open")
    .order("detected_at", { ascending: false })
    .limit(5);

  return data ?? [];
}