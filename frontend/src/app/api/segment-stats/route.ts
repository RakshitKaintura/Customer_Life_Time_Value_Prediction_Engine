import { NextResponse } from "next/server";
import { createServerSupabaseClient } from "@/lib/supabase/server";

type SegmentRow = {
  segment: string | null;
  pct_customers: number | null;
  avg_ltv_36m: number | null;
  avg_max_cac: number | null;
};

export async function GET() {
  const supabase = await createServerSupabaseClient();

  const { data, error } = await supabase
    .from("v_segment_revenue_concentration")
    .select("segment, pct_customers, avg_ltv_36m, avg_max_cac")
    .order("avg_ltv_36m", { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const normalized = (data as SegmentRow[] | null ?? []).map((row) => {
    const avgLtv = Number(row.avg_ltv_36m ?? 0);
    const avgMaxCac = Number(row.avg_max_cac ?? 0);
    const pctCustomers = Number(row.pct_customers ?? 0) / 100;
    const maxCacPct = avgLtv > 0 ? avgMaxCac / avgLtv : 0;

    return {
      segment: String(row.segment ?? "unknown"),
      avg_ltv: avgLtv,
      avg_max_cac: avgMaxCac,
      pct_customers: pctCustomers,
      max_cac_pct: maxCacPct,
    };
  });

  return NextResponse.json({ data: normalized });
}
