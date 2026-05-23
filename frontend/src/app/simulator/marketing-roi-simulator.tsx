"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Legend, CartesianGrid,
} from "recharts";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";
import { Button } from "@/components/ui/button";
import { ltvApi, SegmentStat } from "@/lib/api";

const CHANNELS = ["paid_search", "paid_social", "email", "referral"];

export function MarketingROISimulator() {
  const [budget, setBudget] = useState(50_000);
  const [channel, setChannel] = useState("paid_search");
  const [baselineCPC, setBaselineCPC] = useState(5);
  const [conversionRate, setConversionRate] = useState(0.03);
  const [targetSegment, setTargetSegment] = useState("all");
  const [segmentStats, setSegmentStats] = useState<SegmentStat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const applySegmentStats = (payload: { data?: SegmentStat[] }) => {
    setSegmentStats(payload.data ?? []);
    setLastUpdated(
      new Date().toLocaleString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    );
  };

  const fetchSegmentStats = async () => {
    try {
      const payload = await ltvApi.getSegmentStats();
      applySegmentStats(payload);
      return true;
    } catch {
      // Fallback to Next.js API route that reads Supabase directly.
      const res = await fetch("/api/segment-stats", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`Failed to load segment stats (${res.status})`);
      }
      const payload = await res.json();
      applySegmentStats(payload);
      return true;
    }
  };

  useEffect(() => {
    let active = true;

    async function loadSegmentStats() {
      try {
        setLoading(true);
        setError(null);
        if (active) {
          await fetchSegmentStats();
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unable to load segment stats");
          setSegmentStats([]);
          setLastUpdated(null);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSegmentStats();
    return () => {
      active = false;
    };
  }, []);

  const segmentData = useMemo(() => {
    return segmentStats.reduce<Record<string, SegmentStat>>((acc, seg) => {
      acc[seg.segment] = seg;
      return acc;
    }, {});
  }, [segmentStats]);

  const results = useMemo(() => {
    const totalCustomers = Math.floor((budget / baselineCPC) * conversionRate);

    const baselineRevenue = Object.values(segmentData).reduce(
      (sum, seg) => sum + totalCustomers * seg.pct_customers * seg.avg_ltv * 0.25,
      0
    );

    let ltvRevenue = 0;
    let ltvCustomers = 0;
    const totalAvgLtv = Object.values(segmentData).reduce((s, d) => s + d.avg_ltv, 0);

    const segFilter =
      targetSegment === "all"
        ? Object.entries(segmentData)
        : Object.entries(segmentData).filter(([k]) => k === targetSegment);

    segFilter.forEach(([, seg]) => {
      const segBudget = totalAvgLtv > 0 ? budget * (seg.avg_ltv / totalAvgLtv) : 0;
      const maxCAC = seg.avg_ltv * seg.max_cac_pct;
      const adjCPC = Math.min(baselineCPC * (maxCAC / (baselineCPC * 20)), maxCAC);
      const custCount = Math.floor((segBudget / adjCPC) * conversionRate);
      ltvCustomers += custCount;
      ltvRevenue += custCount * seg.avg_ltv * 0.25;
    });

    const improvement = baselineRevenue > 0 ? ((ltvRevenue - baselineRevenue) / baselineRevenue) * 100 : 0;

    return {
      baseline: {
        customers: totalCustomers,
        revenue: baselineRevenue,
        roas: budget > 0 ? baselineRevenue / budget : 0,
      },
      ltv: {
        customers: ltvCustomers,
        revenue: ltvRevenue,
        roas: budget > 0 ? ltvRevenue / budget : 0,
      },
      improvement,
    };
  }, [budget, baselineCPC, conversionRate, targetSegment, segmentData]);

  const chartData = [
    { metric: "Revenue", baseline: Math.round(results.baseline.revenue), ltv: Math.round(results.ltv.revenue) },
    { metric: "Customers", baseline: results.baseline.customers, ltv: results.ltv.customers },
  ];

  return (
    <div className="space-y-6">
      <div className="chart-container">
        <CardHeader>
          <CardTitle>Simulation Parameters</CardTitle>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {lastUpdated && <span>Updated {lastUpdated}</span>}
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => {
                setLoading(true);
                setError(null);
                fetchSegmentStats()
                  .catch((err) => {
                    setError(err instanceof Error ? err.message : "Unable to load segment stats");
                    setSegmentStats([]);
                    setLastUpdated(null);
                  })
                  .finally(() => setLoading(false));
              }}
              disabled={loading}
            >
              {loading ? "Refreshing..." : "Refresh"}
            </Button>
          </div>
        </CardHeader>
        {loading && (
          <p className="mb-4 text-xs text-muted-foreground">Loading live segment stats...</p>
        )}
        {error && (
          <p className="mb-4 text-xs text-muted-foreground">{error}</p>
        )}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <div>
            <label className="mb-1.5 block text-sm font-medium text-foreground">Total Budget</label>
            <Input type="number" value={budget} onChange={(e) => setBudget(Number(e.target.value))} step={10000} />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-foreground">Channel</label>
            <Select value={channel} onChange={(e) => setChannel(e.target.value)}>
              {CHANNELS.map((c) => (
                <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
              ))}
            </Select>
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-foreground">Baseline CPC (GBP)</label>
            <Input type="number" value={baselineCPC} onChange={(e) => setBaselineCPC(Number(e.target.value))} step={0.5} min={0.1} />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-foreground">Conversion Rate</label>
            <Input type="number" value={conversionRate} onChange={(e) => setConversionRate(Number(e.target.value))} step={0.005} min={0.001} max={0.5} />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-foreground">Target Segment</label>
            <Select value={targetSegment} onChange={(e) => setTargetSegment(e.target.value)}>
              <option value="all">All Segments</option>
              {segmentStats.map((s) => (
                <option key={s.segment} value={s.segment}>{s.segment.replace(/_/g, " ")}</option>
              ))}
            </Select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Baseline Revenue (36m)" value={results.baseline.revenue} format="currency" subtitle="Flat CPC bidding" />
        <StatCard title="LTV-Informed Revenue (36m)" value={results.ltv.revenue} format="currency" subtitle="CAC-capped per segment" trend={results.improvement} />
        <StatCard title="Baseline ROAS" value={`${results.baseline.roas.toFixed(1)}x`} format="text" />
        <StatCard
          title="LTV-Informed ROAS"
          value={`${results.ltv.roas.toFixed(1)}x`}
          format="text"
          trend={results.baseline.roas > 0 ? ((results.ltv.roas - results.baseline.roas) / results.baseline.roas) * 100 : 0}
        />
      </div>

      <div className="chart-container">
        <CardHeader>
          <CardTitle>Baseline vs LTV-Informed Performance</CardTitle>
          <span className="text-sm font-medium text-foreground">
            {results.improvement > 0 ? "+" : ""}{results.improvement.toFixed(1)}% revenue improvement
          </span>
        </CardHeader>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} />
            <XAxis dataKey="metric" tick={{ fontSize: 12, fill: chartAxisTick.fill }} />
            <YAxis tick={chartAxisTick} />
            <Tooltip
              formatter={(v: number, name: string) => [name === "baseline" ? "Baseline" : "LTV-Informed", v.toLocaleString()]}
              contentStyle={chartTooltipStyle}
            />
            <Legend formatter={(v) => (v === "baseline" ? "Baseline" : "LTV-Informed")} />
            <Bar dataKey="baseline" fill="hsl(var(--chart-3))" radius={[3, 3, 0, 0]} name="baseline" />
            <Bar dataKey="ltv" fill="hsl(var(--chart-1))" radius={[3, 3, 0, 0]} name="ltv" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <CardHeader>
          <CardTitle>Recommended Max CAC by Segment</CardTitle>
          <span className="text-xs text-muted-foreground">Based on predicted LTV x CAC fraction</span>
        </CardHeader>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="pb-3 pr-4 font-medium text-muted-foreground">Segment</th>
              <th className="pb-3 pr-4 font-medium text-muted-foreground">Avg LTV 36m</th>
              <th className="pb-3 pr-4 font-medium text-muted-foreground">Max CAC Fraction</th>
              <th className="pb-3 font-medium text-muted-foreground">Max CAC</th>
            </tr>
          </thead>
          <tbody>
            {segmentStats.length > 0 ? (
              segmentStats.map((seg) => (
                <tr key={seg.segment} className="border-b border-border">
                  <td className="py-2.5 pr-4 font-medium capitalize text-foreground">{seg.segment.replace(/_/g, " ")}</td>
                  <td className="py-2.5 pr-4">{formatCurrency(seg.avg_ltv)}</td>
                  <td className="py-2.5 pr-4">{formatPercent(seg.max_cac_pct)}</td>
                  <td className="py-2.5 font-bold text-foreground">{formatCurrency(seg.avg_max_cac)}</td>
                </tr>
              ))
            ) : (
              <tr className="border-b border-border">
                <td className="py-3 text-sm text-muted-foreground" colSpan={4}>
                  No live segment stats available yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
