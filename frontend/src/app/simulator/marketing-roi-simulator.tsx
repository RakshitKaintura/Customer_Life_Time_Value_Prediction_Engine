"use client";

import { useState, useMemo } from "react";
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

const SEGMENT_DATA = {
  champions: { avg_ltv: 15000, max_cac_pct: 0.5, pct_customers: 0.05 },
  high_value: { avg_ltv: 7000, max_cac_pct: 0.4, pct_customers: 0.15 },
  medium_value: { avg_ltv: 2500, max_cac_pct: 0.3, pct_customers: 0.4 },
  low_value: { avg_ltv: 600, max_cac_pct: 0.2, pct_customers: 0.4 },
};

const CHANNELS = ["paid_search", "paid_social", "email", "referral"];

export function MarketingROISimulator() {
  const [budget, setBudget] = useState(50_000);
  const [channel, setChannel] = useState("paid_search");
  const [baselineCPC, setBaselineCPC] = useState(5);
  const [conversionRate, setConversionRate] = useState(0.03);
  const [targetSegment, setTargetSegment] = useState("all");

  const results = useMemo(() => {
    const totalCustomers = Math.floor((budget / baselineCPC) * conversionRate);

    const baselineRevenue = Object.values(SEGMENT_DATA).reduce(
      (sum, seg) => sum + totalCustomers * seg.pct_customers * seg.avg_ltv * 0.25,
      0
    );

    let ltvRevenue = 0;
    let ltvCustomers = 0;
    const totalAvgLtv = Object.values(SEGMENT_DATA).reduce((s, d) => s + d.avg_ltv, 0);

    const segFilter =
      targetSegment === "all"
        ? Object.entries(SEGMENT_DATA)
        : Object.entries(SEGMENT_DATA).filter(([k]) => k === targetSegment);

    segFilter.forEach(([, seg]) => {
      const segBudget = budget * (seg.avg_ltv / totalAvgLtv);
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
  }, [budget, baselineCPC, conversionRate, targetSegment]);

  const chartData = [
    { metric: "Revenue", baseline: Math.round(results.baseline.revenue), ltv: Math.round(results.ltv.revenue) },
    { metric: "Customers", baseline: results.baseline.customers, ltv: results.ltv.customers },
  ];

  return (
    <div className="space-y-6">
      <div className="chart-container">
        <CardHeader>
          <CardTitle>Simulation Parameters</CardTitle>
        </CardHeader>
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
              {Object.keys(SEGMENT_DATA).map((s) => (
                <option key={s} value={s}>{s.replace(/_/g, " ")}</option>
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
            {Object.entries(SEGMENT_DATA).map(([seg, data]) => (
              <tr key={seg} className="border-b border-border">
                <td className="py-2.5 pr-4 font-medium capitalize text-foreground">{seg.replace(/_/g, " ")}</td>
                <td className="py-2.5 pr-4">{formatCurrency(data.avg_ltv)}</td>
                <td className="py-2.5 pr-4">{formatPercent(data.max_cac_pct)}</td>
                <td className="py-2.5 font-bold text-foreground">{formatCurrency(data.avg_ltv * data.max_cac_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
