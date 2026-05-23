"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Topbar } from "@/components/nav/topbar";
import { Select } from "@/components/ui/select";
import { chartAxisTick, chartGridStroke, chartTooltipStyle } from "@/components/ui/chart-theme";

const datasetHighlights = [
  { label: "Dataset", value: "UCI Online Retail" },
  { label: "Rows", value: "541,909" },
  { label: "Columns", value: "8" },
  { label: "Date Range", value: "2010-12-01 to 2011-12-09" },
  { label: "Countries", value: "38" },
];

const qualityHighlights = [
  { label: "Rows Kept After Cleaning", value: "397,884 (73.42%)" },
  { label: "Dropped Rows", value: "144,025" },
  { label: "Unique Customers (Cleaned)", value: "4,338" },
  { label: "Unique Orders (Cleaned)", value: "18,532" },
  { label: "Cleaned Revenue Sum", value: "$8,911,407.90" },
];

const datasetColumns = [
  "InvoiceNo",
  "StockCode",
  "Description",
  "Quantity",
  "InvoiceDate",
  "UnitPrice",
  "CustomerID",
  "Country",
];

const dictionaryRows = [
  { field: "InvoiceNo", type: "string", description: "Invoice identifier. Values starting with C are cancellations/returns.", role: "Used for order counts and return filtering." },
  { field: "StockCode", type: "string", description: "Product code for each line item.", role: "Mapped to heuristic product categories." },
  { field: "Description", type: "string", description: "Product text description.", role: "Interpretability and sample-level explanation." },
  { field: "Quantity", type: "integer", description: "Units bought in each transaction line.", role: "Revenue and basket-size calculations." },
  { field: "InvoiceDate", type: "datetime", description: "Purchase timestamp.", role: "Recency, frequency cadence, cohort windows." },
  { field: "UnitPrice", type: "float", description: "Per-item price in invoice currency.", role: "Monetary features and LTV labels." },
  { field: "CustomerID", type: "string", description: "Customer identifier (null in anonymous rows).", role: "Entity key for customer-level modeling." },
  { field: "Country", type: "string", description: "Country for the transaction.", role: "Geographic segmentation and profile features." },
];

const cleaningRules = [
  "Remove rows with null CustomerID.",
  "Remove cancelled/return transactions where InvoiceNo starts with C.",
  "Remove rows with zero or negative Quantity.",
  "Remove rows with zero or negative UnitPrice.",
  "Remove extreme pricing outliers where UnitPrice >= 10,000.",
  "Normalize strings and compute line_total = Quantity * UnitPrice.",
];

const dropBreakdown = [
  { issue: "Missing CustomerID", rows: 135080 },
  { issue: "Cancelled invoices", rows: 9288 },
  { issue: "Non-positive Quantity", rows: 10624 },
  { issue: "Non-positive UnitPrice", rows: 2517 },
  { issue: "UnitPrice > 10,000", rows: 10 },
];

const countryStats = [
  { country: "United Kingdom", rows: 354321, orders: 16646, customers: 3920, revenue: 7308391.55 },
  { country: "Netherlands", rows: 2359, orders: 94, customers: 9, revenue: 285446.34 },
  { country: "EIRE", rows: 7236, orders: 260, customers: 3, revenue: 265545.9 },
  { country: "Germany", rows: 9040, orders: 457, customers: 94, revenue: 228867.14 },
  { country: "France", rows: 8341, orders: 389, customers: 87, revenue: 209024.05 },
  { country: "Australia", rows: 1182, orders: 57, customers: 9, revenue: 138521.31 },
  { country: "Spain", rows: 2484, orders: 90, customers: 30, revenue: 61577.11 },
  { country: "Switzerland", rows: 1841, orders: 51, customers: 21, revenue: 56443.95 },
  { country: "Belgium", rows: 2031, orders: 98, customers: 25, revenue: 41196.34 },
  { country: "Sweden", rows: 451, orders: 36, customers: 8, revenue: 38378.33 },
];

const monthlyTrendData = [
  { month: "2010-12", orders: 1400, customers: 885, revenue: 572713.89 },
  { month: "2011-01", orders: 987, customers: 741, revenue: 569445.04 },
  { month: "2011-02", orders: 997, customers: 758, revenue: 447137.35 },
  { month: "2011-03", orders: 1321, customers: 974, revenue: 595500.76 },
  { month: "2011-04", orders: 1149, customers: 856, revenue: 469200.36 },
  { month: "2011-05", orders: 1555, customers: 1056, revenue: 678594.56 },
  { month: "2011-06", orders: 1393, customers: 991, revenue: 661213.69 },
  { month: "2011-07", orders: 1331, customers: 949, revenue: 600091.01 },
  { month: "2011-08", orders: 1280, customers: 935, revenue: 645343.9 },
  { month: "2011-09", orders: 1755, customers: 1266, revenue: 952838.38 },
  { month: "2011-10", orders: 1929, customers: 1364, revenue: 1039318.79 },
  { month: "2011-11", orders: 2657, customers: 1664, revenue: 1161817.38 },
  { month: "2011-12", orders: 778, customers: 615, revenue: 518192.79 },
];

const qualityDashboard = [
  { metric: "Missing CustomerID", count: 135080 },
  { metric: "Cancelled Rows", count: 9288 },
  { metric: "Non-positive Quantity", count: 10624 },
  { metric: "Non-positive UnitPrice", count: 2517 },
  { metric: "Price Outliers", count: 10 },
];

const engineeredFeatures = [
  { name: "RFM Core", items: "recency_days, frequency, monetary_avg, monetary_total", why: "Primary CLV signal for repeat behavior and spend." },
  { name: "Order Behavior", items: "orders_count, avg_items_per_order, purchase_variance", why: "Captures consistency and order composition." },
  { name: "Timing Dynamics", items: "avg_days_between_orders, std_days_between_orders, days_to_second_purchase", why: "Early repeat behavior strongly predicts long-term value." },
  { name: "Breadth & Mix", items: "unique_products, unique_categories, first_purchase_category", why: "Higher product breadth often correlates with stronger retention." },
  { name: "Modeling Labels", items: "actual_ltv_12m, actual_ltv_24m, actual_ltv_36m", why: "Ground-truth holdout targets for supervised learning." },
];

const knownLimitations = [
  "This dataset is historical retail data from a specific business context.",
  "No direct marketing spend fields (CAC/channel cost) in the raw table.",
  "No rich firmographic attributes in source transactions.",
  "Country and product effects are present, but seasonality can vary by business.",
  "Transfer to another company requires schema mapping and retraining.",
];

const modelingUsage = [
  { stage: "Ingestion", detail: "Raw CSV is loaded and standardized into snake_case columns." },
  { stage: "Cleaning", detail: "Invalid customers, returns, and invalid price/quantity rows are removed." },
  { stage: "Feature Engineering", detail: "RFM, category, amount buckets, and sequence features are created." },
  { stage: "Split Strategy", detail: "Observation window and holdout window are used for robust evaluation." },
  { stage: "Model Training", detail: "BG/NBD + Gamma-Gamma + transformer/fusion models learn CLV patterns." },
  { stage: "Serving", detail: "Predictions are exposed via dashboard/API for segmentation decisions." },
];

const datasetSampleRows = [
  { InvoiceNo: "536365", StockCode: "85123A", Description: "WHITE HANGING HEART T-LIGHT HOLDER", Quantity: "6", InvoiceDate: "2010-12-01 08:26", UnitPrice: "2.55", CustomerID: "17850", Country: "United Kingdom" },
  { InvoiceNo: "536365", StockCode: "71053", Description: "WHITE METAL LANTERN", Quantity: "6", InvoiceDate: "2010-12-01 08:26", UnitPrice: "3.39", CustomerID: "17850", Country: "United Kingdom" },
  { InvoiceNo: "536370", StockCode: "22728", Description: "ALARM CLOCK BAKELIKE PINK", Quantity: "24", InvoiceDate: "2010-12-01 08:45", UnitPrice: "3.75", CustomerID: "12583", Country: "France" },
  { InvoiceNo: "536370", StockCode: "22727", Description: "ALARM CLOCK BAKELIKE RED", Quantity: "24", InvoiceDate: "2010-12-01 08:45", UnitPrice: "3.75", CustomerID: "12583", Country: "France" },
  { InvoiceNo: "536527", StockCode: "22809", Description: "SET OF 6 T-LIGHTS SANTA", Quantity: "6", InvoiceDate: "2010-12-01 13:04", UnitPrice: "2.95", CustomerID: "12662", Country: "Germany" },
  { InvoiceNo: "536527", StockCode: "84347", Description: "ROTATING SILVER ANGELS T-LIGHT HLDR", Quantity: "6", InvoiceDate: "2010-12-01 13:04", UnitPrice: "2.55", CustomerID: "12662", Country: "Germany" },
  { InvoiceNo: "536403", StockCode: "22867", Description: "HAND WARMER BIRD DESIGN", Quantity: "96", InvoiceDate: "2010-12-01 11:27", UnitPrice: "1.85", CustomerID: "12791", Country: "Netherlands" },
  { InvoiceNo: "536403", StockCode: "POST", Description: "POSTAGE", Quantity: "1", InvoiceDate: "2010-12-01 11:27", UnitPrice: "15.00", CustomerID: "12791", Country: "Netherlands" },
  { InvoiceNo: "536944", StockCode: "22383", Description: "LUNCH BAG SUKI DESIGN", Quantity: "70", InvoiceDate: "2010-12-03 12:20", UnitPrice: "1.65", CustomerID: "12557", Country: "Spain" },
  { InvoiceNo: "536944", StockCode: "22384", Description: "LUNCH BAG PINK POLKADOT", Quantity: "100", InvoiceDate: "2010-12-03 12:20", UnitPrice: "1.45", CustomerID: "12557", Country: "Spain" },
  { InvoiceNo: "537026", StockCode: "84375", Description: "SET OF 20 KIDS COOKIE CUTTERS", Quantity: "12", InvoiceDate: "2010-12-03 16:35", UnitPrice: "2.10", CustomerID: "12395", Country: "Belgium" },
  { InvoiceNo: "537026", StockCode: "21217", Description: "RED RETROSPOT ROUND CAKE TINS", Quantity: "2", InvoiceDate: "2010-12-03 16:35", UnitPrice: "9.95", CustomerID: "12395", Country: "Belgium" },
  { InvoiceNo: "536858", StockCode: "22326", Description: "ROUND SNACK BOXES SET OF4 WOODLAND", Quantity: "30", InvoiceDate: "2010-12-03 10:36", UnitPrice: "2.95", CustomerID: "13520", Country: "Switzerland" },
  { InvoiceNo: "536858", StockCode: "22554", Description: "PLASTERS IN TIN WOODLAND ANIMALS", Quantity: "36", InvoiceDate: "2010-12-03 10:36", UnitPrice: "1.65", CustomerID: "13520", Country: "Switzerland" },
  { InvoiceNo: "536990", StockCode: "21992", Description: "VINTAGE PAISLEY STATIONERY SET", Quantity: "6", InvoiceDate: "2010-12-03 15:14", UnitPrice: "2.95", CustomerID: "12793", Country: "Portugal" },
  { InvoiceNo: "536990", StockCode: "22383", Description: "LUNCH BAG SUKI DESIGN", Quantity: "10", InvoiceDate: "2010-12-03 15:14", UnitPrice: "1.65", CustomerID: "12793", Country: "Portugal" },
  { InvoiceNo: "536389", StockCode: "22941", Description: "CHRISTMAS LIGHTS 10 REINDEER", Quantity: "6", InvoiceDate: "2010-12-01 10:03", UnitPrice: "8.50", CustomerID: "12431", Country: "Australia" },
  { InvoiceNo: "536389", StockCode: "21622", Description: "VINTAGE UNION JACK CUSHION COVER", Quantity: "8", InvoiceDate: "2010-12-01 10:03", UnitPrice: "4.95", CustomerID: "12431", Country: "Australia" },
  { InvoiceNo: "536532", StockCode: "84692", Description: "BOX OF 24 COCKTAIL PARASOLS", Quantity: "50", InvoiceDate: "2010-12-01 13:24", UnitPrice: "0.42", CustomerID: "12433", Country: "Norway" },
  { InvoiceNo: "536532", StockCode: "22444", Description: "GROW YOUR OWN PLANT IN A CAN", Quantity: "96", InvoiceDate: "2010-12-01 13:24", UnitPrice: "1.06", CustomerID: "12433", Country: "Norway" },
];

const formatCurrency = (value: number) =>
  `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

const formatNumber = (value: number) => value.toLocaleString();

type DatasetInsights = {
  datasetHighlights: Array<{ label: string; value: string }>;
  qualityHighlights: Array<{ label: string; value: string }>;
  dropBreakdown: Array<{ issue: string; rows: number }>;
  qualityDashboard: Array<{ metric: string; count: number }>;
  countryStats: Array<{ country: string; rows: number; orders: number; customers: number; revenue: number }>;
  monthlyTrendData: Array<{ month: string; orders: number; customers: number; revenue: number }>;
  datasetSampleRows: Array<Record<string, string>>;
};

export default function DatasetPage() {
  const [insights, setInsights] = useState<DatasetInsights | null>(null);
  const [isLoadingInsights, setIsLoadingInsights] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const loadInsights = async () => {
      setIsLoadingInsights(true);
      try {
        const response = await fetch("/api/dataset-insights", { cache: "no-store" });
        if (!response.ok) throw new Error("Failed to fetch dataset insights");
        const data = (await response.json()) as DatasetInsights;
        if (!cancelled) setInsights(data);
      } catch {
        // Keep fallback demo constants if live fetch fails.
      } finally {
        if (!cancelled) setIsLoadingInsights(false);
      }
    };

    void loadInsights();
    return () => {
      cancelled = true;
    };
  }, []);

  const liveDatasetHighlights = insights?.datasetHighlights ?? datasetHighlights;
  const liveQualityHighlights = insights?.qualityHighlights ?? qualityHighlights;
  const liveDropBreakdown = insights?.dropBreakdown ?? dropBreakdown;
  const liveQualityDashboard = insights?.qualityDashboard ?? qualityDashboard;
  const liveCountryStats = insights?.countryStats ?? countryStats;
  const liveMonthlyTrendData = insights?.monthlyTrendData ?? monthlyTrendData;
  const liveSampleRows = insights?.datasetSampleRows ?? datasetSampleRows;

  const countries = useMemo(
    () => ["All Countries", ...Array.from(new Set(liveSampleRows.map((r) => r.Country))).sort()],
    [liveSampleRows]
  );
  const [selectedCountry, setSelectedCountry] = useState("All Countries");

  const filteredRows = useMemo(() => {
    if (selectedCountry === "All Countries") return liveSampleRows.slice(0, 10);
    return liveSampleRows.filter((row) => row.Country === selectedCountry).slice(0, 10);
  }, [selectedCountry, liveSampleRows]);

  return (
    <div className="page-container">
      <Topbar
        title="Dataset Intelligence"
        subtitle="Complete documentation of the source dataset, quality logic, and modeling relevance"
      />
      <div className="page-content space-y-6">
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">1) Dataset Overview</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            This project uses the UCI Online Retail transactions dataset as the base for customer-level CLV modeling.
            The raw file is transformed into cleaned transactions, customer aggregates, and model features.
          </p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {liveDatasetHighlights.map((item) => (
              <div key={item.label} className="mt-4 rounded-xl border border-border bg-background p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">{item.label}</p>
                <p className="mt-2 text-sm font-medium text-foreground">{item.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">2) Country Explorer</h2>
          <p className="mt-2 text-sm text-muted-foreground">Explore country-level coverage and see real sample rows by selected country.</p>
          <div className="mt-4 grid gap-4 lg:grid-cols-[320px_1fr]">
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Filter Sample Rows</p>
              <Select className="mt-3" value={selectedCountry} onChange={(e) => setSelectedCountry(e.target.value)}>
                {countries.map((country) => (
                  <option key={country} value={country}>{country}</option>
                ))}
              </Select>
              <p className="mt-3 text-xs text-muted-foreground">Showing up to 10 sample rows for the selected country.</p>
            </div>
            <div className="overflow-x-auto rounded-xl border border-border">
              <table className="min-w-full divide-y divide-border text-left text-xs">
                <thead className="bg-background">
                  <tr>
                    <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Country</th>
                    <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Rows</th>
                    <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Orders</th>
                    <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Customers</th>
                    <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Revenue</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {liveCountryStats.map((row) => (
                    <tr key={row.country} className="bg-card">
                      <td className="px-3 py-2 text-foreground">{row.country}</td>
                      <td className="px-3 py-2 text-muted-foreground">{formatNumber(row.rows)}</td>
                      <td className="px-3 py-2 text-muted-foreground">{formatNumber(row.orders)}</td>
                      <td className="px-3 py-2 text-muted-foreground">{formatNumber(row.customers)}</td>
                      <td className="px-3 py-2 text-muted-foreground">{formatCurrency(row.revenue)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">3) Monthly Trend Charts</h2>
          <p className="mt-2 text-sm text-muted-foreground">Monthly order volume and revenue trend over the available dataset window.</p>
          <div className="mt-5 grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="mb-3 text-sm font-medium text-foreground">Orders by Month</p>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={liveMonthlyTrendData}>
                  <CartesianGrid stroke={chartGridStroke} strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={chartAxisTick} />
                  <YAxis tick={chartAxisTick} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value: number) => [formatNumber(value), "Orders"]} />
                  <Line type="monotone" dataKey="orders" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="mb-3 text-sm font-medium text-foreground">Revenue by Month</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={liveMonthlyTrendData}>
                  <CartesianGrid stroke={chartGridStroke} strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={chartAxisTick} />
                  <YAxis tick={chartAxisTick} tickFormatter={(v: number) => `$${Math.round(v / 1000)}k`} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value: number) => [formatCurrency(value), "Revenue"]} />
                  <Bar dataKey="revenue" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">4) Data Quality Dashboard</h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {liveQualityHighlights.map((item) => (
              <div key={item.label} className="rounded-xl border border-border bg-background p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">{item.label}</p>
                <p className="mt-2 text-sm font-medium text-foreground">{item.value}</p>
              </div>
            ))}
          </div>
          <div className="mt-5 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="mb-3 text-sm font-medium text-foreground">Quality Issue Counts</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={liveQualityDashboard} layout="vertical" margin={{ left: 24 }}>
                  <CartesianGrid stroke={chartGridStroke} strokeDasharray="3 3" />
                  <XAxis type="number" tick={chartAxisTick} />
                  <YAxis type="category" dataKey="metric" tick={chartAxisTick} width={150} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value: number) => [formatNumber(value), "Rows"]} />
                  <Bar dataKey="count" fill="hsl(var(--chart-3))" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-sm font-medium text-foreground">Cleaning logic applied in pipeline</p>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                {cleaningRules.map((rule) => (
                  <li key={rule}>{rule}</li>
                ))}
              </ul>
            </div>
          </div>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="min-w-full divide-y divide-border text-left text-xs">
              <thead className="bg-background">
                <tr>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Quality Issue</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Affected Rows</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {liveDropBreakdown.map((row) => (
                  <tr key={row.issue} className="bg-card">
                    <td className="px-3 py-2 text-muted-foreground">{row.issue}</td>
                    <td className="px-3 py-2 text-foreground">{formatNumber(row.rows)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Note: quality issue counts can overlap on the same row, so individual issue totals are not additive.
          </p>
          {isLoadingInsights ? (
            <p className="mt-2 text-xs text-muted-foreground">Refreshing live metrics from database...</p>
          ) : null}
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">5) Sample Raw Rows</h2>
          <div className="mt-6 overflow-x-auto rounded-xl border border-border">
            <table className="min-w-full divide-y divide-border text-left text-xs">
              <thead className="bg-background">
                <tr>
                  {datasetColumns.map((column) => (
                    <th key={column} className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filteredRows.map((row, index) => (
                  <tr key={`${row.InvoiceNo}-${index}`} className="bg-card">
                    {datasetColumns.map((column) => (
                      <td key={`${row.InvoiceNo}-${column}-${index}`} className="px-3 py-2 text-muted-foreground">
                        {row[column as keyof typeof row]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">6) Data Dictionary</h2>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="min-w-full divide-y divide-border text-left text-xs">
              <thead className="bg-background">
                <tr>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Field</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Type</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Description</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Model Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {dictionaryRows.map((row) => (
                  <tr key={row.field} className="bg-card">
                    <td className="px-3 py-2 text-foreground">{row.field}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.type}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.description}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.role}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">7) Engineered Features for CLV</h2>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="min-w-full divide-y divide-border text-left text-xs">
              <thead className="bg-background">
                <tr>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Feature Group</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Examples</th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-[0.2em] text-muted-foreground">Why It Matters</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {engineeredFeatures.map((row) => (
                  <tr key={row.name} className="bg-card">
                    <td className="px-3 py-2 text-foreground">{row.name}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.items}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">8) How This Dataset Flows Through the System</h2>
          <div className="mt-4 grid gap-3">
            {modelingUsage.map((step, index) => (
              <div key={step.stage} className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">
                  {index + 1}. {step.stage}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">{step.detail}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-base font-semibold text-foreground">9) Known Limitations</h2>
          <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
            {knownLimitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-muted-foreground">
            For external company data, this pipeline remains reusable with schema mapping, drift checks, and retraining.
          </p>
        </div>
      </div>
    </div>
  );
}
