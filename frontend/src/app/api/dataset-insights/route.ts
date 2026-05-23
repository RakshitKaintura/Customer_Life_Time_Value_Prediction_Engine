import { NextResponse } from "next/server";
import { createServerSupabaseClient } from "@/lib/supabase/server";

type RawTxn = {
  customer_id: string | null;
  invoice_no: string | null;
  quantity: number | null;
  unit_price: number | null;
};

type CleanTxn = {
  invoice_no: string | null;
  stock_code: string | null;
  description: string | null;
  quantity: number | null;
  invoice_date: string | null;
  unit_price: number | null;
  customer_id: string | null;
  country: string | null;
};

const PAGE_SIZE = 50000;

export async function GET() {
  const supabase = await createServerSupabaseClient();

  const [{ count: rawCount }, { count: cleanedCount }, { count: customerCount }] = await Promise.all([
    supabase.from("raw_transactions").select("id", { count: "exact", head: true }),
    supabase.from("transactions").select("id", { count: "exact", head: true }),
    supabase.from("customers").select("customer_id", { count: "exact", head: true }),
  ]);

  const rawRows = Number(rawCount ?? 0);
  const cleanedRows = Number(cleanedCount ?? 0);
  const droppedRows = Math.max(0, rawRows - cleanedRows);

  let missingCustomerId = 0;
  let cancelledRows = 0;
  let nonPositiveQuantity = 0;
  let nonPositiveUnitPrice = 0;
  let priceOutliers = 0;

  for (let start = 0; start < rawRows; start += PAGE_SIZE) {
    const end = Math.min(start + PAGE_SIZE - 1, rawRows - 1);
    const { data, error } = await supabase
      .from("raw_transactions")
      .select("customer_id, invoice_no, quantity, unit_price")
      .range(start, end);

    if (error) return NextResponse.json({ error: error.message }, { status: 500 });

    const rows = (data ?? []) as RawTxn[];
    for (const row of rows) {
      const invoiceNo = row.invoice_no ?? "";
      const qty = Number(row.quantity ?? 0);
      const unitPrice = Number(row.unit_price ?? 0);
      if (!row.customer_id) missingCustomerId++;
      if (invoiceNo.startsWith("C")) cancelledRows++;
      if (qty <= 0) nonPositiveQuantity++;
      if (unitPrice <= 0) nonPositiveUnitPrice++;
      if (unitPrice >= 10000) priceOutliers++;
    }
  }

  const countryMap = new Map<string, { rows: number; orders: Set<string>; customers: Set<string>; revenue: number }>();
  const monthlyMap = new Map<string, { orders: Set<string>; customers: Set<string>; revenue: number }>();
  const sampleRows: Array<Record<string, string>> = [];
  const sampleCountryCap = new Map<string, number>();
  const globalOrders = new Set<string>();
  let cleanedRevenue = 0;

  for (let start = 0; start < cleanedRows; start += PAGE_SIZE) {
    const end = Math.min(start + PAGE_SIZE - 1, cleanedRows - 1);
    const { data, error } = await supabase
      .from("transactions")
      .select("invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id, country")
      .range(start, end);

    if (error) return NextResponse.json({ error: error.message }, { status: 500 });

    const rows = (data ?? []) as CleanTxn[];
    for (const row of rows) {
      const country = row.country ?? "Unknown";
      const invoiceNo = row.invoice_no ?? "";
      const customerId = row.customer_id ?? "";
      const quantity = Number(row.quantity ?? 0);
      const unitPrice = Number(row.unit_price ?? 0);
      const lineTotal = quantity * unitPrice;

      cleanedRevenue += lineTotal;
      if (invoiceNo) globalOrders.add(invoiceNo);

      if (!countryMap.has(country)) {
        countryMap.set(country, { rows: 0, orders: new Set<string>(), customers: new Set<string>(), revenue: 0 });
      }
      const countryAgg = countryMap.get(country)!;
      countryAgg.rows += 1;
      if (invoiceNo) countryAgg.orders.add(invoiceNo);
      if (customerId) countryAgg.customers.add(customerId);
      countryAgg.revenue += lineTotal;

      const month = (row.invoice_date ?? "").slice(0, 7);
      if (month) {
        if (!monthlyMap.has(month)) monthlyMap.set(month, { orders: new Set<string>(), customers: new Set<string>(), revenue: 0 });
        const monthlyAgg = monthlyMap.get(month)!;
        if (invoiceNo) monthlyAgg.orders.add(invoiceNo);
        if (customerId) monthlyAgg.customers.add(customerId);
        monthlyAgg.revenue += lineTotal;
      }

      const current = sampleCountryCap.get(country) ?? 0;
      if (current < 2 && sampleRows.length < 40) {
        sampleRows.push({
          InvoiceNo: invoiceNo,
          StockCode: row.stock_code ?? "",
          Description: (row.description ?? "").trim().replace(/\s+/g, " "),
          Quantity: String(quantity),
          InvoiceDate: (row.invoice_date ?? "").replace("T", " ").slice(0, 16),
          UnitPrice: unitPrice.toFixed(2),
          CustomerID: customerId,
          Country: country,
        });
        sampleCountryCap.set(country, current + 1);
      }
    }
  }

  const countryStats = Array.from(countryMap.entries())
    .map(([country, v]) => ({
      country,
      rows: v.rows,
      orders: v.orders.size,
      customers: v.customers.size,
      revenue: Number(v.revenue.toFixed(2)),
    }))
    .sort((a, b) => b.revenue - a.revenue)
    .slice(0, 12);

  const monthlyTrendData = Array.from(monthlyMap.entries())
    .map(([month, v]) => ({
      month,
      orders: v.orders.size,
      customers: v.customers.size,
      revenue: Number(v.revenue.toFixed(2)),
    }))
    .sort((a, b) => a.month.localeCompare(b.month));

  return NextResponse.json({
    datasetHighlights: [
      { label: "Dataset", value: "UCI Online Retail" },
      { label: "Rows", value: rawRows.toLocaleString() },
      { label: "Columns", value: "8" },
      { label: "Date Range", value: monthlyTrendData.length > 0 ? `${monthlyTrendData[0].month} to ${monthlyTrendData[monthlyTrendData.length - 1].month}` : "N/A" },
      { label: "Countries", value: countryMap.size.toLocaleString() },
    ],
    qualityHighlights: [
      { label: "Rows Kept After Cleaning", value: `${cleanedRows.toLocaleString()} (${rawRows > 0 ? ((cleanedRows / rawRows) * 100).toFixed(2) : "0.00"}%)` },
      { label: "Dropped Rows", value: droppedRows.toLocaleString() },
      { label: "Unique Customers (Cleaned)", value: Number(customerCount ?? 0).toLocaleString() },
      { label: "Unique Orders (Cleaned)", value: globalOrders.size.toLocaleString() },
      { label: "Cleaned Revenue Sum", value: `$${cleanedRevenue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` },
    ],
    dropBreakdown: [
      { issue: "Missing CustomerID", rows: missingCustomerId },
      { issue: "Cancelled invoices", rows: cancelledRows },
      { issue: "Non-positive Quantity", rows: nonPositiveQuantity },
      { issue: "Non-positive UnitPrice", rows: nonPositiveUnitPrice },
      { issue: "UnitPrice >= 10,000", rows: priceOutliers },
    ],
    qualityDashboard: [
      { metric: "Missing CustomerID", count: missingCustomerId },
      { metric: "Cancelled Rows", count: cancelledRows },
      { metric: "Non-positive Quantity", count: nonPositiveQuantity },
      { metric: "Non-positive UnitPrice", count: nonPositiveUnitPrice },
      { metric: "Price Outliers", count: priceOutliers },
    ],
    countryStats,
    monthlyTrendData,
    datasetSampleRows: sampleRows,
  });
}
