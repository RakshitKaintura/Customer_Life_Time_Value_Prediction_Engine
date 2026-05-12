"use client";

import { useState } from "react";
import { Search, Loader2 } from "lucide-react";
import { ltvApi } from "@/lib/api";
import { CustomerScoreCard } from "./customer-score-card";
import { ColdStartForm } from "./cold-start-form";

export function CustomerSearch() {
  const [query,   setQuery]   = useState("");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState<Record<string, unknown> | null>(null);
  const [error,   setError]   = useState<string | null>(null);
  const [mode,    setMode]    = useState<"existing" | "cold-start">("existing");

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await ltvApi.scoreCustomer(query.trim());
      setResult(data as unknown as Record<string, unknown>);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => { setMode("existing"); setResult(null); setError(null); }}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            mode === "existing"
              ? "bg-blue-600 text-white"
              : "border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          Existing Customer
        </button>
        <button
          onClick={() => { setMode("cold-start"); setResult(null); setError(null); }}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            mode === "cold-start"
              ? "bg-blue-600 text-white"
              : "border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          New Customer (Cold Start)
        </button>
      </div>

      {mode === "existing" ? (
        <>
          {/* Search bar */}
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSearch()}
                placeholder="Enter customer ID (e.g. 17850)"
                className="w-full rounded-lg border border-slate-200 bg-white py-2.5 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              Score
            </button>
          </div>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && <CustomerScoreCard data={result} />}
        </>
      ) : (
        <ColdStartForm />
      )}
    </div>
  );
}