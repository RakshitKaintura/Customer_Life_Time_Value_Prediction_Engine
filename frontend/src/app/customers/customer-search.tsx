"use client";

import { useState } from "react";
import { Search, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ltvApi } from "@/lib/api";
import { CustomerScoreCard } from "./customer-score-card";
import { ColdStartForm } from "./cold-start-form";
import { cn } from "@/lib/utils";

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
      <div className="inline-flex rounded-lg border border-border bg-card p-1">
        <button
          onClick={() => { setMode("existing"); setResult(null); setError(null); }}
          className={cn(
            "rounded-md px-4 py-2 text-sm font-medium transition-colors",
            mode === "existing" ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
          )}
        >
          Existing Customer
        </button>
        <button
          onClick={() => { setMode("cold-start"); setResult(null); setError(null); }}
          className={cn(
            "rounded-md px-4 py-2 text-sm font-medium transition-colors",
            mode === "cold-start" ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
          )}
        >
          New Customer (Cold Start)
        </button>
      </div>

      {mode === "existing" ? (
        <>
          {/* Search bar */}
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSearch()}
                placeholder="Enter customer ID (e.g. 17850)"
                className="pl-10"
              />
            </div>
            <Button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              Score
            </Button>
          </div>

          {error && (
            <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
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
