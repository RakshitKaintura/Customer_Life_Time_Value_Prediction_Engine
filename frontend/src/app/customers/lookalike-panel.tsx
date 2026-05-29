"use client";

import { useState, useEffect } from "react";
import { Users, Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { formatCurrency } from "@/lib/utils";
import { ltvApi } from "@/lib/api";

interface Props {
  customerId: string;
  initialLookalikes: string[];
}

interface LookalikeData {
  candidate_customer_id: string;
  similarity: number;
  ltv_36m: number | null;
  segment: string | null;
}

function formatLookalikeLtv(value: number | null | undefined) {
  return value == null ? "LTV unavailable" : formatCurrency(value);
}

function formatSimilarity(value: number) {
  const percent = value * 100;
  return percent > 99.9 ? `${percent.toFixed(4)}%` : `${percent.toFixed(1)}%`;
}

export function LookalikePanel({ customerId, initialLookalikes }: Props) {
  const [lookalikes, setLookalikes] = useState<LookalikeData[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!customerId) return;
    let isActive = true;
    Promise.resolve().then(() => {
      if (!isActive) return;
      setLoading(true);
      ltvApi
        .getLookalikes(customerId, 10)
        .then((res) => {
          if (isActive) setLookalikes(res.lookalikes as unknown as LookalikeData[]);
        })
        .catch(() => {
          if (isActive) setLookalikes([]);
        })
        .finally(() => {
          if (isActive) setLoading(false);
        });
    });
    return () => {
      isActive = false;
    };
  }, [customerId]);

  if (loading) {
    return (
      <Card>
        <div className="flex items-center gap-2 py-4 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm text-muted-foreground">Loading lookalike customers...</span>
        </div>
      </Card>
    );
  }

  if (lookalikes.length === 0 && initialLookalikes.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Users className="h-4 w-4 text-foreground" />
          <CardTitle>Lookalike Customers</CardTitle>
        </div>
        <span className="text-xs text-muted-foreground">Most similar existing customers (pgvector ANN)</span>
      </CardHeader>
      <div className="space-y-2">
        {lookalikes.length > 0 ? (
          lookalikes.map((l, index) => (
            <div
              key={l.candidate_customer_id}
              className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2"
            >
              <div className="flex items-center gap-3">
                <span className="w-7 text-xs font-medium text-muted-foreground">#{index + 1}</span>
                <span className="font-mono text-sm text-foreground">{l.candidate_customer_id}</span>
                {l.segment && <SegmentBadge segment={l.segment} size="sm" />}
              </div>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">{formatSimilarity(Number(l.similarity))} similar</span>
                <span className="font-medium text-foreground">{formatLookalikeLtv(l.ltv_36m)}</span>
              </div>
            </div>
          ))
        ) : (
          initialLookalikes.slice(0, 5).map((id) => (
            <div key={id} className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2">
              <span className="font-mono text-sm text-foreground">{id}</span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}
