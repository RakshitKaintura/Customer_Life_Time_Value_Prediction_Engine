"use client";

import { useState, useEffect } from "react";
import { Users, Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { SegmentBadge } from "@/components/ui/segment-badge";
import { formatCurrency } from "@/lib/utils";
import { ltvApi } from "@/lib/api";

interface Props {
  customerId:        string;
  initialLookalikes: string[];
}

interface LookalikeData {
  candidate_customer_id: string;
  similarity:            number;
  ltv_36m:               number;
  segment:               string;
}

export function LookalikePanel({ customerId, initialLookalikes }: Props) {
  const [lookalikes, setLookalikes] = useState<LookalikeData[]>([]);
  const [loading,    setLoading]    = useState(false);

  useEffect(() => {
    if (!customerId) return;
    setLoading(true);
    ltvApi
      .getLookalikes(customerId, 10)
      .then(res => setLookalikes(res.lookalikes as unknown as LookalikeData[]))
      .catch(() => setLookalikes([]))
      .finally(() => setLoading(false));
  }, [customerId]);

  if (loading) {
    return (
      <Card>
        <div className="flex items-center gap-2 py-4 text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading lookalike customers…</span>
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
          <Users className="h-4 w-4 text-blue-500" />
          <CardTitle>Lookalike Customers</CardTitle>
        </div>
        <span className="text-xs text-slate-400">
          Most similar existing customers (pgvector ANN)
        </span>
      </CardHeader>
      <div className="space-y-2">
        {lookalikes.length > 0 ? (
          lookalikes.map((l) => (
            <div
              key={l.candidate_customer_id}
              className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2"
            >
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm text-blue-600">
                  {l.candidate_customer_id}
                </span>
                <SegmentBadge segment={l.segment} size="sm" />
              </div>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-slate-500">
                  {(Number(l.similarity) * 100).toFixed(1)}% similar
                </span>
                <span className="font-medium text-slate-900">
                  {formatCurrency(Number(l.ltv_36m))}
                </span>
              </div>
            </div>
          ))
        ) : (
          initialLookalikes.slice(0, 5).map((id) => (
            <div
              key={id}
              className="flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2"
            >
              <span className="font-mono text-sm text-blue-600">{id}</span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}