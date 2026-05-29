"use client";

import { CheckCircle, XCircle, Clock } from "lucide-react";
import { CardHeader, CardTitle } from "@/components/ui/card";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Run = Record<string, any>;

interface Props {
  runs:       Run[];
  dailyLimit: number | null;
}

function fmt(iso: string | null) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function parseResult(str: unknown, key: string): string {
  if (!str) return "—";
  const m = String(str).match(new RegExp(`'${key}':\\s*(\\d+)`));
  return m ? m[1] : "—";
}

export function SyncHistory({ runs, dailyLimit }: Props) {
  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Marketing Sync History</CardTitle>
        <span className="text-xs text-muted-foreground">
          Last {runs.length} runs of the 24-hour marketing_sync worker
          {dailyLimit ? ` · ${dailyLimit} emails/day limit` : ""}
        </span>
      </CardHeader>

      {runs.length === 0 ? (
        <div className="py-10 text-center space-y-2">
          <Clock className="mx-auto h-8 w-8 text-muted-foreground opacity-40" />
          <p className="text-sm font-medium text-muted-foreground">No sync runs yet</p>
          <p className="text-xs text-muted-foreground opacity-60 max-w-sm mx-auto">
            Run the marketing sync worker to populate this log.
            <br />
            <code className="rounded bg-muted px-1 py-0.5 mt-1 inline-block font-mono">
              python -m backend.workers.marketing_sync
            </code>
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr>
                {["Timestamp", "Status", "Airtable synced", "Brevo sent", "Run ID"].map((h) => (
                  <th
                    key={h}
                    className="pb-3 pr-6 text-left font-medium text-muted-foreground whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {runs.map((run, i) => {
                const ok    = run.status === "success";
                const meta  = run.metadata ?? {};
                const atRes = meta?.results?.airtable ?? null;
                const bvRes = meta?.results?.brevo    ?? null;
                const updated = parseResult(atRes, "updated");
                const sent    = parseResult(bvRes,  "sent");

                return (
                  <tr
                    key={run.run_id ?? i}
                    className="hover:bg-accent/50 transition-colors"
                  >
                    <td className="py-2.5 pr-6 font-mono text-muted-foreground whitespace-nowrap">
                      {fmt(run.started_at)}
                    </td>
                    <td className="py-2.5 pr-6">
                      <div className="flex items-center gap-1.5">
                        {ok
                          ? <CheckCircle className="h-3.5 w-3.5 text-foreground" />
                          : <XCircle    className="h-3.5 w-3.5 text-muted-foreground" />
                        }
                        <span className={ok ? "text-foreground font-medium" : "text-muted-foreground"}>
                          {run.status ?? "unknown"}
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-6 tabular-nums text-foreground">
                      {updated !== "—" ? `${updated} records` : <span className="text-muted-foreground">—</span>}
                    </td>
                    <td className="py-2.5 pr-6 tabular-nums text-foreground">
                      {sent !== "—" ? `${sent} emails` : <span className="text-muted-foreground">—</span>}
                    </td>
                    <td className="py-2.5 pr-6 font-mono text-muted-foreground truncate max-w-[160px]">
                      {run.run_id ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
