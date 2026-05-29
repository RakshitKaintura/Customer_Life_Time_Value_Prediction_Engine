"use client";

import { CheckCircle2, AlertCircle, Database, Mail } from "lucide-react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Run = Record<string, any>;

interface BrevoConfig {
  senderName:  string | null;
  senderEmail: string | null;
  dailyLimit:  number | null;
  templates:   Record<string, number | null>;
}

interface AirtableConfig {
  baseId:     string | null;
  tableId:    string | null;
  emailField: string;
}

interface Props {
  pipelineRuns: Run[];
  config: {
    brevo:    BrevoConfig;
    airtable: AirtableConfig;
  };
}

function fmt(iso: string | null) {
  if (!iso) return "Never";
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

/** Mask middle of a long ID: appa2V0Po…SSoe */
function maskId(id: string | null): string {
  if (!id) return "Not configured";
  if (id.length <= 8) return id;
  return `${id.slice(0, 6)}…${id.slice(-4)}`;
}

export function IntegrationStatus({ pipelineRuns, config }: Props) {
  const lastRun      = pipelineRuns[0] ?? null;
  const isConfigured = pipelineRuns.length > 0;
  const lastAt       = lastRun?.started_at ?? null;

  const meta        = lastRun?.metadata ?? {};
  const airtableRes = meta?.results?.airtable ?? "";
  const brevoRes    = meta?.results?.brevo    ?? "";
  const updated     = parseResult(airtableRes, "updated");
  const sent        = parseResult(brevoRes,    "sent");

  const { brevo, airtable } = config;

  const templatesConfigured = Object.values(brevo.templates).filter(Boolean).length;

  const cards = [
    {
      id:          "airtable",
      icon:        Database,
      name:        "Airtable CRM Sync",
      description: "Pushes LTV scores, segments and recommended CAC into your Airtable contact base every 24 hours.",
      stat:        updated !== "—" ? `${updated} records synced` : "Not yet synced",
      details: [
        { label: "Base ID",      value: maskId(airtable.baseId) },
        { label: "Table ID",     value: maskId(airtable.tableId) },
        { label: "Email field",  value: airtable.emailField },
        { label: "Last run",     value: fmt(lastAt) },
      ],
    },
    {
      id:          "brevo",
      icon:        Mail,
      name:        "Brevo Email Engine",
      description: "Sends personalised transactional emails per LTV segment via Brevo templates.",
      stat:        sent !== "—" ? `${sent} / ${brevo.dailyLimit ?? "∞"} emails today` : "Not yet sent",
      details: [
        { label: "Sender",      value: brevo.senderName   ?? "Not configured" },
        { label: "From",        value: brevo.senderEmail  ?? "Not configured" },
        { label: "Daily limit", value: brevo.dailyLimit ? `${brevo.dailyLimit} emails/day` : "Not set" },
        { label: "Templates",   value: `${templatesConfigured} / 4 configured` },
      ],
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {cards.map(({ id, icon: Icon, name, description, stat, details }) => (
        <div key={id} className="chart-container space-y-4">
          {/* Header row */}
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-foreground">
                <Icon className="h-4 w-4" />
              </div>
              <p className="text-sm font-semibold text-foreground">{name}</p>
            </div>
            <div className="flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-secondary px-2.5 py-1 text-xs font-medium">
              {isConfigured
                ? <><CheckCircle2 className="h-3 w-3 text-foreground" /><span className="text-foreground ml-1">Active</span></>
                : <><AlertCircle  className="h-3 w-3 text-muted-foreground" /><span className="text-muted-foreground ml-1">Pending</span></>
              }
            </div>
          </div>

          {/* Description */}
          <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>

          {/* Real config details grid */}
          <div className="rounded-lg border border-border bg-secondary p-3 grid grid-cols-2 gap-x-4 gap-y-2">
            {details.map(({ label, value }) => (
              <div key={label}>
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="text-xs font-medium text-foreground font-mono truncate">{value}</p>
              </div>
            ))}
          </div>

          {/* Stat box */}
          <div className="rounded-lg border border-border bg-secondary px-4 py-3 text-center">
            <p className="text-base font-bold text-foreground font-mono">{stat}</p>
            <p className="text-xs text-muted-foreground mt-0.5">last sync result</p>
          </div>
        </div>
      ))}
    </div>
  );
}
