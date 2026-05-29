import { Topbar } from "@/components/nav/topbar";
import { getCampaignData } from "@/lib/supabase/queries";
import { IntegrationStatus } from "./integration-status";
import { SegmentCards } from "./segment-cards";
import { SyncHistory } from "./sync-history";

export const revalidate = 120;

// Real integration config — read server-side from env vars
function getIntegrationConfig() {
  return {
    brevo: {
      senderName:   process.env.BREVO_SENDER_NAME   ?? null,
      senderEmail:  process.env.BREVO_SENDER_EMAIL  ?? null,
      dailyLimit:   process.env.BREVO_DAILY_LIMIT   ? Number(process.env.BREVO_DAILY_LIMIT)  : null,
      templates: {
        champions:    process.env.BREVO_TEMPLATE_CHAMPIONS ? Number(process.env.BREVO_TEMPLATE_CHAMPIONS) : null,
        high_value:   process.env.BREVO_TEMPLATE_HIGH      ? Number(process.env.BREVO_TEMPLATE_HIGH)      : null,
        medium_value: process.env.BREVO_TEMPLATE_MEDIUM    ? Number(process.env.BREVO_TEMPLATE_MEDIUM)    : null,
        low_value:    process.env.BREVO_TEMPLATE_LOW       ? Number(process.env.BREVO_TEMPLATE_LOW)       : null,
      },
    },
    airtable: {
      baseId:      process.env.AIRTABLE_BASE_ID     ?? null,
      tableId:     process.env.AIRTABLE_TABLE_ID    ?? null,
      emailField:  process.env.AIRTABLE_EMAIL_FIELD ?? "email",
    },
  };
}

export default async function CampaignsPage() {
  const [{ segments, pipelineRuns }, config] = await Promise.all([
    getCampaignData(),
    Promise.resolve(getIntegrationConfig()),
  ]);

  return (
    <div className="page-container">
      <Topbar
        title="Campaign Hub"
        subtitle="Airtable CRM sync · Brevo segment email campaigns · 24-hour marketing sync"
      />
      <div className="page-content space-y-6">
        <IntegrationStatus pipelineRuns={pipelineRuns} config={config} />
        <SegmentCards segments={segments} brevoTemplates={config.brevo.templates} />
        <SyncHistory runs={pipelineRuns} dailyLimit={config.brevo.dailyLimit} />
      </div>
    </div>
  );
}
