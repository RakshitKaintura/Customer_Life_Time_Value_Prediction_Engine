# Airtable Automation (Run Script)

Use this when you want Airtable to call the API automatically for new records.

## Setup
1. Airtable -> Automations -> Create automation.
2. Trigger: "When record created" (Contacts table).
3. Action: "Run script".
4. Paste the script below.
5. Set the input variables in the script editor:
   - `apiUrl` (example: https://your-api-host/webhook/airtable)
   - `contactIdField` (example: contact_id)

## Script
```javascript
const inputConfig = input.config();
const apiUrl = inputConfig.apiUrl;
const contactIdField = inputConfig.contactIdField || "contact_id";
const recordId = inputConfig.recordId;

const table = base.getTable("Contacts");
const record = await table.selectRecordAsync(recordId);

if (!record) {
  throw new Error("Record not found");
}

const payload = {
  contact_id: record.getCellValueAsString(contactIdField),
  vertical: record.getCellValueAsString("vertical"),
  company_size: record.getCellValueAsString("company_size"),
  channel: record.getCellValueAsString("channel"),
  plan_tier: record.getCellValueAsString("plan_tier"),
};

const response = await fetch(apiUrl, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

if (!response.ok) {
  const text = await response.text();
  throw new Error(`Webhook failed: ${response.status} ${text}`);
}
```

## Input variables
Create input variables in Airtable automation:
- `apiUrl` (text)
- `contactIdField` (text, optional)
- `recordId` (record ID from the trigger step)
