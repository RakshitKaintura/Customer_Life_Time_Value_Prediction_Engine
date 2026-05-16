import { formatDate } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle, Clock, RefreshCw } from "lucide-react";

interface RetrainingRun {
  run_id: string;
  triggered_at: string;
  trigger_reason: string;
  status: string;
  new_gini: number | null;
  prev_gini: number | null;
  duration_minutes: number | null;
  deployed: boolean;
}

interface Props {
  data: RetrainingRun[];
}

const STATUS_ICONS: Record<string, React.ReactNode> = {
  success: <CheckCircle className="h-4 w-4 text-foreground" />,
  failed: <XCircle className="h-4 w-4 text-muted-foreground" />,
  running: <RefreshCw className="h-4 w-4 animate-spin text-foreground" />,
  default: <Clock className="h-4 w-4 text-muted-foreground" />,
};

export function RetrainingLog({ data }: Props) {
  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Retraining History</CardTitle>
        <span className="text-xs text-muted-foreground">{data.length} runs</span>
      </CardHeader>
      {data.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No retraining runs yet. Monthly schedule starts after initial deployment.
        </p>
      ) : (
        <div className="space-y-2">
          {data.map((run) => (
            <div key={run.run_id} className="flex items-center gap-3 rounded-lg border border-border bg-card p-3">
              {STATUS_ICONS[run.status] ?? STATUS_ICONS.default}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{run.trigger_reason.replace(/_/g, " ")}</span>
                  <Badge variant={run.deployed ? "success" : "default"}>{run.deployed ? "Deployed" : "Not deployed"}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {formatDate(run.triggered_at)}
                  {run.duration_minutes != null && ` | ${run.duration_minutes.toFixed(1)}min`}
                  {run.new_gini != null && ` | Gini: ${run.new_gini.toFixed(4)}`}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
