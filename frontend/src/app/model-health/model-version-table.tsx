import { formatDate } from "@/lib/utils";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Props {
  fusion:      Record<string, unknown> | null;
  bgnbd:       Record<string, unknown> | null;
  transformer: Record<string, unknown> | null;
}

export function ModelVersionTable({ fusion, bgnbd, transformer }: Props) {
  const models = [
    {
      name:    "XGBoost Fusion",
      version: String(fusion?.model_version ?? "—"),
      trained: String(fusion?.trained_at ?? ""),
      mae:     Number(fusion?.mae_ltv_12m ?? 0),
      gini:    Number(fusion?.gini_coefficient ?? 0),
      status:  fusion ? "active" : "not_trained",
    },
    {
      name:    "BG/NBD + Gamma-Gamma",
      version: String(bgnbd?.model_version ?? "—"),
      trained: String(bgnbd?.fitted_at ?? ""),
      mae:     Number(bgnbd?.mae_ltv_12m ?? 0),
      gini:    null,
      status:  bgnbd ? "active" : "not_trained",
    },
    {
      name:    "Transformer (ONNX)",
      version: String(transformer?.model_version ?? "—"),
      trained: String(transformer?.trained_at ?? ""),
      mae:     Number(transformer?.mae_ltv_12m ?? 0),
      gini:    Number(transformer?.gini_coefficient ?? 0),
      status:  transformer ? "active" : "not_trained",
    },
  ];

  return (
    <div className="chart-container">
      <CardHeader>
        <CardTitle>Model Registry</CardTitle>
        <span className="text-xs text-slate-400">Currently deployed model versions</span>
      </CardHeader>
      <div className="space-y-3">
        {models.map((model) => (
          <div
            key={model.name}
            className="rounded-lg border border-slate-100 bg-slate-50 p-4"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="font-medium text-slate-900 text-sm">{model.name}</p>
                <p className="font-mono text-xs text-slate-500 mt-0.5">{model.version}</p>
              </div>
              <Badge
                variant={model.status === "active" ? "success" : "warning"}
              >
                {model.status === "active" ? "Active" : "Not trained"}
              </Badge>
            </div>
            {model.status === "active" && (
              <div className="mt-3 flex gap-4 text-xs text-slate-500">
                <span>MAE: <strong className="text-slate-700">£{model.mae.toFixed(0)}</strong></span>
                {model.gini != null && (
                  <span>Gini: <strong className="text-slate-700">{model.gini.toFixed(4)}</strong></span>
                )}
                <span>Trained: <strong className="text-slate-700">{formatDate(model.trained)}</strong></span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}