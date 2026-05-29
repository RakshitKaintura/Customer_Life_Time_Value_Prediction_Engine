"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ltvApi, HealthResponse } from "@/lib/api";
import { Loader2, AlertTriangle, CheckCircle2 } from "lucide-react";

interface ApiStatusState {
  loading: boolean;
  data: HealthResponse | null;
  error: string | null;
}

export function ApiStatusBadge() {
  const [state, setState] = useState<ApiStatusState>({
    loading: true,
    data: null,
    error: null,
  });

  useEffect(() => {
    let isActive = true;

    ltvApi
      .health()
      .then((data) => {
        if (!isActive) return;
        setState({ loading: false, data, error: null });
      })
      .catch((error: Error) => {
        if (!isActive) return;
        setState({ loading: false, data: null, error: error.message });
      });

    return () => {
      isActive = false;
    };
  }, []);

  if (state.loading) {
    return (
      <Badge variant="info" className="gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        API: checking
      </Badge>
    );
  }

  if (state.error || !state.data) {
    return (
      <Badge variant="danger" className="gap-1">
        <AlertTriangle className="h-3 w-3" />
        API: offline
      </Badge>
    );
  }

  const environment = state.data.environment ?? "unknown";
  const ok = state.data.status === "ok";

  return (
    <Badge variant={ok ? "success" : "warning"} className="gap-1">
      {ok ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
      API: {state.data.status} ({environment})
    </Badge>
  );
}
