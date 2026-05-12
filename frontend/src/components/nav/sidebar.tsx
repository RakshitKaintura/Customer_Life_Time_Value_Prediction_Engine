"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Users,
  GitBranch,
  Search,
  Calculator,
  Activity,
  TrendingUp,
  Zap,
} from "lucide-react";

const navItems = [
  {
    href:  "/",
    label: "LTV Overview",
    icon:  LayoutDashboard,
    description: "Score distribution & revenue concentration",
  },
  {
    href:  "/cohorts",
    label: "Cohort Analysis",
    icon:  TrendingUp,
    description: "Retention curves & LTV by cohort",
  },
  {
    href:  "/causal",
    label: "Causal Insights",
    icon:  GitBranch,
    description: "What causes high LTV",
  },
  {
    href:  "/customers",
    label: "Customer Lookup",
    icon:  Search,
    description: "Individual predictions & lookalikes",
  },
  {
    href:  "/simulator",
    label: "Marketing ROI",
    icon:  Calculator,
    description: "CAC simulator & channel ROI",
  },
  {
    href:  "/model-health",
    label: "Model Health",
    icon:  Activity,
    description: "MAE, Gini, drift monitoring",
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-slate-200/70 bg-white/70 backdrop-blur">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-slate-200/70 px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-slate-900 shadow-[0_8px_16px_-10px_rgba(15,23,42,0.5)]">
          <Zap className="h-4 w-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold tracking-tight text-slate-900">LTV Engine</p>
          <p className="text-xs text-slate-500">Prediction Dashboard</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        <ul className="space-y-2 px-4">
          {navItems.map((item) => {
            const active = pathname === item.href;
            const Icon   = item.icon;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "group flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition-all",
                    active
                      ? "bg-teal-600/95 text-white shadow-[0_10px_24px_-18px_rgba(15,23,42,0.7)] ring-1 ring-teal-500/40"
                      : "text-slate-700 hover:bg-white/70 hover:text-slate-900"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 shrink-0",
                      active ? "text-white" : "text-slate-400 group-hover:text-slate-700"
                    )}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium tracking-tight">{item.label}</p>
                    <p
                      className={cn(
                        "truncate text-xs",
                        active ? "text-teal-100" : "text-slate-400"
                      )}
                    >
                      {item.description}
                    </p>
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="border-t border-slate-200/70 p-4">
        <div className="flex items-center gap-2 rounded-xl bg-white/70 p-3 shadow-[0_10px_20px_-18px_rgba(15,23,42,0.45)]">
          <div className="h-2 w-2 rounded-full bg-emerald-500" />
          <p className="text-xs text-slate-600">API connected</p>
        </div>
      </div>
    </aside>
  );
}