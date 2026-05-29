"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useClerk } from "@clerk/nextjs";
import {
  LayoutDashboard,
  GitBranch,
  Search,
  Calculator,
  Activity,
  TrendingUp,
  BarChart3,
  ArrowUpRight,
  Sparkles,
  TableProperties,
  Megaphone,
  LogOut,
} from "lucide-react";

const navItems = [
  {
    href:        "/welcome",
    label:       "Welcome",
    icon:        Sparkles,
    description: "Project overview & quick start",
  },
  {
    href:        "/",
    label:       "LTV Overview",
    icon:        LayoutDashboard,
    description: "Score distribution & revenue concentration",
  },
  {
    href:        "/dataset",
    label:       "Dataset",
    icon:        TableProperties,
    description: "Short source data snapshot",
  },
  {
    href:        "/cohorts",
    label:       "Cohort Analysis",
    icon:        TrendingUp,
    description: "Retention curves & LTV by cohort",
  },
  {
    href:        "/causal",
    label:       "Causal Insights",
    icon:        GitBranch,
    description: "What causes high LTV",
  },
  {
    href:        "/customers",
    label:       "Customer Lookup",
    icon:        Search,
    description: "Individual predictions & lookalikes",
  },
  {
    href:        "/simulator",
    label:       "Marketing ROI",
    icon:        Calculator,
    description: "CAC simulator & channel ROI",
  },
  {
    href:        "/campaigns",
    label:       "Campaigns",
    icon:        Megaphone,
    description: "Airtable sync & Brevo emails",
  },
  {
    href:        "/model-health",
    label:       "Model Health",
    icon:        Activity,
    description: "MAE, Gini, drift monitoring",
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const router   = useRouter();
  const { signOut } = useClerk();

  const handleSignOut = async () => {
    await signOut();
    router.push("/welcome");
  };

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-border bg-background/80 backdrop-blur">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="relative flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-foreground">
          <BarChart3 className="h-4 w-4 text-background" />
          <ArrowUpRight className="absolute -right-0.5 -top-0.5 h-3 w-3 text-background" />
        </div>
        <div>
          <p className="text-sm font-semibold tracking-tight text-foreground">LTV Engine</p>
          <p className="text-xs text-muted-foreground">Prediction Dashboard</p>
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
                      ? "bg-foreground text-background shadow-[0_12px_24px_-20px_rgba(0,0,0,0.55)]"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 shrink-0",
                      active ? "text-background" : "text-muted-foreground group-hover:text-foreground"
                    )}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium tracking-tight">{item.label}</p>
                    <p
                      className={cn(
                        "truncate text-xs",
                        active ? "text-background/70" : "text-muted-foreground"
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
      <div className="border-t border-border p-4">
        <button
          type="button"
          onClick={handleSignOut}
          className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl border border-border bg-card px-3 py-2 text-sm font-medium text-muted-foreground transition hover:bg-accent hover:text-foreground"
        >
          <LogOut className="h-4 w-4" />
          Sign Out
        </button>
        <div className="flex items-center gap-2 rounded-xl border border-border bg-card p-3">
          <div className="h-2 w-2 rounded-full bg-foreground" />
          <p className="text-xs text-muted-foreground">API connected</p>
        </div>
      </div>
    </aside>
  );
}
