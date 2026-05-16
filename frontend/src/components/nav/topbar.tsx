import { Bell, RefreshCw } from "lucide-react";
import { ThemeToggle } from "@/components/ui/theme-toggle";

interface TopbarProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

export function Topbar({ title, subtitle, actions }: TopbarProps) {
  return (
    <header className="flex min-h-16 flex-wrap items-center justify-between gap-3 border-b border-border bg-background/80 px-4 py-3 sm:px-6 backdrop-blur">
      <div>
        <h1 className="text-lg font-semibold tracking-tight text-foreground sm:text-xl">{title}</h1>
        {subtitle && (
          <p className="text-xs text-muted-foreground sm:text-sm">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        {actions}
        <button className="hidden items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-sm text-muted-foreground shadow-[0_10px_20px_-18px_rgba(0,0,0,0.35)] transition hover:-translate-y-0.5 hover:text-foreground sm:flex">
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
        <ThemeToggle />
        <button className="relative rounded-lg border border-transparent p-2 text-muted-foreground transition hover:border-border hover:bg-accent">
          <Bell className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}
