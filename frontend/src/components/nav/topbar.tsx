import { Bell, RefreshCw } from "lucide-react";

interface TopbarProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

export function Topbar({ title, subtitle, actions }: TopbarProps) {
  return (
    <header className="flex h-16 items-center justify-between border-b border-slate-200/70 bg-white/75 px-6 backdrop-blur">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {subtitle && (
          <p className="text-sm text-slate-500">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-3">
        {actions}
        <button className="flex items-center gap-1.5 rounded-lg border border-slate-200/80 bg-white/80 px-3 py-1.5 text-sm text-slate-700 shadow-[0_10px_20px_-18px_rgba(15,23,42,0.5)] transition hover:-translate-y-0.5 hover:border-teal-200 hover:text-slate-900">
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
        <button className="relative rounded-lg border border-transparent p-2 text-slate-500 transition hover:border-slate-200/70 hover:bg-white/70">
          <Bell className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}