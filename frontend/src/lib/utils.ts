import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value >= 1_000_000) return `£${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000)     return `£${(value / 1_000).toFixed(1)}K`;
  return `£${value.toFixed(0)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value == null) return "—";
  return new Intl.NumberFormat("en-GB").format(Math.round(value));
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("en-GB", {
    day:   "numeric",
    month: "short",
    year:  "numeric",
  });
}

export const SEGMENT_CONFIG = {
  champions: {
    label: "Champions",
    color: "#6366f1",
    bg:    "bg-indigo-100",
    text:  "text-indigo-800",
    border:"border-indigo-200",
    dot:   "bg-indigo-500",
  },
  high_value: {
    label: "High Value",
    color: "#3b82f6",
    bg:    "bg-blue-100",
    text:  "text-blue-800",
    border:"border-blue-200",
    dot:   "bg-blue-500",
  },
  medium_value: {
    label: "Medium Value",
    color: "#06b6d4",
    bg:    "bg-cyan-100",
    text:  "text-cyan-800",
    border:"border-cyan-200",
    dot:   "bg-cyan-500",
  },
  low_value: {
    label: "Low Value",
    color: "#94a3b8",
    bg:    "bg-slate-100",
    text:  "text-slate-700",
    border:"border-slate-200",
    dot:   "bg-slate-400",
  },
} as const;

export type SegmentKey = keyof typeof SEGMENT_CONFIG;

export function getSegmentConfig(segment: string) {
  return SEGMENT_CONFIG[segment as SegmentKey] ?? SEGMENT_CONFIG.low_value;
}