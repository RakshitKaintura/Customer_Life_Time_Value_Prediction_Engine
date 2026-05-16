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
    color: "hsl(var(--chart-1))",
    bg:    "bg-muted",
    text:  "text-foreground",
    border:"border-border",
    dot:   "bg-foreground",
  },
  high_value: {
    label: "High Value",
    color: "hsl(var(--chart-2))",
    bg:    "bg-muted",
    text:  "text-foreground",
    border:"border-border",
    dot:   "bg-foreground/80",
  },
  medium_value: {
    label: "Medium Value",
    color: "hsl(var(--chart-3))",
    bg:    "bg-muted",
    text:  "text-foreground",
    border:"border-border",
    dot:   "bg-foreground/60",
  },
  low_value: {
    label: "Low Value",
    color: "hsl(var(--chart-4))",
    bg:    "bg-muted",
    text:  "text-foreground",
    border:"border-border",
    dot:   "bg-foreground/40",
  },
} as const;

export type SegmentKey = keyof typeof SEGMENT_CONFIG;

export function getSegmentConfig(segment: string) {
  return SEGMENT_CONFIG[segment as SegmentKey] ?? SEGMENT_CONFIG.low_value;
}