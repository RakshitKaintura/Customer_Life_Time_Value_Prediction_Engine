import { cn, getSegmentConfig } from "@/lib/utils";

interface SegmentBadgeProps {
  segment: string;
  className?: string;
  size?: "sm" | "md";
}

export function SegmentBadge({
  segment,
  className,
  size = "md",
}: SegmentBadgeProps) {
  const cfg = getSegmentConfig(segment);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-medium",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm",
        cfg.bg,
        cfg.text,
        className
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", cfg.dot)} />
      {cfg.label}
    </span>
  );
}