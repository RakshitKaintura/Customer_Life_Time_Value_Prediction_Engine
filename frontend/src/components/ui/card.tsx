import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}

export function Card({ children, className, hover = false }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-slate-200/70 bg-white/85 p-6 shadow-[0_16px_30px_-24px_rgba(15,23,42,0.5)] backdrop-blur-sm",
        hover && "transition-all hover:-translate-y-0.5 hover:shadow-[0_18px_36px_-24px_rgba(15,23,42,0.6)]",
        className
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-4 flex items-center justify-between", className)}>
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <h3 className={cn("text-sm font-semibold text-slate-700", className)}>
      {children}
    </h3>
  );
}

export function CardValue({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <p className={cn("text-2xl font-bold text-slate-900", className)}>
      {children}
    </p>
  );
}