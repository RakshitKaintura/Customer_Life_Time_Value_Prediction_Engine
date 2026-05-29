"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  TrendingUp,
  GitBranch,
  Search,
  Activity,
  TableProperties,
  Megaphone,
} from "lucide-react";

const items = [
  { href: "/",            label: "Overview",  icon: LayoutDashboard },
  { href: "/dataset",     label: "Dataset",   icon: TableProperties },
  { href: "/cohorts",     label: "Cohorts",   icon: TrendingUp },
  { href: "/causal",      label: "Causal",    icon: GitBranch },
  { href: "/customers",   label: "Customers", icon: Search },
  { href: "/campaigns",   label: "Campaigns", icon: Megaphone },
  { href: "/model-health",label: "Health",    icon: Activity },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background/95 px-2 py-2 backdrop-blur lg:hidden">
      <ul className="grid grid-cols-7 gap-1">
        {items.map((item) => {
          const Icon   = item.icon;
          const active = pathname === item.href;
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-md px-2 py-2 text-[11px] transition-colors",
                  active ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="truncate">{item.label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
