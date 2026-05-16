"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import type { ComponentType, ReactNode, SVGProps } from "react";
import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

type FadeInProps = {
  children: ReactNode;
  className?: string;
  delay?: number;
};

export function FadeIn({ children, className, delay = 0 }: FadeInProps) {
  return (
    <motion.div
      className={className}
      variants={fadeUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-120px" }}
      transition={{ duration: 0.5, ease: "easeOut", delay }}
    >
      {children}
    </motion.div>
  );
}

type SectionHeaderProps = {
  eyebrow: string;
  title: string;
  description?: string;
  badge?: string;
};

export function SectionHeader({ eyebrow, title, description, badge }: SectionHeaderProps) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-6">
      <div className="max-w-2xl">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">{eyebrow}</p>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">{title}</h2>
        {description ? (
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground sm:text-base">{description}</p>
        ) : null}
      </div>
      {badge ? (
        <span className="rounded-full border border-border bg-card px-3 py-1 text-xs uppercase tracking-[0.25em] text-muted-foreground">
          {badge}
        </span>
      ) : null}
    </div>
  );
}

type FeatureCardProps = {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  title: string;
  description: string;
};

export function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.25 }}
      className="group rounded-2xl border border-border bg-card p-6 shadow-[0_12px_40px_-32px_rgba(0,0,0,0.6)]"
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-background">
        <Icon className="h-5 w-5 text-foreground" />
      </div>
      <h3 className="mt-4 text-base font-semibold text-foreground">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
      <div className="mt-6 h-px w-full bg-linear-to-r from-transparent via-border to-transparent" />
    </motion.div>
  );
}

type MetricCardProps = {
  value: string;
  label: string;
  detail?: string;
};

export function MetricCard({ value, label, detail }: MetricCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-card px-5 py-6">
      <div className="text-2xl font-semibold text-foreground">{value}</div>
      <div className="mt-2 text-xs uppercase tracking-[0.3em] text-muted-foreground">{label}</div>
      {detail ? <div className="mt-3 text-sm text-muted-foreground">{detail}</div> : null}
    </div>
  );
}

type CodeCardProps = {
  title: string;
  code: string;
  accent?: string;
};

export function CodeCard({ title, code, accent }: CodeCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-muted-foreground">
        <span>{title}</span>
        {accent ? <span className="text-muted-foreground">{accent}</span> : null}
      </div>
      <pre className="mt-4 overflow-x-auto rounded-xl bg-foreground p-4 text-xs text-background">
        <code className="whitespace-pre-wrap leading-relaxed">{code}</code>
      </pre>
    </div>
  );
}

type TerminalMetric = {
  label: string;
  value: string;
  note?: string;
};

type TerminalCardProps = {
  title: string;
  status: string;
  metrics: TerminalMetric[];
};

export function TerminalCard({ title, status, metrics }: TerminalCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-[0_18px_44px_-36px_rgba(0,0,0,0.7)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-3 w-3 items-center justify-center rounded-full bg-foreground" />
          <span className="text-xs uppercase tracking-[0.3em] text-muted-foreground">{title}</span>
        </div>
        <span className="rounded-full border border-border bg-background px-3 py-1 text-[11px] uppercase tracking-[0.3em] text-muted-foreground">
          {status}
        </span>
      </div>
      <div className="mt-6 space-y-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{metric.label}</span>
            <div className="text-right">
              <div className="text-foreground">{metric.value}</div>
              {metric.note ? <div className="text-xs text-muted-foreground">{metric.note}</div> : null}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-6 rounded-xl border border-border bg-background px-4 py-3 text-xs text-muted-foreground">
        inference_latency: <span className="text-foreground">7.8ms</span> · model_status: <span className="text-foreground">green</span>
      </div>
    </div>
  );
}

type InlineChipProps = {
  text: string;
};

export function InlineChip({ text }: InlineChipProps) {
  return (
    <span className="rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground">
      {text}
    </span>
  );
}

type LandingNavProps = {
  className?: string;
};

export function LandingNav({ className }: LandingNavProps) {
  return (
    <nav className={cn("sticky top-0 z-50 border-b border-border bg-background/70 backdrop-blur", className)}>
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/welcome" className="flex items-center gap-3 text-sm font-semibold text-foreground">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-foreground text-background">
            L
          </span>
          LTV Engine
        </Link>
        <div className="hidden items-center gap-6 text-sm text-muted-foreground md:flex">
          <Link href="/docs" className="transition-colors hover:text-foreground">Documentation</Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 transition-colors hover:text-foreground"
          >
            GitHub
            <ArrowUpRight className="h-4 w-4" />
          </a>
          <Link href="/auth" className="transition-colors hover:text-foreground">Sign In</Link>
          <Link
            href="/auth"
            className="rounded-lg border border-border bg-foreground text-background px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition hover:bg-foreground/90"
          >
            Get Started
          </Link>
        </div>
        <div className="md:hidden">
          <Link
            href="/auth"
            className="rounded-lg border border-border bg-foreground text-background px-3 py-2 text-xs font-semibold uppercase tracking-[0.2em]"
          >
            Get Started
          </Link>
        </div>
      </div>
    </nav>
  );
}

type FooterColumn = {
  title: string;
  links: { label: string; href: string }[];
};

type LandingFooterProps = {
  columns: FooterColumn[];
};

export function LandingFooter({ columns }: LandingFooterProps) {
  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto grid max-w-6xl gap-10 px-6 py-16 sm:grid-cols-2 lg:grid-cols-4">
        {columns.map((column) => (
          <div key={column.title}>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">{column.title}</p>
            <div className="mt-4 space-y-2 text-sm text-muted-foreground">
              {column.links.map((link) => (
                <a key={link.label} href={link.href} className="block transition-colors hover:text-foreground">
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="border-t border-border py-6 text-center text-xs text-muted-foreground">
        © 2026 LTV Engine. All rights reserved.
      </div>
    </footer>
  );
}
