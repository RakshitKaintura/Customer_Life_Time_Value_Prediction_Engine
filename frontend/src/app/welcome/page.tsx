"use client";

import Link from "next/link";
import {
  Activity,
  Box,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Radar,
  Shield,
  Timer,
  Zap,
} from "lucide-react";
import {
  CodeCard,
  FadeIn,
  FeatureCard,
  InlineChip,
  LandingFooter,
  LandingNav,
  MetricCard,
  SectionHeader,
  TerminalCard,
} from "@/components/landing/landing-components";

const featureItems = [
  {
    icon: Layers,
    title: "Hybrid ML Architecture",
    description: "Blend probabilistic models, transformers, and XGBoost fusion for explainable, high-accuracy LTV.",
  },
  {
    icon: Database,
    title: "Probabilistic Modeling",
    description: "BG/NBD and Gamma-Gamma forecasting tuned for repeat purchase intensity and value curves.",
  },
  {
    icon: Zap,
    title: "Real-Time Inference",
    description: "Sub-10ms ONNX inference with streaming feature refresh and low-latency scoring APIs.",
  },
  {
    icon: GitBranch,
    title: "Causal Analysis",
    description: "Counterfactual estimation, uplift segmentation, and intervention impact tracking at scale.",
  },
  {
    icon: Radar,
    title: "Monitoring & Drift",
    description: "Automated drift detection, retraining orchestration, and model health SLAs.",
  },
  {
    icon: Shield,
    title: "Enterprise Integrations",
    description: "Secure connectors for CRMs, data lakes, and analytics stacks with audit-grade logging.",
  },
];

const architectureLayers = [
  {
    title: "Data Layer",
    detail: "Streaming ingestion, CDC, and unified identity graph.",
  },
  {
    title: "Feature Store",
    detail: "Real-time feature materialization with point-in-time consistency.",
  },
  {
    title: "Probabilistic Engine",
    detail: "BG/NBD + Gamma-Gamma calibration and cohort priors.",
  },
  {
    title: "Transformer Models",
    detail: "Sequence embeddings for high-resolution behavior signals.",
  },
  {
    title: "Scoring API",
    detail: "FastAPI inference gateway with rate-limited tenants.",
  },
  {
    title: "Monitoring Layer",
    detail: "Drift alerts, retraining automation, and audit trail.",
  },
];

const metrics = [
  { value: "15+", label: "ML models", detail: "Calibrated ensembles" },
  { value: "<10ms", label: "Inference", detail: "ONNX optimized" },
  { value: "99.9%", label: "Uptime", detail: "Multi-region" },
  { value: "45+", label: "Test suites", detail: "Regression coverage" },
];

const terminalMetrics = [
  { label: "Predicted LTV", value: "$1,842.22", note: "High value" },
  { label: "Retention Score", value: "0.91", note: "+4.2% vs cohort" },
  { label: "Churn Probability", value: "0.07", note: "Low risk" },
  { label: "Cohort Rank", value: "Top 8%", note: "Q4 enterprise" },
];

const footerColumns = [
  {
    title: "Product",
    links: [
      { label: "Platform", href: "/" },
      { label: "Monitoring", href: "/model-health" },
      { label: "Integrations", href: "/" },
    ],
  },
  {
    title: "Developers",
    links: [
      { label: "Documentation", href: "/docs" },
      { label: "API Reference", href: "/docs" },
      { label: "SDKs", href: "/docs" },
    ],
  },
  {
    title: "Company",
    links: [
      { label: "Security", href: "/" },
      { label: "Compliance", href: "/" },
      { label: "Careers", href: "/" },
    ],
  },
  {
    title: "Social",
    links: [
      { label: "GitHub", href: "https://github.com" },
      { label: "LinkedIn", href: "https://linkedin.com" },
      { label: "Status", href: "/" },
    ],
  },
];

const requestSnippet = `POST /predict\n\n{\n  "customer_id": "A1021",\n  "country": "US",\n  "transactions": [\n    { "amount": 120.5, "timestamp": "2026-05-01" },\n    { "amount": 89.4, "timestamp": "2026-05-12" }\n  ]\n}`;

const responseSnippet = `{\n  "predicted_ltv": 1842.22,\n  "risk_segment": "high_value",\n  "retention_probability": 0.91,\n  "churn_probability": 0.07\n}`;

export default function WelcomePage() {
  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,hsl(var(--border))_0%,transparent_55%)] opacity-30" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-size-[120px_120px] opacity-10" />
      <LandingNav />
      <main className="relative z-10 mx-auto max-w-6xl px-6 pb-24 pt-16 lg:pt-24">
        <section className="grid items-center gap-12 lg:grid-cols-[1.1fr_0.9fr]">
          <FadeIn>
            <div>
              <span className="inline-flex items-center rounded-full border border-border bg-card px-3 py-1 text-xs uppercase tracking-[0.3em] text-muted-foreground">
                Production-ready ML Infrastructure
              </span>
              <h1 className="mt-6 text-4xl font-semibold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
                Predict Customer Lifetime Value with Precision
              </h1>
              <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
                A premium LTV platform that fuses probabilistic modeling, deep learning, causal inference, and real-time scoring so every segment is measurable, explainable, and ready for production.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-4">
                <Link
                  href="/auth"
                  className="rounded-lg border border-foreground bg-foreground px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.3em] text-background transition hover:bg-foreground/90"
                >
                  Get Started
                </Link>
                <Link
                  href="/docs"
                  className="rounded-lg border border-border bg-transparent px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.3em] text-foreground transition hover:bg-accent"
                >
                  View Docs
                </Link>
              </div>
              <div className="mt-8 flex flex-wrap gap-3">
                {[
                  "FastAPI",
                  "PyTorch",
                  "ONNX",
                  "Real-time inference",
                  "Enterprise ready",
                ].map((item) => (
                  <InlineChip key={item} text={item} />
                ))}
              </div>
            </div>
          </FadeIn>
          <FadeIn delay={0.1}>
            <TerminalCard title="ltv_engine" status="live" metrics={terminalMetrics} />
          </FadeIn>
        </section>

        <FadeIn className="mt-24">
          <SectionHeader
            eyebrow="Features"
            title="Enterprise-grade intelligence with a minimal surface area"
            description="Everything you need to deliver reliable LTV predictions, packaged with observability and compliance controls built in."
            badge="Vercel-class"
          />
          <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {featureItems.map((item) => (
              <FeatureCard key={item.title} {...item} />
            ))}
          </div>
        </FadeIn>

        <FadeIn className="mt-24">
          <SectionHeader
            eyebrow="Architecture"
            title="Composable LTV Stack"
            description="Modular layers that map cleanly to data, modeling, and scoring workflows across your org."
            badge="Production pipeline"
          />
          <div className="mt-10 grid gap-10 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="rounded-2xl border border-[#1F1F1F] bg-[#0A0A0A] p-8">
              <div className="grid gap-6 sm:grid-cols-2">
                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Database className="h-4 w-4" />
                    Data plane orchestration
                  </div>
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Cpu className="h-4 w-4" />
                    Multi-model inference
                  </div>
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Activity className="h-4 w-4" />
                    Live health and drift scoring
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Box className="h-4 w-4" />
                    Versioned feature catalog
                  </div>
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Timer className="h-4 w-4" />
                    SLA-aware scheduling
                  </div>
                  <div className="flex items-center gap-3 text-sm text-white">
                    <Radar className="h-4 w-4" />
                    Automated retraining triggers
                  </div>
                </div>
              </div>
            </div>
            <div className="relative rounded-2xl border border-[#1F1F1F] bg-black p-8">
              <div className="absolute left-6 top-10 bottom-10 w-px bg-white/10" />
              <div className="space-y-6">
                {architectureLayers.map((layer) => (
                  <div key={layer.title} className="relative pl-10">
                    <div className="absolute left-4 top-2.5 h-2.5 w-2.5 rounded-full bg-white" />
                    <div className="rounded-xl border border-[#1F1F1F] bg-[#0A0A0A] px-4 py-3">
                      <p className="text-sm font-semibold text-white">{layer.title}</p>
                      <p className="mt-1 text-xs text-zinc-400">{layer.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>

        <FadeIn className="mt-24">
          <SectionHeader
            eyebrow="Metrics"
            title="Measured for scale"
            description="Built for high-volume scoring, compliance, and continuous validation."
            badge="Operational"
          />
          <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {metrics.map((metric) => (
              <MetricCard key={metric.label} {...metric} />
            ))}
          </div>
        </FadeIn>

        <FadeIn className="mt-24">
          <SectionHeader
            eyebrow="API"
            title="Ship predictions with a single endpoint"
            description="Clean request/response contracts that integrate directly into your product and analytics stack."
            badge="Predict API"
          />
          <div className="mt-10 grid gap-6 lg:grid-cols-2">
            <CodeCard title="Request" code={requestSnippet} accent="POST /predict" />
            <CodeCard title="Response" code={responseSnippet} accent="200 OK" />
          </div>
        </FadeIn>

        <FadeIn className="mt-24">
          <div className="rounded-2xl border border-[#1F1F1F] bg-[#0A0A0A] px-8 py-12 text-center">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500">Deploy now</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white">Deploy your LTV infrastructure</h2>
            <p className="mt-4 text-sm text-zinc-400">
              Launch production-grade LTV scoring with a stack built for ML teams that move fast and operate at scale.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <Link
                href="/auth"
                className="rounded-lg border border-white bg-white px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.3em] text-black transition hover:bg-zinc-200"
              >
                Start Free
              </Link>
              <Link
                href="/docs"
                className="rounded-lg border border-[#1F1F1F] bg-transparent px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.3em] text-white transition hover:bg-white/10"
              >
                Read Documentation
              </Link>
            </div>
          </div>
        </FadeIn>
      </main>
      <LandingFooter columns={footerColumns} />
    </div>
  );
}
