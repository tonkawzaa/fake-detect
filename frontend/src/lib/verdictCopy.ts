// Shared label/color mapping for AnalyzeReport.verdict and
// reconstruction_check.verdict, used by VerdictCard.tsx and ReportTable.tsx
// so the two can't drift apart on what a given verdict string means.

export const VERDICT_COPY: Record<string, { label: string; shortLabel: string; colorVar: string }> = {
  likely_ai: { label: "Likely AI-generated", shortLabel: "Likely AI", colorVar: "--danger" },
  likely_real: { label: "Likely a real photo", shortLabel: "Likely real", colorVar: "--safe" },
  uncertain: { label: "Uncertain", shortLabel: "Uncertain", colorVar: "--warn" },
};

export const BAND_COPY: Record<string, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence (models disagreed)",
};

export const RECON_VERDICT_COPY: Record<string, string> = {
  likely_ai: "consistent with a Stable-Diffusion-family image",
  likely_real: "consistent with a real photo",
  uncertain: "inconclusive",
};
