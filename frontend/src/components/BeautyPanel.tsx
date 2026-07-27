import type { BeautyOut } from "@/lib/types";
import { Bar } from "./Gauge";

export const BEAUTY_LEVEL_COLOR: Record<string, string> = {
  None: "--safe",
  Light: "--accent",
  Moderate: "--warn",
  Heavy: "--danger",
};

const SUBSCORE_LABELS: Record<string, { label: string; hint: string }> = {
  f1_skin_hf_ratio: { label: "Skin sharpness vs. eyes/background", hint: "Primary signal: smoothing leaves eyes and background untouched." },
  f2_noise_residual: { label: "Sensor noise mismatch", hint: "Filters strip natural noise selectively from skin." },
  f3_lbp_entropy_drop: { label: "Skin texture (pore) loss", hint: "Smoothing collapses fine-grain skin texture." },
  f4_fft_highfreq: { label: "High-frequency detail loss", hint: "Frequency-domain check for lost texture." },
  f6_tone_uniformity: { label: "Skin tone evenness", hint: "Whitening/evening-out reduces local tone variance." },
  f7_geometry: { label: "Face-shape deviation", hint: "Compared to a baseline photo population — not a filter detector on its own." },
  f8_contour_warp: { label: "Jaw contour warp", hint: "Weak, low-confidence signal for slimming/liquify warps." },
};

export function BeautyPanel({ beauty }: { beauty: BeautyOut }) {
  const colorVar = BEAUTY_LEVEL_COLOR[beauty.level] ?? "--accent";
  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Beauty mode / retouching</h3>
        <span
          className="rounded-full px-3 py-1 text-sm font-semibold"
          style={{ background: `color-mix(in srgb, var(${colorVar}) 18%, transparent)`, color: `var(${colorVar})` }}
        >
          {beauty.level}
        </span>
      </div>
      <p className="mt-1 text-sm text-muted">
        Fused score: {(beauty.score * 100).toFixed(0)}%
        {!beauty.calibrated && " · using default (uncalibrated) thresholds"}
      </p>

      <div className="mt-4 flex flex-col gap-4">
        {Object.entries(beauty.subscores).map(([key, value]) => {
          const meta = SUBSCORE_LABELS[key] ?? { label: key, hint: "" };
          return <Bar key={key} value={value} label={meta.label} hint={meta.hint} />;
        })}
      </div>

      {beauty.notes.length > 0 && (
        <ul className="mt-4 flex flex-col gap-1 border-t border-card-border pt-3 text-xs text-muted">
          {beauty.notes.map((note, i) => (
            <li key={i}>⚠️ {note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
