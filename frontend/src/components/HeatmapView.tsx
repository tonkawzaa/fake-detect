"use client";

import { useState } from "react";

export function HeatmapView({
  previewUrl,
  heatmapPng,
}: {
  previewUrl: string;
  heatmapPng: string | null;
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Where the model looked</h3>
        {heatmapPng && (
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            className="rounded-full border border-card-border px-3 py-1 text-sm hover:border-accent"
          >
            {showHeatmap ? "Show original" : "Show saliency heatmap"}
          </button>
        )}
      </div>
      <div className="mt-4 flex justify-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={showHeatmap && heatmapPng ? heatmapPng : previewUrl}
          alt={showHeatmap ? "Saliency heatmap overlay" : "Uploaded photo"}
          className="max-h-96 rounded-xl border border-card-border object-contain"
        />
      </div>
      {!heatmapPng && <p className="mt-2 text-center text-xs text-muted">Heatmap unavailable for this image.</p>}
      <p className="mt-3 text-xs text-muted">
        Saliency map (gradient of the AI-probability logit w.r.t. input pixels) over the whole frame.
      </p>
    </div>
  );
}
