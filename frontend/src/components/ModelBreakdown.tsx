"use client";

import { useState } from "react";
import type { ModelScoreOut } from "@/lib/types";

export function ModelBreakdown({ models }: { models: ModelScoreOut[] }) {
  const [open, setOpen] = useState(false);
  if (models.length === 0) return null;

  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between text-left"
      >
        <h3 className="text-lg font-semibold">Per-model breakdown</h3>
        <span className="text-muted">{open ? "▲ Hide" : "▼ Show"}</span>
      </button>

      {open && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-card-border text-left text-muted">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 font-medium">Full frame</th>
                <th className="pb-2 font-medium">Face crop</th>
                <th className="pb-2 font-medium">Combined</th>
                <th className="pb-2 font-medium">Ensemble weight</th>
                <th className="pb-2 font-medium">Eval AUC</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-card-border last:border-0">
                  <td className="py-2 pr-2 font-medium">{m.name}</td>
                  <td className="py-2 pr-2">{(m.ai_probability_full * 100).toFixed(1)}%</td>
                  <td className="py-2 pr-2">{m.ai_probability_face != null ? `${(m.ai_probability_face * 100).toFixed(1)}%` : "—"}</td>
                  <td className="py-2 pr-2">{(m.ai_probability_combined * 100).toFixed(1)}%</td>
                  <td className="py-2 pr-2">{(m.weight * 100).toFixed(0)}%</td>
                  <td className="py-2 pr-2">{m.eval_auc != null ? m.eval_auc.toFixed(3) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
