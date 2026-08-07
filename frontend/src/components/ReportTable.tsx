"use client";

import type { FeedbackVote, ReportRow } from "@/lib/reportRow";
import { VERDICT_COPY, BAND_COPY } from "@/lib/verdictCopy";
import { FeedbackButtons } from "./FeedbackButtons";

function StatusCell({ row }: { row: ReportRow }) {
  if (row.status === "pending") return <span className="text-xs text-muted">Queued</span>;
  if (row.status === "loading")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted">
        <span className="h-3 w-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        Analyzing…
      </span>
    );
  if (row.status === "error") return <span className="text-xs font-medium text-danger">Failed</span>;
  return <span className="text-xs font-medium text-safe">Done</span>;
}

function Badge({ label, colorVar }: { label: string; colorVar: string }) {
  return (
    <span
      className="whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-semibold"
      style={{ background: `color-mix(in srgb, var(${colorVar}) 18%, transparent)`, color: `var(${colorVar})` }}
    >
      {label}
    </span>
  );
}

export function ReportTable({
  rows,
  selectedId,
  onSelect,
  onVote,
}: {
  rows: ReportRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onVote: (rowId: string, vote: FeedbackVote) => void;
}) {
  const doneCount = rows.filter((r) => r.status === "done" || r.status === "error").length;

  return (
    <div className="rounded-2xl border border-card-border bg-card p-4 sm:p-6">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Report ({doneCount}/{rows.length})</h3>
        <p className="text-xs text-muted">Click a row for full details</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-card-border text-left text-xs uppercase tracking-wide text-muted">
              <th className="py-2 pr-3 font-medium">Photo</th>
              <th className="py-2 pr-3 font-medium">Status</th>
              <th className="py-2 pr-3 font-medium">Verdict</th>
              <th className="py-2 pr-3 font-medium">AI probability</th>
              <th className="py-2 pr-3 font-medium">Confidence</th>
              <th className="py-2 pr-3 font-medium">Feedback</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const verdict = row.report?.verdict ? VERDICT_COPY[row.report.verdict] : null;
              const isSelectable = row.status === "done" || row.status === "error";
              return (
                <tr
                  key={row.id}
                  onClick={() => isSelectable && onSelect(row.id)}
                  className={`border-b border-card-border/60 last:border-b-0
                    ${isSelectable ? "cursor-pointer hover:bg-accent/5" : ""}
                    ${selectedId === row.id ? "bg-accent/10" : ""}`}
                >
                  <td className="py-2 pr-3">
                    <div className="flex items-center gap-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={row.previewUrl} alt={row.file.name} className="h-10 w-10 rounded-lg border border-card-border object-cover" />
                      <span className="max-w-[10rem] truncate text-xs text-muted" title={row.file.name}>
                        {row.file.name}
                      </span>
                    </div>
                  </td>
                  <td className="py-2 pr-3">
                    <StatusCell row={row} />
                  </td>
                  <td className="py-2 pr-3">
                    {verdict ? (
                      <Badge label={verdict.shortLabel} colorVar={verdict.colorVar} />
                    ) : row.status === "error" ? (
                      <span className="text-xs text-muted">{row.error ?? "—"}</span>
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-sm">
                    {row.report?.ai_probability != null ? `${(row.report.ai_probability * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-2 pr-3 text-xs text-muted">
                    {row.report?.confidence_band ? BAND_COPY[row.report.confidence_band] : "—"}
                  </td>
                  <td className="py-2 pr-3">
                    {row.report?.verdict ? (
                      <FeedbackButtons
                        compact
                        feedback={row.feedback}
                        submitting={!!row.feedbackSubmitting}
                        onVote={(vote) => onVote(row.id, vote)}
                      />
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
