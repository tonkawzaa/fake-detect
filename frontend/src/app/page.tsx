"use client";

import { useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { ReportTable } from "@/components/ReportTable";
import { VerdictCard } from "@/components/VerdictCard";
import { ProvenancePanel } from "@/components/ProvenancePanel";
import { ModelBreakdown } from "@/components/ModelBreakdown";
import { HeatmapView } from "@/components/HeatmapView";
import { LimitationsFooter } from "@/components/LimitationsFooter";
import { FeedbackStatsPanel } from "@/components/FeedbackStatsPanel";
import { analyzeImage, getFeedbackStats, submitFeedback } from "@/lib/api";
import type { ReportRow, FeedbackVote } from "@/lib/reportRow";
import type { FeedbackStatsOut } from "@/lib/types";

export default function Home() {
  const [rows, setRows] = useState<ReportRow[]>([]);
  const [running, setRunning] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStatsOut | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  async function handleFiles(files: File[]) {
    const newRows: ReportRow[] = files.map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      status: "pending",
    }));
    setRows((prev) => [...prev, ...newRows]);
    setRunning(true);

    // Sequential on purpose: the backend's /analyze handler runs the model
    // pipeline synchronously in a single event loop, so concurrent requests
    // from the client would just queue behind each other anyway -- doing it
    // sequentially here lets the table fill in row-by-row instead of all
    // finishing at once at the end.
    for (const row of newRows) {
      setRows((prev) => prev.map((r) => (r.id === row.id ? { ...r, status: "loading" } : r)));
      try {
        const report = await analyzeImage(row.file);
        setRows((prev) => prev.map((r) => (r.id === row.id ? { ...r, status: "done", report } : r)));
      } catch (e) {
        setRows((prev) =>
          prev.map((r) =>
            r.id === row.id ? { ...r, status: "error", error: e instanceof Error ? e.message : "Analysis failed." } : r
          )
        );
      }
    }
    setRunning(false);
  }

  function reset() {
    for (const row of rows) URL.revokeObjectURL(row.previewUrl);
    setRows([]);
    setSelectedId(null);
    setRunning(false);
  }

  async function handleVote(rowId: string, vote: FeedbackVote) {
    const row = rows.find((r) => r.id === rowId);
    if (!row?.report?.verdict) return;
    setRows((prev) => prev.map((r) => (r.id === rowId ? { ...r, feedbackSubmitting: true } : r)));
    try {
      await submitFeedback({ analysis_id: row.report.analysis_id, verdict: row.report.verdict, is_correct: vote === "up" });
      setRows((prev) => prev.map((r) => (r.id === rowId ? { ...r, feedback: vote, feedbackSubmitting: false } : r)));
    } catch {
      setRows((prev) => prev.map((r) => (r.id === rowId ? { ...r, feedbackSubmitting: false } : r)));
    }
  }

  async function toggleStats() {
    if (feedbackStats) {
      setFeedbackStats(null);
      return;
    }
    setStatsLoading(true);
    try {
      setFeedbackStats(await getFeedbackStats());
    } finally {
      setStatsLoading(false);
    }
  }

  const selectedRow = rows.find((r) => r.id === selectedId) ?? null;

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 px-4 py-10 sm:px-6">
      <header className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-bold sm:text-3xl">AI Image Detector</h1>
        <p className="max-w-xl text-sm text-muted">
          Local, offline analysis of any photo — is it AI-generated? Select one photo or several at once.
        </p>
        <button onClick={toggleStats} className="text-xs text-muted underline hover:text-accent" disabled={statsLoading}>
          {feedbackStats ? "Hide feedback stats" : statsLoading ? "Loading…" : "View feedback stats"}
        </button>
      </header>

      {feedbackStats && <FeedbackStatsPanel stats={feedbackStats} />}

      {rows.length === 0 && <UploadZone onFiles={handleFiles} disabled={running} />}

      {rows.length > 0 && (
        <div className="flex flex-col gap-6">
          {running && (
            <div className="flex items-center justify-center gap-3 text-sm text-muted">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              Analyzing {rows.filter((r) => r.status === "done" || r.status === "error").length + 1} of {rows.length}…
            </div>
          )}

          <ReportTable rows={rows} selectedId={selectedId} onSelect={setSelectedId} onVote={handleVote} />

          {selectedRow && (
            <div className="flex flex-col gap-6">
              {selectedRow.status === "error" ? (
                <div className="rounded-2xl border border-danger/30 bg-danger/5 p-6 text-center">
                  <p className="font-medium text-danger">Analysis failed for {selectedRow.file.name}</p>
                  <p className="mt-1 text-sm text-muted">{selectedRow.error}</p>
                </div>
              ) : selectedRow.report ? (
                <>
                  <VerdictCard report={selectedRow.report} />
                  <HeatmapView previewUrl={selectedRow.previewUrl} heatmapPng={selectedRow.report.heatmap_png} />
                  {selectedRow.report.provenance && <ProvenancePanel provenance={selectedRow.report.provenance} />}
                  <ModelBreakdown models={selectedRow.report.models} />
                  <LimitationsFooter limitations={selectedRow.report.limitations} />
                </>
              ) : null}
            </div>
          )}

          <div className="flex justify-center gap-3">
            {!running && <UploadZone onFiles={handleFiles} disabled={running} />}
          </div>

          {!running && (
            <button
              onClick={reset}
              className="self-center rounded-full border border-card-border px-5 py-2 text-sm font-medium hover:border-accent"
            >
              Start over
            </button>
          )}
        </div>
      )}

      <footer className="mt-auto pt-8 text-center text-xs text-muted">
        Runs fully locally — your photos are sent only to the backend on your machine.
      </footer>
    </main>
  );
}
