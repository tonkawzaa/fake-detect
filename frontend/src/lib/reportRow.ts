import type { AnalyzeReport } from "./types";

export type RowStatus = "pending" | "loading" | "done" | "error";

export type FeedbackVote = "up" | "down";

export interface ReportRow {
  id: string;
  file: File;
  previewUrl: string;
  status: RowStatus;
  report?: AnalyzeReport;
  error?: string;
  // Lives on the row (not local component state) because VerdictCard stays
  // mounted across row selection changes -- local state would leak a vote
  // from one row onto another when switching selectedId.
  feedback?: FeedbackVote;
  feedbackSubmitting?: boolean;
}
