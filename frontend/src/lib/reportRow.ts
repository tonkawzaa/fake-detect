import type { AnalyzeReport } from "./types";

export type RowStatus = "pending" | "loading" | "done" | "error";

export interface ReportRow {
  id: string;
  file: File;
  previewUrl: string;
  status: RowStatus;
  report?: AnalyzeReport;
  error?: string;
}
