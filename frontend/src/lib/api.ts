import type { AnalyzeReport, FeedbackIn, FeedbackOut, FeedbackStatsOut, ModelInfoOut } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function analyzeImage(file: File): Promise<AnalyzeReport> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed with status ${res.status}`);
  }
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfoOut> {
  const res = await fetch(`${API_URL}/model-info`);
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return res.json();
}

export async function submitFeedback(payload: FeedbackIn): Promise<FeedbackOut> {
  const res = await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return res.json();
}

export async function getFeedbackStats(): Promise<FeedbackStatsOut> {
  const res = await fetch(`${API_URL}/feedback/stats`);
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return res.json();
}
