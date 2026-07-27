import type { AnalyzeReport, ModelInfoOut } from "./types";

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
