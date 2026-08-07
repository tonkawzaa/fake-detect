// Mirrors backend/app/schemas.py

export interface ModelScoreOut {
  name: string;
  ai_probability: number;
  weight: number;
  eval_auc: number | null;
}

export interface ModelAccuracyOut {
  accuracy: number | null;
  auc: number | null;
  n: number | null;
  out_of_fold: boolean;
  per_generator: Record<string, number> | null;
  note: string | null;
}

export interface ReconstructionCheckOut {
  reconstruction_error: number;
  p_ai: number | null;
  verdict: "likely_ai" | "likely_real" | "uncertain" | string;
  calibrated: boolean;
  note: string;
}

export interface ProvenanceOut {
  exif_present: boolean;
  camera_make: string | null;
  camera_model: string | null;
  software: string | null;
  flagged_editor_software: boolean;
  c2pa_present: boolean;
  c2pa_claim_generator: string | null;
  c2pa_is_generative_ai: boolean;
  c2pa_actions: string[];
  xmp_present: boolean;
  xmp_digital_source_type: string | null;
  xmp_is_generative_ai: boolean;
}

export interface AnalyzeReport {
  analysis_id: string;
  status: "ok"; // always "ok" -- kept for API back-compat, nothing sets it to anything else
  message: string;
  verdict: "likely_ai" | "likely_real" | "uncertain" | null;
  ai_probability: number | null;
  confidence_band: "high" | "medium" | "low" | null;
  calibrated: boolean;
  verdict_source: "ensemble" | "c2pa" | "xmp" | string;
  models: ModelScoreOut[];
  model_accuracy: ModelAccuracyOut | null;
  provenance: ProvenanceOut | null;
  heatmap_png: string | null;
  reconstruction_check: ReconstructionCheckOut | null;
  limitations: string[];
}

export interface ModelInfoOut {
  ensemble_models: string[];
  calibrated: boolean;
  model_accuracy: ModelAccuracyOut | null;
  limitations: string[];
}

export interface FeedbackIn {
  analysis_id: string;
  verdict: string;
  is_correct: boolean;
}

export interface FeedbackOut {
  status: "ok";
  analysis_id: string;
}

export interface VerdictBreakdownOut {
  total: number;
  correct_pct: number | null;
  incorrect_pct: number | null;
}

export interface FeedbackStatsOut {
  total: number;
  correct_pct: number | null;
  incorrect_pct: number | null;
  by_verdict: Record<string, VerdictBreakdownOut>;
}
