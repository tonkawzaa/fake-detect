// Mirrors backend/app/schemas.py

export interface ModelScoreOut {
  name: string;
  ai_probability_full: number;
  ai_probability_face: number;
  ai_probability_combined: number;
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

export interface BeautyOut {
  score: number;
  level: "None" | "Light" | "Moderate" | "Heavy" | string;
  subscores: Record<string, number>;
  raw: Record<string, number>;
  guard_multiplier: number;
  notes: string[];
  calibrated: boolean;
}

export interface FaceQualityOut {
  width: number;
  height: number;
  blur_score: number;
  blur_label: string;
  coverage: number;
}

export interface FaceOut {
  count: number;
  bbox: [number, number, number, number] | null;
  quality: FaceQualityOut | null;
  notes: string[];
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
}

export interface AnalyzeReport {
  status: "ok" | "no_face" | "low_quality";
  message: string;
  verdict: "likely_ai" | "likely_real" | "uncertain" | null;
  ai_probability: number | null;
  confidence_band: "high" | "medium" | "low" | null;
  calibrated: boolean;
  models: ModelScoreOut[];
  model_accuracy: ModelAccuracyOut | null;
  beauty: BeautyOut | null;
  face: FaceOut;
  provenance: ProvenanceOut | null;
  heatmap_png: string | null;
  reconstruction_check: ReconstructionCheckOut | null;
  limitations: string[];
}

export interface ModelInfoOut {
  ensemble_models: string[];
  calibrated: boolean;
  model_accuracy: ModelAccuracyOut | null;
  beauty_calibrated: boolean;
  limitations: string[];
}
