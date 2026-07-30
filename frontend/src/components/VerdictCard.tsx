import type { AnalyzeReport } from "@/lib/types";
import { Gauge } from "./Gauge";
import { VERDICT_COPY, BAND_COPY, RECON_VERDICT_COPY } from "@/lib/verdictCopy";

export function VerdictCard({ report }: { report: AnalyzeReport }) {
  if (report.ai_probability === null || !report.verdict) {
    return (
      <div className="rounded-2xl border border-card-border bg-card p-6">
        <p className="text-muted">AI-detection unavailable for this image.</p>
      </div>
    );
  }

  const verdict = VERDICT_COPY[report.verdict] ?? VERDICT_COPY.uncertain;
  const acc = report.model_accuracy;

  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-center sm:justify-around">
        <Gauge value={report.ai_probability} label="AI probability (confidence in this image)" colorVar={verdict.colorVar} />
        <div className="flex flex-col items-center gap-2 sm:items-start">
          <span
            className="rounded-full px-3 py-1 text-sm font-semibold"
            style={{ background: `color-mix(in srgb, var(${verdict.colorVar}) 18%, transparent)`, color: `var(${verdict.colorVar})` }}
          >
            {verdict.label}
          </span>
          {report.confidence_band && <span className="text-sm text-muted">{BAND_COPY[report.confidence_band]}</span>}
          {report.verdict_source === "c2pa" ? (
            <span className="text-xs text-muted">
              Forced by C2PA content credentials
              {report.provenance?.c2pa_claim_generator ? ` (${report.provenance.c2pa_claim_generator})` : ""} declaring
              this as generative AI — the model ensemble below was not the deciding factor.
            </span>
          ) : report.verdict_source === "xmp" ? (
            <span className="text-xs text-muted">
              Forced by an IPTC digitalSourceType tag in the image&apos;s XMP metadata
              {report.provenance?.xmp_digital_source_type ? ` (${report.provenance.xmp_digital_source_type})` : ""} declaring
              this as generative AI — unlike C2PA this isn&apos;t cryptographically signed, just metadata, but the model
              ensemble below was not the deciding factor.
            </span>
          ) : (
            !report.calibrated && (
              <span className="text-xs text-warn">
                Uncalibrated ensemble average — run scripts/evaluate.py to calibrate this probability.
              </span>
            )
          )}
        </div>
      </div>

      <div className="mt-6 border-t border-card-border pt-4">
        <p className="text-sm font-medium">Model accuracy (measured on a held-out eval set)</p>
        {acc?.accuracy != null ? (
          <p className="mt-1 text-sm text-muted">
            <span className="font-semibold text-foreground">{(acc.accuracy * 100).toFixed(1)}%</span> accuracy
            {acc.auc != null && <> · AUC {acc.auc.toFixed(3)}</>} on {acc.n} held-out images
            {acc.out_of_fold && " (out-of-fold, not fit on these images)"}.
            <br />
            <span className="text-xs">
              This is different from the AI-probability gauge above, which is this model&apos;s confidence on{" "}
              <em>this specific photo</em> — not a measured accuracy rate.
            </span>
          </p>
        ) : (
          <p className="mt-1 text-sm text-muted">{acc?.note ?? "Not yet measured."}</p>
        )}
      </div>

      {report.reconstruction_check && (
        <div className="mt-4 border-t border-card-border pt-4">
          <p className="text-sm font-medium">Reconstruction-error check (secondary signal, ran because the main models disagreed)</p>
          <p className="mt-1 text-sm text-muted">
            LPIPS reconstruction distance <span className="font-semibold text-foreground">{report.reconstruction_check.reconstruction_error.toFixed(3)}</span>
            {report.reconstruction_check.calibrated && report.reconstruction_check.p_ai !== null ? (
              <> — {RECON_VERDICT_COPY[report.reconstruction_check.verdict] ?? report.reconstruction_check.verdict}.</>
            ) : (
              <> — not calibrated yet, no verdict to show (run scripts/calibrate_reconstruction.py).</>
            )}
            <br />
            <span className="text-xs">{report.reconstruction_check.note}</span>
          </p>
        </div>
      )}
    </div>
  );
}
