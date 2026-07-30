import type { ProvenanceOut } from "@/lib/types";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-sm">
      <span className="text-muted">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

export function ProvenancePanel({ provenance }: { provenance: ProvenanceOut }) {
  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <h3 className="text-lg font-semibold">Provenance</h3>

      <div className="mt-3 divide-y divide-card-border">
        <Row label="Camera EXIF present" value={provenance.exif_present ? "Yes" : "No"} />
        {provenance.exif_present && (
          <>
            <Row label="Camera" value={[provenance.camera_make, provenance.camera_model].filter(Boolean).join(" ") || "—"} />
            <Row label="Software tag" value={provenance.software ?? "—"} />
            {provenance.flagged_editor_software && (
              <Row label="⚠️ Editor detected" value={<span className="text-warn">Yes, in Software tag</span>} />
            )}
          </>
        )}
        <Row label="C2PA content credentials" value={provenance.c2pa_present ? "Present" : "Not found"} />
        {provenance.c2pa_present && (
          <>
            <Row label="Claim generator" value={provenance.c2pa_claim_generator ?? "—"} />
            {provenance.c2pa_is_generative_ai && (
              <Row label="⚠️ Generative AI claim" value={<span className="text-danger">Yes</span>} />
            )}
          </>
        )}
        <Row label="XMP digitalSourceType" value={provenance.xmp_present ? "Present" : "Not found"} />
        {provenance.xmp_present && provenance.xmp_digital_source_type && (
          <>
            <Row label="Value" value={provenance.xmp_digital_source_type} />
            {provenance.xmp_is_generative_ai && (
              <Row label="⚠️ Generative AI claim" value={<span className="text-danger">Yes (unsigned metadata)</span>} />
            )}
          </>
        )}
      </div>
      {!provenance.exif_present && !provenance.c2pa_present && !provenance.xmp_present && (
        <p className="mt-3 text-xs text-muted">
          No EXIF, C2PA, or XMP data found — common after social-media re-uploads, and not evidence either way.
        </p>
      )}
    </div>
  );
}
