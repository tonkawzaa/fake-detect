"use client";

import { useCallback, useRef, useState } from "react";

const MAX_BYTES = 15 * 1024 * 1024;
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

export function UploadZone({
  onFiles,
  disabled,
}: {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndEmit = useCallback(
    (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;
      const files = Array.from(fileList);
      const rejected: string[] = [];
      const accepted: File[] = [];
      for (const file of files) {
        if (!ACCEPTED.includes(file.type)) {
          rejected.push(`${file.name} (unsupported type)`);
        } else if (file.size > MAX_BYTES) {
          rejected.push(`${file.name} (over 15MB)`);
        } else {
          accepted.push(file);
        }
      }
      setError(rejected.length > 0 ? `Skipped: ${rejected.join(", ")}` : null);
      if (accepted.length > 0) onFiles(accepted);
    },
    [onFiles]
  );

  return (
    <div>
      <div
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!disabled) validateAndEmit(e.dataTransfer.files);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-8 py-16 text-center transition-colors cursor-pointer
          ${dragOver ? "border-accent bg-accent/5" : "border-card-border bg-card"}
          ${disabled ? "opacity-60 cursor-not-allowed" : "hover:border-accent/60"}`}
      >
        <div className="text-4xl">📷</div>
        <p className="text-base font-medium">Drop one or more portrait photos here, or click to browse</p>
        <p className="text-sm text-muted">JPG, PNG, or WEBP · up to 15MB each</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          multiple
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            validateAndEmit(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
    </div>
  );
}
