const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export function Gauge({
  value,
  label,
  colorVar = "--accent",
}: {
  value: number; // 0..1
  label: string;
  colorVar?: string;
}) {
  const pct = clamp01(value) * 100;
  const r = 54;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - clamp01(value));

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-36 w-36">
        <svg viewBox="0 0 120 120" className="h-36 w-36 -rotate-90">
          <circle cx="60" cy="60" r={r} strokeWidth="10" className="fill-none stroke-card-border" />
          <circle
            cx="60"
            cy="60"
            r={r}
            strokeWidth="10"
            strokeLinecap="round"
            className="fill-none transition-all duration-500"
            style={{ stroke: `var(${colorVar})`, strokeDasharray: c, strokeDashoffset: offset }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold">{pct.toFixed(1)}%</span>
        </div>
      </div>
      <span className="text-sm text-muted">{label}</span>
    </div>
  );
}

export function Bar({ value, label, hint }: { value: number; label: string; hint?: string }) {
  const pct = clamp01(value) * 100;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-card-border">
        <div
          className="h-full rounded-full bg-accent transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </div>
  );
}
