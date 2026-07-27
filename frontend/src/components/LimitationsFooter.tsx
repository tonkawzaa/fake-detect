export function LimitationsFooter({ limitations }: { limitations: string[] }) {
  if (limitations.length === 0) return null;
  return (
    <div className="rounded-2xl border border-warn/30 bg-warn/5 p-5">
      <h3 className="text-sm font-semibold text-warn">Limitations</h3>
      <ul className="mt-2 flex flex-col gap-1.5 text-xs text-muted">
        {limitations.map((l, i) => (
          <li key={i}>• {l}</li>
        ))}
      </ul>
    </div>
  );
}
