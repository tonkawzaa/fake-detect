import type { FeedbackStatsOut } from "@/lib/types";
import { VERDICT_COPY } from "@/lib/verdictCopy";

export function FeedbackStatsPanel({ stats }: { stats: FeedbackStatsOut }) {
  if (stats.total === 0) {
    return (
      <div className="rounded-2xl border border-card-border bg-card p-6">
        <p className="text-sm font-medium">Feedback report</p>
        <p className="mt-1 text-sm text-muted">No feedback yet — vote 👍 / 👎 on an analyzed photo to start building this report.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-card-border bg-card p-6">
      <p className="text-sm font-medium">Feedback report</p>
      <p className="mt-1 text-sm text-muted">
        Based on <span className="font-semibold text-foreground">{stats.total}</span> user rating{stats.total === 1 ? "" : "s"}:{" "}
        <span className="font-semibold" style={{ color: "var(--safe)" }}>
          {stats.correct_pct?.toFixed(1)}% marked correct
        </span>
        {" · "}
        <span className="font-semibold" style={{ color: "var(--danger)" }}>
          {stats.incorrect_pct?.toFixed(1)}% marked incorrect
        </span>
      </p>

      {Object.keys(stats.by_verdict).length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-card-border pt-3 text-sm text-muted">
          {Object.entries(stats.by_verdict).map(([verdict, breakdown]) => {
            const copy = VERDICT_COPY[verdict] ?? { shortLabel: verdict, colorVar: "--muted" };
            return (
              <li key={verdict}>
                <span style={{ color: `var(${copy.colorVar})` }}>{copy.shortLabel}</span>: {breakdown.total} rating
                {breakdown.total === 1 ? "" : "s"}
                {breakdown.correct_pct !== null && ` · ${breakdown.correct_pct.toFixed(1)}% correct`}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
