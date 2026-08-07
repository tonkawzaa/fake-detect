import type { FeedbackVote } from "@/lib/reportRow";

function voteButtonStyle(selected: boolean, dimmed: boolean, colorVar: string) {
  if (selected) {
    return {
      borderColor: `var(${colorVar})`,
      background: `color-mix(in srgb, var(${colorVar}) 18%, transparent)`,
      color: `var(${colorVar})`,
    };
  }
  return { opacity: dimmed ? 0.4 : 1 };
}

export function FeedbackButtons({
  feedback,
  submitting,
  onVote,
  compact = false,
}: {
  feedback: FeedbackVote | undefined;
  submitting: boolean;
  onVote: (vote: FeedbackVote) => void;
  compact?: boolean;
}) {
  const voted = feedback !== undefined;
  const buttonClass = compact
    ? "flex h-7 w-7 items-center justify-center rounded-full border border-card-border text-sm transition-colors hover:border-accent disabled:cursor-default disabled:hover:border-card-border"
    : "rounded-full border border-card-border px-5 py-2 text-sm font-medium transition-colors hover:border-accent disabled:cursor-default disabled:hover:border-card-border";

  const buttons = (
    <>
      <button
        type="button"
        disabled={voted || submitting}
        onClick={(e) => {
          e.stopPropagation();
          onVote("up");
        }}
        title="Got it right"
        className={buttonClass}
        style={voteButtonStyle(feedback === "up", voted && feedback !== "up", "--safe")}
      >
        {compact ? "👍" : "👍 Got it right"}
      </button>
      <button
        type="button"
        disabled={voted || submitting}
        onClick={(e) => {
          e.stopPropagation();
          onVote("down");
        }}
        title="Got it wrong"
        className={buttonClass}
        style={voteButtonStyle(feedback === "down", voted && feedback !== "down", "--danger")}
      >
        {compact ? "👎" : "👎 Got it wrong"}
      </button>
    </>
  );

  if (compact) {
    return (
      <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
        {buttons}
      </div>
    );
  }

  return (
    <div className="mt-4 border-t border-card-border pt-4">
      <p className="text-sm font-medium">Was this verdict correct?</p>
      <div className="mt-2 flex items-center gap-2">
        {buttons}
        {voted && <span className="text-xs text-muted">Thanks for the feedback!</span>}
      </div>
    </div>
  );
}
