import { Link } from "react-router-dom";
import { Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { Button, EmptyState } from "./ui";

export function PageHeader({
  title,
  desc,
  right,
}: {
  title: string;
  desc?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div>
        <h1 className="text-lg font-bold tracking-tight text-terminal-text">{title}</h1>
        {desc && <p className="mt-0.5 max-w-2xl text-xs text-terminal-muted">{desc}</p>}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}

export function NoRun({ message }: { message?: string }) {
  return (
    <div className="rounded-lg border border-terminal-border bg-terminal-panel">
      <EmptyState
        title="No run selected"
        hint={message || "Launch a backtest or select an existing run to view this module."}
        icon={<Inbox className="h-7 w-7" />}
      />
      <div className="flex justify-center pb-8">
        <Link to="/backtests">
          <Button variant="primary">Go to Backtests</Button>
        </Link>
      </div>
    </div>
  );
}

export function RunningNotice() {
  return (
    <EmptyState
      title="Run in progress"
      hint="Charts and tables populate automatically when the run completes."
    />
  );
}
