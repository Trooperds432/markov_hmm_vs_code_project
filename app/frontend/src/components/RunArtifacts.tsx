import { useEffect, useRef, useState } from "react";
import { Download, FileText } from "lucide-react";
import { api, downloadFile } from "../lib/api";
import { useOutputs } from "../lib/hooks";
import type { RunMeta } from "../lib/types";
import { Panel, EmptyState, cn } from "./ui";

function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function LogViewer({ run }: { run: RunMeta }) {
  const [content, setContent] = useState("");
  const boxRef = useRef<HTMLPreElement>(null);
  const live = run.status === "running" || run.status === "queued";

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const res = await api.logs(run.run_id, 0);
        if (!cancelled) setContent(res.content || "");
      } catch {
        /* ignore */
      }
      if (!cancelled && live) timer = window.setTimeout(tick, 1500);
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [run.run_id, run.status, live]);

  useEffect(() => {
    if (boxRef.current && live) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [content, live]);

  return (
    <Panel
      title="Engine Log"
      subtitle={live ? "streaming…" : "job.log"}
      right={
        <button
          onClick={() => downloadFile(run.run_id, "job.log")}
          className="text-terminal-muted hover:text-terminal-text"
          title="Download log"
        >
          <Download className="h-3.5 w-3.5" />
        </button>
      }
    >
      <pre
        ref={boxRef}
        className={cn(
          "max-h-[360px] overflow-auto whitespace-pre-wrap break-words px-4 py-3 font-mono text-[11px] leading-relaxed text-terminal-muted",
          live && "border-l-2 border-accent/40"
        )}
      >
        {content || "No log output yet."}
      </pre>
    </Panel>
  );
}

export function OutputsList({ run }: { run: RunMeta }) {
  const { data } = useOutputs(run.run_id, run.status);
  const files = data?.files ?? [];

  if (run.status !== "completed") {
    return (
      <Panel title="Output Artifacts">
        <EmptyState title="No artifacts yet" hint="Artifacts appear after the run completes." />
      </Panel>
    );
  }

  return (
    <Panel title="Output Artifacts" subtitle={`${files.length} files`}>
      <div className="max-h-[360px] divide-y divide-terminal-border/50 overflow-auto">
        {files.map((f) => (
          <button
            key={f.name}
            onClick={() => downloadFile(run.run_id, f.name)}
            className="flex w-full items-center justify-between gap-3 px-4 py-2 text-left text-xs hover:bg-terminal-panel2/60"
          >
            <span className="flex items-center gap-2 truncate">
              <FileText className="h-3.5 w-3.5 shrink-0 text-terminal-dim" />
              <span className="truncate font-mono text-terminal-text">{f.name}</span>
            </span>
            <span className="flex items-center gap-3 shrink-0">
              <span className="text-terminal-dim">{fmtBytes(f.size)}</span>
              <Download className="h-3.5 w-3.5 text-terminal-muted" />
            </span>
          </button>
        ))}
        {files.length === 0 && <EmptyState title="No files" />}
      </div>
    </Panel>
  );
}
