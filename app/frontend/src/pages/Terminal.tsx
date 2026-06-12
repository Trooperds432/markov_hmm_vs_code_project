import { useState } from "react";
import { RotateCw, ChevronRight } from "lucide-react";
import { useRunContext } from "../lib/runContext";
import { useRunActions } from "../lib/hooks";
import { useCommandExecutor, COMMAND_HELP } from "../lib/commands";
import { PageHeader } from "../components/PageHeader";
import { Button, Panel } from "../components/ui";
import { RunMetaBar } from "../components/RunSummary";
import { LogViewer, OutputsList } from "../components/RunArtifacts";

interface Line {
  prompt: string;
  result: string;
  ok: boolean;
}

export function Terminal() {
  const { selected } = useRunContext();
  const { rerun } = useRunActions();
  const execute = useCommandExecutor();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<Line[]>([]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const raw = input.trim();
    if (!raw || busy) return;
    setBusy(true);
    const res = await execute(raw);
    setBusy(false);
    setHistory((h) => [...h, { prompt: raw, result: res.message, ok: res.ok }].slice(-12));
    if (res.ok) setInput("");
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Terminal"
        desc="Type engine commands directly. Press ⌘K anywhere for the command palette."
        right={
          selected ? (
            <Button variant="ghost" onClick={() => rerun.mutate(selected.run_id)} loading={rerun.isPending}>
              <RotateCw className="h-3.5 w-3.5" /> Re-run
            </Button>
          ) : undefined
        }
      />

      <Panel
        title="Command Console"
        subtitle="RUN NVDA · QUICK TSLA · PAPER NVDA · PORTFOLIO · UNIVERSE · COMPARE MODELS · STRESS · OPEN HMM"
      >
        <div className="space-y-2 px-4 py-3 font-mono text-[12px] leading-relaxed">
          {history.length === 0 && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-terminal-dim">
              {COMMAND_HELP.map((c) => (
                <button
                  key={c.cmd}
                  onClick={() => setInput(c.cmd)}
                  className="text-bull/80 hover:text-bull"
                  title={c.desc}
                >
                  {c.cmd}
                </button>
              ))}
            </div>
          )}
          {history.map((l, i) => (
            <div key={i} className="space-y-0.5">
              <div className="flex items-center gap-2 text-terminal-muted">
                <ChevronRight className="h-3.5 w-3.5 text-bull" />
                <span className="text-terminal-text">{l.prompt}</span>
              </div>
              <div className={`pl-5 ${l.ok ? "text-terminal-muted" : "text-bear"}`}>{l.result}</div>
            </div>
          ))}
          <form onSubmit={submit} className="flex items-center gap-2">
            <ChevronRight className="h-3.5 w-3.5 text-bull" />
            <input
              autoFocus
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              placeholder="enter command…"
              spellCheck={false}
              className="flex-1 bg-transparent text-terminal-text outline-none placeholder:text-terminal-dim disabled:opacity-60"
            />
          </form>
        </div>
      </Panel>

      {selected ? (
        <>
          <RunMetaBar run={selected} />
          <Panel title="Command Reference" subtitle="Underlying engine invocation">
            <pre className="overflow-auto px-4 py-3 font-mono text-[11px] leading-relaxed text-terminal-muted">
              {selected.command?.join(" ") || "—"}
            </pre>
          </Panel>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <LogViewer run={selected} />
            <OutputsList run={selected} />
          </div>
        </>
      ) : (
        <Panel title="Active run">
          <div className="px-4 py-8 text-center text-sm text-terminal-muted">
            No run selected yet. Launch one above (e.g.{" "}
            <code className="font-mono text-bull">QUICK TSLA</code>) to stream logs and artifacts here.
          </div>
        </Panel>
      )}
    </div>
  );
}
