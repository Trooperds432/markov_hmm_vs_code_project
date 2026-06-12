import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Beaker,
  Brain,
  Briefcase,
  ChevronRight,
  Cpu,
  Flame,
  LayoutDashboard,
  LineChart,
  RefreshCw,
  ScanLine,
  Search,
  Settings as SettingsIcon,
  TerminalSquare,
  Zap,
} from "lucide-react";
import { useCreateRun } from "../lib/hooks";
import { useRunContext } from "../lib/runContext";
import {
  COMMAND_HELP,
  isCommandLike,
  parseCommand,
  useCommandExecutor,
} from "../lib/commands";
import { StatusBadge } from "./ui";

interface PaletteCtx {
  open: boolean;
  setOpen: (v: boolean) => void;
}
const Ctx = createContext<PaletteCtx>({ open: false, setOpen: () => {} });
export const usePalette = () => useContext(Ctx);

export function PaletteProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return <Ctx.Provider value={{ open, setOpen }}>{children}</Ctx.Provider>;
}

const NAV: { label: string; path: string; icon: ReactNode }[] = [
  { label: "Dashboard", path: "/", icon: <LayoutDashboard className="h-4 w-4" /> },
  { label: "Backtests", path: "/backtests", icon: <LineChart className="h-4 w-4" /> },
  { label: "Security Analysis", path: "/security", icon: <ScanLine className="h-4 w-4" /> },
  { label: "HMM Regime Lab", path: "/hmm", icon: <Brain className="h-4 w-4" /> },
  { label: "Neural Network", path: "/nn", icon: <Cpu className="h-4 w-4" /> },
  { label: "Portfolio", path: "/portfolio", icon: <Briefcase className="h-4 w-4" /> },
  { label: "Paper Trading", path: "/paper", icon: <Activity className="h-4 w-4" /> },
  { label: "Stress Tests", path: "/stress", icon: <Flame className="h-4 w-4" /> },
  { label: "Experiments", path: "/experiments", icon: <Beaker className="h-4 w-4" /> },
  { label: "Terminal", path: "/terminal", icon: <TerminalSquare className="h-4 w-4" /> },
  { label: "Settings", path: "/settings", icon: <SettingsIcon className="h-4 w-4" /> },
];

export function CommandPalette() {
  const { open, setOpen } = usePalette();
  const navigate = useNavigate();
  const createRun = useCreateRun();
  const execute = useCommandExecutor();
  const { runs, setSelectedId } = useRunContext();
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setQ("");
      setError(null);
    }
  }, [open]);

  useEffect(() => {
    setError(null);
  }, [q]);

  const commandLike = isCommandLike(q);
  const parsed = useMemo(() => (commandLike ? parseCommand(q) : null), [q, commandLike]);

  const navItems = useMemo(
    () =>
      NAV.filter((n) => n.label.toLowerCase().includes(q.toLowerCase())).map((n) => ({
        ...n,
        run: () => {
          navigate(n.path);
          setOpen(false);
        },
      })),
    [q, navigate, setOpen]
  );

  const runItems = useMemo(
    () =>
      runs
        .filter(
          (r) =>
            r.run_id.includes(q) || (r.label || "").toLowerCase().includes(q.toLowerCase())
        )
        .slice(0, 5),
    [runs, q]
  );

  if (!open) return null;

  const runCommand = async () => {
    if (!commandLike || busy) return;
    setBusy(true);
    setError(null);
    const res = await execute(q);
    setBusy(false);
    if (res.ok) {
      if (res.kind === "run") navigate("/backtests");
      setOpen(false);
    } else {
      setError(res.message);
    }
  };

  const quickRun = async () => {
    await createRun.mutateAsync({
      mode: "research",
      config: "quick_synthetic_provider.json",
      quick: true,
      plots: false,
      label: "palette synthetic",
    });
    navigate("/backtests");
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && commandLike) {
      e.preventDefault();
      void runCommand();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[12vh] backdrop-blur-sm"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-xl border border-terminal-borderlit bg-terminal-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-terminal-border px-3.5 py-3">
          <Search className="h-4 w-4 text-terminal-muted" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a command (RUN NVDA, OPEN HMM) or search modules…"
            className="flex-1 bg-transparent font-mono text-sm text-terminal-text outline-none placeholder:font-sans placeholder:text-terminal-dim"
          />
          <kbd className="rounded border border-terminal-border px-1.5 py-0.5 text-[10px] text-terminal-dim">ESC</kbd>
        </div>
        <div className="max-h-[52vh] overflow-y-auto p-2">
          {commandLike && (
            <>
              <div className="micro px-2 py-1">Command</div>
              {parsed && parsed.kind !== "error" ? (
                <button
                  onClick={runCommand}
                  disabled={busy}
                  className="flex w-full items-center gap-2.5 rounded border border-terminal-borderlit bg-terminal-panel2 px-2.5 py-2 text-left text-sm text-terminal-text hover:border-accent disabled:opacity-60"
                >
                  <ChevronRight className="h-4 w-4 text-bull" />
                  <span className="flex-1">{parsed.label}</span>
                  <kbd className="rounded border border-terminal-border px-1.5 py-0.5 text-[10px] text-terminal-dim">↵</kbd>
                </button>
              ) : (
                <div className="flex items-center gap-2.5 rounded px-2.5 py-2 text-sm text-terminal-muted">
                  <AlertTriangle className="h-4 w-4 text-side" />
                  {parsed?.kind === "error" ? parsed.message : "Keep typing…"}
                </div>
              )}
              {error && (
                <div className="flex items-center gap-2.5 rounded px-2.5 py-1 text-xs text-bear">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {error}
                </div>
              )}
            </>
          )}

          <div className="micro px-2 pb-1 pt-3">Actions</div>
          <button
            onClick={quickRun}
            className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-terminal-text hover:bg-terminal-panel2"
          >
            <Zap className="h-4 w-4 text-side" />
            Run synthetic quick backtest
          </button>
          <button
            onClick={() => {
              window.location.reload();
            }}
            className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-terminal-text hover:bg-terminal-panel2"
          >
            <RefreshCw className="h-4 w-4 text-accent" />
            Reload terminal
          </button>

          {!commandLike && q === "" && (
            <>
              <div className="micro px-2 pb-1 pt-3">Command reference</div>
              <div className="grid grid-cols-1 gap-0.5 px-1 sm:grid-cols-2">
                {COMMAND_HELP.map((c) => (
                  <button
                    key={c.cmd}
                    onClick={() => setQ(c.cmd)}
                    className="flex items-baseline gap-2 rounded px-1.5 py-1 text-left hover:bg-terminal-panel2"
                  >
                    <code className="font-mono text-[11px] text-bull">{c.cmd}</code>
                    <span className="truncate text-[11px] text-terminal-dim">{c.desc}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="micro px-2 pb-1 pt-3">Navigate</div>
          {navItems.map((n) => (
            <button
              key={n.path}
              onClick={n.run}
              className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-terminal-text hover:bg-terminal-panel2"
            >
              <span className="text-terminal-muted">{n.icon}</span>
              {n.label}
            </button>
          ))}

          {runItems.length > 0 && (
            <>
              <div className="micro px-2 pb-1 pt-3">Select run</div>
              {runItems.map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => {
                    setSelectedId(r.run_id);
                    setOpen(false);
                  }}
                  className="flex w-full items-center justify-between gap-2.5 rounded px-2.5 py-2 text-left text-sm hover:bg-terminal-panel2"
                >
                  <span className="flex items-center gap-2">
                    <span className="font-mono text-xs text-terminal-muted">{r.run_id}</span>
                    <span className="text-terminal-text">{r.label}</span>
                  </span>
                  <StatusBadge status={r.status} />
                </button>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
