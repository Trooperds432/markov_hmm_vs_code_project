import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateRun } from "./hooks";
import { useRunContext } from "./runContext";
import type { RunRequest } from "./types";

export const DEFAULT_CONFIG = "default.json";

export type ParsedCommand =
  | { kind: "run"; label: string; request: RunRequest }
  | { kind: "navigate"; label: string; path: string }
  | { kind: "error"; label: string; message: string };

const OPEN_TARGETS: Record<string, { path: string; label: string }> = {
  STATS: { path: "/", label: "Dashboard" },
  DASHBOARD: { path: "/", label: "Dashboard" },
  HOME: { path: "/", label: "Dashboard" },
  BACKTESTS: { path: "/backtests", label: "Backtests" },
  RUNS: { path: "/backtests", label: "Backtests" },
  SECURITY: { path: "/security", label: "Security Analysis" },
  HMM: { path: "/hmm", label: "HMM Regime Lab" },
  REGIME: { path: "/hmm", label: "HMM Regime Lab" },
  NN: { path: "/nn", label: "Neural Network" },
  NEURAL: { path: "/nn", label: "Neural Network" },
  PORTFOLIO: { path: "/portfolio", label: "Portfolio" },
  PAPER: { path: "/paper", label: "Paper Trading" },
  STRESS: { path: "/stress", label: "Stress Tests" },
  EXPERIMENTS: { path: "/experiments", label: "Experiments" },
  TERMINAL: { path: "/terminal", label: "Terminal" },
  SETTINGS: { path: "/settings", label: "Settings" },
};

const TICKER_RE = /^[A-Za-z][A-Za-z0-9.\-]{0,11}$/;

const COMMAND_KEYWORDS = [
  "RUN",
  "BACKTEST",
  "QUICK",
  "PAPER",
  "PORTFOLIO",
  "UNIVERSE",
  "STRESS",
  "COMPARE",
  "OPEN",
];

export interface CommandExample {
  cmd: string;
  desc: string;
}

export const COMMAND_HELP: CommandExample[] = [
  { cmd: "RUN NVDA", desc: "Backtest a ticker (research mode)" },
  { cmd: "QUICK TSLA", desc: "Fast, reduced backtest" },
  { cmd: "PAPER NVDA", desc: "Paper-trading simulation" },
  { cmd: "PORTFOLIO", desc: "Run the configured portfolio" },
  { cmd: "UNIVERSE", desc: "Run the configured universe" },
  { cmd: "COMPARE MODELS", desc: "Compare model variants" },
  { cmd: "STRESS", desc: "Run with stress tests enabled" },
  { cmd: "OPEN HMM", desc: "Jump to a module (STATS, HMM, NN, …)" },
];

export function isCommandLike(raw: string): boolean {
  const head = raw.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
  return COMMAND_KEYWORDS.includes(head);
}

function ticker(t?: string): string | undefined {
  return t ? t.toUpperCase() : undefined;
}

export function parseCommand(raw: string): ParsedCommand | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const tokens = trimmed.split(/\s+/);
  const cmd = tokens[0].toUpperCase();
  const arg = tokens[1];
  const validTicker = !!arg && TICKER_RE.test(arg);

  switch (cmd) {
    case "RUN":
    case "BACKTEST":
      if (!validTicker)
        return { kind: "error", label: trimmed, message: "Usage: RUN <TICKER>  (e.g. RUN NVDA)" };
      return {
        kind: "run",
        label: `Backtest ${ticker(arg)}`,
        request: { mode: "research", config: DEFAULT_CONFIG, ticker: ticker(arg), label: `cmd: run ${ticker(arg)}` },
      };
    case "QUICK":
      if (!validTicker)
        return { kind: "error", label: trimmed, message: "Usage: QUICK <TICKER>  (e.g. QUICK TSLA)" };
      return {
        kind: "run",
        label: `Quick backtest ${ticker(arg)}`,
        request: {
          mode: "research",
          config: DEFAULT_CONFIG,
          ticker: ticker(arg),
          quick: true,
          plots: false,
          label: `cmd: quick ${ticker(arg)}`,
        },
      };
    case "PAPER":
      if (!validTicker)
        return { kind: "error", label: trimmed, message: "Usage: PAPER <TICKER>  (e.g. PAPER NVDA)" };
      return {
        kind: "run",
        label: `Paper simulation ${ticker(arg)}`,
        request: { mode: "paper", config: DEFAULT_CONFIG, ticker: ticker(arg), label: `cmd: paper ${ticker(arg)}` },
      };
    case "PORTFOLIO":
      return {
        kind: "run",
        label: "Run portfolio",
        request: { mode: "portfolio", config: DEFAULT_CONFIG, label: "cmd: portfolio" },
      };
    case "UNIVERSE":
      return {
        kind: "run",
        label: "Run universe",
        request: { mode: "universe", config: DEFAULT_CONFIG, label: "cmd: universe" },
      };
    case "STRESS":
      return {
        kind: "run",
        label: validTicker ? `Stress test ${ticker(arg)}` : "Run with stress tests",
        request: {
          mode: "research",
          config: DEFAULT_CONFIG,
          ticker: validTicker ? ticker(arg) : null,
          enable_stress_tests: true,
          label: "cmd: stress",
        },
      };
    case "COMPARE":
      if ((arg || "").toUpperCase() !== "MODELS")
        return { kind: "error", label: trimmed, message: "Usage: COMPARE MODELS" };
      return {
        kind: "run",
        label: "Compare models",
        request: { mode: "compare_models", config: DEFAULT_CONFIG, label: "cmd: compare models" },
      };
    case "OPEN": {
      const target = OPEN_TARGETS[(arg || "").toUpperCase()];
      if (!target)
        return {
          kind: "error",
          label: trimmed,
          message: "Usage: OPEN <STATS|HMM|NN|SECURITY|PORTFOLIO|PAPER|STRESS|EXPERIMENTS|SETTINGS>",
        };
      return { kind: "navigate", label: `Open ${target.label}`, path: target.path };
    }
    default:
      return { kind: "error", label: trimmed, message: `Unknown command: ${cmd}` };
  }
}

export interface CommandResult {
  ok: boolean;
  message: string;
  kind: ParsedCommand["kind"] | "empty";
}

export function useCommandExecutor() {
  const navigate = useNavigate();
  const createRun = useCreateRun();
  const { setSelectedId } = useRunContext();

  return useCallback(
    async (raw: string): Promise<CommandResult> => {
      const parsed = parseCommand(raw);
      if (!parsed) return { ok: false, message: "Empty command", kind: "empty" };
      if (parsed.kind === "error") return { ok: false, message: parsed.message, kind: "error" };
      if (parsed.kind === "navigate") {
        navigate(parsed.path);
        return { ok: true, message: parsed.label, kind: "navigate" };
      }
      try {
        const run = await createRun.mutateAsync(parsed.request);
        setSelectedId(run.run_id);
        return { ok: true, message: `${parsed.label} — launched ${run.run_id}`, kind: "run" };
      } catch (e: any) {
        return { ok: false, message: e?.message || "Run failed to start", kind: "error" };
      }
    },
    [navigate, createRun, setSelectedId]
  );
}
