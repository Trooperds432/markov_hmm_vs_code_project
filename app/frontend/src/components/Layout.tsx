import { NavLink, useLocation } from "react-router-dom";
import {
  Activity,
  Beaker,
  Brain,
  Briefcase,
  Command,
  Cpu,
  Flame,
  LayoutDashboard,
  LineChart,
  LogOut,
  ScanLine,
  Settings as SettingsIcon,
  TerminalSquare,
} from "lucide-react";
import type { ReactNode } from "react";
import { useAuth } from "../lib/auth";
import { useRunContext } from "../lib/runContext";
import { StatusBadge, Tag } from "./ui";
import { usePalette } from "./CommandPalette";
import { fmtDate } from "../lib/format";

const PRIMARY = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/backtests", label: "Backtests", icon: LineChart },
  { to: "/portfolio", label: "Portfolio", icon: Briefcase },
  { to: "/paper", label: "Paper Trading", icon: Activity },
];

const MODULES = [
  { to: "/security", label: "Security Analysis", icon: ScanLine },
  { to: "/hmm", label: "HMM Regime Lab", icon: Brain },
  { to: "/nn", label: "Neural Network", icon: Cpu },
  { to: "/stress", label: "Stress Tests", icon: Flame },
  { to: "/experiments", label: "Experiments", icon: Beaker },
  { to: "/terminal", label: "Terminal", icon: TerminalSquare },
];

function NavItem({ to, label, icon: Icon, end }: any) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] transition-colors ${
          isActive
            ? "bg-accent/10 text-accent"
            : "text-terminal-muted hover:bg-terminal-panel2 hover:text-terminal-text"
        }`
      }
    >
      <Icon className="h-[15px] w-[15px]" />
      {label}
    </NavLink>
  );
}

function RunSelector() {
  const { runs, selectedId, setSelectedId, selected } = useRunContext();
  return (
    <div className="flex items-center gap-2">
      <span className="micro hidden sm:block">Active Run</span>
      <select
        value={selectedId || ""}
        onChange={(e) => setSelectedId(e.target.value)}
        className="max-w-[260px] cursor-pointer rounded border border-terminal-border bg-terminal-panel px-2 py-1 font-mono text-xs text-terminal-text outline-none focus:border-accent"
      >
        {runs.length === 0 && <option value="">No runs yet</option>}
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.run_id} · {r.label || r.mode} [{r.status}]
          </option>
        ))}
      </select>
      {selected && <StatusBadge status={selected.status} />}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { setOpen } = usePalette();
  const { required, authed, logout } = useAuth();
  const { selected } = useRunContext();
  const loc = useLocation();

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="flex w-[224px] shrink-0 flex-col border-r border-terminal-border bg-terminal-panel/60">
        <div className="flex items-center gap-2 px-4 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent/15 text-accent">
            <TerminalSquare className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <div className="text-[13px] font-bold tracking-wide text-terminal-text">QUANT TERMINAL</div>
            <div className="text-[9px] uppercase tracking-[0.16em] text-terminal-dim">HMM · NN Research</div>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-2.5 pb-4">
          <div className="micro px-2 pb-1 pt-2">Trading</div>
          {PRIMARY.map((n) => (
            <NavItem key={n.to} {...n} />
          ))}
          <div className="micro px-2 pb-1 pt-4">Research Modules</div>
          {MODULES.map((n) => (
            <NavItem key={n.to} {...n} />
          ))}
          <div className="micro px-2 pb-1 pt-4">System</div>
          <NavItem to="/settings" label="Settings" icon={SettingsIcon} />
        </nav>

        <div className="border-t border-terminal-border px-3 py-3">
          <p className="text-[9px] leading-relaxed text-terminal-dim">
            Research & educational use only. Not investment advice. Paper trading is
            simulation-only.
          </p>
          {required && authed && (
            <button
              onClick={logout}
              className="mt-2 flex items-center gap-1.5 text-[11px] text-terminal-muted hover:text-bear"
            >
              <LogOut className="h-3 w-3" /> Lock terminal
            </button>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-terminal-border bg-terminal-panel/40 px-5 py-2.5">
          <RunSelector />
          <div className="flex items-center gap-3">
            {selected?.finished_at && (
              <span className="hidden text-[11px] text-terminal-dim md:block">
                Updated {fmtDate(selected.finished_at)}
              </span>
            )}
            <Tag tone="accent">LIVE NETWORK OFF</Tag>
            <button
              onClick={() => setOpen(true)}
              className="flex items-center gap-2 rounded border border-terminal-border bg-terminal-panel px-2.5 py-1.5 text-xs text-terminal-muted hover:border-terminal-borderlit hover:text-terminal-text"
            >
              <Command className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Command</span>
              <kbd className="rounded bg-terminal-panel2 px-1 text-[10px]">⌘K</kbd>
            </button>
          </div>
        </header>

        <main key={loc.pathname} className="flex-1 overflow-y-auto px-5 py-5">
          {children}
        </main>
      </div>
    </div>
  );
}
