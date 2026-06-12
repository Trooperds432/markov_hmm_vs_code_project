import clsx from "clsx";
import { Loader2 } from "lucide-react";
import type { ReactNode, SelectHTMLAttributes, InputHTMLAttributes } from "react";
import type { RunStatus } from "../lib/types";

export const cn = clsx;

export function Panel({
  children,
  className,
  title,
  subtitle,
  right,
}: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <section
      className={cn(
        "rounded-lg border border-terminal-border bg-terminal-panel shadow-panel",
        className
      )}
    >
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 border-b border-terminal-border px-4 py-2.5">
          <div>
            {title && (
              <h3 className="text-[13px] font-semibold tracking-wide text-terminal-text">
                {title}
              </h3>
            )}
            {subtitle && <p className="micro mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

const STATUS_STYLES: Record<RunStatus, string> = {
  queued: "bg-side/15 text-side border-side/30",
  running: "bg-accent/15 text-accent border-accent/30",
  completed: "bg-bull/15 text-bull border-bull/30",
  failed: "bg-bear/15 text-bear border-bear/30",
  cancelled: "bg-terminal-dim/15 text-terminal-muted border-terminal-border",
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
        STATUS_STYLES[status] || STATUS_STYLES.cancelled
      )}
    >
      {(status === "running" || status === "queued") && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {status}
    </span>
  );
}

export function Tag({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "bull" | "bear" | "side" | "accent" }) {
  const styles = {
    neutral: "border-terminal-border bg-terminal-panel2 text-terminal-muted",
    bull: "border-bull/30 bg-bull/10 text-bull",
    bear: "border-bear/30 bg-bear/10 text-bear",
    side: "border-side/30 bg-side/10 text-side",
    accent: "border-accent/30 bg-accent/10 text-accent",
  }[tone];
  return (
    <span className={cn("inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide", styles)}>
      {children}
    </span>
  );
}

type BtnVariant = "primary" | "ghost" | "danger" | "subtle";

export function Button({
  children,
  variant = "subtle",
  className,
  loading,
  ...rest
}: {
  children: ReactNode;
  variant?: BtnVariant;
  loading?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const styles: Record<BtnVariant, string> = {
    primary: "bg-accent text-black hover:bg-accent/90 border-accent font-semibold",
    ghost: "bg-transparent text-terminal-text hover:bg-terminal-panel2 border-terminal-border",
    danger: "bg-transparent text-bear hover:bg-bear/10 border-bear/40",
    subtle: "bg-terminal-panel2 text-terminal-text hover:bg-terminal-borderlit border-terminal-border",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded border px-2.5 py-1.5 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        styles[variant],
        className
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="micro">{label}</span>
      {children}
      {hint && <span className="text-[10px] text-terminal-dim">{hint}</span>}
    </label>
  );
}

const inputBase =
  "w-full rounded border border-terminal-border bg-terminal-bg px-2 py-1.5 text-xs text-terminal-text outline-none transition-colors focus:border-accent placeholder:text-terminal-dim";

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn(inputBase, props.className)} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cn(inputBase, "appearance-none cursor-pointer", props.className)} />;
}

export function Checkbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-terminal-text">
      <span
        onClick={() => onChange(!checked)}
        className={cn(
          "flex h-4 w-4 items-center justify-center rounded border transition-colors",
          checked ? "border-accent bg-accent" : "border-terminal-borderlit bg-terminal-bg"
        )}
      >
        {checked && <span className="h-1.5 w-1.5 rounded-sm bg-black" />}
      </span>
      {label}
    </label>
  );
}

export function KpiCard({
  label,
  value,
  sub,
  tone = "flat",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "pos" | "neg" | "flat";
}) {
  const color = tone === "pos" ? "text-bull" : tone === "neg" ? "text-bear" : "text-terminal-text";
  return (
    <div className="rounded-lg border border-terminal-border bg-terminal-panel px-4 py-3 shadow-panel">
      <div className="micro">{label}</div>
      <div className={cn("mt-1.5 font-mono text-2xl font-semibold tnum", color)}>{value}</div>
      {sub && <div className="mt-1 text-[11px] text-terminal-muted tnum">{sub}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-xs text-terminal-muted">
      <Loader2 className="h-4 w-4 animate-spin" />
      {label || "Loading…"}
    </div>
  );
}

export function EmptyState({ title, hint, icon }: { title: string; hint?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      {icon && <div className="text-terminal-dim">{icon}</div>}
      <p className="text-sm font-medium text-terminal-muted">{title}</p>
      {hint && <p className="max-w-md text-xs text-terminal-dim">{hint}</p>}
    </div>
  );
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <h2 className="text-sm font-semibold uppercase tracking-[0.12em] text-terminal-text">{children}</h2>
      {right}
    </div>
  );
}
