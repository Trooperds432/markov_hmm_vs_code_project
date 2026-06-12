import { useState } from "react";
import { Lock, TerminalSquare } from "lucide-react";
import { useAuth } from "../lib/auth";
import { Button, TextInput } from "./ui";

export function LoginGate() {
  const { login } = useAuth();
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr("");
    const ok = await login(pw);
    setBusy(false);
    if (!ok) setErr("Incorrect password.");
  };

  return (
    <div className="grid-bg flex h-screen items-center justify-center">
      <form
        onSubmit={submit}
        className="w-full max-w-sm rounded-xl border border-terminal-border bg-terminal-panel p-7 shadow-panel"
      >
        <div className="mb-5 flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-accent/15 text-accent">
            <TerminalSquare className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-bold tracking-wide text-terminal-text">QUANT TERMINAL</div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-terminal-dim">Access required</div>
          </div>
        </div>
        <label className="micro mb-1 block">App Password</label>
        <div className="relative">
          <Lock className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-terminal-dim" />
          <TextInput
            type="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            placeholder="••••••••"
            className="pl-8"
            autoFocus
          />
        </div>
        {err && <p className="mt-2 text-xs text-bear">{err}</p>}
        <Button type="submit" variant="primary" className="mt-4 w-full" loading={busy}>
          Unlock
        </Button>
        <p className="mt-4 text-[10px] leading-relaxed text-terminal-dim">
          Research & educational tool. Not investment advice. No live or broker execution.
        </p>
      </form>
    </div>
  );
}
