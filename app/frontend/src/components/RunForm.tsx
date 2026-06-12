import { useMemo, useState } from "react";
import { ChevronDown, Play, ShieldCheck, AlertTriangle } from "lucide-react";
import { useConfigs, useCreateRun, useMeta } from "../lib/hooks";
import { api } from "../lib/api";
import type { RunMode, RunRequest } from "../lib/types";
import { Button, Checkbox, Field, Select, TextInput, cn } from "./ui";
import { useRunContext } from "../lib/runContext";

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: { field: string; message: string }[];
  resolved?: Record<string, any>;
}

const numOrNull = (v: string) => (v.trim() === "" ? null : Number(v));
const strOrNull = (v: string) => (v.trim() === "" ? null : v.trim());

export function RunForm({ defaultMode = "research" as RunMode }: { defaultMode?: RunMode }) {
  const { data: configsData } = useConfigs();
  const { data: meta } = useMeta();
  const createRun = useCreateRun();
  const { setSelectedId } = useRunContext();

  const [form, setForm] = useState({
    mode: defaultMode,
    config: "quick_synthetic_provider.json",
    label: "",
    profile: "",
    ticker: "",
    benchmark: "",
    start: "",
    end: "",
    quick: true,
    plots: false,
  });
  const [adv, setAdv] = useState({
    n_states: "",
    covariance_type: "",
    commission_bps: "",
    slippage_bps: "",
    train_window: "",
    retrain_frequency: "",
    random_seed: "",
  });
  const [toggles, setToggles] = useState({
    enable_diagnostics: false,
    enable_stress_tests: false,
    enable_nn_confirmation: false,
    enable_observable_markov: false,
    enable_forecast_benchmarks: false,
    enable_volatility_forecasts: false,
  });
  const [showAdv, setShowAdv] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [validating, setValidating] = useState(false);

  const configs = configsData?.configs ?? [];
  const modes = meta?.modes ?? (["research", "backtest", "paper", "portfolio", "compare_models", "nested", "universe"] as RunMode[]);
  const profiles = meta?.profiles ?? [];
  const covTypes = meta?.covariance_types ?? ["diag", "full", "tied", "spherical"];

  const payload = useMemo<RunRequest>(() => {
    const out: RunRequest = {
      mode: form.mode,
      config: form.config,
      quick: form.quick,
      plots: form.plots,
      label: strOrNull(form.label) || undefined,
      profile: strOrNull(form.profile) || undefined,
      ticker: strOrNull(form.ticker),
      benchmark: strOrNull(form.benchmark),
      start: strOrNull(form.start),
      end: strOrNull(form.end),
      n_states: numOrNull(adv.n_states),
      covariance_type: strOrNull(adv.covariance_type),
      commission_bps: numOrNull(adv.commission_bps),
      slippage_bps: numOrNull(adv.slippage_bps),
      train_window: numOrNull(adv.train_window),
      retrain_frequency: numOrNull(adv.retrain_frequency),
      random_seed: numOrNull(adv.random_seed),
      ...toggles,
    };
    return out;
  }, [form, adv, toggles]);

  const validate = async () => {
    setValidating(true);
    try {
      const res = await api.validate(payload);
      setValidation(res);
    } catch (e) {
      setValidation({ valid: false, errors: [(e as Error).message], warnings: [] });
    } finally {
      setValidating(false);
    }
  };

  const launch = async () => {
    const run = await createRun.mutateAsync(payload);
    setSelectedId(run.run_id);
    setValidation(null);
  };

  return (
    <div className="space-y-4 p-4">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Mode">
          <Select value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value as RunMode })}>
            {modes.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Base Config">
          <Select value={form.config} onChange={(e) => setForm({ ...form, config: e.target.value })}>
            {configs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Ticker" hint="Blank = use config">
          <TextInput value={form.ticker} onChange={(e) => setForm({ ...form, ticker: e.target.value })} placeholder="e.g. AAPL" />
        </Field>
        <Field label="Benchmark">
          <TextInput value={form.benchmark} onChange={(e) => setForm({ ...form, benchmark: e.target.value })} placeholder="e.g. SPY" />
        </Field>
        <Field label="Start">
          <TextInput value={form.start} onChange={(e) => setForm({ ...form, start: e.target.value })} placeholder="2020-01-01" />
        </Field>
        <Field label="End">
          <TextInput value={form.end} onChange={(e) => setForm({ ...form, end: e.target.value })} placeholder="optional" />
        </Field>
        <Field label="Risk Profile">
          <Select value={form.profile} onChange={(e) => setForm({ ...form, profile: e.target.value })}>
            <option value="">— none —</option>
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Label">
          <TextInput value={form.label} onChange={(e) => setForm({ ...form, label: e.target.value })} placeholder="run note" />
        </Field>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <Checkbox label="Quick mode (fast, reduced)" checked={form.quick} onChange={(v) => setForm({ ...form, quick: v })} />
        <Checkbox label="Generate plots" checked={form.plots} onChange={(v) => setForm({ ...form, plots: v })} />
      </div>

      <button
        onClick={() => setShowAdv((v) => !v)}
        className="flex items-center gap-1.5 text-xs font-medium text-terminal-muted hover:text-terminal-text"
      >
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", showAdv && "rotate-180")} />
        Advanced parameters
      </button>

      {showAdv && (
        <div className="space-y-3 rounded-lg border border-terminal-border bg-terminal-bg/40 p-3">
          <div className="grid grid-cols-3 gap-3">
            <Field label="HMM States">
              <TextInput value={adv.n_states} onChange={(e) => setAdv({ ...adv, n_states: e.target.value })} placeholder="3" />
            </Field>
            <Field label="Covariance">
              <Select value={adv.covariance_type} onChange={(e) => setAdv({ ...adv, covariance_type: e.target.value })}>
                <option value="">default</option>
                {covTypes.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Random Seed">
              <TextInput value={adv.random_seed} onChange={(e) => setAdv({ ...adv, random_seed: e.target.value })} placeholder="42" />
            </Field>
            <Field label="Commission (bps)">
              <TextInput value={adv.commission_bps} onChange={(e) => setAdv({ ...adv, commission_bps: e.target.value })} placeholder="7.5" />
            </Field>
            <Field label="Slippage (bps)">
              <TextInput value={adv.slippage_bps} onChange={(e) => setAdv({ ...adv, slippage_bps: e.target.value })} placeholder="3.0" />
            </Field>
            <Field label="Train Window">
              <TextInput value={adv.train_window} onChange={(e) => setAdv({ ...adv, train_window: e.target.value })} placeholder="126" />
            </Field>
            <Field label="Retrain Freq">
              <TextInput value={adv.retrain_frequency} onChange={(e) => setAdv({ ...adv, retrain_frequency: e.target.value })} placeholder="63" />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-2 pt-1">
            <Checkbox label="Diagnostics" checked={toggles.enable_diagnostics} onChange={(v) => setToggles({ ...toggles, enable_diagnostics: v })} />
            <Checkbox label="Stress tests" checked={toggles.enable_stress_tests} onChange={(v) => setToggles({ ...toggles, enable_stress_tests: v })} />
            <Checkbox label="NN confirmation" checked={toggles.enable_nn_confirmation} onChange={(v) => setToggles({ ...toggles, enable_nn_confirmation: v })} />
            <Checkbox label="Observable Markov" checked={toggles.enable_observable_markov} onChange={(v) => setToggles({ ...toggles, enable_observable_markov: v })} />
            <Checkbox label="Forecast benchmarks" checked={toggles.enable_forecast_benchmarks} onChange={(v) => setToggles({ ...toggles, enable_forecast_benchmarks: v })} />
            <Checkbox label="Volatility forecasts" checked={toggles.enable_volatility_forecasts} onChange={(v) => setToggles({ ...toggles, enable_volatility_forecasts: v })} />
          </div>
        </div>
      )}

      {validation && (
        <div
          className={cn(
            "rounded-lg border p-3 text-xs",
            validation.valid ? "border-bull/30 bg-bull/5" : "border-bear/30 bg-bear/5"
          )}
        >
          <div className="flex items-center gap-1.5 font-semibold">
            {validation.valid ? (
              <ShieldCheck className="h-4 w-4 text-bull" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-bear" />
            )}
            {validation.valid ? "Configuration valid" : "Configuration invalid"}
          </div>
          {validation.errors.map((er, i) => (
            <p key={i} className="mt-1 text-bear">• {er}</p>
          ))}
          {validation.warnings.map((w, i) => (
            <p key={i} className="mt-1 text-side">• {w.field}: {w.message}</p>
          ))}
          {validation.resolved && (
            <p className="mt-1.5 font-mono text-terminal-muted">
              {validation.resolved.ticker} vs {validation.resolved.benchmark} · {validation.resolved.n_states} states · {validation.resolved.covariance_type}
            </p>
          )}
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <Button variant="ghost" onClick={validate} loading={validating}>
          <ShieldCheck className="h-3.5 w-3.5" /> Validate
        </Button>
        <Button variant="primary" onClick={launch} loading={createRun.isPending}>
          <Play className="h-3.5 w-3.5" /> Launch Run
        </Button>
        {createRun.isError && (
          <span className="text-xs text-bear">{(createRun.error as Error).message}</span>
        )}
      </div>
    </div>
  );
}
