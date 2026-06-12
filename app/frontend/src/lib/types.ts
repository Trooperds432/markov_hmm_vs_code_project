export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type RunMode =
  | "research"
  | "backtest"
  | "paper"
  | "portfolio"
  | "compare_models"
  | "nested"
  | "universe"
  | "diagnostics"
  | "ci_smoke";

export interface RunRequest {
  mode: RunMode;
  config: string;
  ticker?: string | null;
  benchmark?: string | null;
  start?: string | null;
  end?: string | null;
  profile?: string | null;
  quick?: boolean;
  plots?: boolean;
  label?: string | null;
  n_states?: number | null;
  covariance_type?: string | null;
  commission_bps?: number | null;
  slippage_bps?: number | null;
  train_window?: number | null;
  retrain_frequency?: number | null;
  random_seed?: number | null;
  enable_diagnostics?: boolean | null;
  enable_stress_tests?: boolean | null;
  enable_nn_confirmation?: boolean | null;
  enable_observable_markov?: boolean | null;
  enable_forecast_benchmarks?: boolean | null;
  enable_volatility_forecasts?: boolean | null;
}

export interface Metrics {
  strategy_sharpe?: number | null;
  strategy_total_return?: number | null;
  strategy_max_drawdown?: number | null;
  win_rate?: number | null;
  strategy_cagr?: number | null;
  strategy_sortino?: number | null;
  strategy_calmar?: number | null;
  turnover?: number | null;
  exposure_adjusted_cagr?: number | null;
  avg_abs_position?: number | null;
  buy_hold_total_return?: number | null;
  trades?: number | null;
  ticker?: string | null;
  benchmark?: string | null;
}

export interface ConfigWarning {
  field: string;
  message: string;
}

export interface RunMeta {
  run_id: string;
  status: RunStatus;
  mode: RunMode;
  label: string;
  request: RunRequest;
  command: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_sec?: number | null;
  return_code?: number | null;
  error?: string | null;
  metrics?: Metrics | null;
  config_warnings?: ConfigWarning[];
  ticker?: string | null;
  benchmark?: string | null;
}

export interface SeriesData {
  available: boolean;
  reason: string;
  points: number;
  dates?: string[];
  price?: (number | null)[];
  strategy_equity?: (number | null)[];
  buy_hold_equity?: (number | null)[];
  position?: (number | null)[];
  drawdown?: (number | null)[];
  prob_bull?: (number | null)[];
  prob_bear?: (number | null)[];
  prob_sideways?: (number | null)[];
  confidence?: (number | null)[];
  regime?: (string | null)[];
}

export interface TableData {
  available: boolean;
  reason: string;
  columns: string[];
  rows: Record<string, any>[];
  total_rows?: number;
}

export interface StatsData {
  available: boolean;
  reason: string;
  stats: Record<string, number | string | null>;
}

export interface OutputFile {
  name: string;
  size: number;
  kind: string;
}

export interface AppMeta {
  disclaimer: string;
  profiles: string[];
  modes: RunMode[];
  covariance_types: string[];
  app_password_required: boolean;
}

export interface DependencyRow {
  package?: string;
  status?: string;
  version?: string | null;
  [key: string]: any;
}
