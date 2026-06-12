import type {
  AppMeta,
  DependencyRow,
  OutputFile,
  RunMeta,
  RunRequest,
  SeriesData,
  StatsData,
  TableData,
} from "./types";

const PW_KEY = "qt_app_password";

export function getPassword(): string {
  return localStorage.getItem(PW_KEY) || "";
}
export function setPassword(value: string): void {
  localStorage.setItem(PW_KEY, value);
}
export function clearPassword(): void {
  localStorage.removeItem(PW_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers = new Headers(opts.headers);
  if (opts.body) headers.set("Content-Type", "application/json");
  const pw = getPassword();
  if (pw) headers.set("X-App-Password", pw);
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; app_password_required: boolean }>("/api/health"),
  authStatus: () => request<{ required: boolean }>("/api/auth/status"),
  login: (password: string) =>
    request<{ ok: boolean }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  meta: () => request<AppMeta>("/api/meta"),
  dependencies: () =>
    request<{ dependencies: DependencyRow[] }>("/api/dependencies"),
  configs: () => request<{ configs: string[] }>("/api/configs"),
  getConfig: (name: string) =>
    request<{ name: string; content: Record<string, any> }>(
      `/api/configs/${encodeURIComponent(name)}`
    ),
  validate: (body: Partial<RunRequest>) =>
    request<{
      valid: boolean;
      errors: string[];
      warnings: { field: string; message: string }[];
      resolved?: Record<string, any>;
    }>("/api/configs/validate", { method: "POST", body: JSON.stringify(body) }),

  listRuns: () => request<{ runs: RunMeta[] }>("/api/runs"),
  getRun: (id: string) => request<RunMeta>(`/api/runs/${id}`),
  runStatus: (id: string) =>
    request<{
      run_id: string;
      status: string;
      duration_sec?: number | null;
      error?: string | null;
      metrics?: any;
    }>(`/api/runs/${id}/status`),
  logs: (id: string, offset = 0) =>
    request<{ content: string; offset: number; eof: boolean }>(
      `/api/runs/${id}/logs?offset=${offset}`
    ),
  createRun: (body: RunRequest) =>
    request<RunMeta>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  rerun: (id: string) =>
    request<RunMeta>(`/api/runs/${id}/rerun`, { method: "POST" }),
  cancel: (id: string) =>
    request<{ ok: boolean }>(`/api/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: string) =>
    request<{ ok: boolean }>(`/api/runs/${id}`, { method: "DELETE" }),

  stats: (id: string) => request<StatsData>(`/api/runs/${id}/stats`),
  series: (id: string) => request<SeriesData>(`/api/runs/${id}/series`),
  outputs: (id: string) =>
    request<{ files: OutputFile[] }>(`/api/runs/${id}/outputs`),
  table: (id: string, name: string, limit?: number) =>
    request<TableData>(
      `/api/runs/${id}/table/${name}${limit ? `?limit=${limit}` : ""}`
    ),
};

export async function downloadFile(runId: string, filename: string): Promise<void> {
  const headers = new Headers();
  const pw = getPassword();
  if (pw) headers.set("X-App-Password", pw);
  const res = await fetch(
    `/api/runs/${runId}/download/${encodeURIComponent(filename)}`,
    { headers }
  );
  if (!res.ok) throw new ApiError(res.status, "download failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
