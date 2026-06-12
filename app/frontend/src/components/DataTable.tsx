import type { RunMeta, TableData } from "../lib/types";
import { useTable } from "../lib/hooks";
import { DataState } from "./DataState";
import { Panel } from "./ui";

function cell(v: any): { text: string; numeric: boolean } {
  if (v === null || v === undefined || v === "") return { text: "—", numeric: false };
  if (typeof v === "boolean") return { text: v ? "true" : "false", numeric: false };
  if (typeof v === "number") {
    if (Number.isNaN(v)) return { text: "—", numeric: true };
    if (Number.isInteger(v)) return { text: v.toLocaleString(), numeric: true };
    const abs = Math.abs(v);
    const digits = abs !== 0 && abs < 0.01 ? 6 : abs < 1 ? 4 : abs < 1000 ? 3 : 2;
    return { text: v.toLocaleString(undefined, { maximumFractionDigits: digits }), numeric: true };
  }
  const s = String(v);
  const asNum = Number(s);
  if (s.trim() !== "" && !Number.isNaN(asNum)) {
    return cell(asNum);
  }
  return { text: s, numeric: false };
}

export function DataTable({ data, maxHeight = 420 }: { data?: TableData; maxHeight?: number }) {
  if (!data) return null;
  return (
    <div className="overflow-auto" style={{ maxHeight }}>
      <table className="w-full text-xs">
        <thead className="sticky top-0 z-10 bg-terminal-panel">
          <tr className="border-b border-terminal-border text-left text-terminal-dim">
            {data.columns.map((c) => (
              <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr key={i} className="border-b border-terminal-border/40 hover:bg-terminal-panel2/50">
              {data.columns.map((c) => {
                const { text, numeric } = cell(row[c]);
                return (
                  <td
                    key={c}
                    className={`whitespace-nowrap px-3 py-1.5 ${
                      numeric ? "text-right font-mono tnum text-terminal-text" : "text-terminal-muted"
                    }`}
                  >
                    {text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {data.total_rows != null && data.total_rows > data.rows.length && (
        <div className="px-3 py-2 text-[10px] text-terminal-dim">
          Showing {data.rows.length} of {data.total_rows} rows
        </div>
      )}
    </div>
  );
}

export function TableCard({
  run,
  file,
  title,
  subtitle,
  limit,
  maxHeight,
  className,
}: {
  run?: RunMeta;
  file: string;
  title: string;
  subtitle?: string;
  limit?: number;
  maxHeight?: number;
  className?: string;
}) {
  const { data, isLoading, error } = useTable(run?.run_id, file, run?.status, limit);
  return (
    <Panel title={title} subtitle={subtitle} className={className}>
      <DataState
        loading={isLoading}
        error={error}
        available={data?.available}
        reason={data?.reason}
        emptyTitle="Not produced by this run"
      >
        <DataTable data={data} maxHeight={maxHeight} />
      </DataState>
    </Panel>
  );
}
