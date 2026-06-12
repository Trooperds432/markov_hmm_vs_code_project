import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import { api } from "../lib/api";
import { useDependencies, useMeta } from "../lib/hooks";
import { PageHeader } from "../components/PageHeader";
import { Panel, Tag, cn } from "../components/ui";

export function Settings() {
  const { data: meta } = useMeta();
  const { data: deps } = useDependencies();
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 10000 });

  const rows = deps?.dependencies ?? [];

  return (
    <div className="space-y-5">
      <PageHeader title="Settings" desc="Environment status, dependency doctor, and engine metadata." />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Panel title="System">
          <div className="space-y-2.5 p-4 text-xs">
            <Row label="Backend">
              <Tag tone={health?.status === "ok" ? "bull" : "bear"}>{health?.status === "ok" ? "online" : "offline"}</Tag>
            </Row>
            <Row label="Auth gate">
              <span className="text-terminal-text">{meta?.app_password_required ? "enabled" : "disabled"}</span>
            </Row>
            <Row label="Profiles">
              <span className="font-mono text-terminal-text">{meta?.profiles?.join(", ") || "—"}</span>
            </Row>
            <Row label="Modes">
              <span className="font-mono text-terminal-muted">{meta?.modes?.join(", ") || "—"}</span>
            </Row>
            <Row label="Covariance types">
              <span className="font-mono text-terminal-muted">{meta?.covariance_types?.join(", ") || "—"}</span>
            </Row>
          </div>
        </Panel>

        <Panel title="Dependency Doctor" subtitle={`${rows.length} packages`}>
          <div className="max-h-[320px] overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-terminal-panel">
                <tr className="border-b border-terminal-border text-left text-terminal-dim">
                  <th className="px-4 py-2 font-medium">Package</th>
                  <th className="px-4 py-2 font-medium">Group</th>
                  <th className="px-4 py-2 text-right font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((d) => {
                  const ok = d.status === "available";
                  return (
                    <tr key={d.package} className="border-b border-terminal-border/40">
                      <td className="px-4 py-1.5 font-mono text-terminal-text">{d.package}</td>
                      <td className="px-4 py-1.5 text-terminal-muted">{d.group}</td>
                      <td className="px-4 py-1.5">
                        <span className={cn("flex items-center justify-end gap-1.5", ok ? "text-bull" : "text-terminal-dim")}>
                          {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                          {d.status}
                          {d.version ? ` ${d.version}` : ""}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel title="Disclaimer">
        <p className="p-4 text-xs leading-relaxed text-terminal-muted">{meta?.disclaimer}</p>
      </Panel>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="micro">{label}</span>
      {children}
    </div>
  );
}
