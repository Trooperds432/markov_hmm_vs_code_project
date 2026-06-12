import { useState } from "react";
import { useRunContext } from "../lib/runContext";
import { useOutputs } from "../lib/hooks";
import { PageHeader } from "../components/PageHeader";
import { Panel, Select } from "../components/ui";
import { RunForm } from "../components/RunForm";
import { TableCard } from "../components/DataTable";

export function Experiments() {
  const { selected } = useRunContext();
  const { data: outputs } = useOutputs(selected?.run_id, selected?.status);
  const csvFiles = (outputs?.files ?? []).filter((f) => f.kind === "csv").sort((a, b) => a.name.localeCompare(b.name));
  const [file, setFile] = useState("");

  return (
    <div className="space-y-5">
      <PageHeader
        title="Experiments"
        desc="Model comparison, nested cross-validation, and a raw output explorer for any artifact produced by a run."
      />

      <Panel title="Experiment Run" subtitle="mode = compare_models / nested / universe">
        <RunForm defaultMode="compare_models" />
      </Panel>

      <Panel
        title="Output Explorer"
        subtitle={selected ? `${csvFiles.length} CSV artifacts` : "Select a completed run"}
        right={
          csvFiles.length > 0 && (
            <Select value={file} onChange={(e) => setFile(e.target.value)} className="max-w-[260px]">
              <option value="">Choose an artifact…</option>
              {csvFiles.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </Select>
          )
        }
      >
        {selected && file ? (
          <div className="p-3">
            <TableCard
              run={selected}
              file={file.replace(/\.csv$/, "")}
              title={file}
              limit={400}
              maxHeight={520}
            />
          </div>
        ) : (
          <div className="p-8 text-center text-sm text-terminal-muted">
            {selected
              ? "Pick an artifact above to inspect its contents."
              : "Select a completed run to explore its outputs."}
          </div>
        )}
      </Panel>
    </div>
  );
}
