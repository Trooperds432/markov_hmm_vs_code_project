import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useRuns } from "./hooks";
import type { RunMeta } from "./types";

const KEY = "qt_selected_run";

interface RunContextValue {
  runs: RunMeta[];
  selectedId?: string;
  selected?: RunMeta;
  setSelectedId: (id: string) => void;
  loading: boolean;
}

const RunContext = createContext<RunContextValue>({
  runs: [],
  setSelectedId: () => {},
  loading: false,
});

export function RunProvider({ children }: { children: ReactNode }) {
  const { data, isLoading } = useRuns();
  const runs = data?.runs ?? [];
  const [selectedId, setSelected] = useState<string | undefined>(
    () => localStorage.getItem(KEY) || undefined
  );

  useEffect(() => {
    if (runs.length === 0) return;
    const exists = selectedId && runs.some((r) => r.run_id === selectedId);
    if (!exists) {
      const completed = runs.find((r) => r.status === "completed");
      setSelected((completed ?? runs[0]).run_id);
    }
  }, [runs, selectedId]);

  const setSelectedId = (id: string) => {
    setSelected(id);
    localStorage.setItem(KEY, id);
  };

  const selected = useMemo(
    () => runs.find((r) => r.run_id === selectedId),
    [runs, selectedId]
  );

  return (
    <RunContext.Provider value={{ runs, selectedId, selected, setSelectedId, loading: isLoading }}>
      {children}
    </RunContext.Provider>
  );
}

export const useRunContext = () => useContext(RunContext);
