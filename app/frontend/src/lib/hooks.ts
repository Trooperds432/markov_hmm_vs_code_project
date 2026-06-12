import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { api } from "./api";
import type { RunMeta, RunRequest, RunStatus } from "./types";

const ACTIVE: RunStatus[] = ["queued", "running"];

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: (query) => {
      const data = query.state.data;
      const active = data?.runs?.some((r) => ACTIVE.includes(r.status));
      return active ? 2500 : 7000;
    },
  });
}

export function useRun(id?: string) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id!),
    enabled: !!id,
    refetchInterval: (query) =>
      ACTIVE.includes((query.state.data as RunMeta)?.status) ? 2000 : false,
  });
}

export function useSeries(id?: string, status?: RunStatus) {
  return useQuery({
    queryKey: ["series", id],
    queryFn: () => api.series(id!),
    enabled: !!id && status === "completed",
  });
}

export function useStats(id?: string, status?: RunStatus) {
  return useQuery({
    queryKey: ["stats", id],
    queryFn: () => api.stats(id!),
    enabled: !!id && status === "completed",
  });
}

export function useTable(id: string | undefined, name: string, status?: RunStatus, limit?: number) {
  return useQuery({
    queryKey: ["table", id, name, limit],
    queryFn: () => api.table(id!, name, limit),
    enabled: !!id && status === "completed",
  });
}

export function useOutputs(id?: string, status?: RunStatus) {
  return useQuery({
    queryKey: ["outputs", id],
    queryFn: () => api.outputs(id!),
    enabled: !!id && status === "completed",
  });
}

export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: 60_000 });
}

export function useConfigs() {
  return useQuery({ queryKey: ["configs"], queryFn: api.configs, staleTime: 60_000 });
}

export function useDependencies() {
  return useQuery({ queryKey: ["dependencies"], queryFn: api.dependencies, staleTime: 30_000 });
}

export function useCreateRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunRequest) => api.createRun(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export function useRunActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["runs"] });
  return {
    rerun: useMutation({ mutationFn: (id: string) => api.rerun(id), onSuccess: invalidate }),
    cancel: useMutation({ mutationFn: (id: string) => api.cancel(id), onSuccess: invalidate }),
    remove: useMutation({ mutationFn: (id: string) => api.deleteRun(id), onSuccess: invalidate }),
  };
}
