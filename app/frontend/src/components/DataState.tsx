import type { ReactNode } from "react";
import { AlertTriangle, FileQuestion } from "lucide-react";
import { EmptyState, Spinner } from "./ui";

export function DataState({
  loading,
  error,
  available,
  reason,
  children,
  emptyTitle = "Data unavailable",
}: {
  loading?: boolean;
  error?: unknown;
  available?: boolean;
  reason?: string;
  children: ReactNode;
  emptyTitle?: string;
}) {
  if (loading) return <Spinner />;
  if (error) {
    return (
      <EmptyState
        title="Could not load data"
        hint={(error as Error)?.message || "Request failed."}
        icon={<AlertTriangle className="h-7 w-7" />}
      />
    );
  }
  if (available === false) {
    return (
      <EmptyState
        title={emptyTitle}
        hint={reason || "This output was not produced for the current run configuration."}
        icon={<FileQuestion className="h-7 w-7" />}
      />
    );
  }
  return <>{children}</>;
}
