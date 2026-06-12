from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from . import registry, settings
from .loaders import load_stats_metrics


class JobRunner:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def submit(self, run_id: str, argv: list[str], run_directory: Path) -> None:
        self._executor.submit(self._run, run_id, argv, run_directory)

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(run_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True
        return False

    def _run(self, run_id: str, argv: list[str], run_directory: Path) -> None:
        pending = registry.read_meta(run_id) or {}
        if pending.get("status") == "cancelled":
            # Cancelled while still queued (no live process to terminate); do not start.
            return
        registry.update_meta(
            run_id,
            status="running",
            started_at=registry.now_iso(),
        )
        log_path = run_directory / "job.log"
        started = time.monotonic()
        return_code: int | None = None
        error: str | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {' '.join(argv)}\n\n")
                log_file.flush()
                proc = subprocess.Popen(
                    argv,
                    cwd=str(settings.REPO_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._lock:
                    self._procs[run_id] = proc
                try:
                    return_code = proc.wait(timeout=settings.JOB_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return_code = -9
                    error = f"Run exceeded the {settings.JOB_TIMEOUT_SEC}s time limit."
        except Exception as exc:  # noqa: BLE001
            error = f"Failed to launch run: {exc}"
        finally:
            with self._lock:
                self._procs.pop(run_id, None)

        duration = round(time.monotonic() - started, 2)
        meta = registry.read_meta(run_id) or {}
        if meta.get("status") == "cancelled":
            registry.update_meta(
                run_id,
                finished_at=registry.now_iso(),
                duration_sec=duration,
                return_code=return_code,
            )
            return

        if return_code == 0:
            metrics = load_stats_metrics(run_directory)
            registry.update_meta(
                run_id,
                status="completed",
                finished_at=registry.now_iso(),
                duration_sec=duration,
                return_code=0,
                metrics=metrics,
                error=None,
            )
        else:
            if error is None:
                error = _tail_log(log_path)
            registry.update_meta(
                run_id,
                status="failed",
                finished_at=registry.now_iso(),
                duration_sec=duration,
                return_code=return_code,
                error=error,
            )


def _tail_log(log_path: Path, max_chars: int = 1200) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "Run failed (no log available)."
    text = text.strip()
    if len(text) > max_chars:
        text = "... " + text[-max_chars:]
    return text or "Run failed with a non-zero exit code."


def utc_now() -> datetime:
    return datetime.now(UTC)
