---
name: Replit port binding for backend workflows
description: Why backend servers must bind 0.0.0.0 (not 127.0.0.1) for Replit workflow port detection
---

A uvicorn/server process bound to `127.0.0.1` starts and serves fine locally, but the
Replit workflow `waitForPort` probe marks the workflow **failed** ("Expected port to open,
but it didn't") because the probe connects from outside loopback.

**Rule:** Bind backend workflow servers to `0.0.0.0` even when only the local Vite dev
server proxies to them. Use `uvicorn ... --host 0.0.0.0 --port 8000`.

**Why:** The port-open detector cannot see a loopback-only listener; detection failure
blocks the preview/workflow from being marked ready.

**How to apply:** Any `outputType: "console"` backend workflow with `waitForPort` must
bind `0.0.0.0`. The frontend (port 5000 webview) can still proxy to `127.0.0.1:8000`.
