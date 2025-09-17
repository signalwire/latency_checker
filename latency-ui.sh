#!/usr/bin/env bash
# Manage the Audio Latency Analyzer web UI.
#
# Usage:
#   ./latency-ui.sh start    Start the server in the background
#   ./latency-ui.sh stop     Stop the running server
#   ./latency-ui.sh restart  Stop then start
#   ./latency-ui.sh status   Show running state
#   ./latency-ui.sh logs     Tail the log (Ctrl-C to exit)
#
# Environment variables (all optional):
#   LATENCY_UI_HOST      Host to bind (default: 127.0.0.1)
#   LATENCY_UI_PORT      Port to listen on (default: 8000)
#   LATENCY_UI_LOG       Log file path (default: ./latency-ui.log)
#   LATENCY_UI_PID       PID file path (default: ./latency-ui.pid)
#   LATENCY_UI_BIN       Path to latency-ui binary (default: latency-ui on PATH)

set -euo pipefail

HOST="${LATENCY_UI_HOST:-127.0.0.1}"
PORT="${LATENCY_UI_PORT:-8000}"
LOG_FILE="${LATENCY_UI_LOG:-./latency-ui.log}"
PID_FILE="${LATENCY_UI_PID:-./latency-ui.pid}"
BIN="${LATENCY_UI_BIN:-latency-ui}"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "Already running (pid $(cat "$PID_FILE"))"
    return 0
  fi

  if ! command -v "$BIN" >/dev/null 2>&1; then
    echo "Error: '$BIN' not found on PATH. Install with: pip install .[web]" >&2
    exit 1
  fi

  echo "Starting latency-ui on $HOST:$PORT (log: $LOG_FILE)"
  nohup "$BIN" --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  # Detach from the job table so bash won't try to wait/reap us later
  disown "$pid" 2>/dev/null || true
  echo "$pid" > "$PID_FILE"

  # Watch the process for up to 8 seconds. Uvicorn takes ~4-5 seconds to
  # fail-and-exit when the port is already bound by another process, so
  # our window needs to cover that. If the process exits during this
  # window, it failed — report it. If it survives, it started.
  local deadline=$((SECONDS + 8))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Failed to start — last log lines:" >&2
      tail -n 20 "$LOG_FILE" >&2 2>/dev/null || true
      exit 1
    fi
    sleep 0.25
  done

  echo "Started (pid $pid)"
}

cmd_stop() {
  if ! is_running; then
    echo "Not running"
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping pid $pid"
  kill "$pid"

  # Wait up to 10s for graceful shutdown
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "Graceful stop timed out; sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_FILE"
  echo "Stopped"
}

cmd_restart() {
  cmd_stop
  cmd_start
}

cmd_status() {
  if is_running; then
    echo "Running (pid $(cat "$PID_FILE")) on $HOST:$PORT"
    echo "Log: $LOG_FILE"
  else
    echo "Not running"
    if [[ -f "$PID_FILE" ]]; then
      echo "Stale pid file: $PID_FILE"
    fi
  fi
}

cmd_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "No log file yet ($LOG_FILE)"
    exit 1
  fi
  tail -n 100 -f "$LOG_FILE"
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac
