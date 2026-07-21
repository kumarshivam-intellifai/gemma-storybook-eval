#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTFOO_BIN="$ROOT/node_modules/.bin/promptfoo"
MODEL="${MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
SERVED_MODEL="gemma-4-26b-a4b-nvfp4"
PORT=38080
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PREFIX_CACHING="${PREFIX_CACHING:-true}"
RUN_ID="${RUN_ID:-gemma-4-26b-a4b-nvfp4_full_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT/results/$RUN_ID"
SERVER_LOG="$RUN_DIR/vllm_server.log"
SERVER_PID=""

VLLM_BIN="${VLLM_BIN:-}"
if [[ -z "$VLLM_BIN" ]]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -n "$VLLM_BIN" ]]; then
  candidate="$(dirname "$VLLM_BIN")/python"
  if [[ -x "$candidate" ]]; then
    PYTHON_BIN="$candidate"
  fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for command_name in node curl setsid; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command_name" >&2
    exit 2
  fi
done
if [[ ! -x "$PROMPTFOO_BIN" ]]; then
  printf 'Missing Promptfoo. Run npm ci in %s\n' "$ROOT" >&2
  exit 2
fi
if [[ -z "$VLLM_BIN" || ! -x "$VLLM_BIN" ]]; then
  printf 'Missing vLLM. Activate the Python environment or set VLLM_BIN=/path/to/vllm.\n' >&2
  exit 2
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  printf 'Missing Python. Set PYTHON_BIN=/path/to/python.\n' >&2
  exit 2
fi

case "${PREFIX_CACHING,,}" in
  true|1|yes|on)
    PREFIX_CACHE_ARG=(--enable-prefix-caching)
    ;;
  false|0|no|off)
    PREFIX_CACHE_ARG=(--no-enable-prefix-caching)
    ;;
  *)
    printf 'PREFIX_CACHING must be true or false, got: %s\n' "$PREFIX_CACHING" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR" "$ROOT/.promptfoo"
export PATH="$(dirname "$VLLM_BIN"):$(dirname "$PYTHON_BIN"):$PATH"
export PROMPTFOO_PYTHON="$PYTHON_BIN"
export PROMPTFOO_CONFIG_DIR="$ROOT/.promptfoo"
export REQUEST_TIMEOUT_MS="${REQUEST_TIMEOUT_MS:-300000}"
export PROMPTFOO_DISABLE_UPDATE=true

printf 'Run id: %s\n' "$RUN_ID"
printf 'Results: %s\n' "$RUN_DIR"
printf 'Model: %s\n' "$MODEL"
printf 'Prefix caching: %s\n' "$PREFIX_CACHING"

setsid "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --dtype auto \
  "${PREFIX_CACHE_ARG[@]}" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'vLLM exited before becoming ready. See %s\n' "$SERVER_LOG" >&2
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'Timed out waiting for vLLM. See %s\n' "$SERVER_LOG" >&2
  exit 1
fi

printf 'vLLM is ready; starting the 64-case Promptfoo evaluation.\n'
cd "$ROOT"
set +e
"$PROMPTFOO_BIN" eval \
  --config promptfooconfig_full.yaml \
  --no-cache \
  --no-share \
  --max-concurrency 1 \
  --output "$RUN_DIR/results.json" "$RUN_DIR/results.html"
EVAL_STATUS=$?
set -e

if [[ -s "$RUN_DIR/results.json" ]]; then
  "$PYTHON_BIN" "$ROOT/summarize_full_results.py" \
    "$RUN_DIR/results.json" "$RUN_DIR" \
    | tee "$RUN_DIR/summary_stdout.json"
  "$PYTHON_BIN" "$ROOT/regrade_full_results.py" \
    "$RUN_DIR/results.json" "$RUN_DIR" \
    | tee "$RUN_DIR/regraded_summary_stdout.json"
fi

printf 'Promptfoo JSON: %s\n' "$RUN_DIR/results.json"
printf 'Promptfoo HTML: %s\n' "$RUN_DIR/results.html"
printf 'Compact summary: %s\n' "$RUN_DIR/summary.json"
printf 'Per-case CSV: %s\n' "$RUN_DIR/cases.csv"
printf 'Audited summary: %s\n' "$RUN_DIR/regraded_summary.json"
printf 'Audited per-case CSV: %s\n' "$RUN_DIR/regraded_cases.csv"
printf 'vLLM log: %s\n' "$SERVER_LOG"
exit "$EVAL_STATUS"
