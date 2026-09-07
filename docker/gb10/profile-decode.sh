#!/usr/bin/env bash
# Profile decode steps of the running GB10 serve stack and summarize them.
#
# The server must have been started with VLLM_PROFILE_DIR set (run.sh /
# run-vision.sh mount it at /root/profiles and enable the torch profiler):
#   VLLM_PROFILE_DIR=~/vllm-profiles NODE_RANK=0 ./docker/gb10/run-vision.sh
#   VLLM_PROFILE_DIR=~/vllm-profiles NODE_RANK=1 ... ./docker/gb10/run-vision.sh
# Then, on the head node:
#   VLLM_PROFILE_DIR=~/vllm-profiles ./docker/gb10/profile-decode.sh
#
# Steps: one unprofiled warm request (prefix cache, JIT), /start_profile,
# CONCURRENCY x PROFILE_REQUESTS chat requests of MAX_TOKENS tokens,
# /stop_profile, then summarize-profile.py on rank 0's newest trace.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
URL="${URL:-http://127.0.0.1:30001}"
PROFILE_DIR="${VLLM_PROFILE_DIR:-${HOME}/vllm-profiles}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1}"
PROFILE_REQUESTS="${PROFILE_REQUESTS:-2}"
PY="${PY:-python3}"

model="$(curl -fsS --max-time 30 "${URL}/v1/models" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "model ${model}; traces -> ${PROFILE_DIR}"

request() {  # request <max_tokens> -> completion token count on stdout
  local body status
  body="$(mktemp)"
  status="$(curl -sS --max-time 900 -o "${body}" -w '%{http_code}' \
    "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"temperature\":0,\"seed\":0,\"max_tokens\":$1,\"chat_template_kwargs\":{\"thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed explanation of how a hash table handles collisions, with examples.\"}]}")"
  if [ "${status}" != "200" ]; then
    echo "chat request failed: HTTP ${status}" >&2
    cat "${body}" >&2
    echo >&2
    rm -f "${body}"
    return 1
  fi
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["usage"]["completion_tokens"])' "${body}"
  rm -f "${body}"
}

echo "warm request (not profiled)"
request 16 >/dev/null

# The API server writes a CPU-only "*.async_llm*" trace next to the worker's.
newest_worker_trace() {
  ls -1t "${PROFILE_DIR}"/*.pt.trace.json* 2>/dev/null | grep -v async_llm | head -n1 || true
}

before="$(newest_worker_trace)"
post() {  # post <path> <timeout>; prints the server response on failure
  local body status
  body="$(mktemp)"
  status="$(curl -sS -X POST --max-time "$2" -o "${body}" -w '%{http_code}' "${URL}$1")"
  if [ "${status}" != "200" ]; then
    echo "POST $1 failed: HTTP ${status}" >&2
    cat "${body}" >&2
    echo >&2
    echo "both ranks must be started with VLLM_PROFILE_DIR (each node parses its own --profiler-config)" >&2
    rm -f "${body}"
    return 1
  fi
  rm -f "${body}"
}

post /start_profile 60
echo "profiling ${CONCURRENCY}x${PROFILE_REQUESTS} requests of ${MAX_TOKENS} tokens"
total=0
for _ in $(seq "${PROFILE_REQUESTS}"); do
  pids=()
  tmp="$(mktemp -d)"
  for c in $(seq "${CONCURRENCY}"); do
    request "${MAX_TOKENS}" > "${tmp}/${c}" &
    pids+=($!)
  done
  wait "${pids[@]}"
  for c in $(seq "${CONCURRENCY}"); do
    total=$((total + $(cat "${tmp}/${c}")))
  done
  rm -rf "${tmp}"
done
post /stop_profile 600
echo "generated ${total} tokens under the profiler"

# Rank 0's trace is written by the head container into the mounted dir.
for _ in $(seq 60); do
  newest="$(newest_worker_trace)"
  if [ -n "${newest}" ] && [ "${newest}" != "${before}" ]; then
    break
  fi
  sleep 2
done
if [ -z "${newest:-}" ] || [ "${newest}" = "${before}" ]; then
  echo "no new trace under ${PROFILE_DIR}; was the server started with VLLM_PROFILE_DIR?" >&2
  exit 1
fi
echo "trace ${newest}"
"$PY" "${HERE}/summarize-profile.py" "${newest}" --tokens "${total}" --top "${TOP:-30}"
echo
echo "torch key_averages table: ${PROFILE_DIR}/profiler_out_0.txt"
