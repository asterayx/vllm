#!/usr/bin/env bash
# boot-shape-warmup.sh — burn common Triton / DSpark shape buckets after ready.
#
# Why: mid-serve JIT on TP=2 leaves a peer stuck in NCCL until the 600s
# ProcessGroup watchdog kills the pair (Mia issue #117 pattern). This script
# is non-fatal: failures WARN and exit 0 from the launcher's perspective when
# invoked with `|| echo WARN`.
#
# Usage: boot-shape-warmup.sh [base_url] [model]
#   base_url default http://127.0.0.1:30001/v1
#   model    default deepseek-v4-flash-0731
#
# Env:
#   DSPARK_WARMUP_REQ_TIMEOUT   curl --max-time (default 240)
#   DSPARK_WARMUP_MAX_CONCURRENCY  chat concurrency ladder cap (default 6)
#   VLLM_API_KEY / DSPARK_WARMUP_BEARER  optional bearer
set -u

BASE="${1:-http://127.0.0.1:30001/v1}"
MODEL="${2:-deepseek-v4-flash-0731}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${DSPARK_WARMUP_REQ_TIMEOUT:-240}"
MAX_CONCURRENCY="${DSPARK_WARMUP_MAX_CONCURRENCY:-6}"
NONCE="$$-$(date +%s)"

case "$MAX_CONCURRENCY" in
  ''|*[!0-9]*|0) MAX_CONCURRENCY=6 ;;
esac

AUTH_ARGS=()
if [ -n "${DSPARK_WARMUP_BEARER:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${DSPARK_WARMUP_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

ok=0
fail=0

mk_prompt() {
  local n=$1 tag=$2 body
  body=$(printf 'warm %.0s' $(seq 1 "$n"))
  printf '[warmup %s %s] Filler: %s Reply with OK.' "$NONCE" "$tag" "$body"
}

post_chat() {
  local tag=$1 words=$2 out=$3
  local prompt payload
  prompt=$(mk_prompt "$words" "$tag")
  payload=$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":8,"temperature":0,"stream":false}' \
    "$MODEL" "$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
  if "$CURL_BIN" -fsS "${AUTH_ARGS[@]}" \
      -H 'Content-Type: application/json' \
      --max-time "$REQ_TIMEOUT" \
      -d "$payload" \
      "${BASE}/chat/completions" >"$out" 2>"${out}.err"; then
    ok=$((ok + 1))
    echo "boot-shape-warmup: OK  ${tag}"
  else
    fail=$((fail + 1))
    echo "boot-shape-warmup: FAIL ${tag} ($(tr '\n' ' ' <"${out}.err" | head -c 160))" >&2
  fi
}

post_completion() {
  local tag=$1 tokens=$2 out=$3
  # Exact-ish ladder for DFlash prepare-inputs BLOCK_SIZE = next_pow2(s+6).
  local prompt payload
  prompt=$(python3 - "$tokens" "$NONCE" "$tag" <<'PY'
import sys
n, nonce, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
# 'hello' + (n-1)x' hello' is ~n tokens on DeepSeek tokenizers; nonce busts cache.
body = "hello" + (" hello" * max(n - 1, 0))
print(f"[ladder {nonce} {tag}] {body}")
PY
)
  payload=$(printf '{"model":"%s","prompt":%s,"max_tokens":1,"temperature":0,"stream":false}' \
    "$MODEL" "$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
  if "$CURL_BIN" -fsS "${AUTH_ARGS[@]}" \
      -H 'Content-Type: application/json' \
      --max-time "$REQ_TIMEOUT" \
      -d "$payload" \
      "${BASE}/completions" >"$out" 2>"${out}.err"; then
    ok=$((ok + 1))
    echo "boot-shape-warmup: OK  ${tag}"
  else
    fail=$((fail + 1))
    echo "boot-shape-warmup: FAIL ${tag} ($(tr '\n' ' ' <"${out}.err" | head -c 160))" >&2
  fi
}

echo "boot-shape-warmup: base=${BASE} model=${MODEL} max_c=${MAX_CONCURRENCY}"

# Bucket ladder → BLOCK {8,16,32,64,128,256} via next_pow2(s+6)
for s in 1 6 20 45 100 200; do
  post_completion "ladder-s${s}" "$s" "${tmpdir}/ladder-${s}.json"
done

# Chat concurrency arms (batch-keyed kernels)
for c in 1 2 4 6; do
  if [ "$c" -gt "$MAX_CONCURRENCY" ]; then
    continue
  fi
  pids=()
  for i in $(seq 1 "$c"); do
    post_chat "c${c}-${i}" 64 "${tmpdir}/c${c}-${i}.json" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
done

# Medium + multi-chunk-ish prefill
post_chat "medium" 2048 "${tmpdir}/medium.json"
post_chat "longchunk" 9000 "${tmpdir}/longchunk.json"

# Sampling arms: top_k / top_p constexpr combos (greedy arms above skip these)
for profile in k-only p-only kp; do
  case "$profile" in
    k-only) extra='"top_k":50' ;;
    p-only) extra='"top_p":0.9' ;;
    kp) extra='"top_k":50,"top_p":0.9' ;;
  esac
  prompt=$(mk_prompt 32 "sample-${profile}")
  payload=$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":8,"temperature":0.8,%s,"stream":false}' \
    "$MODEL" "$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$extra")
  out="${tmpdir}/sample-${profile}.json"
  if "$CURL_BIN" -fsS "${AUTH_ARGS[@]}" \
      -H 'Content-Type: application/json' \
      --max-time "$REQ_TIMEOUT" \
      -d "$payload" \
      "${BASE}/chat/completions" >"$out" 2>"${out}.err"; then
    ok=$((ok + 1))
    echo "boot-shape-warmup: OK  sample-${profile}"
  else
    fail=$((fail + 1))
    echo "boot-shape-warmup: FAIL sample-${profile}" >&2
  fi
done

echo "boot-shape-warmup: done ok=${ok} fail=${fail}"
# Non-fatal contract: always exit 0 so launchers can WARN without failing boot.
exit 0
