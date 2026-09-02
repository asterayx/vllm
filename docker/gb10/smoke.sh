#!/usr/bin/env bash
# Wait until the GB10 serve container is ready, then curl /v1/models.
# Run on the head node after both ranks are up:
#   ./docker/gb10/smoke.sh
set -euo pipefail

NAME="${NAME:-dspark-tp2-rank${NODE_RANK:-0}}"
HOST="${SMOKE_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-30001}"
TIMEOUT="${SMOKE_TIMEOUT:-1800}"
INTERVAL="${SMOKE_INTERVAL:-5}"

ready_log() {
  docker logs "${NAME}" 2>&1 | grep -q "Available routes are:"
}

deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
  if ! docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null | grep -q true; then
    echo "container ${NAME} is not running" >&2
    docker logs "${NAME}" 2>&1 | tail -n 80 >&2 || true
    exit 1
  fi
  if ready_log; then
    break
  fi
  sleep "${INTERVAL}"
done

if ! ready_log; then
  echo "timed out waiting for ${NAME} to publish routes (${TIMEOUT}s)" >&2
  docker logs "${NAME}" 2>&1 | tail -n 80 >&2 || true
  exit 1
fi

curl -fsS --max-time 30 "http://${HOST}:${PORT}/v1/models"
echo
