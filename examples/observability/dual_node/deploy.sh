#!/usr/bin/env bash
# Deploy Prometheus + Grafana + node_exporter for a 2-node vLLM / DGX Spark pair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NODE_EXPORTER_VERSION="${NODE_EXPORTER_VERSION:-1.9.1}"
OFFICIAL_DASHBOARDS="${OFFICIAL_DASHBOARDS:-$ROOT/../dashboards/grafana}"

usage() {
  cat <<'EOF'
Usage: ./deploy.sh <command>

Commands (run in this order on the vLLM head / Spark-1):

  1. init                 Create config.env from the example if missing
  2. check                Verify Docker, RoCE twins, and config
  3. install-node-exporter
                          Install node_exporter on THIS machine (needs sudo)
  4. generate             Render prometheus.yml and copy Grafana dashboards
  5. up                   Start Prometheus + Grafana (docker compose)
  6. status               Print scrape targets and UI URLs
  7. down                 Stop Prometheus + Grafana

  all                     init → check → generate → install-node-exporter → up → status
  remote-node-exporter    Print the exact commands to run on Spark-2

Edit config.env (especially SPARK2_NODE_EXPORTER) before `all` or `up`.
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

compose() {
  # Prefer Compose V2 (`docker compose`). The Python V1 `docker-compose`
  # binary breaks against current Docker Engine API sockets.
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    die "Docker Compose V2 is required (docker compose). Install the compose plugin."
  fi
}

# Chat UIs often paste [https://x](https://x). Grafana needs a raw URL.
_strip_md_url() {
  local v="$1"
  case "$v" in
    \[http://*\]\(*\) | \[https://*\]\(*\))
      v="${v#\[}"
      v="${v%%\]*}"
      ;;
    \<http://*\> | \<https://*\>)
      v="${v#\<}"
      v="${v%\>}"
      ;;
  esac
  printf '%s' "$v"
}

load_config() {
  if [[ ! -f "$ROOT/config.env" ]]; then
    die "config.env not found. Run: ./deploy.sh init"
  fi
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/config.env"
  set +a
  : "${GRAFANA_BIND:=127.0.0.1}"
  : "${GRAFANA_PASSWORD:=admin}"
  : "${GRAFANA_DOMAIN:=}"
  : "${GRAFANA_ROOT_URL:=http://${GRAFANA_BIND}:3000/}"
  : "${GRAFANA_SERVE_FROM_SUB_PATH:=false}"
  : "${GRAFANA_COOKIE_SECURE:=true}"
  : "${PROMETHEUS_LISTEN:=127.0.0.1:9090}"
  : "${VLLM_METRICS_TARGET:=127.0.0.1:30001}"
  : "${NODE_EXPORTER_LISTEN:=127.0.0.1:9100}"
  : "${SPARK1_LABEL:=spark-1}"
  : "${SPARK1_NODE_EXPORTER:=127.0.0.1:9100}"
  : "${SPARK2_LABEL:=spark-2}"
  : "${SPARK2_NODE_EXPORTER:=}"
  : "${ROCE_DEVICE_REGEX:=rocep1s0f1|roceP2p1s0f1}"
  GRAFANA_ROOT_URL="$(_strip_md_url "$GRAFANA_ROOT_URL")"
  GRAFANA_DOMAIN="$(_strip_md_url "$GRAFANA_DOMAIN")"
  if [[ "${GRAFANA_ROOT_URL}" != http://* && "${GRAFANA_ROOT_URL}" != https://* ]]; then
    die "GRAFANA_ROOT_URL must be a raw URL, got: ${GRAFANA_ROOT_URL}"
  fi
}

cmd_init() {
  if [[ -f "$ROOT/config.env" ]]; then
    log "config.env already exists"
    return 0
  fi
  cp "$ROOT/config.env.example" "$ROOT/config.env"
  log "Wrote $ROOT/config.env"
  log "Set SPARK2_NODE_EXPORTER to Spark-2's Wi-Fi IP:9100, then continue."
}

cmd_check() {
  load_config
  need_cmd docker
  docker info >/dev/null 2>&1 || die "docker daemon is not running / not accessible"

  log "RoCE / netdev on this host:"
  if command -v rdma >/dev/null 2>&1; then
    rdma link || true
  else
    warn "rdma not installed (optional; apt install rdma-core)"
  fi
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev || true
  fi

  local missing=0
  local dev
  IFS='|' read -r -a _devs <<<"$ROCE_DEVICE_REGEX"
  for dev in "${_devs[@]}"; do
    if [[ -d "/sys/class/infiniband/$dev" ]]; then
      local state
      state="$(cat "/sys/class/infiniband/$dev/ports/1/state" 2>/dev/null || echo unknown)"
      log "  $dev  $state"
    else
      warn "RoCE device $dev not found under /sys/class/infiniband"
      missing=1
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    warn "RoCE devices missing on this machine. Dashboards will be empty until node_exporter sees them."
  fi

  if [[ -z "${SPARK2_NODE_EXPORTER}" ]]; then
    warn "SPARK2_NODE_EXPORTER is empty — Prometheus will scrape only this node."
    warn "On Spark-2 run: ip -br addr   then put <wifi>:9100 in config.env"
  fi

  if curl -fsS --max-time 2 "http://${VLLM_METRICS_TARGET}/metrics" >/dev/null 2>&1; then
    log "vLLM /metrics reachable at ${VLLM_METRICS_TARGET}"
  else
    warn "vLLM /metrics not reachable at ${VLLM_METRICS_TARGET} (start vllm serve first)"
  fi
}

_node_exporter_arch() {
  case "$(uname -m)" in
    aarch64 | arm64) echo arm64 ;;
    x86_64 | amd64) echo amd64 ;;
    *) die "unsupported arch: $(uname -m)" ;;
  esac
}

cmd_install_node_exporter() {
  load_config
  need_cmd curl
  need_cmd sudo

  local arch tarball url tmp
  arch="$(_node_exporter_arch)"
  tarball="node_exporter-${NODE_EXPORTER_VERSION}.linux-${arch}.tar.gz"
  url="https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/${tarball}"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  log "Downloading $url"
  curl -fsSL "$url" -o "$tmp/$tarball"
  tar -xzf "$tmp/$tarball" -C "$tmp"
  sudo install -m 0755 \
    "$tmp/node_exporter-${NODE_EXPORTER_VERSION}.linux-${arch}/node_exporter" \
    /usr/local/bin/node_exporter

  # Run as root: User=nobody cannot read some ConnectX /sys/class/infiniband
  # counters on DGX Spark, so node_infiniband_* never appears.
  sudo tee /etc/systemd/system/node_exporter.service >/dev/null <<UNIT
[Unit]
Description=Prometheus Node Exporter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/node_exporter \\
  --web.listen-address=${NODE_EXPORTER_LISTEN} \\
  --collector.infiniband \\
  --collector.netdev
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable --now node_exporter
  sudo systemctl restart node_exporter
  sleep 2
  _check_node_exporter_metrics
}

_check_node_exporter_metrics() {
  local metrics
  metrics="$(mktemp)"
  # Do not pipe curl to grep -q: grep closes the pipe early, curl exits 23,
  # and with pipefail the script reports a false "no InfiniBand metrics".
  if ! curl -fsS --max-time 5 "http://${NODE_EXPORTER_LISTEN}/metrics" -o "$metrics"; then
    rm -f "$metrics"
    die "node_exporter did not start on ${NODE_EXPORTER_LISTEN}"
  fi
  if grep -q '^node_infiniband_' "$metrics"; then
    log "node_exporter is up at ${NODE_EXPORTER_LISTEN} and exporting InfiniBand metrics"
    grep -E '^node_infiniband_(info|port_data_transmitted_bytes_total|rate_bytes_per_second)' \
      "$metrics" | head -n 8 || true
    rm -f "$metrics"
    return 0
  fi
  warn "node_exporter is up but no node_infiniband_* metrics yet"
  if [[ ! -d /sys/class/infiniband ]]; then
    warn "/sys/class/infiniband is missing (rdma-core / mlx5 not loaded?)"
  else
    log "host IB devices:"
    ls -1 /sys/class/infiniband || true
  fi
  rm -f "$metrics"
}

cmd_generate() {
  load_config
  mkdir -p "$ROOT/generated/dashboards"

  local spark2_block=""
  if [[ -n "${SPARK2_NODE_EXPORTER}" ]]; then
    spark2_block="$(
      cat <<EOF
      - targets: ["${SPARK2_NODE_EXPORTER}"]
        labels:
          spark: ${SPARK2_LABEL}
          role: worker
EOF
    )"
  fi

  cat >"$ROOT/generated/prometheus.yml" <<EOF
global:
  scrape_interval: 2s
  evaluation_interval: 15s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["${VLLM_METRICS_TARGET}"]
        labels:
          spark: ${SPARK1_LABEL}
          role: head

  - job_name: node
    static_configs:
      - targets: ["${SPARK1_NODE_EXPORTER}"]
        labels:
          spark: ${SPARK1_LABEL}
          role: head
${spark2_block}
EOF

  if [[ -d "$OFFICIAL_DASHBOARDS" ]]; then
    cp -f "$OFFICIAL_DASHBOARDS/performance_statistics.json" \
      "$ROOT/generated/dashboards/performance_statistics.json"
    cp -f "$OFFICIAL_DASHBOARDS/query_statistics.json" \
      "$ROOT/generated/dashboards/query_statistics.json"
  else
    warn "Official dashboards not found at $OFFICIAL_DASHBOARDS"
  fi
  cp -f "$ROOT/grafana/roce.json" "$ROOT/generated/dashboards/roce.json"
  log "Wrote generated/prometheus.yml and generated/dashboards/"
}

cmd_up() {
  load_config
  cmd_generate
  export GRAFANA_BIND GRAFANA_PASSWORD GRAFANA_DOMAIN GRAFANA_ROOT_URL
  export GRAFANA_SERVE_FROM_SUB_PATH GRAFANA_COOKIE_SECURE PROMETHEUS_LISTEN
  log "Grafana root_url=${GRAFANA_ROOT_URL} sub_path=${GRAFANA_SERVE_FROM_SUB_PATH}"
  compose up -d --force-recreate grafana
  log "Waiting for Grafana..."
  local i
  for i in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://${GRAFANA_BIND}:3000/dash/api/health" >/dev/null 2>&1 \
      || curl -fsS --max-time 2 "http://${GRAFANA_BIND}:3000/api/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  cmd_status
}

cmd_down() {
  compose down
}

cmd_status() {
  load_config
  local prom_host="${PROMETHEUS_LISTEN}"
  log "Grafana:     http://${GRAFANA_BIND}:3000   (admin / ${GRAFANA_PASSWORD})"
  if [[ -n "${GRAFANA_ROOT_URL}" ]]; then
    log "Public URL:  ${GRAFANA_ROOT_URL}"
  fi
  log "Prometheus:  http://${prom_host}"
  log "Dashboards:  folder 'vLLM' → Performance Statistics, Query Statistics, RoCE / Dual Spark"
  if curl -fsS --max-time 2 "http://${prom_host}/api/v1/targets" >/dev/null 2>&1; then
    if command -v jq >/dev/null 2>&1; then
      curl -fsS --max-time 2 "http://${prom_host}/api/v1/targets" | jq -r '
        .data.activeTargets[]?
        | "  \(.scrapeUrl)  health=\(.health)  spark=\(.labels.spark // "") job=\(.labels.job // "")"
      '
    else
      log "Prometheus targets API is up (install jq to print each target)"
    fi
  else
    warn "Prometheus API not reachable at http://${prom_host}"
  fi
}

cmd_remote() {
  load_config
  cat <<EOF
On Spark-2 (worker), install only node_exporter:

  1. Copy this directory (or just deploy.sh + config.env):
       scp -r $ROOT <user>@<spark2>:vllm-obs/

  2. On Spark-2, set NODE_EXPORTER_LISTEN to THAT machine's Wi-Fi IP:
       NODE_EXPORTER_LISTEN=<spark2-wifi>:9100

  3. Run:
       cd vllm-obs && ./deploy.sh install-node-exporter

  4. Back on Spark-1, put the same address in config.env:
       SPARK2_NODE_EXPORTER=<spark2-wifi>:9100
     then:
       ./deploy.sh generate && ./deploy.sh up

Do not start Prometheus/Grafana on Spark-2.
Prefer docker/gb10/run.sh or run-vision.sh (metrics on head :30001).
If you set NCCL yourself:
  export VLLM_HOST_IP=192.168.100.x          # QSFP .100 address
  export NCCL_SOCKET_IFNAME=enp1s0f1np1
  export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
EOF
}

cmd_all() {
  cmd_init
  load_config
  if [[ -z "${SPARK2_NODE_EXPORTER}" ]]; then
    warn "SPARK2_NODE_EXPORTER is empty. Continuing with Spark-1 only."
  fi
  cmd_check
  cmd_generate
  cmd_install_node_exporter
  cmd_up
}

cmd="${1:-}"
case "$cmd" in
  init) cmd_init ;;
  check) cmd_check ;;
  install-node-exporter) cmd_install_node_exporter ;;
  generate) cmd_generate ;;
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  remote-node-exporter) cmd_remote ;;
  all) cmd_all ;;
  -h | --help | help | "") usage ;;
  *)
    usage
    die "unknown command: $cmd"
    ;;
esac
