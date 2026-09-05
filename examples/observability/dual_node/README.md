# Dual-node Prometheus + Grafana (DGX Spark / bare metal)

Full operator path (official image, serve, proxy, Grafana, Cloudflare
tunnel from scratch): [`docker/gb10/README.md`](../../../docker/gb10/README.md).

Ported onto `cursor/spark-v0280-dsv4-df88`. Serve defaults still match
`docker/gb10/run.sh` / `run-vision.sh` (`VLLM_METRICS_TARGET=127.0.0.1:30001`).

Local, no-Kubernetes stack for a two-node vLLM deployment. It starts
Prometheus and Grafana on the head node, scrapes vLLM `/metrics` plus
`node_exporter` InfiniBand/RoCE counters on both machines, and auto-loads:

- Official **Performance Statistics** and **Query Statistics** dashboards
- **Spark vLLM — Serving Performance** — decode/prefill throughput (dual
  Y-axis), TTFT/ITL/e2e, DFlash2 acceptance, KV, token sizes (model
  defaults to All)
- A **RoCE / Dual Spark** dashboard for the two ConnectX-7 twins on the
  plugged QSFP (`rocep1s0f1` + `roceP2p1s0f1`)

Prometheus and Grafana use host networking so they can reach vLLM and
`node_exporter` on the host. They bind **127.0.0.1**. cloudflared
publishes Grafana. The head scrapes the worker `node_exporter` over
**RoCE** (`192.168.101.13:9100`), not Wi-Fi.

## Prerequisites

- Docker with **Compose V2** (`docker compose version`) on the vLLM head
  (Spark-1). The old Python `docker-compose` v1 binary is not supported.
- `sudo` on both nodes to install `node_exporter`
- QSFP DAC already up (see `rdma link` / `ibdev2netdev`)

Default `config.env.example` matches this layout:

| Interface | Address | Role |
| --- | --- | --- |
| `enP2p1s0f1np1` / `roceP2p1s0f1` | `192.168.101.12` / `.13` | RoCE, `VLLM_HOST_IP`, worker `:9100` |
| loopback | `127.0.0.1` | Grafana `:3000`, Prometheus `:9091`, head `:9100` |
| Wi-Fi | roaming | SSH / LAN clients only |

## Step-by-step (Spark-1 / head)

```bash
cd examples/observability/dual_node
chmod +x deploy.sh
./deploy.sh init
```

Edit `config.env` (quote `ROCE_DEVICE_REGEX` — it contains `|`).
The example is already loopback + RoCE scrape:

1. Keep `GRAFANA_BIND=127.0.0.1` and
   `NODE_EXPORTER_LISTEN` / `SPARK1_NODE_EXPORTER=127.0.0.1:9100`.
2. Set `SPARK2_NODE_EXPORTER=192.168.101.13:9100` after Spark-2 has
   `node_exporter` on that RoCE address.
3. Keep `VLLM_METRICS_TARGET=127.0.0.1:30001` when using
   `docker/gb10/run.sh` or `docker/gb10/run-vision.sh`.

```bash
./deploy.sh check
./deploy.sh install-node-exporter
./deploy.sh generate
./deploy.sh up
./deploy.sh status
```

Or, after `config.env` is filled:

```bash
./deploy.sh all
```

Open `http://<GRAFANA_BIND>:3000` (default user/password `admin` / `admin`).
Dashboards are in folder **vLLM**. Use **Spark vLLM — Serving Performance**
for decode/prefill, DFlash2 accept rate, KV, and tokens. Throughput puts
prefill on the left axis and decode on the right. Set refresh to 2s on
the RoCE board. The official Performance Statistics board defaults to
`granite-33-2b-instruct` — pick **All** or your served model, or use the
Serving board instead (All by default).

## Cloudflare Tunnel (`/dash`)

Creating a **new** tunnel (`login` / `tunnel create` / `route dns` /
systemd) is in
[`docker/gb10/README.md`](../../../docker/gb10/README.md#cloudflare-tunnel-create-and-dash).
This section is the Grafana `/dash` ingress only.

On the head, publish Grafana at `https://token.asteraix.com/dash`.
This is the only observability path; do not keep a `/telemetry`
ingress. `sudo cloudflared service install` reads `/etc/cloudflared`,
not `$HOME/.cloudflared` — use
[`cloudflared/install-service.sh`](cloudflared/install-service.sh).

1. Add these lines to `config.env` (hostname must match DNS):

   ```bash
   GRAFANA_DOMAIN=token.asteraix.com
   GRAFANA_ROOT_URL=https://token.asteraix.com/dash/
   GRAFANA_SERVE_FROM_SUB_PATH=true
   ```

2. Recreate Grafana so it serves under `/dash`:

   ```bash
   ./deploy.sh up
   ```

3. Install the systemd unit from `/etc/cloudflared/config.yml`
   (`./cloudflared/install-service.sh`). Delete any leftover
   `/telemetry` rule.

4. Restart the tunnel (`sudo systemctl restart cloudflared`).

Cloudflare **URL** must be `http://127.0.0.1:3000` with Path `dash`.
Do not set the origin to `http://127.0.0.1:3000/dash` — the public
path is already forwarded, and you would get `/dash/dash`.

Login at `https://token.asteraix.com/dash` with `admin` / `admin`.
Change that password; this URL is on the public internet.

## Spark-2 / worker

Do **not** start Prometheus or Grafana on the worker. Only `node_exporter`:

```bash
./deploy.sh remote-node-exporter   # prints the copy/install commands
```

On Spark-2:

```bash
# after copying this directory
# NODE_EXPORTER_LISTEN=192.168.101.13:9100
./deploy.sh install-node-exporter
```

Then on Spark-1 keep `SPARK2_NODE_EXPORTER=192.168.101.13:9100` and run
`./deploy.sh generate && ./deploy.sh up`.

## vLLM / NCCL (both nodes)

This example is meant to sit on top of the post-#6 Spark serve scripts.
Those already set `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, host networking,
and `/metrics` on the API process (rank 0 only):

```bash
# Text 0731
NODE_RANK=0 ./docker/gb10/run.sh
# or Vision-Exp (PR #6)
NODE_RANK=0 ./docker/gb10/run-vision.sh

# Worker
NODE_RANK=1 VLLM_HOST_IP=192.168.101.13 HEADLESS=--headless \
  ./docker/gb10/run.sh          # or run-vision.sh
```

Then scrape `http://127.0.0.1:30001/metrics` on the head. Do not point
`VLLM_HOST_IP` at Wi-Fi or `tun0`. If you serve outside those scripts,
keep:

```bash
export VLLM_HOST_IP=192.168.101.12   # .13 on Spark-2
export NCCL_SOCKET_IFNAME=enP2p1s0f1np1
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_IB_DISABLE=0
```

## Commands

| Command | What it does |
| --- | --- |
| `./deploy.sh init` | Create `config.env` from the example |
| `./deploy.sh check` | Docker, RoCE sysfs, vLLM `/metrics` |
| `./deploy.sh install-node-exporter` | Install and start systemd `node_exporter` |
| `./deploy.sh generate` | Render `generated/prometheus.yml` and copy dashboards |
| `./deploy.sh up` | `docker compose up -d` |
| `./deploy.sh status` | URLs and scrape target health |
| `./deploy.sh down` | Stop the compose stack |
| `./deploy.sh remote-node-exporter` | Instructions for Spark-2 |

## Notes

- `ibstat` is optional (`sudo apt install infiniband-diags`). `rdma link` is enough.
- `f0` twins (`rocep1s0f0`, `roceP2p1s0f0`) are the unused QSFP; leave them out.
- RoCE graphs are bursty under tensor-parallel allreduce; look at peaks and
  twin balance, not the time average.
- Bind Grafana / Prometheus / head `node_exporter` to `127.0.0.1`.
  Bind the worker `node_exporter` to the RoCE IP. Do not use Wi-Fi.
- `node_exporter` must run as root on Spark. `User=nobody` cannot read some
  ConnectX sysfs counters, so `node_infiniband_*` stays empty. Re-run
  `./deploy.sh install-node-exporter` or `sudo systemctl restart node_exporter`
  after updating the unit.
