# Dual-node Prometheus + Grafana (DGX Spark / bare metal)

Local, no-Kubernetes stack for a two-node vLLM deployment. It starts
Prometheus and Grafana on the head node, scrapes vLLM `/metrics` plus
`node_exporter` InfiniBand/RoCE counters on both machines, and auto-loads:

- Official **Performance Statistics** and **Query Statistics** dashboards
- A **RoCE / Dual Spark** dashboard for the two ConnectX-7 twins on the
  plugged QSFP (`rocep1s0f1` + `roceP2p1s0f1`)

Prometheus and Grafana use host networking so they can reach vLLM and
`node_exporter` on the host. Scrape traffic is intended to stay on the
management / Wi-Fi NIC, not the 200 GbE DAC.

## Prerequisites

- Docker with **Compose V2** (`docker compose version`) on the vLLM head
  (Spark-1). The old Python `docker-compose` v1 binary is not supported.
- `sudo` on both nodes to install `node_exporter`
- QSFP DAC already up (see `rdma link` / `ibdev2netdev`)

Default `config.env.example` matches this layout:

| Interface | Address | Role |
| --- | --- | --- |
| `enp1s0f1np1` / `rocep1s0f1` | `192.168.100.10` | RoCE rail 0, `VLLM_HOST_IP` |
| `enP2p1s0f1np1` / `roceP2p1s0f1` | `192.168.101.10` | RoCE rail 1 |
| `wlP9s9` | `192.168.8.134` | Grafana + node_exporter listen |

## Step-by-step (Spark-1 / head)

```bash
cd examples/observability/dual_node
chmod +x deploy.sh
./deploy.sh init
```

Edit `config.env` (quote `ROCE_DEVICE_REGEX` — it contains `|`):

1. Set `GRAFANA_BIND` and `NODE_EXPORTER_LISTEN` / `SPARK1_NODE_EXPORTER`
   to this machine's Wi-Fi IP (`ip -br addr`).
2. Set `SPARK2_NODE_EXPORTER=<spark2-wifi>:9100` after Spark-2 has
   `node_exporter`. Leave it empty to bring the stack up on Spark-1 only.
3. Keep `VLLM_METRICS_TARGET=127.0.0.1:30001` when using
   `docker/gb10/run.sh` or `docker/gb10/run-vision.sh` (their default
   `VLLM_PORT`). Override if you set `VLLM_PORT`.

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
Dashboards are in folder **vLLM**. Set refresh to 2s on the RoCE board.

## Spark-2 / worker

Do **not** start Prometheus or Grafana on the worker. Only `node_exporter`:

```bash
./deploy.sh remote-node-exporter   # prints the copy/install commands
```

On Spark-2:

```bash
# after copying this directory
# set NODE_EXPORTER_LISTEN to Spark-2's Wi-Fi IP in config.env
./deploy.sh install-node-exporter
```

Then on Spark-1 put that address in `SPARK2_NODE_EXPORTER` and run
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
NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
  ./docker/gb10/run.sh          # or run-vision.sh
```

Then scrape `http://127.0.0.1:30001/metrics` on the head. Do not point
`VLLM_HOST_IP` at Wi-Fi or `tun0`. If you serve outside those scripts,
keep:

```bash
export VLLM_HOST_IP=192.168.100.10   # .11 on Spark-2
export NCCL_SOCKET_IFNAME=enp1s0f1np1
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
- Bind Grafana / `node_exporter` to Wi-Fi, not `0.0.0.0`, on a box that also
  has `tun0`.
