# Telemetry for the Qwen3.8-Flash-Next RTX PRO 5000 deployment

Prometheus + Grafana stack for `../qwen3_8_flash_next_rtx_pro_5000.sh`:

| Source | Port (127.0.0.1) | What |
|---|---|---|
| vLLM `/metrics` | 8000 | throughput, TTFT/TPOT/ITL/E2E, queue, KV cache, prefix cache, MTP acceptance |
| dcgm-exporter | 9400 | GPU util, tensor/DRAM activity, memory, power, clocks, PCIe |
| node-exporter | 9100 | CPU, memory per NUMA node, RDMA port counters, mlx5 ethtool (PFC pause, `*_bytes_phy`) |
| roce-hw-counters | (textfile) | RoCE `hw_counters`: CNP/ECN, retransmits, sequence errors |
| Prometheus | 9090 | 30-day retention (`PROM_RETENTION`) |
| Grafana | 3000 | dashboard "vLLM · Qwen3.8-Flash-Next · RTX PRO 5000" |

Requires Docker with the NVIDIA Container Toolkit.

```bash
cd examples/deployment/telemetry
GRAFANA_ADMIN_PASSWORD=<password> docker compose up -d
# Grafana listens on 127.0.0.1 by default; GRAFANA_ADDR=0.0.0.0 exposes it on the LAN.
ssh -L 3000:127.0.0.1:3000 <server>   # then open http://localhost:3000/telemetry/
```

Grafana is served under `/telemetry/`. To publish it through the same
Cloudflare tunnel as the API, start it with the public URL and add an ingress
rule before the catch-all:

```bash
GRAFANA_ROOT_URL=https://t.example.com/telemetry/ GRAFANA_COOKIE_SECURE=true \
GRAFANA_ADMIN_PASSWORD=<password> docker compose up -d
```

```yaml
  - hostname: t.example.com
    path: ^/telemetry(/|$)
    service: http://127.0.0.1:3000
```

Put a Cloudflare Access application on that path as well; Grafana's login is
then the second factor, not the only one.

Variables:

- `DCGM_EXPORTER_IMAGE`: dcgm-exporter image; pick a current tag from Docker Hub `nvidia/dcgm-exporter` (or NGC) if
  the default does not support your driver.
- `ROCE_NETDEV_REGEX`: netdevs for the ethtool collector (default
  `^(ens2f|ens16f).*`). The dashboard's `netdev` variable lists the same names.

Notes:

- vLLM's `--api-key` does not guard `/metrics`, so Prometheus scrapes it
  without a key. Keep the Cloudflare tunnel restricted to `/v1/`.
- RDMA traffic bypasses the kernel: use the RDMA/`*_bytes_phy` panels, not
  netdev counters, to see RoCE throughput.
- `DCGM_FI_PROF_*` (tensor/DRAM/PCIe activity) may be empty on GPUs or
  drivers without DCGM profiling support.
