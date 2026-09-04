# DGX Spark — official v0.28 image and serve stack

Operator guide for **2× NVIDIA DGX Spark (GB10 / SM12x)** serving
**DeepSeek-V4-Flash** (text 0731 or Vision-Exp) with stock vLLM + DSpark.

This tree is **`cursor/spark-v0280-dsv4-df88`**, rebased on official tag
**`v0.28.0`**. Daily work (observability, compat proxy) sits on
`cursor/spark-obs-codex-df88`, which is based on that Spark line — **not**
on fork `main`. Do not merge this line with `main`.

| | Official (this tree) | Older fork `main` |
| --- | --- | --- |
| Git | `v0.28.0` + Spark patches | `v0.26.1rc0` + early Spark PRs |
| Default image | `vllm-gb10:v0.28.0-dsv4-spark` (+ `.N` release) | `vllm-gb10:dspark` (tag name only) |
| Banner | `0.28.0+dsv4.spark.N` (`docker/gb10/VERSION`) | leftover `_version.py` can lie |
| Serve | `run-*-image.sh` (baked `/opt/vllm`) | often bind-mounts `~/src/vllm` |

`vllm-gb10:dspark` is **not** the official v0.28 image unless you rebuilt
that tag from this tree. Set `VLLM_GB10_IMAGE` explicitly if you reuse it.

## Contents

1. [Spark version](#spark-version)
2. [Layout and ports](#layout-and-ports)
3. [Checkout](#checkout)
4. [Build the official image](#build-the-official-image)
5. [Copy the image to the worker](#copy-the-image-to-the-worker)
6. [Weights](#weights)
7. [Serve vLLM](#serve-vllm)
8. [Compat proxy + Codex / Grok / OpenCode](#compat-proxy--codex--grok--opencode)
9. [Prometheus + Grafana](#prometheus--grafana)
10. [Cloudflare Tunnel (create and /dash)](#cloudflare-tunnel-create-and-dash)
11. [Verify](#verify)
12. [Scripts](#scripts)

## Spark version

This line keeps upstream `v0.28.0` and adds **our** release on top.
Single source of truth: [`VERSION`](VERSION).

| | Current |
| --- | --- |
| Package / serve banner | `0.28.0+dsv4.spark.2` |
| Family tag (moving latest) | `vllm-gb10:v0.28.0-dsv4-spark` |
| Release tag (pin this publish) | `vllm-gb10:v0.28.0-dsv4-spark.2` |

`build.sh` / `pack-venv.sh` apply **both** tags. `vllm --version`,
`importlib.metadata.version("vllm")`, `vllm._version.__version__`, and
the image label `org.opencontainers.image.version` all show the Spark
version. Docker tags cannot contain `+`, so the image uses `.2` while
Python uses `+dsv4.spark.2`.

```bash
./docker/gb10/version.sh
# after a build:
docker image inspect vllm-gb10:v0.28.0-dsv4-spark \
  --format '{{index .Config.Labels "org.opencontainers.image.version"}}'
# pin a published cut:
VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark.2 \
  NODE_RANK=0 ./docker/gb10/run-vision-image.sh
```

Next official image: edit `VERSION` to `0.28.0+dsv4.spark.3` and rebuild.
Do not reuse `.2` for a different tree.

## Layout and ports

Default pair used by the scripts and
`examples/observability/dual_node/config.env.example`:

| Interface | Address | Role |
| --- | --- | --- |
| `enp1s0f1np1` / `rocep1s0f1` | `192.168.100.10` / `.11` | RoCE rail 0, `VLLM_HOST_IP` |
| `enP2p1s0f1np1` / `roceP2p1s0f1` | `192.168.101.10` / `.11` | RoCE rail 1 |
| Wi-Fi (example `wlP9s9`) | `192.168.8.134` | Grafana / `node_exporter` / LAN clients |

`VLLM_HOST_IP` and NCCL stay on the **QSFP DAC**. Do not point them at
Wi-Fi or `tun0`.

| Port | Process | Bind |
| --- | --- | --- |
| **30001** | vLLM OpenAI + `/metrics` (head, rank 0) | `0.0.0.0` |
| **30000** | `spark-compat-proxy` (`reasoning` → `reasoning_content`) | `0.0.0.0` |
| **3000** | Grafana | `127.0.0.1` when cloudflared is local |
| **9091** | Prometheus (9090 is often taken) | `127.0.0.1` |
| **9100** | `node_exporter` | Wi-Fi IP, both nodes |
| **29500** | vLLM `--master-port` | RoCE |

Clients (Codex / Grok / OpenCode) talk to **:30000**, not :30001.

## Checkout

On **both** Sparks:

```bash
cd ~/src/vllm   # or wherever this repo lives
git fetch origin cursor/spark-obs-codex-df88 cursor/spark-v0280-dsv4-df88
git checkout cursor/spark-obs-codex-df88
git pull --ff-only origin cursor/spark-obs-codex-df88
```

Expect tip on this branch to sit on `285a71df4`
(`cursor/spark-v0280-dsv4-df88`) plus the observability / proxy commits.

## Build the official image

Build **on a Spark** (`aarch64` + GPU). Do not cross-build from x86.

Default family tag: `vllm-gb10:v0.28.0-dsv4-spark`
(override with `VLLM_GB10_IMAGE`). The same build also tags
`vllm-gb10:v0.28.0-dsv4-spark.N` from [`VERSION`](VERSION).

Pins inside `docker/Dockerfile.gb10`: CUDA 13.0 devel Ubuntu 24.04,
PyTorch cu130, FlashInfer **0.6.18**, official **b12x==1.2.6**,
**humming-kernels[cu13]==0.1.12**, `TORCH_CUDA_ARCH_LIST=12.1a`.
On SM121 the leftover W8A8 block-FP8 list is Humming before Triton
(DeepGEMM / Cutlass / b12x block-FP8 are off).

### From-scratch (recommended for a publishable image)

Compiles vLLM in Docker. Rebuilds reuse the BuildKit uv cache.

```bash
# on the head Spark, this tree
./docker/gb10/build.sh
# tags vllm-gb10:v0.28.0-dsv4-spark and vllm-gb10:v0.28.0-dsv4-spark.2
#   VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark MAX_JOBS=8 ./docker/gb10/build.sh
```

Do not pass `--no-cache` unless a cached `RUN` layer is known-bad.

### Pack a host venv (faster iterate, same tag)

Only if the venv was compiled from **this** v0.28 tree
(`vllm._C_stable_libtorch` present). Do **not** pack an old later-main
`~/.venvs/vllm028`.

```bash
source ~/.cargo/env
./docker/gb10/build-venv.sh
VENV=~/.venvs/vllm028 INSTALL_B12X=1 INSTALL_HUMMING=1 ./docker/gb10/pack-venv.sh
```

`pack-venv.sh` refuses a venv that cannot import `_C_stable_libtorch`.

To inject humming into an **already packed** official image without
recompiling (wheel + leftover kernel order + `VERSION` stamp):

```bash
./docker/gb10/pack-humming.sh
./docker/gb10/sync-image.sh roccen@192.168.100.11
```

After restart, worker logs should show
`Selected HummingFP8ScaledMMLinearKernel for FP8 block-scaled linear`
and **no** `NVIDIA_GB10` W8A8 Block FP8 config warnings. `pack-humming.sh`
must overlay `kernels/linear/__init__.py`; installing the wheel alone
leaves baked `/opt/vllm` on the old Triton-first order.

### Confirm the tag

```bash
docker image inspect vllm-gb10:v0.28.0-dsv4-spark \
  --format '{{.Id}} {{.Created}} {{index .Config.Labels "org.opencontainers.image.version"}}'
docker image ls 'vllm-gb10:v0.28.0-dsv4-spark*'
```

## Copy the image to the worker

Layers are already gzip-compressed. Do not add `ssh -C`.

```bash
# on the head, over ConnectX
./docker/gb10/sync-image.sh roccen@192.168.100.11
# or a different tag:
#   ./docker/gb10/sync-image.sh roccen@192.168.100.11 vllm-gb10:v0.28.0-dsv4-spark
```

Trusted DAC, two terminals (no SSH crypto):

```bash
# worker
nc -l 18765 | docker load
# head
docker save vllm-gb10:v0.28.0-dsv4-spark | nc -N 192.168.100.11 18765
```

Weights are **not** in the image. Rsync Hugging Face cache if needed:

```bash
rsync -aHAX --info=progress2 ~/.cache/huggingface/ \
  roccen@192.168.100.11:.cache/huggingface/
```

## Weights

On **both** nodes (offline serve: `HF_HUB_OFFLINE=1`):

```bash
# Vision-Exp (default for run-vision-*.sh)
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/DeepSeek-V4-Flash-Vision-Exp

# Text 0731 (run.sh / run-image.sh)
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir ~/models/DeepSeek-V4-Flash-0731
```

Override with `MODEL_HOST` / `MODEL_PATH` if the directory differs.

## Serve vLLM

Stop leftover containers first (same default port **30001**):

```bash
docker rm -f dspark-tp2-rank0 dspark-tp2-rank1 \
  dspark-vision-tp2-rank0 dspark-vision-tp2-rank1
```

**Worker first, then head.** Same image tag on both nodes.

### Official image (baked `/opt/vllm` — use this to publish)

Host `git checkout` does **not** change what the container runs.

```bash
# worker
VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark \
  NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
  ./docker/gb10/run-vision-image.sh

# head
VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark \
  NODE_RANK=0 ./docker/gb10/run-vision-image.sh
```

Text 0731: `run-image.sh` instead of `run-vision-image.sh`.

### Dev (bind-mount this checkout over `/opt/vllm`)

```bash
NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
  ./docker/gb10/run-vision.sh
NODE_RANK=0 ./docker/gb10/run-vision.sh
```

### Defaults these scripts already set

- `network=host`, `--gpus all`, IB devices mounted
- `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1`
- `--kv-cache-dtype fp8_ds_mla`, `--max-model-len 524288`
- Vision: DSpark **k=3**, capture sizes `[1,2,4,8,12,16,24]`
- Text 0731: DSpark **k=5**, capture sizes include **36**
- `/metrics` only on the head (`NODE_RANK=0`)

If the vision tower OOMs: `GPU_MEMORY_UTILIZATION=0.82 MAX_NUM_SEQS=4`.

### Wait until ready

```bash
# Vision container name is dspark-vision-tp2-rank0
NAME=dspark-vision-tp2-rank0 ./docker/gb10/smoke.sh
# or text:
#   ./docker/gb10/smoke.sh
```

`smoke.sh` waits for `Available routes are:` then curls `/v1/models`.
First boot can take many minutes (graphs + FlashInfer autotune).

```bash
docker logs -f dspark-vision-tp2-rank0
curl -fsS http://127.0.0.1:30001/metrics | head
curl -fsS http://127.0.0.1:30001/v1/models
```

AutoTuner `tactic=-1` lines are untuned-bucket fallbacks, not a failed boot.

## Compat proxy + Codex / Grok / OpenCode

v0.28 emits `reasoning`. Codex / Grok still read `reasoning_content`.
`spark-compat-proxy` aliases the field on JSON and SSE.

```bash
cd docker/gb10/compat-proxy
./install.sh          # rust-toolchain.toml pins rustc 1.88.0
sudo cp spark-compat-proxy.service /etc/systemd/system/
# edit User= and --public-base (Wi-Fi IP if clients are not on the head)
sudo systemctl enable --now spark-compat-proxy
```

`--public-base` is what the generated configs put in `base_url` / `baseURL`.

```bash
export VLLM_API_KEY=dummy
curl -fsS http://127.0.0.1:30000/healthz
# browser index
curl -fsS http://127.0.0.1:30000/configs
# downloads (Content-Disposition: attachment)
curl -fsS http://127.0.0.1:30000/configs/codex.toml -o ~/.codex/config.toml
curl -fsS http://127.0.0.1:30000/configs/grok.toml >> ~/.grok/config.toml
mkdir -p ~/.config/opencode
curl -fsS http://127.0.0.1:30000/configs/opencode.json \
  -o ~/.config/opencode/opencode.json
```

Offline copies (localhost `base_url`): [`client-configs/`](client-configs/).

Write files without serving:

```bash
spark-compat-proxy write-configs --out /tmp/spark-client-configs \
  --public-base http://192.168.8.134:30000
```

| Client | File | Notes |
| --- | --- | --- |
| Codex CLI | `~/.codex/config.toml` | `wire_api = "responses"` only (chat removed) |
| Grok Build | `~/.grok/config.toml` | merge `[model."dsv4-spark"]`; `grok --model dsv4-spark` |
| OpenCode | `~/.config/opencode/opencode.json` | provider id `spark` (no dots) |

Python `codex_proxy.py` is the prototype. Use the Rust binary.

## Prometheus + Grafana

Local compose stack on the **head** only. Scrapes head `:30001/metrics`
and `node_exporter` InfiniBand/RoCE on both machines. Host networking.
Scrape traffic stays on Wi-Fi, not the 200 GbE DAC.

Needs Docker **Compose V2** (`docker compose version`).

```bash
cd examples/observability/dual_node
./deploy.sh init
```

Edit `config.env` (quote `ROCE_DEVICE_REGEX` — it contains `|`):

1. `GRAFANA_BIND=127.0.0.1` when cloudflared is on this host.
2. `NODE_EXPORTER_LISTEN` / `SPARK1_NODE_EXPORTER` = this Wi-Fi IP `:9100`.
3. `VLLM_METRICS_TARGET=127.0.0.1:30001` (override if you set `VLLM_PORT`).
4. Leave `SPARK2_NODE_EXPORTER` empty until the worker exporter is up.

```bash
./deploy.sh check
./deploy.sh install-node-exporter
./deploy.sh generate
./deploy.sh up
./deploy.sh status
# or, after config.env is filled: ./deploy.sh all
```

Worker: **do not** start Prometheus/Grafana.

```bash
# on head, prints the copy/install commands
./deploy.sh remote-node-exporter
```

On Spark-2: set `NODE_EXPORTER_LISTEN=<spark2-wifi>:9100` in that node's
`config.env`, then `./deploy.sh install-node-exporter`. Back on the head,
set `SPARK2_NODE_EXPORTER` to the same address and
`./deploy.sh generate && ./deploy.sh up`.

`node_exporter` must run as **root** on Spark (`User=nobody` cannot read
ConnectX sysfs; `node_infiniband_*` stays empty). Re-run
`install-node-exporter` after changing the unit.

Open `http://<GRAFANA_BIND>:3000` (default `admin` / `admin`). Folder
**vLLM**:

- **Spark vLLM — Serving Performance** (decode/prefill, DFlash2 accept, KV)
- Official Performance / Query Statistics
- **RoCE / Dual Spark** (`rocep1s0f1` + `roceP2p1s0f1`)

Set RoCE refresh to 2s. Official Performance defaults to
`granite-33-2b-instruct` — pick **All** or your served model.

LAN-only (no tunnel): you may bind Grafana to the Wi-Fi IP instead of
loopback. Do not bind `0.0.0.0` on a box that also has `tun0`.

## Cloudflare Tunnel (create and /dash)

Publish Grafana at `https://<host>/dash` only. Do **not** add `/telemetry`.
Do **not** put vLLM `:30001` or the compat proxy on the public internet
unless you intend to; this section is Grafana-only.

Hostname in the checked-in examples is `token.asterayx.com`. Replace it
everywhere (DNS, `config.yml`, `GRAFANA_DOMAIN`, `GRAFANA_ROOT_URL`) if
yours differs. A typo (`asteryax.com`) will 404.

### 1. Install `cloudflared` (head, aarch64)

```bash
curl -fsSL -o /tmp/cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

### 2. Login and create the tunnel (first time)

```bash
# opens a browser / prints a URL; authorize the zone (asterayx.com)
cloudflared tunnel login

# named tunnel; writes ~/.cloudflared/<TUNNEL_UUID>.json
cloudflared tunnel create spark-obs
cloudflared tunnel list
```

Copy the UUID from `tunnel list`. That is `TUNNEL_ID` below.

Point DNS at the tunnel (Cloudflare creates the CNAME):

```bash
cloudflared tunnel route dns spark-obs token.asterayx.com
```

Confirm in the Cloudflare dashboard: `token.asterayx.com` → CNAME →
`<TUNNEL_ID>.cfargotunnel.com`, proxied.

If the tunnel **already exists**, skip `create` / `route dns` and only
edit ingress.

### 3. `~/.cloudflared/config.yml`

`/dash` **must** be listed **above** any catch-all `/`. Origin is
`http://127.0.0.1:3000` — **not** `http://127.0.0.1:3000/dash`
(that would become `/dash/dash`).

Full template:
[`examples/observability/dual_node/cloudflared/config.yml.example`](../../examples/observability/dual_node/cloudflared/config.yml.example).
Ingress snippet only:
[`dash-ingress.yml`](../../examples/observability/dual_node/cloudflared/dash-ingress.yml).

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/roccen/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: token.asterayx.com
    path: /dash(/.*)?
    service: http://127.0.0.1:3000
    originRequest:
      httpHostHeader: token.asterayx.com
      disableChunkedEncoding: true

  # optional: whatever you already publish on `/` (not Grafana)
  # - hostname: token.asterayx.com
  #   service: http://127.0.0.1:REPLACE_ROOT_PORT

  - service: http_status:404
```

Delete any leftover `/telemetry` rule.

### 4. systemd

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared --no-pager
# after editing config.yml:
sudo systemctl restart cloudflared
```

Foreground debug: `cloudflared tunnel run spark-obs`.

### 5. Tell Grafana it lives under `/dash`

In `examples/observability/dual_node/config.env`:

```bash
GRAFANA_BIND=127.0.0.1
GRAFANA_DOMAIN=token.asterayx.com
GRAFANA_ROOT_URL=https://token.asterayx.com/dash/
GRAFANA_SERVE_FROM_SUB_PATH=true
```

Raw URL only — do not paste markdown (`[https://x](https://x)`).

```bash
cd examples/observability/dual_node
./deploy.sh generate
./deploy.sh up
```

Login: `https://token.asterayx.com/dash` with `admin` / `admin`.
**Change that password**; the URL is on the public internet.

## Verify

```bash
# image + API
docker image ls vllm-gb10:v0.28.0-dsv4-spark
NAME=dspark-vision-tp2-rank0 ./docker/gb10/smoke.sh
curl -fsS http://127.0.0.1:30001/v1/models

# proxy
curl -fsS http://127.0.0.1:30000/healthz
curl -fsSI http://127.0.0.1:30000/configs/codex.toml | grep -i content-disposition

# metrics + Grafana
curl -fsS http://127.0.0.1:30001/metrics | grep -E '^vllm_' | head
curl -fsS http://127.0.0.1:3000/api/health
curl -fsS http://127.0.0.1:3000/dash/api/health   # after sub-path is on

# tunnel
cloudflared tunnel info spark-obs
curl -fsSI https://token.asterayx.com/dash | head
```

## Scripts

| Script | What it does |
| --- | --- |
| `VERSION` / `version.sh` | Spark release `0.28.0+dsv4.spark.N` + image tags |
| `build.sh` | From-scratch official image on Spark |
| `build-venv.sh` / `install-vllm.sh` | Host compile into a uv venv |
| `pack-venv.sh` | Pack that venv into the same image tag (no compile) |
| `pack-humming.sh` | Inject `humming-kernels` into an existing image |
| `sync-image.sh` | `docker save \| ssh docker load` over ConnectX |
| `run-vision-image.sh` / `run-image.sh` | Official serve (no source mount) |
| `run-vision.sh` / `run.sh` | Dev serve (bind-mount host tree) |
| `smoke.sh` | Wait for routes, `GET /v1/models` |
| `compat-proxy/` | Production `reasoning` alias + config downloads |
| `../observability/dual_node/deploy.sh` | Prometheus, Grafana, `node_exporter` |

More detail: [`compat-proxy/README.md`](compat-proxy/README.md),
[`client-configs/README.md`](client-configs/README.md),
[`examples/observability/dual_node/README.md`](../../examples/observability/dual_node/README.md).
