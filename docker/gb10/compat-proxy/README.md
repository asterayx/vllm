# spark-compat-proxy

Qwen3.8 Spark serving guide:
[`README.qwen38_nvfp4_spark_tp2.md`](../../../examples/online_serving/README.qwen38_nvfp4_spark_tp2.md).

Production OpenAI-compat sidecar for 2× DGX Spark. Listens on **:30000**,
forwards to vLLM **:30001** by default, and copies vLLM `reasoning` onto
`reasoning_content` (JSON + SSE). Connection-pooled, keep-alive, 600s
upstream timeout, graceful SIGTERM.

This branch restores the existing proxy from commit `0648ef21c`, including its
UTF-8 stream-boundary fix. For Qwen3.8, run
`bash examples/online_serving/qwen38_nvfp4_spark_proxy.sh` from the repository
root after building. It uses loopback port 30000, upstream port 18029, model
`qwen38-nvfp4`, and a 16384-token client context limit. `--context-window` controls
the advertised Grok/OpenCode limit; it must match the serving configuration.

Also serves downloadable client configs:

| URL | Client | Install as |
| --- | --- | --- |
| `/configs/codex.toml` | Codex CLI | `~/.codex/config.toml` |
| `/configs/grok.toml` | Grok Build | `~/.grok/config.toml` |
| `/configs/opencode.json` | OpenCode | `~/.config/opencode/opencode.json` |
| `/configs` | HTML index | browser |
| `/healthz` | liveness | `curl` |

## Build (on the Spark or any aarch64/x86 host)

```bash
cd docker/gb10/compat-proxy
# rust-toolchain.toml pins 1.88.0 (some rustup 1.95 images are missing rust-std).
./install.sh
# writes /usr/local/bin/spark-compat-proxy (sudo if needed)
# or:
cargo build --release
sudo install -m755 target/release/spark-compat-proxy /usr/local/bin/
```

## Run

```bash
# public-base is what Codex/Grok/OpenCode put in base_url (Wi-Fi IP if remote)
spark-compat-proxy \
  --listen 0.0.0.0:30000 \
  --upstream http://127.0.0.1:30001 \
  --public-base http://127.0.0.1:30000
```

Write configs to disk without serving:

```bash
spark-compat-proxy write-configs --out /tmp/spark-client-configs \
  --public-base http://192.168.8.134:30000
```

systemd (`User=` must be this host's login, not a leftover `roccen`):

```bash
INSTALL_SYSTEMD=1 ./install.sh
# or: PUBLIC_BASE=http://<wifi>:30000 INSTALL_SYSTEMD=1 ./install.sh
sudo systemctl reset-failed spark-compat-proxy
sudo systemctl enable --now spark-compat-proxy
# 203/EXEC = /usr/local/bin/spark-compat-proxy missing or not +x
ls -l /usr/local/bin/spark-compat-proxy
```

## Download configs from the running proxy

```bash
export VLLM_API_KEY=dummy
curl -fsS http://127.0.0.1:30000/healthz
curl -fsS http://127.0.0.1:30000/configs/codex.toml -o ~/.codex/config.toml
curl -fsS http://127.0.0.1:30000/configs/grok.toml -o ~/.grok/config.toml
mkdir -p ~/.config/opencode
curl -fsS http://127.0.0.1:30000/configs/opencode.json \
  -o ~/.config/opencode/opencode.json
```

Or open `http://<head-wifi>:30000/configs` in a browser.

The Python script `../codex_proxy.py` is the original prototype. Use this
binary for production.
