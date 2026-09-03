# spark-compat-proxy

Production OpenAI-compat sidecar for 2× DGX Spark. Listens on **:30000**,
forwards to vLLM **:30001**, and copies v0.28 `reasoning` onto
`reasoning_content` (JSON + SSE). Connection-pooled, keep-alive, 600s
upstream timeout, graceful SIGTERM.

Also serves downloadable client configs:

| URL | Client | Install as |
| --- | --- | --- |
| `/configs/codex.toml` | Codex CLI | `~/.codex/config.toml` |
| `/configs/grok.toml` | Grok Build | `~/.grok/config.toml` |
| `/configs/opencode.json` | OpenCode | `~/.config/opencode/opencode.json` |
| `/configs` | HTML index | browser |
| `/healthz` | liveness | `curl` |

Checked-in copies (localhost `base_url`) live in
[`../client-configs/`](../client-configs/).

## Build (on the Spark or any aarch64/x86 host)

```bash
cd docker/gb10/compat-proxy
# rust-toolchain.toml pins 1.88.0 (some rustup 1.95 images are missing rust-std).
./install.sh
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

systemd (edit `User=` and `--public-base`):

```bash
sudo cp spark-compat-proxy.service /etc/systemd/system/
sudo systemctl enable --now spark-compat-proxy
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
