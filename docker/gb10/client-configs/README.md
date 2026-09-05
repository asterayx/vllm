# Client configs (Codex / Grok Build / OpenCode)

These files talk to **spark-compat-proxy** on `:30000` (not vLLM `:30001`).
The proxy copies v0.28 `reasoning` onto `reasoning_content`.

Prefer the live copies from a running proxy — they bake in `--public-base`:

```bash
# on the Spark head, or any machine that can reach it
export VLLM_API_KEY=dummy
curl -fsS http://127.0.0.1:30000/configs          # HTML index
curl -fsS http://127.0.0.1:30000/configs/codex.toml -o ~/.codex/config.toml
curl -fsS http://127.0.0.1:30000/configs/grok.toml >> ~/.grok/config.toml
mkdir -p ~/.config/opencode
curl -fsS http://127.0.0.1:30000/configs/opencode.json \
  -o ~/.config/opencode/opencode.json
```

Or generate locally without serving:

```bash
spark-compat-proxy write-configs --out /tmp/spark-client-configs \
  --public-base http://192.168.8.134:30000
```

The checked-in examples below use `http://127.0.0.1:30000`. Edit `base_url` /
`baseURL` if Codex/Grok/OpenCode run on another machine.
