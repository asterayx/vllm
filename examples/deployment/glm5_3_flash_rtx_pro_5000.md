# GLM-5.3-Flash on 4x RTX PRO 5000 — 部署运行手册

本文记录 `glm5.3-flash-rtxpro5000` 分支上实际在用的命令：环境安装、vLLM 启动、
对外暴露、客户端配置、基准测试与排障。机器、Cloudflare Tunnel 和监控与 Qwen
部署共用，见 [qwen3_8_flash_next_rtx_pro_5000.md](qwen3_8_flash_next_rtx_pro_5000.md)。

- 机器：4x NVIDIA RTX PRO 5000 72GB Blackwell (sm_120)，双路 CPU，GPU 之间只有 PCIe
- 模型：`nvidia/GLM-5.3-Flash-NVFP4`（`Glm5NextForConditionalGeneration`，约 191 GB）
  - 45 层，hidden 4096，288 专家 top-8，`moe_intermediate_size` 2048（TP4 下每卡 512）
  - 注意力：KDA 线性注意力 + 稀疏 MLA（Kpool indexer，DeepGEMM）
  - MTP 草稿层为 BF16（未量化）
- 对外入口：`https://t.asteraix.com/v1`，模型名 `glm-5.3-flash`
- Qwen 与 GLM 都占满 4 张卡，同一时间只能运行一个

```bash
export G=$HOME/models/nvidia/GLM-5.3-Flash-NVFP4
```

## 1. 安装

```bash
git clone -b glm5.3-flash-rtxpro5000 https://github.com/asterayx/vllm.git ~/vllm
cd ~/vllm
uv venv --python 3.12 && source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . \
  "b12x==1.5.0" "nvidia-cutlass-dsl[cu13]==4.7.1" "quack-kernels==0.6.5" \
  --torch-backend=auto
uv pip install ninja "flashinfer-cubin==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/

.venv/bin/python -c "from vllm.utils.flashinfer import has_flashinfer; print(has_flashinfer())"  # True
.venv/bin/ninja --version
```

- FlashInfer 会在首次启动时 JIT 编译 `fp4_gemm_cutlass_sm120`、`fused_moe_120`、
  `sparse_mla_sm120`（需要 `ninja` 和 `nvcc`，各需数分钟），结果缓存在 `~/.cache/flashinfer/`。
- 启动脚本会自动把 `.venv/bin`（ninja）和 `/usr/local/cuda/bin`（nvcc）加入 `PATH`。
- 不要对 vLLM 环境里的包使用 `uv pip install -U`。

## 2. 启动 vLLM

生产启动命令（API key 与 Qwen 共用 `~/.asteraux_vllm_key`）：

```bash
cd ~/vllm
VLLM_EMIT_REASONING_CONTENT=1 VLLM_API_KEY=$(cat ~/.asteraux_vllm_key) \
MAX_LEN=262144 MAX_SEQS=32 NUMA=1 NCCL_LL=1 SPEC=3 TOOLS=1 MODEL=$G \
  ./examples/deployment/glm5_3_flash_rtx_pro_5000.sh --moe-backend b12x \
  --host 127.0.0.1 --served-model-name glm-5.3-flash 2>&1 | tee glm53.log
```

| 设置 | 作用 |
|---|---|
| `TP=4`（默认） | 4 卡张量并行 |
| `--moe-backend b12x` | sm_120 NVFP4 MoE kernel；MTP 草稿层（BF16）自动用 FlashInfer CUTLASS |
| `SPEC=3` | MTP 投机解码 3 个草稿 token；mt-bench 接受长度 2.85 |
| `--kv-cache-dtype fp8`（脚本内置） | `fp8_ds_mla` 格式，后端 `FLASHINFER_MLA_SPARSE_SM120` |
| `--reasoning-parser glm47`（脚本内置） | 思考内容放入 `reasoning` 字段 |
| `TOOLS=1` | `--enable-auto-tool-choice --tool-call-parser glm47` |
| `NUMA=1` | `--numa-bind`，GPU0/1 → NUMA 0，GPU2/3 → NUMA 1 |
| `NCCL_LL=1` | `NCCL_P2P_LEVEL=SYS` + `nccl_ll_tuner.c`（同 Qwen） |
| `MAX_LEN=262144` / `MAX_SEQS=32` | 256K 上下文；模型原生支持 1M |
| `VLLM_EMIT_REASONING_CONTENT=1` | 额外输出 `reasoning_content`，供 Grok Build 显示 CoT |

显存（每卡）：权重 45.5 GiB，加 MTP 后 49.2 GiB。KV cache 不开 MTP 约 160 万 token，
开 MTP 约 97 万 token。

启动耗时：权重在页缓存中时约 2–3 分钟；首次启动另需 FlashInfer JIT 编译与 autotune。

停止：`pkill -f "vllm.entrypoints.cli.main serve"`。若 worker 卡死不退出，用
`pkill -9 -f "VLLM::|VllmWorker|vllm.entrypoints.cli.main serve"`，再用 `nvidia-smi`
确认显存已释放。

## 3. 对外暴露与监控

与 Qwen 完全相同：cloudflared 仍转发到 `127.0.0.1:8000`，Prometheus 仍抓取
`:8000/metrics`，无需修改。见 Qwen 手册第 3、6 节。

## 4. 客户端配置

在 Qwen 手册第 4 节的配置基础上，把模型名换成 `glm-5.3-flash`：

```toml
# ~/.codex/config.toml
model = "glm-5.3-flash"
```

```toml
# ~/.grok/config.toml
[model.asteraix-glm]
model = "glm-5.3-flash"
base_url = "https://t.asteraix.com/v1"
name = "GLM-5.3 Flash (asteraix)"
env_key = "ASTERAIX_API_KEY"
api_backend = "chat_completions"
context_window = 262144
max_completion_tokens = 32768
reasoning_effort = "medium"
```

pi：在 `~/.pi/agent/models.json` 的 `models` 中加一项，`"id": "glm-5.3-flash"`，
其余字段与 Qwen 相同。

## 5. 验证

```bash
K=$ASTERAIX_API_KEY; B=https://t.asteraix.com/v1
curl -s $B/models -H "Authorization: Bearer $K" | jq -r '.data[].id'   # glm-5.3-flash

# 思考与回答应分别出现在 reasoning 与 content
curl -s $B/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"37*43 等于多少？"}],"max_tokens":1024}' \
  | jq '.choices[0].message | {reasoning: .reasoning[0:200], content}'

# 工具调用：应返回结构化 tool_calls
curl -s $B/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{
 "model":"glm-5.3-flash","messages":[{"role":"user","content":"北京现在几点？"}],
 "tools":[{"type":"function","function":{"name":"get_time","description":"Get current time in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
}' | jq '.choices[0].message.tool_calls'
```

## 6. 基准测试

```bash
export OPENAI_API_KEY=$(cat ~/.asteraux_vllm_key)   # 启用了 VLLM_API_KEY 时

# 真实对话（mt-bench，默认采样 temperature 1.0 / top_p 0.95）
for c in 1 8 16; do
  .venv/bin/vllm bench serve --model glm-5.3-flash --tokenizer $G \
    --dataset-name hf --dataset-path philschmid/mt-bench \
    --num-prompts $((c == 1 ? 80 : c*8)) --max-concurrency $c 2>&1 \
    | grep -E "Output token throughput|Mean TPOT|Acceptance"
done

# 随机数据（1024 输入 / 512 输出）
for c in 1 8 16; do
  .venv/bin/vllm bench serve --model glm-5.3-flash --tokenizer $G \
    --dataset-name random --random-input-len 1024 --random-output-len 512 \
    --num-prompts $((c*8)) --max-concurrency $c 2>&1 \
    | grep -E "Output token throughput|Mean TPOT|Mean TTFT"
done
```

参考结果（TP4，输出 tok/s，并发 1 / 8 / 16）：

| 配置 | random 1024/512 | mt-bench |
|---|---|---|
| FlashInfer CUTLASS，无 MTP | 90 / 397 / 583 | — |
| b12x + `NUMA=1 NCCL_LL=1` + MTP2 | 100 / 431 / 595 | 152 / 567 / 755 |
| b12x + `NUMA=1 NCCL_LL=1` + MTP3（推荐） | — | 165 / 562 / 808 |

MTP 接受长度（mt-bench，并发 1）：MTP2 为 2.41（greedy 2.51），MTP3 为 2.85。
random 数据的接受长度只有 1.3–1.8，不能用来评估投机解码。

## 7. 已知问题与排障

| 现象 | 原因 / 处理 |
|---|---|
| `No valid attention backend found` | `has_flashinfer()` 为 False：装 `flashinfer-cubin`，或保证 `nvcc` 在 `PATH`（脚本已自动处理 `/usr/local/cuda/bin`） |
| `FileNotFoundError: 'ninja'` | FlashInfer JIT 需要 ninja：`uv pip install ninja` |
| 启动卡在 `Running FlashInfer autotune`，日志只剩 `No available shared memory broadcast block` | 旧版本中各卡 autotune 缓存不一致导致 all-reduce 死锁，本分支已修复。若仍出现：`rm -rf ~/.cache/vllm/flashinfer_autotune_cache` 后重启，或临时加 `--no-enable-flashinfer-autotune` |
| `Free memory on device ... is less than desired GPU memory utilization` | 上一个 vLLM 进程未退出，按第 2 节用 `pkill -9` 清理 |
| 日志中 `Triton kernel JIT compilation during inference: _topp_sb_* / _kpool_tail_seed_kernel` | 首次遇到该 shape 时现编，只影响第一次请求 |
