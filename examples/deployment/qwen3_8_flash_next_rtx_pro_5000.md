# Qwen3.8-Flash-Next on 4x RTX PRO 5000 — 部署运行手册

本文记录 `qwen3.8-next-flash-rtxpro5000` 分支上实际在用的全部命令：环境安装、
vLLM 启动、Cloudflare Tunnel 对外暴露、客户端（Codex / Grok Build / pi）配置、
Prometheus + Grafana 监控，以及基准测试与排障。

- 机器：4x NVIDIA RTX PRO 5000 72GB Blackwell (sm_120)，双路 CPU，GPU 之间只有 PCIe
    - GPU0/1 + CX7 (`rocep37s0f*`, `ens2f*`) 在 NUMA 0
    - GPU2/3 + CX7 (`rocep241s0f*`, `ens16f*`) 在 NUMA 1
- 模型：`nvidia/Qwen3.8-Flash-Next-NVFP4`（512 专家，NVFP4；MTP 层为 FP8）
- 对外入口：`https://t.asteraix.com/v1`（OpenAI 兼容 API），`https://t.asteraix.com/telemetry/`（Grafana）

下文中 `M` 为模型目录，例如：

```bash
export M=$HOME/models/nvidia/Qwen3.8-Flash-Next-NVFP4
```

## 1. 安装

```bash
git clone -b qwen3.8-next-flash-rtxpro5000 https://github.com/asterayx/vllm.git ~/vllm
cd ~/vllm
uv venv --python 3.12 && source .venv/bin/activate

# vLLM 与 b12x 必须在同一次解析中安装，否则 b12x 的宽松依赖会把 torch 升级掉
VLLM_USE_PRECOMPILED=1 uv pip install -e . \
  "b12x==1.5.0" "nvidia-cutlass-dsl[cu13]==4.7.1" "quack-kernels==0.6.5" \
  --torch-backend=auto

# flashinfer-cubin is only on the FlashInfer index, so `-e .` skips it. Without
# it (or nvcc on PATH) vLLM disables FlashInfer backends.
uv pip install "flashinfer-cubin==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/
.venv/bin/python -c "from vllm.utils.flashinfer import has_flashinfer; print(has_flashinfer())"  # True

uv pip show torch torchvision b12x nvidia-cutlass-dsl | grep -E "^(Name|Version)"
# 期望: torch 2.13.0, torchvision 0.28.0, b12x 1.5.0, nvidia-cutlass-dsl 4.7.1
```

不要对 vLLM 环境里的包使用 `uv pip install -U`。

### v0.34.0 集成版（分支 `rtxpro5000-v0.34.0`）

上面的分支名和版本号是 v0.31 时期的。v0.34.0 集成版在 Python 3.12、
PyTorch 2.14.0+cu130、FlashInfer 0.7.3、CUDA toolkit 13.5 下验证过；其余依赖的
版本以该分支的 `examples/deployment/rtxpro5000_v034_merge.md` 为准。

启动前确认两件事，否则 FlashInfer PCIe IPC 的自动调优会把全部候选跳过并存下一份
空缓存，表现为启动要 40 分钟以上、日志里反复出现
`Skipped 151 unsupported tactic(s)` 和 `Saved 0 configs`：

1. **CUDA 根目录要显式给出。** `nvcc` 是转发脚本的机器上，自动识别会得到错误的
   根目录，调优计时用的辅助模块编译时找不到 `cuda_runtime_api.h`。

    ```bash
    export CUDA_HOME=/usr/local/cuda-13.5          # 含 include/ 和 bin/nvcc 的目录
    export FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
    ```

2. **调优器要读到 venv 自带的 cuBLAS。** 它按不带版本号的 `libcublas.so` /
   `libcublasLt.so` 查找，venv 里只有带版本号的文件，于是会读到系统那份的版本号，
   与缓存里记录的不一致，已有缓存被判为不匹配。在 venv 之外建一个只放软链接的
   目录：

    ```bash
    CUDA_LIB=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
    mkdir -p ~/vllm-libshim
    for n in libcublas libcublasLt libcudart; do
      ln -sf "$CUDA_LIB/$n.so.13" ~/vllm-libshim/$n.so
    done
    export LD_LIBRARY_PATH=$HOME/vllm-libshim:$CUDA_LIB
    export LIBRARY_PATH=$HOME/vllm-libshim:$CUDA_LIB
    ```

调优结果按配置哈希分目录存放（开不开草稿是两个目录），同一配置第二次启动直接
复用。正常的冷启动约 6–9 分钟，其中调优只占几秒。

### 可选开关

下面的开关都登记在 `vllm/envs.py`，设为 `1` 才生效，并且会进入编译缓存的键。
"已测"指 xzss 四卡 C1 测试（PR #28 正文的结果）打开了它；
`qwen3_8_flash_next_rtx_pro_5000.sh` 默认不设这些，PCIe IPC 要用 `PCIE_IPC=1`。

| 开关 | 状态 | 作用 |
| --- | --- | --- |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` | 已测 | 小批量 all-reduce 走 FlashInfer PCIe IPC，不走 NCCL |
| `VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS` | 已测（16） | 超过这个 token 数的 all-reduce 交给下一个后端 |
| `QWEN4_EXP_TRITON_GEMM` | 默认开，已测 | SM120 上稠密投影用 Triton 解码 GEMM；`VLLM_BATCH_INVARIANT=1` 时不用 |
| `QWEN4_EXP_TRITON_MOE` | 已测 | 小 token 数的 NVFP4 专家用 Triton 实现，替代 b12x |
| `QWEN4_EXP_TRITON_MOE_MAX_M` | 已测（96） | 上一项生效的最大 token 数 |
| `QWEN4_EXP_TRITON_MOE_DIRECT_MAX_M` | 已测（4） | 不超过这个 token 数时，每个（token，专家）对独占一个块，省掉按专家分组的那一趟；0 为始终分组 |
| `QWEN4_EXP_TRITON_MOE_DOWNBF16` | 实验 | 强制开或关 bf16 下投影变体 |
| `DPSK_NVTX` | 调试 | 主机侧 NVTX 区间，配合 nsys 使用 |

`DPSK_*` 的其余开关属于 DeepSeek-V4.1 和 PP 路径，见
`rtxpro5000_perf_probe.md` 和 `vllm/envs.py` 里的说明。

可选：跑 sm_120 相关单测（需要 `uv pip install -r requirements/test/cuda.in`）：

```bash
.venv/bin/python -m pytest -q tests/models/qwen4_exp/test_hc_ops.py \
  tests/models/qwen4_exp/test_qsa_reference.py
.venv/bin/python -m pytest -q tests/kernels/moe/test_b12x.py -k small_batch
```

## 2. 启动 vLLM

生成 API key（只做一次）：

```bash
openssl rand -hex 32 > ~/.asteraux_vllm_key && chmod 600 ~/.asteraux_vllm_key
```

推荐的生产启动命令：

```bash
cd ~/vllm
VLLM_EMIT_REASONING_CONTENT=1 VLLM_API_KEY=$(cat ~/.asteraux_vllm_key) \
MAX_LEN=262144 MAX_SEQS=32 NUMA=1 NCCL_LL=1 MODEL=$M TP=4 SPEC=3 \
  ./examples/deployment/qwen3_8_flash_next_rtx_pro_5000.sh \
  --moe-backend b12x --kv-cache-dtype fp8 --host 127.0.0.1 \
  --served-model-name qwen3.8-flash-next \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

| 设置 | 作用 |
| --- | --- |
| `TP=4` | 4 卡张量并行（每卡 MoE intermediate 160，b12x 补齐到 192） |
| `--moe-backend b12x` | sm_120 专用 NVFP4 MoE kernel，比默认 CUTLASS 快 12–15% |
| `SPEC=3` | MTP 投机解码 3 个草稿 token（取舍见下文"草稿 token 数的取舍"）；MTP 层的 FP8 MoE 自动选后端（`SPEC_MOE=auto`） |
| `--kv-cache-dtype fp8` | FP8 KV cache，约 455 万 token |
| `NUMA=1` | `--numa-bind`：worker 与其 PLE 表分片绑定到 GPU 所在 NUMA 节点 |
| `NCCL_LL=1` | `NCCL_P2P_LEVEL=SYS` + `nccl_ll_tuner.c`：≤320 KB 的 TP all-reduce 用 LL 协议（~31 µs → 12–17 µs） |
| `MAX_LEN=262144` / `MAX_SEQS=32` | 256K 上下文，最多 32 个并发序列 |
| `--host 127.0.0.1` | 只本机监听，外部只能经 Cloudflare Tunnel |
| `VLLM_API_KEY` | `/v1/*` 需要 Bearer key（注意：`/metrics`、`/tokenize` 等不受保护） |
| `--served-model-name` | 客户端使用的模型名 |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | 函数调用（agent 必需） |
| `VLLM_EMIT_REASONING_CONTENT=1` | Chat Completions 额外输出 `reasoning_content`，Grok Build 等客户端才能显示 CoT |
| PLE (n-gram) 表 | 默认 `cpu_offload=True`，约 51 GB 放在锁页主机内存，经 UVA 读取 |

可选开关（默认不开）：

- `PCIE_IPC=1`：FlashInfer PCIe IPC all-reduce。单路约 +8%，高并发无收益，占用约 1.7 GiB KV cache。
  `VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS=N` 限制其接管的 token 数。
- `--override-generation-config '{"max_new_tokens": 32768}'`：客户端未设 `max_tokens` 时的默认上限。
- `CHECK=1`：启动前跑 sm_120 单测；`TEXT_ONLY=1`：不加载视觉塔。

停止：在启动终端按 `Ctrl-C`，或 `pkill -f "vllm.entrypoints.cli.main serve"`。

## 3. Cloudflare Tunnel

```bash
cloudflared tunnel login                       # 选择 asteraix.com
cloudflared tunnel create asteraix-llm
cloudflared tunnel route dns asteraix-llm t.asteraix.com
```

`systemd` 服务读取的是 `/etc/cloudflared/config.yml`（`cloudflared service install`
会复制一份过去），以后只改这一份：

```yaml
tunnel: asteraix-llm
credentials-file: /home/<user>/.cloudflared/<UUID>.json
# QUIC uploads large agent requests (~0.5 MB per turn) at ~11 KB/s here.
protocol: http2

ingress:
  - hostname: t.asteraix.com
    path: ^/v1/
    service: http://127.0.0.1:8000
    originRequest:
      connectTimeout: 30s
      keepAliveTimeout: 300s
  - hostname: t.asteraix.com
    path: ^/telemetry(/|$)
    service: http://127.0.0.1:3000
  - hostname: t.asteraix.com
    service: http_status:404
  - service: http_status:404
```

```bash
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress rule https://t.asteraix.com/telemetry/
sudo systemctl restart cloudflared
```

- 本机的透明代理（sing-box / Clash TUN）会接管 cloudflared 的出站连接，导致隧道频繁断线
  （`connection with edge closed`，客户端 502）与上传缓慢。让 cloudflared 直连，在 sing-box
  `route.rules` 最前面加：

  ```json
  { "process_name": ["cloudflared"], "outbound": "direct" },
  { "ip_cidr": ["198.41.192.0/24", "198.41.200.0/24"], "outbound": "direct" },
  { "domain_suffix": ["argotunnel.com"], "outbound": "direct" },
  ```

  验证：`sudo mtr -rwc 20 -i 0.2 -T -P 7844 198.41.192.47` 延迟应为真实网络延迟（而非 0.1 ms）。
- 只放行 `/v1/` 与 `/telemetry`，其余（`/metrics`、`/scale_elastic_ep` 等）返回 404。
- Cloudflare 后台：WAF / Bot Fight Mode 对 `t.asteraix.com` 加 Skip 规则，避免拦截 API 客户端。
- 建议在 Zero Trust 中仅为 `t.asteraix.com/telemetry` 建 Access 应用；不要给整个域名加，否则 `/v1` 会被拦。
- 非流式请求 100 秒无输出会被 Cloudflare 返回 524；客户端应使用流式。

## 4. 客户端配置

各客户端都从环境变量读取 key：

```bash
echo 'export ASTERAIX_API_KEY="<key>"' >> ~/.zshrc && source ~/.zshrc
```

### Codex (`~/.codex/config.toml`)

```toml
model = "qwen3.8-flash-next"
model_provider = "asteraix"
model_context_window = 262144
model_reasoning_effort = "medium"
show_raw_agent_reasoning = true

[model_providers.asteraix]
name = "Asteraix vLLM"
base_url = "https://t.asteraix.com/v1"
env_key = "ASTERAIX_API_KEY"
wire_api = "responses"

[profiles.fast]
model_reasoning_effort = "none"
```

### Grok Build (`~/.grok/config.toml`)

```toml
[model.asteraix-qwen]
model = "qwen3.8-flash-next"
base_url = "https://t.asteraix.com/v1"
name = "Qwen3.8 Flash Next (asteraix)"
env_key = "ASTERAIX_API_KEY"
api_backend = "chat_completions"
context_window = 262144
max_completion_tokens = 32768
reasoning_effort = "medium"

[models]
default = "asteraix-qwen"
```

### pi (`~/.pi/agent/models.json`)

```json
{
  "providers": {
    "asteraix": {
      "baseUrl": "https://t.asteraix.com/v1",
      "api": "openai-completions",
      "apiKey": "ASTERAIX_API_KEY",
      "models": [
        {
          "id": "qwen3.8-flash-next",
          "name": "Qwen3.8 Flash Next (asteraix)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

思考强度：`reasoning_effort` / `reasoning.effort` 为 `none` 时关闭思考，其余值开启；
`low/medium/high` 是否有区别取决于模型对话模板是否使用 `reasoning_effort` 变量。

## 5. 验证

```bash
K=$ASTERAIX_API_KEY; B=https://t.asteraix.com/v1
curl -s $B/models -H "Authorization: Bearer $K" | jq -r '.data[].id'           # qwen3.8-flash-next
curl -s -o /dev/null -w "%{http_code}\n" $B/models                                # 401
curl -s -o /dev/null -w "%{http_code}\n" https://t.asteraix.com/metrics            # 404

# 工具调用：应返回结构化 tool_calls
curl -s $B/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{
 "model":"qwen3.8-flash-next","messages":[{"role":"user","content":"北京现在几点？"}],
 "tools":[{"type":"function","function":{"name":"get_time","description":"Get current time in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
}' | jq '.choices[0].message.tool_calls'

# Responses API 流式：应看到 reasoning_text.delta 与 output_text.delta
curl -sN $B/responses -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","input":"23*47 等于多少？","stream":true}' \
  | grep -o '"type":"response\.[a-z_.]*"' | sort | uniq -c
```

## 6. 监控（Prometheus + Grafana）

前置：Docker（含 compose 插件）与 NVIDIA Container Toolkit。

```bash
cd ~/vllm/examples/deployment/telemetry
GRAFANA_ROOT_URL=https://t.asteraix.com/telemetry/ GRAFANA_COOKIE_SECURE=true \
GRAFANA_ADMIN_PASSWORD='<strong password>' docker compose up -d
docker compose ps
curl -s 127.0.0.1:19090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job)\t\(.health)\t\(.lastError)"'
```

- 端口（均监听 127.0.0.1）：Prometheus 19090（`PROM_PORT`）、Grafana 3000、node-exporter 9100、dcgm-exporter 9400。
- `GRAFANA_ADMIN_PASSWORD` 只在首次初始化生效；之后改密码：
  `docker compose exec grafana grafana cli admin reset-admin-password '<new password>'`
- 更新仪表板：`git pull && docker compose restart grafana`。
- 面板说明见 [telemetry/README.md](telemetry/README.md)。

## 7. 基准测试

```bash
# 随机数据（1024 输入 / 512 输出）
for c in 1 8 16; do
  echo "== concurrency $c"
  .venv/bin/python -m vllm.entrypoints.cli.main bench serve --model qwen3.8-flash-next \
    --tokenizer $M --dataset-name random --random-input-len 1024 --random-output-len 512 \
    --ignore-eos --num-prompts $((c*8)) --max-concurrency $c 2>&1 \
    | grep -E "Output token throughput|Median TPOT|Median TTFT"
done

# 真实对话（mt-bench）
.venv/bin/python -m vllm.entrypoints.cli.main bench serve --model qwen3.8-flash-next \
  --tokenizer $M --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1 --num-warmups 2
```

（启用 `VLLM_API_KEY` 后，本机压测需 `export OPENAI_API_KEY=$(cat ~/.asteraux_vllm_key)`。）

参考结果（TP4，random 1024/512，输出 tok/s，并发 1 / 8 / 16）：

| 配置 | 吞吐 |
| --- | --- |
| CUTLASS + MTP3 | 143 / 580 / 889 |
| b12x + MTP2 | 145 / 721 / 1030 |
| b12x + MTP2 + `NUMA=1 NCCL_LL=1`（当前） | ~193 / 770 / 1114 |
| 同上 + `PCIE_IPC=1` | 209 / 748 / 1100 |
| 同上，`SPEC=3` | 189 / 731 / 1081 |

mt-bench 并发 1：186 tok/s，MTP 接受率 54.6%，接受长度 2.09。

草稿 token 数的取舍（v0.34.0 集成版，TP4，真实文本，每用户 tok/s = 1000 / TPOT 中位数；
2 测了两次）：

| `SPEC` | 并发 1，10K→4000 | 并发 1，3K→1000 | 并发 8，3K→1000 总吞吐 | 接受长度 | 每步耗时 |
| --- | --- | --- | --- | --- | --- |
| 1 | 241 | 240 | 906 | 1.77 | 7.3 ms |
| 2 | 273–275 | 272–274 | 971–979 | 2.31 | 8.4 ms |
| 3 | 291 | 283 | 956 | 2.69 | 9.3 ms |
| 4 | 250 | 250 | 902 | 2.91 | 11.6 ms |

每多起草一个 token 约多花 0.9–1.1 ms，各位置在前一位被接受的条件下接受率都在
70–75%。真实文本上 `SPEC=3` 在并发 1 时比 2 高 3–6%，并发 8 时低约 2%；上表
random 输入的旧结果里 3 在各并发下都更低，因为随机输入的接受率低。

同一批 50 条请求逐条配对（并发 1），`SPEC=3` 的 TPOT 比 `SPEC=2` 低 3.1–4.4%，
10K→4000 上的 95% 区间为 2.2–7.8% 和 1.9–6.2%（对两次 `SPEC=2` 各一）；两次
`SPEC=2` 之间只差 0.4–0.7%。并发 8 的那约 2% 没有做配对分析。

默认用 3，取单用户延迟；以并发总吞吐为先时用 2。

All-reduce 微基准（`~/ar.py`，TP4，µs）：

| tokens | 默认 | `P2P_LEVEL=SYS` | SYS + LL 插件 |
| --- | --- | --- | --- |
| 1 | 30.9 | 21.9 | 12.7 |
| 24 | — | 25.8 | 16.6 |
| 8192 | 2149 | 2112 | 2104 |

## 8. 已知问题与排障

| 现象 | 原因 / 处理 |
| --- | --- |
| `FLASHINFER_* requires FlashInfer's ... API` 但 FlashInfer 可以 import | 未装 `flashinfer-cubin` 且 `nvcc` 不在 PATH；按第 1 节安装 cubin |
| `torchvision::nms does not exist` | `uv pip install -U` 升级了 torch；按第 1 节重新安装 |
| b12x 在 TP4 下 NaN / illegal memory access | b12x < 1.5.0 不支持 intermediate 192；升级到 1.5.0 |
| Grok Build 只显示 "Waiting for response" | 需要 `VLLM_EMIT_REASONING_CONTENT=1` |
| Codex 不显示思考 | `show_raw_agent_reasoning = true` |
| Codex 提示 `Missing environment variable` | 启动 Codex 的 shell 中未 `export ASTERAIX_API_KEY` |
| Grok 间歇性 502；cloudflared 日志 `connection with edge closed` 与 8000 端口 `context canceled` 同时出现 | cloudflared 经本机 sing-box TUN 出站，代理链路断开会中断其上全部请求。按第 3 节让 cloudflared 直连（176 KB 上传 16.6 s → 2.0 s） |
| 客户端长时间 "Waiting for response"，服务端 `Running: 0` | cloudflared 默认 QUIC 上传大请求极慢（176 KB 需 16.6 s，边缘节点直传 1 s）；日志有 `Body length 0` / `context canceled`。配置 `protocol: http2` 后重启 cloudflared |
| `https://t.asteraix.com/telemetry` 404 | 规则写在了 `~/.cloudflared/config.yml`；服务读 `/etc/cloudflared/config.yml` |
| 容器挂载文件变成 root 拥有的空目录 | 挂载源不存在时 Docker 会自建目录；删除后 `git pull`（仓库 `.gitignore` 忽略 `*.csv`，counters 文件需 `git add -f`） |
| Prometheus `bind: address already in use` | 9090 被占用；已改为 19090（`PROM_PORT`） |
| `--data-parallel-size 2` 并未得到独立副本 | MoE 模型的 DP 会把 MoE 跨全部 GPU 切分（AllGather+ReduceScatter）；要独立副本需起两个进程 |
| NCCL 走 CX7 RDMA 报 `ibv_modify_qp ... Invalid argument` | CPU1 侧网卡无 IPv4（GID 不一致）且对端为本机地址；单机不建议走网卡 |
| `ens2f0np0` 与 `ens2f1np1` 同 IP 192.168.100.20 | 多机 RDMA 前需修正 |
