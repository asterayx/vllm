# vLLM v0.34.0 合并版：C1 性能与 profiling

两种配置各完成240个正式响应（4×50单轮＋40多轮），全部成功；另各有32个预热请求。无草稿四个单轮测点约156–157 tok/s，较历史基线约快7%；MTP约273–277 tok/s，中位数较历史基线快3.9%–7.3%，但按原有p10–p90区间标准均记为“无可测差异”。每种配置均保留64步Nsight trace及30秒Python栈采样。

测试源码：`asterayx/vllm` 分支 `integration/rtxpro5000-v0.34.0`，提交 `d77cff6e2823637d2eab077caa53d78cb6dfce7c`。本地源码在 `/workspace/vllm-integration-v0.34.0`；GPU 测试在 xzss 的 0/1/4/5 四张 RTX PRO 5000 72GB 上执行。2/3/6/7 上既有服务保留。

独立运行环境：Python 3.12.15、Torch 2.14.0+cu130、FlashInfer 0.7.3、NCCL 2.30.7+cuda13.3，编译工具链 CUDA 13.5。包版本字符串仍含 editable 安装时的 `g644379a54`，本次实际源码以以上 Git 提交为准。官方 v0.34.0 预编译扩展与该源码使用同一套 C/CUDA 实现。

V2 表示草稿关闭，V1 表示 MTP 2-token；这两个测试标签不是 Model Runner 版本。新分支使用 MRV2。TP4、FP8 KV、CPU pinned FP8 PLE、NCCL SYS、ardelva-lltune v5、PCIe IPC、小批量 Triton GEMM/MoE 开关沿用原方法。新版自动 attention block 为 1600 tokens；旧版无草稿为 1568。

模型为`/home/roccen/models/nvidia/Qwen3.8-Flash-Next-NVFP4`；最大上下文262144、max_num_seqs=64、gpu_memory_utilization=.95。C1指并发请求数为1。服务只绑定测试端口18775，完整启动参数及环境覆盖见各配置`command.json`。

**测量方法**：未修改的 `ardelva_bench.py` 与原 benchmark 客户端环境，`--conc 1 --min-requests 50 --req-per-conc 10 --warmup 8 --seed 1`。顺序固定为 3K→1000、10K→1000、32K→1000、10K→4000、多轮 60K/+5K/5 轮。模型默认采样 temperature=1/top_p=.95/top_k=20，单轮 ignore_eos=true。前四组每组 50 个测量请求；多轮 8 会话×5轮，每轮500 token。

每用户 tok/s = 1000 / TPOT p50；MTP 平均接受长度含 bonus，使用 `1 + Δspec_decode_num_accepted_tokens_total / Δspec_decode_num_drafts_total`；估算每步 = TPOT p50 × 接受长度。它不是直接测量的 GPU 单步耗时。

**无草稿：C1**

| 负载 | 请求数 | 每用户 tok/s | aggregate out tok/s | TPOT p50/p99 ms | TTFT p50/p99 ms | 接受长度 | 估算每步 ms | 重复/乱码候选 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realtext3k-1000 | 50 | 156.92 | 151.86 | 6.373/6.376 | 235.32/237.2 | 1.0000 | 6.3728 | 0 |
| realtext10k-1000 | 50 | 156.72 | 148.0 | 6.381/6.386 | 434.5/438.85 | 1.0000 | 6.3807 | 0 |
| realtext32k-1000 | 50 | 156.15 | 133.54 | 6.404/6.408 | 1285.94/1296.61 | 1.0000 | 6.4040 | 0 |
| realtext10k-4000 | 50 | 156.67 | 156.07 | 6.383/6.392 | 100.92/119.03 | 1.0000 | 6.3828 | 12 |
| multiturn | 40 | 155.76 | 111.29 | 6.42/6.438 | 592.09/4680.96 | 1.0000 | 6.4200 | 1 |

**MTP 2-token：C1**

| 负载 | 请求数 | 每用户 tok/s | aggregate out tok/s | TPOT p50/p99 ms | TTFT p50/p99 ms | 接受长度 | 估算每步 ms | 重复/乱码候选 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realtext3k-1000 | 50 | 277.19 | 258.56 | 3.608/3.988 | 238.16/851.78 | 2.3106 | 8.3358 | 0 |
| realtext10k-1000 | 50 | 273.18 | 245.15 | 3.661/4.091 | 436.01/443.97 | 2.2687 | 8.3046 | 0 |
| realtext32k-1000 | 50 | 274.02 | 204.39 | 3.649/4.004 | 1485.62/1495.99 | 2.3090 | 8.4264 | 0 |
| realtext10k-4000 | 50 | 276.83 | 274.3 | 3.612/3.869 | 97.93/117.84 | 2.3090 | 8.3411 | 10 |
| multiturn | 40 | 212.27 | 136.89 | 4.711/4.973 | 581.15/4880.81 | 1.8210 | 8.5788 | 0 |

筛查规则沿用旧轮次：替换字符、混合字母表和连续重复块。这里只计异常候选，不能替代正确性或质量评估。固定 ignore_eos 的长输出可能包含回答完成后的延续。原始文本、逐请求耗时、计数器窗口保留在各配置的 `bench` 目录。

本轮未重跑模型正确性套件；请求成功仅代表服务和输出采集成功。

**与旧轮次的对照**

基线为 issue #55 round5，旧源码 `6b2fea9068`、Torch 2.13.0+cu132、FlashInfer 0.7.0.post1。不是同一时刻交错运行的纯源码 A/B；版本、依赖和默认运行器均有变化，其他四卡服务的负载未控制。区间标准沿用之前方法：TPOT p10–p90 重叠记为“无可测差异”。

| 配置 | 负载 | 旧 TPOT p50 | 新 TPOT p50 | 旧 p10–p90 | 新 p10–p90 | 每用户速度变化 | 区间结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | realtext3k-1000 | 6.8183 | 6.3728 | 6.8158–6.8222 | 6.3698–6.3749 | +6.99% | 本轮区间更低 |
| V2 | realtext10k-1000 | 6.8319 | 6.3807 | 6.8287–6.8349 | 6.3787–6.3843 | +7.07% | 本轮区间更低 |
| V2 | realtext32k-1000 | 6.8521 | 6.4040 | 6.8489–6.8556 | 6.4007–6.4066 | +7.00% | 本轮区间更低 |
| V2 | realtext10k-4000 | 6.8336 | 6.3828 | 6.8313–6.8417 | 6.3810–6.3876 | +7.06% | 本轮区间更低 |
| V1 | realtext3k-1000 | 3.8298 | 3.6076 | 3.5787–4.1323 | 3.4265–3.8634 | +6.16% | 无可测差异 |
| V1 | realtext10k-1000 | 3.8411 | 3.6605 | 3.5778–4.1111 | 3.4471–3.9987 | +4.93% | 无可测差异 |
| V1 | realtext32k-1000 | 3.9164 | 3.6493 | 3.6522–4.2016 | 3.4128–3.8412 | +7.32% | 无可测差异 |
| V1 | realtext10k-4000 | 3.7518 | 3.6123 | 3.5208–4.0154 | 3.4175–3.8099 | +3.86% | 无可测差异 |

**Nsight Systems**

每种配置只采 TP0，首个文本输出后请求采集，延后50步、最多64步；CUDA/NVTX/OSRT，禁用CPU sampling和context-switch，CUDA graph按node展开。正式性能阶段已预加载注入器但未开启采集，与此前 profiling 方法一致；注入器的潜在基线开销没有另行标定。使用官方 `nvtx_gpu_proj_trace` 投影确定稳定解码起点，随后核对 graph replay 次数和最后一步。原始报告中起点前的预填充/额外数据不计入每步统计。

| 配置 | 有效步数 | kernel 次/步 | 累计 kernel ms/步 | GPU窗口 ms/步 | 通信 kernel ms/步 | 通信占累计时间 |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | 64 | 1635 | 7.4629 | 6.4025 | 0.6000 | 8.04% |
| V1 | 64 | 1859 | 8.7443 | 8.3351 | 1.0205 | 11.67% |

累计 kernel 时间包含并发 stream 的重叠，因此可能大于 GPU 窗口跨度；两者之差不能解释为 CPU 调度开销。GPU 窗口是首个到最后一个 kernel 的跨度，不能替代整个客户端请求吞吐。通信为名称可识别的 PCIe IPC/NCCL kernel 的累计时间，不是暴露在关键路径上的通信开销。

采集仅代表一个TP worker，不是四卡累计值。Nsight将该卡标为logical GPU 0 / physical GPU ID 4；按UUID与PCI地址`0000:01:00.0`核对，它对应本机`nvidia-smi`的GPU 0。Nsight内部physical编号与nvidia-smi索引不能直接互换，映射保存于`gpu-index-map.csv`。

**MTP验证与草稿拆分**

| graph ID | 归属 | replay次数 | kernel次/步 | 累计kernel ms/步 |
| --- | --- | --- | --- | --- |
| 9020 | 目标模型验证图（M=3） | 64 | 1637 | 7.0217 |
| 9377 | MTP第一个草稿（draft prefill，M=3） | 64 | 85 | 0.6035 |
| 9440 | MTP第二个草稿（decode，M=1） | 64 | 80 | 0.5293 |
| None | 图外采样、logits、通信和元数据等 | N/A | 57 | 0.5898 |

三个图均为64次replay，所有kernel名称计数均为64的整数倍，末步完整。归属依据图的执行顺序、M=3/M=1 kernel符号，以及当前源码`TargetDependentARSpeculator.propose()`先运行draft prefill、再运行multi-step decode的逻辑；不是额外插入的逐层NVTX测量。两次草稿图合计1.1328 ms/步；图外0.5898 ms仍含目标logits及共享工作，不能把验证图7.0217 ms称为完整验证阶段的全部开销。

历史64步trace的V2/V1累计kernel为7.1711/8.9127 ms，GPU窗口为6.3648/8.6157 ms；本轮分别为7.4629/8.7443 ms和6.4025/8.3351 ms。无草稿正式请求约7%的速度提升未在这段GPU窗口中表现为同等下降，因此不能据此断言提升来自GPU kernel加速。短窗口上下文位置、依赖版本和profiler扰动不同，未作因果归因。

V2 含短段 Nsight 采集的整个4000-token请求：首末文本间 33.011 秒，按3999个token间隔估算 121.14 tok/s。它包含采集窗口之外的时间，单列记录，不纳入正式C1表，也不当作64步窗口的吞吐。

**V2：前15个 kernel**

| ID/短名 | 次/步 | 均值 µs/次 | ms/步 | 累计时间占比 |
| --- | --- | --- | --- | --- |
| K1 `CuteSkinnyGemm M=1, K=2560` | 73 | 13.315 | 0.9720 | 13.02% |
| K2 `HcDownSiluFma M=1` | 96 | 8.233 | 0.7903 | 10.59% |
| K3 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 97 | 7.885 | 0.7648 | 10.25% |
| K4 `_hc_combine_norm_kernel` | 95 | 6.717 | 0.6381 | 8.55% |
| K5 `PCIe IPC push RSAG` | 97 | 5.714 | 0.5543 | 7.43% |
| K6 `_fc1_kernel` | 48 | 8.037 | 0.3858 | 5.17% |
| K7 `CuteSkinnyGemm M=1, K=1536` | 48 | 7.602 | 0.3649 | 4.89% |
| K8 `internal::gemvx::kernel variant 7` | 49 | 6.401 | 0.3136 | 4.20% |
| K9 `vllm::moe::topkGating` | 48 | 5.202 | 0.2497 | 3.35% |
| K10 `internal::gemvx::kernel variant 6` | 12 | 20.753 | 0.2490 | 3.34% |
| K11 `_fc2_kernel` | 48 | 4.663 | 0.2238 | 3.00% |
| K12 `cublasLt::splitKreduce_kernel` | 48 | 4.217 | 0.2024 | 2.71% |
| K13 `CuteSkinnyGemm M=1, K=2560 (shared gate/up)` | 48 | 3.490 | 0.1675 | 2.24% |
| K14 `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 36 | 4.210 | 0.1516 | 2.03% |
| K15 `moe_align_block_size_kernel` | 48 | 2.994 | 0.1437 | 1.93% |

完整符号与此排序一一对应，见 `V2/analysis/window-kernels.json`；同名通用 GEMM 不按猜测分配给某层。

通信 kernel 明细：

| kernel | 次/步 | µs/次 | ms/步 |
| --- | --- | --- | --- |
| PCIe IPC push RSAG | 97 | 5.714 | 0.5543 |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 1 | 38.221 | 0.0382 |
| ncclDevKernel_AllReduce_Sum_u8_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 1 | 7.495 | 0.0075 |

CUDA API：使用相邻首/末 NVTX 起点之间的63个完整主机迭代区间，按边界裁剪 API 时间。时间可跨线程重叠，不能与 GPU 时间相加。

| API | 调用数 | 窗口累计 ms | 均摊 ms/步 |
| --- | --- | --- | --- |
| cudaEventSynchronize_v3020 | 64 | 390.6039 | 6.2001 |
| cudaGraphLaunch_v10000 | 63 | 6.4691 | 0.1027 |
| cudaLaunchKernel_v7000 | 1890 | 6.1502 | 0.0976 |
| cudaMemcpyAsync_v3020 | 882 | 4.2613 | 0.0676 |
| cuLaunchKernelEx | 1008 | 3.0462 | 0.0484 |
| cudaEventQuery_v3020 | 630 | 0.7524 | 0.0119 |
| cudaEventRecordWithFlags_v11010 | 378 | 0.5354 | 0.0085 |
| cudaStreamWaitEvent_v3020 | 252 | 0.3068 | 0.0049 |
| cudaLaunchKernelExC_v11060 | 63 | 0.2829 | 0.0045 |
| cuKernelGetName | 1827 | 0.2626 | 0.0042 |

Python 栈：主线程 2978 个样本，leaf 前五如下。这是采样栈比例，不是 CPU 利用率；跨连续长输出请求，包含请求边界。

| leaf 栈帧 | 样本权重占比 |
| --- | --- |
| sched_yield (utils.py) | 54.77% |
| wait (shm_broadcast.py) | 6.65% |
| memory_fence (shm_broadcast.py) | 2.32% |
| wait (shm_broadcast.py) | 2.15% |
| __call__ (driver.py) | 1.21% |

V1 含短段 Nsight 采集的整个4000-token请求：首末文本间 21.378 秒，按3999个token间隔估算 187.06 tok/s。它包含采集窗口之外的时间，单列记录，不纳入正式C1表，也不当作64步窗口的吞吐。

**V1：前15个 kernel**

| ID/短名 | 次/步 | 均值 µs/次 | ms/步 | 累计时间占比 |
| --- | --- | --- | --- | --- |
| K1 `CuteSkinnyGemm M=3, K=2560` | 38 | 24.544 | 0.9327 | 10.67% |
| K2 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(T1::Params)` | 149 | 5.795 | 0.8635 | 9.87% |
| K3 `PCIe IPC push RSAG` | 103 | 8.377 | 0.8628 | 9.87% |
| K4 `HcDownSiluFma M=3` | 98 | 8.220 | 0.8056 | 9.21% |
| K5 `_fc1_kernel` | 48 | 12.887 | 0.6186 | 7.07% |
| K6 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 68 | 9.058 | 0.6160 | 7.04% |
| K7 `CuteSkinnyGemm M=1, K=2560` | 3 | 170.077 | 0.5102 | 5.84% |
| K8 `CuteSkinnyGemm M=3, K=1536` | 49 | 7.692 | 0.3769 | 4.31% |
| K9 `_fc2_kernel` | 48 | 7.367 | 0.3536 | 4.04% |
| K10 `_hc_combine_norm_kernel` | 101 | 2.666 | 0.2692 | 3.08% |
| K11 `vllm::moe::topkGating` | 50 | 5.220 | 0.2610 | 2.98% |
| K12 `gdn_decode_post_conv_mtp_kernel<float,3,true,4>` | 36 | 5.160 | 0.1858 | 2.12% |
| K13 `_topk_topp_kernel` | 1 | 155.834 | 0.1558 | 1.78% |
| K14 `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 7 | 21.315 | 0.1492 | 1.71% |
| K15 `moe_align_block_size_kernel` | 48 | 2.981 | 0.1431 | 1.64% |

完整符号与此排序一一对应，见 `V1/analysis/window-kernels.json`；同名通用 GEMM 不按猜测分配给某层。

通信 kernel 明细：

| kernel | 次/步 | µs/次 | ms/步 |
| --- | --- | --- | --- |
| PCIe IPC push RSAG | 103 | 8.377 | 0.8628 |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 7 | 21.315 | 0.1492 |
| ncclDevKernel_AllReduce_Sum_u8_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 1 | 8.431 | 0.0084 |

CUDA API：使用相邻首/末 NVTX 起点之间的63个完整主机迭代区间，按边界裁剪 API 时间。时间可跨线程重叠，不能与 GPU 时间相加。

| API | 调用数 | 窗口累计 ms | 均摊 ms/步 |
| --- | --- | --- | --- |
| cudaEventSynchronize_v3020 | 64 | 506.8509 | 8.0453 |
| cudaGraphLaunch_v10000 | 189 | 8.7415 | 0.1388 |
| cudaMemcpyAsync_v3020 | 1701 | 8.4590 | 0.1343 |
| cudaLaunchKernel_v7000 | 1764 | 6.8097 | 0.1081 |
| cuLaunchKernelEx | 1764 | 6.0475 | 0.0960 |
| cudaEventQuery_v3020 | 1008 | 1.3672 | 0.0217 |
| cudaEventRecordWithFlags_v11010 | 567 | 0.8632 | 0.0137 |
| cudaStreamIsCapturing_v10000 | 1134 | 0.3910 | 0.0062 |
| cudaStreamWaitEvent_v3020 | 252 | 0.3755 | 0.0060 |
| cuKernelGetName | 1827 | 0.3258 | 0.0052 |

Python 栈：主线程 2984 个样本，leaf 前五如下。这是采样栈比例，不是 CPU 利用率；跨连续长输出请求，包含请求边界。

| leaf 栈帧 | 样本权重占比 |
| --- | --- |
| sched_yield (utils.py) | 43.10% |
| __call__ (driver.py) | 4.36% |
| memory_fence (shm_broadcast.py) | 2.31% |
| dynamic_func (<string>) | 1.68% |
| wait (shm_broadcast.py) | 1.64% |

**采集记录与限制**

首次无草稿运行完成3K测点后，旧抓包解析器将采集边界截断的 `/metrics` 响应判为错误，外层调度器停止实例。已验证50条测量输出完整且逐条匹配原生JSON；修正仅针对监控GET连接的解析。该次数据单列在 `V2-attempt1`，正式表使用重新启动后连续完成的五组测试。模型和官方benchmark未修改。

V2原始dmon贯穿运行，但带绝对时间的dmon是在长输出正式窗口中途补开；前三组不提供伪精确的逐测点GPU利用率区间。V1从启动即使用带时间戳dmon。

启动采用独立缓存和已验证的CUDA_HOME/动态库路径设置，保留OOM计数监控和本次进程oom_score_adj=1000；沿用用户授权，不设内存余量停止线。两种配置均在测试与采集完成后请求停止，常规退出等待超时后由本次监督器强制清理（exit -9，reason=requested_stop），采集文件已完整保存；这不是测试中途失败。

最终核查：`verification.json` 保存源码/包清单、GPU进程和OOM计数的前后对照。

复现脚本：`env.sh`、`run-server.py`、`run-benchmark.py`、`profile-launcher.py`、`profile-client.py`、`collect-profile.py`。完整命令和覆盖环境见每种配置的 `command.json` 与 `bench/status.json`。本地原始 `.nsys-rep` 和 `.speedscope.json` 均保留。

本地数据索引：`V1/benchmark-summary.json`、`V2/benchmark-summary.json`（性能及接受率）；`performance-comparison.json`（历史区间对照）；`profile-summary.json`（64步汇总）；`V1/profile/trace.nsys-rep`、`V2/profile/trace.nsys-rep`（Nsight原始报告）；`V1/profile/worker.speedscope.json`、`V2/profile/worker.speedscope.json`（Python采样）；各配置的`analysis/`保存支持查询返回的证据。完整原始抓包保留在xzss的`/home/roccen/vllm-v034-c1`，本地同步了解析输出及原生benchmark JSON，未复制pcap。

最终状态：两份源码工作区干净，原服务及合并版包清单均未改变；benchmark与输入哈希通过；原有四个GPU进程PID/UUID集合不变，本次测试GPU无残留计算进程；系统OOM计数42→42。
