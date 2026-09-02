#!/usr/bin/env bash
# Dev (bind-mount host source over /opt/vllm):
#   NODE_RANK=0 ./docker/gb10/run.sh
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless ./docker/gb10/run.sh
# Release (baked /opt/vllm, no source mount): ./docker/gb10/run-image.sh
set -euo pipefail

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/DeepSeek-V4-Flash-0731}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-0731}"
MOUNT_VLLM_SRC="${MOUNT_VLLM_SRC:-1}"
VLLM_CACHE="${VLLM_CACHE:-${HOME}/vllm-cache}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
B12X_CACHE="${B12X_CACHE:-${HOME}/.cache/b12x}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
VLLM_HOST_IP="${VLLM_HOST_IP:-${MASTER_ADDR}}"
NODE_RANK="${NODE_RANK:-0}"
HEADLESS="${HEADLESS:-}"
NAME="${NAME:-dspark-tp2-rank${NODE_RANK}}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"

src_mount=()
pythonpath_args=()
if [ "${MOUNT_VLLM_SRC}" != "0" ]; then
  VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
  src_mount=(-v "${VLLM_SRC}:/opt/vllm")
  pythonpath_args=(-e PYTHONPATH=/opt/vllm)
  echo "dev: mounting ${VLLM_SRC} -> /opt/vllm"
else
  echo "image: using baked /opt/vllm (no host source mount)"
fi

mkdir -p "${VLLM_CACHE}" "${HF_CACHE}/flashinfer" "${B12X_CACHE}"
docker rm -f "${NAME}" 2>/dev/null || true

# 6 seqs * (1 + DSpark k=5) = 36; include 36 so capture is not truncated to 32.
# Vision-Exp (run-vision.sh) overrides these for k=3 / next_n=4.
# JSON defaults cannot live in ${VAR:-...} (bash cuts at the first `}`).
if [ -z "${CUGRAPH_CFG+x}" ]; then
  CUGRAPH_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,24,32,36]}'
fi
if [ -z "${SPECULATIVE_CONFIG+x}" ]; then
  SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
fi

docker run -d --name "${NAME}" \
  --gpus all --ipc=host --network host --privileged \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v "${MODEL_HOST}:${MODEL_PATH}:ro" \
  "${src_mount[@]}" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${HF_CACHE}/flashinfer:/root/.cache/flashinfer" \
  -v "${B12X_CACHE}:/root/.cache/b12x" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${pythonpath_args[@]}" \
  -e MOUNT_VLLM_SRC="${MOUNT_VLLM_SRC}" \
  -e VLLM_HOST_IP="${VLLM_HOST_IP}" \
  -e VLLM_LOGGING_COLOR=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/root/.cache/b12x/cute_compile \
  -e B12X_PRINT_COMPILE_PROGRESS="${B12X_PRINT_COMPILE_PROGRESS:-1}" \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e TP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -w / \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'export PATH=/opt/venv/bin:${PATH}
       if [ "${MOUNT_VLLM_SRC:-1}" != "0" ]; then
         export PYTHONPATH=/opt/vllm${PYTHONPATH:+:$PYTHONPATH}
         for site in /opt/venv/lib/python*/site-packages; do
           [ -d "$site" ] || continue
           rm -f "$site"/__editable__.vllm* "$site"/__editable___vllm*
           echo /opt/vllm > "$site"/_vllm_relocated.pth
         done
       else
         # Image mode: do not put /opt/vllm on PYTHONPATH. That loads the
         # source tree as a regular package and hides cmake .so files that
         # the uv editable finder would otherwise add to vllm.__path__.
         _pp=""
         _save_ifs="$IFS"
         IFS=:
         for _p in ${PYTHONPATH:-}; do
           [ -z "$_p" ] && continue
           [ "$_p" = "/opt/vllm" ] && continue
           _pp="${_pp:+$_pp:}$_p"
         done
         IFS="$_save_ifs"
         if [ -n "$_pp" ]; then
           export PYTHONPATH="$_pp"
         else
           unset PYTHONPATH
         fi
       fi
       /opt/venv/bin/python - <<'"'"'PY'"'"'
import importlib.metadata as m
import os
from pathlib import Path

ver = os.environ.get("VLLM_VERSION_OVERRIDE", "0.28.0")
text = "Metadata-Version: 2.1\nName: vllm\nVersion: %s\n" % ver
try:
    print("vllm metadata", m.version("vllm"), flush=True)
except m.PackageNotFoundError:
    dests = [Path("/tmp") / ("vllm-%s.dist-info" % ver)]
    try:
        import site
        dests.extend(
            Path(p) / ("vllm-%s.dist-info" % ver)
            for p in site.getsitepackages()
            if p
        )
    except Exception:
        pass
    wrote = False
    for dest in dests:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "METADATA").write_text(text)
            print("wrote stub", dest, flush=True)
            wrote = True
        except OSError as e:
            print("could not write", dest, e, flush=True)
    if not wrote:
        raise SystemExit("could not create vllm package metadata")

# cmake install may leave extensions under build/ rather than vllm/.
# PYTHONPATH and a plain .pth only see the source package dir.
pkg = Path("/opt/vllm/vllm")
names = (
    "_C_stable_libtorch",
    "_moe_C_stable_libtorch",
    "_C",
    "_moe_C",
)
search_roots = [pkg]
search_roots.extend(Path("/opt/vllm").glob("build*"))
search_roots.extend(Path("/opt/vllm").glob("_skbuild*"))
search_roots.extend(Path("/opt/venv").glob("lib/python*/site-packages"))
search_roots.extend(Path("/opt/venv").glob("lib/python*/site-packages/vllm"))
for name in names:
    already = sorted(pkg.glob(f"{name}*.so")) if pkg.is_dir() else []
    if already:
        print("have", already[0], flush=True)
        continue
    found = []
    for root in search_roots:
        if not root.is_dir():
            continue
        found.extend(p for p in root.glob(f"{name}*.so") if p.is_file())
        found.extend(p for p in root.glob(f"**/{name}*.so") if p.is_file())
    uniq, seen = [], set()
    for p in found:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    print(f"candidates {name}: {uniq}", flush=True)
    if not uniq or not pkg.is_dir():
        continue
    src = uniq[0]
    dst = pkg / src.name
    try:
        if not dst.exists():
            dst.symlink_to(src)
            print("linked", dst, "->", src, flush=True)
    except OSError as e:
        print("could not link", dst, e, flush=True)
PY
       if [ -d /tmp/vllm-0.28.0.dist-info ] || [ -d /tmp/vllm-"${VLLM_VERSION_OVERRIDE:-0.28.0}".dist-info ]; then
         export PYTHONPATH="/tmp${PYTHONPATH:+:$PYTHONPATH}"
       fi
       exec /opt/venv/bin/python -m vllm.entrypoints.cli.main "$@"' \
  vllm \
  serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}" \
  --host 0.0.0.0 --port "${VLLM_PORT:-30001}" \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size "${MAX_CUGRAPH:-36}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.87}" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --linear-backend "${LINEAR_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --compilation-config "${CUGRAPH_CFG}" \
  ${EXTRA_VLLM_ARGS:-} \
  ${HEADLESS}

echo "started ${NAME}; logs: docker logs -f ${NAME}"
