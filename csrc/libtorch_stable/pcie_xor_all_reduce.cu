// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Paired posted-write XOR-butterfly all-reduce.
//
// Numeric contract (elementwise, every rank), matching
// vllm.distributed.device_communicators.pcie_xor_butterfly:
//   partial = fp32(input)
//   for round in 0 .. log2(world)-1:
//     partner = rank XOR (1 << round)
//     partial = lower_rank_partial + higher_rank_partial
//   output = round-to-nearest-even(partial)
// fp16/bf16 -> fp32 is exact, so round 0 posts the original bits and later
// rounds post the fp32 partial. Flags live in the receiver's slab. Readers
// spin on local memory. The epoch counter lives on device so CUDA-graph
// replay does not bake a host epoch. One CTA avoids a grid-wide barrier.

#include "ops.h"
#include "torch_utils.h"

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>

  #include <string>
  #include <type_traits>

namespace {

constexpr int kThreads = 512;
constexpr int kFlagBytes = 128;
constexpr unsigned long long kErrTimeout = 1;
constexpr unsigned long long kErrMismatch = 2;
constexpr unsigned long long kErrFuture = 3;
constexpr unsigned long long kErrEpoch = 4;

struct XorLaunch {
  void* slabs[8];
  float* partial;
  unsigned long long* state;
  const void* input;
  void* output;
  int64_t flag_base;
  int64_t payload_base;
  int64_t slot_stride;
  int rank;
  int rounds;
  int numel;
};

void xor_cuda_check(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess,
                  std::string(what) + cudaGetErrorString(err));
}

__device__ __forceinline__ unsigned long long ld_gpu_u64(
    const unsigned long long* addr) {
  unsigned long long value;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ void st_gpu_u64(unsigned long long* addr,
                                           unsigned long long value) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(addr), "l"(value)
               : "memory");
}

__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(
    const unsigned long long* addr) {
  unsigned long long value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ unsigned long long ld_acquire_sys_u64(
    const unsigned long long* addr) {
  unsigned long long value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ void st_release_sys_u64(unsigned long long* addr,
                                                   unsigned long long value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(value)
               : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_u32(unsigned* addr,
                                                   unsigned value) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(addr), "r"(value)
               : "memory");
}

__device__ __forceinline__ unsigned ld_relaxed_sys_u32(const unsigned* addr) {
  unsigned value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ void st_relaxed_sys_u16(unsigned short* addr,
                                                   unsigned short value) {
  asm volatile("st.relaxed.sys.global.u16 [%0], %1;" ::"l"(addr), "h"(value)
               : "memory");
}

__device__ __forceinline__ unsigned short ld_cg_u16(
    const unsigned short* addr) {
  unsigned short value;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ void st_relaxed_sys_f32(float* addr, float value) {
  unsigned bits = __float_as_uint(value);
  st_relaxed_sys_u32(reinterpret_cast<unsigned*>(addr), bits);
}

__device__ __forceinline__ float ld_cg_f32(const float* addr) {
  unsigned bits = ld_relaxed_sys_u32(reinterpret_cast<const unsigned*>(addr));
  return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void fail(unsigned long long* state,
                                     unsigned long long code, int rank,
                                     int round) {
  unsigned long long old = atomicCAS(state + 1, 0ull, code);
  if (old == 0) {
    printf(
        "DSV41 PCIe XOR all-reduce fault: code=%llu rank=%d round=%d "
        "(1=timeout 2=metadata 3=future-epoch 4=epoch). Output zeroed.\n",
        code, rank, round);
  }
}

__device__ __forceinline__ char* payload_at(char* slab, const XorLaunch& L,
                                            int parity, int round) {
  int index = parity * L.rounds + round;
  return slab + L.payload_base + static_cast<int64_t>(index) * L.slot_stride;
}

__device__ __forceinline__ unsigned long long* flag_epoch(char* slab,
                                                          const XorLaunch& L,
                                                          int parity,
                                                          int round) {
  int index = parity * L.rounds + round;
  return reinterpret_cast<unsigned long long*>(
      slab + L.flag_base + static_cast<int64_t>(index) * kFlagBytes);
}

__device__ __forceinline__ void publish_flag(unsigned long long* epoch_addr,
                                             unsigned long long epoch,
                                             int numel, int round) {
  auto* meta = reinterpret_cast<unsigned*>(epoch_addr + 1);
  st_relaxed_sys_u32(meta, static_cast<unsigned>(numel));
  st_relaxed_sys_u32(meta + 1, static_cast<unsigned>(round));
  __threadfence_system();
  st_release_sys_u64(epoch_addr, epoch);
}

__device__ __forceinline__ bool wait_flag(unsigned long long* epoch_addr,
                                          unsigned long long epoch, int numel,
                                          int round, unsigned long long* state,
                                          int rank) {
  unsigned long long timeout = ld_gpu_u64(state + 2);
  unsigned long long start = globaltimer();
  auto* meta = reinterpret_cast<unsigned*>(epoch_addr + 1);
  for (;;) {
    unsigned long long seen = ld_relaxed_sys_u64(epoch_addr);
    if (seen == epoch) {
      unsigned long long confirmed = ld_acquire_sys_u64(epoch_addr);
      unsigned got_numel = ld_relaxed_sys_u32(meta);
      unsigned got_round = ld_relaxed_sys_u32(meta + 1);
      if (confirmed != epoch || got_numel != static_cast<unsigned>(numel) ||
          got_round != static_cast<unsigned>(round)) {
        fail(state, kErrMismatch, rank, round);
        return false;
      }
      return true;
    }
    if (seen > epoch) {
      fail(state, kErrFuture, rank, round);
      return false;
    }
    if (globaltimer() - start >= timeout) {
      fail(state, kErrTimeout, rank, round);
      return false;
    }
    __nanosleep(64);
  }
}

// bf16 device ops exist from sm_80 up. sm_75 still builds fp16/fp32, and the
// preprocessor strips the bf16 intrinsics from that cubin. The host launch of
// the bf16 kernel is likewise compiled out of the sm_75 pass.
template <typename T>
__device__ __forceinline__ float local_to_f32(T value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else if constexpr (std::is_same<T, half>::value) {
    return __half2float(value);
  } else {
  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
      !defined(__CUDA_ARCH__)
    return __bfloat162float(value);
  #else
    return 0.f;
  #endif
  }
}

template <typename T>
__device__ __forceinline__ T f32_to_dtype(float value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else if constexpr (std::is_same<T, half>::value) {
    return __float2half(value);
  } else {
  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
      !defined(__CUDA_ARCH__)
    return __float2bfloat16(value);
  #else
    return T{};
  #endif
  }
}

__device__ __forceinline__ void post_f32(float* dst, const float* src,
                                         int numel) {
  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    st_relaxed_sys_f32(dst + i, src[i]);
  }
  __threadfence_system();
}

template <typename T>
__device__ __forceinline__ void post_raw(T* dst, const T* src, int numel) {
  if constexpr (std::is_same<T, float>::value) {
    post_f32(dst, src, numel);
  } else {
    for (int i = threadIdx.x; i < numel; i += blockDim.x) {
      unsigned short bits;
      if constexpr (std::is_same<T, half>::value) {
        bits = __half_as_ushort(src[i]);
      } else {
  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
      !defined(__CUDA_ARCH__)
        bits = __bfloat16_as_ushort(src[i]);
  #else
        bits = 0;
  #endif
      }
      st_relaxed_sys_u16(reinterpret_cast<unsigned short*>(dst + i), bits);
    }
    __threadfence_system();
  }
}

template <typename T>
__device__ __forceinline__ float peer_to_f32(const T* addr) {
  if constexpr (std::is_same<T, float>::value) {
    return ld_cg_f32(addr);
  }
  unsigned short bits =
      ld_cg_u16(reinterpret_cast<const unsigned short*>(addr));
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(__ushort_as_half(bits));
  } else {
    return __uint_as_float(static_cast<unsigned>(bits) << 16);
  }
}

template <typename T>
__device__ __forceinline__ void accum_peer(float* partial, const T* peer,
                                           int numel, bool local_is_lower) {
  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    float mine = partial[i];
    float other = peer_to_f32(peer + i);
    partial[i] = local_is_lower ? (mine + other) : (other + mine);
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1) pcie_xor_butterfly(XorLaunch L) {
  const int tx = threadIdx.x;
  __shared__ unsigned long long epoch;
  __shared__ int bad;

  auto* input = static_cast<const T*>(L.input);
  auto* output = static_cast<T*>(L.output);

  if (tx == 0) {
    bad = ld_gpu_u64(L.state + 1) != 0;
    epoch = 0;
    if (!bad) {
      unsigned long long prev = ld_gpu_u64(L.state);
      if (prev == ~0ull) {
        fail(L.state, kErrEpoch, L.rank, -1);
        bad = 1;
      } else {
        epoch = prev + 1;
      }
    }
  }
  __syncthreads();

  if (!bad) {
    for (int i = tx; i < L.numel; i += blockDim.x) {
      L.partial[i] = local_to_f32(input[i]);
    }
  }
  __syncthreads();

  for (int round = 0; round < L.rounds && !bad; ++round) {
    int partner = L.rank ^ (1 << round);
    int parity = static_cast<int>(epoch & 1ull);
    char* peer_slab = static_cast<char*>(L.slabs[partner]);
    char* self_slab = static_cast<char*>(L.slabs[L.rank]);
    char* dst = payload_at(peer_slab, L, parity, round);
    char* inbox = payload_at(self_slab, L, parity, round);
    bool raw = round == 0 && !std::is_same<T, float>::value;
    if (raw) {
      post_raw(reinterpret_cast<T*>(dst), input, L.numel);
    } else {
      post_f32(reinterpret_cast<float*>(dst), L.partial, L.numel);
    }
    __syncthreads();
    if (tx == 0) {
      publish_flag(flag_epoch(peer_slab, L, parity, round), epoch, L.numel,
                   round);
      bad = !wait_flag(flag_epoch(self_slab, L, parity, round), epoch, L.numel,
                       round, L.state, L.rank);
    }
    __syncthreads();
    if (bad) {
      break;
    }
    // Every reader synchronizes with the publisher's release.
    ld_acquire_sys_u64(flag_epoch(self_slab, L, parity, round));
    bool local_is_lower = L.rank < partner;
    if (raw) {
      accum_peer(L.partial, reinterpret_cast<const T*>(inbox), L.numel,
                 local_is_lower);
    } else {
      accum_peer(L.partial, reinterpret_cast<const float*>(inbox), L.numel,
                 local_is_lower);
    }
    __syncthreads();
  }

  if (bad) {
    for (int i = tx; i < L.numel; i += blockDim.x) {
      output[i] = f32_to_dtype<T>(0.f);
    }
    return;
  }
  for (int i = tx; i < L.numel; i += blockDim.x) {
    output[i] = f32_to_dtype<T>(L.partial[i]);
  }
  if (tx == 0) {
    st_gpu_u64(L.state, epoch);
  }
}

template <typename T>
void launch_typed(const XorLaunch& launch, cudaStream_t stream) {
  pcie_xor_butterfly<T><<<1, kThreads, 0, stream>>>(launch);
  xor_cuda_check(cudaGetLastError(), "PCIe XOR all-reduce launch failed: ");
}

}  // namespace

void pcie_xor_all_reduce(const std::vector<int64_t>& peer_ptrs,
                         torch::stable::Tensor& state,
                         torch::stable::Tensor& partial,
                         torch::stable::Tensor& inp, torch::stable::Tensor& out,
                         int64_t rank, int64_t world_size, int64_t flag_base,
                         int64_t payload_base, int64_t slot_stride) {
  STD_TORCH_CHECK(inp.is_cuda(), "PCIe XOR all-reduce requires CUDA tensors");
  STD_TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                  "PCIe XOR all-reduce requires contiguous tensors");
  STD_TORCH_CHECK(inp.numel() == out.numel(), "numel mismatch");
  STD_TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "dtype mismatch");
  STD_TORCH_CHECK(inp.numel() > 0, "empty all-reduce");
  STD_TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
                  "world_size must be 2, 4, or 8");
  STD_TORCH_CHECK(static_cast<int64_t>(peer_ptrs.size()) == world_size,
                  "peer pointer count");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "rank");
  STD_TORCH_CHECK(state.scalar_type() == torch::headeronly::ScalarType::Long,
                  "state must be int64");
  STD_TORCH_CHECK(state.numel() >= 3 && state.is_cuda(), "state");
  STD_TORCH_CHECK(
      partial.scalar_type() == torch::headeronly::ScalarType::Float &&
          partial.is_cuda() && partial.numel() >= inp.numel(),
      "partial");
  STD_TORCH_CHECK(
      slot_stride >= inp.numel() * static_cast<int64_t>(sizeof(float)),
      "slot_stride");
  STD_TORCH_CHECK(inp.numel() <= 2147483647LL, "numel");

  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  XorLaunch launch{};
  for (int i = 0; i < world_size; ++i) {
    STD_TORCH_CHECK(peer_ptrs[i] != 0, "null peer slab");
    launch.slabs[i] = reinterpret_cast<void*>(peer_ptrs[i]);
  }
  launch.partial = static_cast<float*>(partial.mutable_data_ptr());
  launch.state = static_cast<unsigned long long*>(state.mutable_data_ptr());
  launch.input = inp.const_data_ptr();
  launch.output = out.mutable_data_ptr();
  launch.flag_base = flag_base;
  launch.payload_base = payload_base;
  launch.slot_stride = slot_stride;
  launch.rank = static_cast<int>(rank);
  launch.numel = static_cast<int>(inp.numel());
  launch.rounds = 0;
  for (int64_t w = world_size; w > 1; w >>= 1) {
    launch.rounds += 1;
  }

  switch (inp.scalar_type()) {
    case torch::headeronly::ScalarType::Float:
      launch_typed<float>(launch, stream);
      break;
    case torch::headeronly::ScalarType::Half:
      launch_typed<half>(launch, stream);
      break;
  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
      !defined(__CUDA_ARCH__)
    case torch::headeronly::ScalarType::BFloat16:
      launch_typed<nv_bfloat16>(launch, stream);
      break;
  #endif
    default:
      STD_TORCH_CHECK(false,
                      "PCIe XOR all-reduce supports fp32, fp16, and bf16");
  }
}

void pcie_xor_close_peer(int64_t ptr) {
  if (ptr == 0) {
    return;
  }
  cudaError_t err = cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
  xor_cuda_check(err, "cudaIpcCloseMemHandle failed: ");
}
#else
void pcie_xor_all_reduce(const std::vector<int64_t>& peer_ptrs,
                         torch::stable::Tensor& state,
                         torch::stable::Tensor& partial,
                         torch::stable::Tensor& inp, torch::stable::Tensor& out,
                         int64_t rank, int64_t world_size, int64_t flag_base,
                         int64_t payload_base, int64_t slot_stride) {
  (void)peer_ptrs;
  (void)state;
  (void)partial;
  (void)inp;
  (void)out;
  (void)rank;
  (void)world_size;
  (void)flag_base;
  (void)payload_base;
  (void)slot_stride;
  STD_TORCH_CHECK(false, "PCIe XOR all-reduce is CUDA-only");
}

void pcie_xor_close_peer(int64_t ptr) {
  (void)ptr;
  STD_TORCH_CHECK(false, "PCIe XOR all-reduce is CUDA-only");
}
#endif
