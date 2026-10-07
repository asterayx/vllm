// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Minimal NCCL tuner: prefer ring/LL for single-node allreduce up to
// LL_TUNER_MAX_BYTES (default 327679); everything else keeps NCCL's choice.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int ncclResult_t;
typedef void (*ncclDebugLogger_t)(int, unsigned long, const char*, int, const char*, ...);

#define FUNC_ALLREDUCE 4
#define ALGO_RING 1
#define PROTO_LL 0

typedef struct { size_t nNodes, maxBytes; } Ctx;

static ncclResult_t setup(void** ctx, size_t nRanks, size_t nNodes) {
  Ctx* c = malloc(sizeof(Ctx));
  if (!c) return 1;
  const char* env = getenv("LL_TUNER_MAX_BYTES");
  c->nNodes = nNodes;
  c->maxBytes = env ? strtoull(env, NULL, 10) : 327679;
  if (getenv("LL_TUNER_VERBOSE"))
    fprintf(stderr, "ll_tuner: nRanks=%zu nNodes=%zu maxBytes=%zu\n", nRanks, nNodes,
            c->maxBytes);
  *ctx = c;
  return 0;
}

static ncclResult_t initV5(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                           ncclDebugLogger_t log, void* nvlInfo, void* constants) {
  return setup(ctx, nRanks, nNodes);
}

static ncclResult_t initV4(size_t nRanks, size_t nNodes, ncclDebugLogger_t log,
                           void** ctx) {
  return setup(ctx, nRanks, nNodes);
}

static ncclResult_t getCollInfo(void* ctx, int collType, size_t nBytes, int numPipeOps,
                                float** cost, int numAlgo, int numProto, int regBuff,
                                int* nChannels) {
  // NCCL passes a contiguous float[numAlgo][numProto] cast to float**.
  float* table = (float*)cost;
  float* ringLL = &table[ALGO_RING * numProto + PROTO_LL];
  Ctx* c = ctx;
  if (c && collType == FUNC_ALLREDUCE && c->nNodes == 1 && nBytes <= c->maxBytes &&
      ALGO_RING < numAlgo && PROTO_LL < numProto && *ringLL >= 0.0f)
    *ringLL = 0.0f;
  return 0;
}

static ncclResult_t finalize(void* ctx) { free(ctx); return 0; }

typedef struct {
  const char* name;
  ncclResult_t (*init)(void**, uint64_t, size_t, size_t, ncclDebugLogger_t, void*, void*);
  ncclResult_t (*getCollInfo)(void*, int, size_t, int, float**, int, int, int, int*);
  ncclResult_t (*finalize)(void*);
} TunerV5;

typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t, size_t, ncclDebugLogger_t, void**);
  ncclResult_t (*getCollInfo)(void*, int, size_t, int, float**, int, int, int, int*);
  ncclResult_t (*destroy)(void*);
} TunerV4;

const TunerV5 ncclTunerPlugin_v5 = {"ll_tuner", initV5, getCollInfo, finalize};
const TunerV4 ncclTunerPlugin_v4 = {"ll_tuner", initV4, getCollInfo, finalize};
