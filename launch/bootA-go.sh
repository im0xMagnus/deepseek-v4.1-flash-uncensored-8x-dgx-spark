#!/usr/bin/env bash
# bootA-go.sh -- first TP8 boot: tonyd2wild boot 10, at eight ranks, 300K context. One variable per boot.
# Differences from his boot10-go.sh: TP8/nnodes 8 (in the launcher), ENGRAM_LOCAL=0 (weights are node-local), port 8888,
# GMU 0.77 not 0.80: at eight ranks the NCCL communicators take ~24 GiB per rank before the model loads, so only
# 96-97 of 121.7 GiB is free at vLLM's startup check and 0.80 (97.35 GiB) is refused (boot A attempts 2-3).
export EXP_NAME=bootA IMAGE=vllm-dsv41:overlay5 \
       GMU="${GMU:-0.77}" MAXLEN=300000 SEQS=8 MAX_BATCHED=8192 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
       SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 ENGRAM_LOCAL=0 TEXT_ONLY=0 PARSERS=1 THINKING=false EP="${EP:-0}"
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC k=$SPEC_K eager=$EAGER cg=$CUDAGRAPH_MODE ep=$EP parsers=$PARSERS"
bash "$(dirname "$0")/boot_dsv41_tp8.sh"
echo "boot_dsv41_tp8 exit=$? $(date -u +%FT%TZ)"
