#!/usr/bin/env bash
# bootB-go.sh -- second boot: identical to boot A except --max-model-len 1048576 (the 1M step).
# Tony's top-k fix is validated to 300K row widths; 1M was served once (boot 7, eager) but not re-run on the fixed
# stack. Run the needle ladder (200K, 500K, 900K) before trusting it; if a >300K request crashes, return to boot A.
export EXP_NAME=bootB IMAGE=vllm-dsv41:overlay5 \
       GMU=0.77 MAXLEN=1048576 SEQS=8 MAX_BATCHED=8192 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
       SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 ENGRAM_LOCAL=0 TEXT_ONLY=0 PARSERS=1 THINKING=false EP="${EP:-0}"
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC k=$SPEC_K eager=$EAGER cg=$CUDAGRAPH_MODE ep=$EP"
bash "$(dirname "$0")/boot_dsv41_tp8.sh"
echo "boot_dsv41_tp8 exit=$? $(date -u +%FT%TZ)"
