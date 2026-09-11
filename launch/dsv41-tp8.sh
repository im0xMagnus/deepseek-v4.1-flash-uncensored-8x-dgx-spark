#!/usr/bin/env bash
# dsv41-tp8.sh <rank>  -- DeepSeek-V4.1-Flash (dealignai UNCENSORED-FP8) TP8 across 8 DGX Spark GB10.
#
# Derived from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark launch/dsv41-tp4.sh (Tech2Wild/Kai, 2026-09-10).
# Changed ONLY where eight nodes and our fabric differ from his four:
#   - 8 ranks, our fabric IPs, RoCE GID index probed per NIC, our NCCL env (same pattern as our GLM job)
#   - weights node-local on every rank (no NFS), so ENGRAM_LOCAL defaults to 0 (rows are already local)
#   - --tensor-parallel-size 8 --nnodes 8; API on :8888 so the client config only changes the model id
#   - EP=1 knob adds --enable-expert-parallel (gate for the moe_intermediate_size/8 question)
#   - --ulimit nofile=1048576: NCCL 2.30 at eight ranks ran out of file descriptors at comm init (boot A, "Too many open files")
# Everything else (patches, memory guards, DSpark, graphs, parsers) is his, deliberately.
#
# Knobs (export before running, SAME on all eight):
#   IMAGE        default vllm-dsv41:overlay5 (fallback: vllm/vllm-openai:deepseekv41-flash-0909-arm64)
#   EXP_NAME     label + per-experiment compile cache dir (default bootA)
#   GMU          --gpu-memory-utilization (default 0.80)
#   MAXLEN       --max-model-len (default 300000)
#   SEQS         --max-num-seqs (default 8)
#   MAX_BATCHED  --max-num-batched-tokens (default 8192)
#   EAGER        1 => --enforce-eager. 0 (default) => CUDA graphs, CUDAGRAPH_MODE (default FULL_AND_PIECEWISE)
#   CG_SIZES     capture sizes; default for dspark = multiples of k and k+1 up to SEQS*(k+1) (exact graphs, FlashInfer #5015)
#   SPEC_ADAPT   adaptive verification (default false: padded rows hang SM12x sparse MLA, FlashInfer #5015)
#   SPEC         none | dspark (default dspark, k=SPEC_K default 5)
#   ENGRAM_DISK  1 (default) => Engram tables read from the shards on disk (patch), not loaded into memory
#   ENGRAM_LOCAL 0 (default here; weights are local on every rank). 1 => mount /var/tmp/engram-local copy
#   TEXT_ONLY    0 (default here; vision on) | 1 => --language-model-only
#   THINKING     false (default) => --default-chat-template-kwargs '{"thinking": false}'
#   PARSERS      1 (default here) => deepseek_v41 tool + reasoning parsers (Rust parser ext in the image)
#   EP           0 (default) | 1 => --enable-expert-parallel
#   RUST_FE      0 (default) => VLLM_USE_RUST_FRONTEND=0
#   PATCH_DIR    default $HOME/dsv41-recipe/patch (Tony's repo clone; mounts.txt lives there)
#   VLLM_EXTRA   extra vllm serve args;  NCCL_EXTRA extra "-e K=V" docker env pairs
#   DRYRUN       1 => run every check as a warning and print the docker command instead of starting it
set -euo pipefail
NODE_RANK="${1:?usage: dsv41-tp8.sh <0..7>}"
DRYRUN="${DRYRUN:-0}"
die(){ if [ "$DRYRUN" = 1 ]; then echo "WARN(dryrun): $1" >&2; else echo "$1" >&2; exit "${2:-1}"; fi; }

IMAGE="${IMAGE:-vllm-dsv41:overlay5}"
EXP_NAME="${EXP_NAME:-bootA}"
GMU="${GMU:-0.80}"
MAXLEN="${MAXLEN:-300000}"
SEQS="${SEQS:-8}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
EAGER="${EAGER:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${CG_SIZES:-}"
SPEC="${SPEC:-dspark}"
SPEC_K="${SPEC_K:-5}"
ENGRAM_DISK="${ENGRAM_DISK:-1}"
ENGRAM_LOCAL="${ENGRAM_LOCAL:-0}"
TEXT_ONLY="${TEXT_ONLY:-0}"
THINKING="${THINKING:-false}"
PARSERS="${PARSERS:-1}"
EP="${EP:-0}"
RUST_FE="${RUST_FE:-0}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
NCCL_EXTRA="${NCCL_EXTRA:-}"

NAME="vllm_dsv41"
MODEL_DIR="DeepSeek-V4.1-Flash-UNCENSORED-FP8"
MODEL_HOST="/data/models/$MODEL_DIR"          # node-local on every rank
CACHE_HOST_PATH="/var/tmp/dsv41-vllm-cache"
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
TOK_ARGS=""
case "$IMAGE" in *deepseekv41-flash-0909*) TOK_ARGS="--tokenizer-mode deepseek_v41" ;; esac

# ---- rank -> fabric IP (same map as the GLM job) ----
FAB=(NODE_PREFIX_PLACEHOLDER.10 NODE_PREFIX_PLACEHOLDER.11 NODE_PREFIX_PLACEHOLDER.12 NODE_PREFIX_PLACEHOLDER.13 \
     NODE_PREFIX_PLACEHOLDER.14 NODE_PREFIX_PLACEHOLDER.15 NODE_PREFIX_PLACEHOLDER.16 NODE_PREFIX_PLACEHOLDER.17)
[ "$NODE_RANK" -ge 0 ] && [ "$NODE_RANK" -le 7 ] || { echo "rank must be 0-7" >&2; exit 2; }
HOST_IP="${FAB[$NODE_RANK]}"
HEAD_IP="${FAB[0]}"; MPORT="29551"; PORT="8888"
[ "$NODE_RANK" = "0" ] && HEADLESS="" || HEADLESS="--headless"

# RoCEv2 GID index is per-NIC and moves across firmware updates -- probe it (from our GLM launcher).
GIDX=3
for i in 0 1 2 3 4 5 6 7; do
  t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null)
  g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null)
  case "$t" in *"RoCE v2"*) case "$g" in *ffff*) GIDX=$i; break ;; esac ;; esac
done
echo "rank $NODE_RANK  ip=$HOST_IP  NCCL_IB_GID_INDEX=$GIDX"

# ---- preflight (fail loudly; never let Docker invent an empty dir over a missing mount) ----
test -f "$MODEL_HOST/config.json" || die "MODEL MISSING at $MODEL_HOST" 3
test -f "$MODEL_HOST/model-00048-of-00048.safetensors" || die "MODEL INCOMPLETE at $MODEL_HOST (shard 48 missing)" 3
n=$(ls "$MODEL_HOST"/*.safetensors 2>/dev/null | wc -l); [ "$n" -eq 48 ] || die "MODEL INCOMPLETE: $n/48 shards" 3
PATCH_DIR="${PATCH_DIR:-$HOME/dsv41-recipe/patch}"
PATCH_MOUNTS=""
if [ -f "$PATCH_DIR/mounts.txt" ]; then
  while read -r f rel; do
    [ -z "$f" ] && continue
    if [ "$ENGRAM_DISK" != "1" ] && { [ "$f" = "engram.py" ] || [ "$f" = "weight_utils.py" ] || [ "$f" = "model_state.py" ]; }; then continue; fi
    test -f "$PATCH_DIR/$f" || die "PATCH FILE MISSING: $PATCH_DIR/$f" 3
    PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$SITE/$rel:ro"
  done < "$PATCH_DIR/mounts.txt"
else
  die "no mounts.txt in $PATCH_DIR (clone tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark to ~/dsv41-recipe)" 3
fi
if [ "$ENGRAM_DISK" = "1" ]; then
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=${ENGRAM_THREADS:-32} -e DSV41_ENGRAM_DISK_CHUNK=${ENGRAM_CHUNK:-16}"
else
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=0"
fi
ENGRAM_LOCAL_HOST="${ENGRAM_LOCAL_HOST:-/var/tmp/engram-local/$MODEL_DIR}"
ENGRAM_LOCAL_MOUNT=""
if [ "$ENGRAM_DISK" = "1" ] && [ "$ENGRAM_LOCAL" = "1" ] && [ -f "$ENGRAM_LOCAL_HOST/engram-local.json" ]; then
  ENGRAM_LOCAL_MOUNT="-v $ENGRAM_LOCAL_HOST:/engram-local:ro"
  ENGRAM_ENV="$ENGRAM_ENV -e DSV41_ENGRAM_DIR=/engram-local"
fi
mkdir -p "$CACHE_HOST_PATH"
DOCKER=docker
if [ "$DRYRUN" = 1 ]; then DOCKER="echo DRYRUN: docker"; else
  docker rm -f "$NAME" 2>/dev/null || true
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "WARN: page-cache drop denied (sudoers rule missing?)" >&2
fi
AVAIL_GB=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
[ "$AVAIL_GB" -ge 100 ] || die "MemAvailable ${AVAIL_GB} GiB < 100 GiB, refusing to boot (is the GLM job still up?)" 4

GRAPH_ENV=""
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS=(--enforce-eager)
else
  if [ "$ENGRAM_DISK" = "1" ] && ! grep -q '^model_state.py ' "$PATCH_DIR/mounts.txt"; then
    die "EAGER=0 + ENGRAM_DISK=1 needs the Engram prestage patch (model_state.py in mounts.txt)" 3
  fi
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then
      K="$SPEC_K"
      CG_SIZES=$( { seq "$K" "$K" $((K * SEQS)); seq $((K + 1)) $((K + 1)) $(((K + 1) * SEQS)); } | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$SEQS" | paste -sd, -)
    fi
  fi
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}")
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
if [ "$EAGER" = "1" ]; then SPEC_ADAPT=false; else SPEC_ADAPT="${SPEC_ADAPT:-false}"; fi
if [ "$SPEC" = "dspark" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":${SPEC_ADAPT}}"
else SPEC_ARGS=""; fi
if [ "$TEXT_ONLY" = "1" ]; then TEXT_ARGS="--language-model-only"; else TEXT_ARGS=""; fi
if [ "$PARSERS" = "1" ]; then PARSER_ARGS="--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41"; else PARSER_ARGS=""; fi
if [ "$EP" = "1" ]; then EP_ARGS="--enable-expert-parallel"; else EP_ARGS=""; fi

# shellcheck disable=SC2086
$DOCKER run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$MODEL_HOST:/models/$MODEL_DIR:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  $PATCH_MOUNTS $ENGRAM_LOCAL_MOUNT \
  -e VLLM_HOST_IP=$HOST_IP -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_RUST_FRONTEND=$RUST_FE -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  $ENGRAM_ENV $GRAPH_ENV \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=NODE_PREFIX_PLACEHOLDER.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_CROSS_NIC=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_BUFFSIZE=16777216 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $NCCL_EXTRA \
  "$IMAGE" \
    "/models/$MODEL_DIR" \
    --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port "$PORT" \
    $TOK_ARGS \
    --tensor-parallel-size 8 $EP_ARGS --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
    --engram-config '{"cpu_offload": false}' \
    --default-chat-template-kwargs "{\"thinking\": $THINKING}" \
    $TEXT_ARGS $PARSER_ARGS $SPEC_ARGS "${GRAPH_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 8 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS $VLLM_EXTRA

echo "launched $NAME rank=$NODE_RANK exp=$EXP_NAME image=$IMAGE patches=$PATCH_DIR gmu=$GMU maxlen=$MAXLEN seqs=$SEQS eager=$EAGER cg=${CUDAGRAPH_MODE}[${CG_SIZES}] spec=$SPEC k=$SPEC_K adapt=$SPEC_ADAPT ep=$EP engram_disk=$ENGRAM_DISK engram_local=${ENGRAM_LOCAL_MOUNT:+yes} text_only=$TEXT_ONLY parsers=$PARSERS avail=${AVAIL_GB}GiB"
[ "$DRYRUN" = 1 ] && exit 0
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; docker logs --tail 40 "$NAME" >&2; exit 1; }
