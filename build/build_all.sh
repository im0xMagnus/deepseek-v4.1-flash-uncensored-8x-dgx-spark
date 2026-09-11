#!/usr/bin/env bash
# build_all.sh -- build tonyd2wild's five-stage image chain ON ONE NODE (the build node; run it there, not from the Mac).
# Produces vllm-dsv41:overlay5, the serving image. Then tools/fanout_image.sh copies it to the other seven.
#
# Stages (all from ~/dsv41-recipe/build, pinned 592540c6):
#   1  overlay1 = vllm/vllm-openai:nightly-8a728663... (arm64 digest pinned below) + vLLM branch dsv41-feat python tree
#                + _C_stable_libtorch rebuilt for sm_121a (build_stable_ext.sh inside a 'v41build' container).
#                Tony's repo ships the Dockerfile and the extension build but not the container setup; this script does it.
#   3  overlay3 = + FlashInfer 0.7.0rc1 from source (build_overlay3.sh, SHAs pinned inside)
#   4  overlay4 = + prebuilt mxfp8_gemm_cutlass_sm120 (build_overlay4.sh)
#   5  overlay5 = + sparse_mla_sm120 rebuilt under the runtime env + GPU verify (build_overlay5.sh)  <- needs a FREE GPU
# Budget: ~2-3 h on one GB10. Needs ~60 GiB free in /var/lib/docker and MemAvailable >= 60 GiB (stage 1 is cmake -j 20).
# Idempotent: each stage is skipped when its image tag already exists (pass FORCE=1 to rebuild everything).
set -euo pipefail
R="${RECIPE_DIR:-$HOME/dsv41-recipe}"
SRC="${SRC_DIR:-/data/v41build/src}"
BASE_TAG="vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c"
BASE_DIGEST="sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0"   # arm64 manifest, checked 2026-09-11
BRANCH_SHA="${VLLM_BRANCH_SHA:-e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba}"   # the commit Tony's patches target; dsv41-feat was force-pushed
                                                                        # 2026-09-11 00:49 UTC, merged (#56214) and deleted 09:11 UTC. Fetch by sha.
LOG="${LOG:-$HOME/dsv41-build-$(date -u +%Y%m%dT%H%M%SZ).log}"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s); stamp(){ echo "[$(date -u +%H:%M:%SZ) +$(( ($(date +%s)-t0)/60 ))m] $*"; }
have(){ docker image inspect "$1" >/dev/null 2>&1; }

stamp "build_all start host=$(hostname) recipe=$R ($(git -C "$R" rev-parse --short HEAD)) log=$LOG"
AV=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
[ "$AV" -ge 60 ] || { echo "MemAvailable ${AV} GiB < 60 GiB: is a vLLM job still running on this node? refusing." >&2; exit 4; }
docker ps --format '{{.Names}}' | grep -qE '^vllm_' && { echo "a vllm_ container is running on this node; stop it first (tools/stop_all.sh)." >&2; exit 4; }

# ---------- stage 1: overlay1 ----------
if [ "${FORCE:-0}" = 1 ] || ! have vllm-dsv41:overlay1; then
  stamp "stage 1: pull base by digest"
  docker pull "vllm/vllm-openai@$BASE_DIGEST"
  docker tag "vllm/vllm-openai@$BASE_DIGEST" "$BASE_TAG"
  if [ ! -d "$SRC/.git" ]; then
    stamp "stage 1: clone vllm (default branch; the pinned commit is fetched by sha)"
    mkdir -p "$(dirname "$SRC")"; git clone https://github.com/vllm-project/vllm.git "$SRC"
  fi
  git -C "$SRC" fetch origin "$BRANCH_SHA" && git -C "$SRC" checkout -q "$BRANCH_SHA"
  echo "vllm @ $(git -C "$SRC" rev-parse HEAD)" | tee "$SRC/.built-from"
  stamp "stage 1: start v41build container (base image + git/cmake/ninja)"
  docker rm -f v41build >/dev/null 2>&1 || true
  docker run -d --name v41build --gpus all --network host --memory "${BUILD_MEM:-96g}" \
    -v "$SRC:/src" --entrypoint sleep "$BASE_TAG" infinity >/dev/null
  docker exec v41build bash -c 'set -e; (command -v git && command -v cmake && command -v ninja) >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git >/dev/null; pip install -q cmake ninja; }; ls /usr/local/cuda/bin/nvcc; git config --global --add safe.directory /src'
  stamp "stage 1: build _C_stable_libtorch for sm_121a (cmake -j 20, ~30-60 min)"
  bash "$R/build/build_stable_ext.sh" | tail -15
  SO=$(docker exec v41build bash -c 'ls /src/build/_C_stable_libtorch*.so 2>/dev/null | head -1')
  [ -n "$SO" ] || { echo "stage 1 FAILED: no _C_stable_libtorch*.so produced (see $LOG)" >&2; exit 1; }
  stamp "stage 1: assemble overlay context (python tree + extension) and docker build"
  CTX=$(mktemp -d /var/tmp/ov1.XXXX); cp "$R/build/Dockerfile.overlay" "$CTX/Dockerfile"
  cp -r "$SRC/vllm" "$CTX/vllm"; docker cp "v41build:$SO" "$CTX/vllm/"
  find "$CTX/vllm" -name __pycache__ -type d -prune -exec rm -rf {} +
  ls -la "$CTX/vllm/"_C_stable_libtorch*.so
  DOCKER_BUILDKIT=1 docker build -t vllm-dsv41:overlay1 "$CTX" | tail -5
  rm -rf "$CTX"; docker rm -f v41build >/dev/null
  stamp "stage 1 done: $(docker images vllm-dsv41:overlay1 --format '{{.Size}}')"
else stamp "stage 1: overlay1 present, skipping"; fi

# ---------- stage 3: overlay3 (FlashInfer 0.7.0rc1) ----------
if [ "${FORCE:-0}" = 1 ] || ! have vllm-dsv41:overlay3; then
  stamp "stage 3: FlashInfer 0.7.0rc1 from source + sparse-MLA prewarm (~30-45 min)"
  bash "$R/build/build_overlay3.sh"; have vllm-dsv41:overlay3 || { echo "stage 3 FAILED (see /var/tmp/v41overlay3/build.err)" >&2; exit 1; }
else stamp "stage 3: overlay3 present, skipping"; fi

# ---------- stage 4: overlay4 (mxfp8 gemm prebuilt) ----------
if [ "${FORCE:-0}" = 1 ] || ! have vllm-dsv41:overlay4; then
  stamp "stage 4: prebuild mxfp8_gemm_cutlass_sm120 (MAX_JOBS 4, ~15 min)"
  bash "$R/build/build_overlay4.sh"; tail -3 /tmp/build-overlay4.log; have vllm-dsv41:overlay4 || { echo "stage 4 FAILED (see /tmp/build-overlay4.log)" >&2; exit 1; }
else stamp "stage 4: overlay4 present, skipping"; fi

# ---------- stage 5: overlay5 (sparse_mla under runtime env + GPU verify) ----------
if [ "${FORCE:-0}" = 1 ] || ! have vllm-dsv41:overlay5; then
  stamp "stage 5: rebuild sparse_mla_sm120 under the runtime env + verify (needs the GPU free, ~20 min)"
  cp "$R/build/prewarm5.py" "$R/build/verify5.py" /tmp/
  bash "$R/build/build_overlay5.sh"; tail -6 /tmp/build-overlay5b.log; have vllm-dsv41:overlay5 || { echo "stage 5 FAILED (see /tmp/build-overlay5b.log)" >&2; exit 1; }
  grep -q "VERIFY mxfp8: HIT" /tmp/build-overlay5b.log && grep -q "VERIFY sparse_mla: HIT" /tmp/build-overlay5b.log \
    || echo "WARNING: verify5 did not report HIT for both kernels; first request will JIT (slow but not fatal)"
else stamp "stage 5: overlay5 present, skipping"; fi

docker images 'vllm-dsv41' --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
stamp "build_all done: vllm-dsv41:overlay5 id=$(docker image inspect vllm-dsv41:overlay5 --format '{{.Id}}' | cut -c8-19)"
