#!/usr/bin/env bash
# boot_dsv41_tp8.sh -- fan out one launch over the 8 nodes with identical knobs: workers 7..1 first, head 0 last.
# Run from the operator machine (needs BatchMode ssh to every node). Mirrors tonyd2wild launch/boot_dsv41.sh.
# Usage: export knobs (see dsv41-tp8.sh) then: bash boot_dsv41_tp8.sh
set -u
J="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
LAUNCHER="${LAUNCHER:-\$HOME/dsv41-tp8/launch/dsv41-tp8.sh}"
KNOBS=""
for k in DRAFT_SAMPLE NCCL_TUNED SERVED_NAMES TP BASE PORT MPORT EXP_NAME IMAGE GMU MAXLEN SEQS MAX_BATCHED EAGER CUDAGRAPH_MODE CG_SIZES SPEC SPEC_K SPEC_ADAPT ENGRAM_DISK ENGRAM_LOCAL ENGRAM_THREADS ENGRAM_CHUNK TEXT_ONLY THINKING PARSERS EP RUST_FE PATCH_DIR VLLM_EXTRA NCCL_EXTRA; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS $k='$v'"
done
echo "knobs:$KNOBS"
TP="${TP:-8}"; BASE="${BASE:-0}"
run() { # rank
  local r="$1"; local host="$USER_@$PREFIX.$((10 + BASE + r))"
  echo "== rank $r $host =="; $J "$host" "export $KNOBS; bash $LAUNCHER $r" 2>&1 | tail -3
}
set -e
for r in $(seq $((TP - 1)) -1 1); do run "$r"; done
sleep 5
run 0
echo "all eight launched $(date +%FT%T)"
