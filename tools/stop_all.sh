#!/usr/bin/env bash
# stop_all.sh -- stop every V4.1 container (and, with --glm, the GLM job) on all 8 nodes, head first, saving logs.
# Mirrors tonyd2wild tools/launch.sh's stop phase: a new worker must never join a still-live old head's rendezvous.
set -u
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
NAMES="vllm_dsv41"; [ "${1:-}" = "--glm" ] && NAMES="vllm_dsv41 ${PREV_JOB:-vllm_previous_job}"
P="${LOGDIR:-$HOME/.local/state/dsv41/stop-$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$P"
for r in 0 1 2 3 4 5 6 7; do
  h="$USER_@$PREFIX.$((10 + r))"
  for n in $NAMES; do
    if $J "$h" "docker ps -a --format '{{.Names}}' | grep -q '^$n\$'" 2>/dev/null; then
      $J "$h" "docker logs -t $n 2>&1 | tail -400" > "$P/rank$r-$n.log" 2>&1
      $J "$h" "docker rm -f $n >/dev/null" && echo "  stopped $n on rank $r (log $P/rank$r-$n.log)"
    fi
  done
done
echo "remaining:"; for r in 0 1 2 3 4 5 6 7; do echo "  rank $r: $($J "$USER_@$PREFIX.$((10 + r))" 'docker ps -q --filter name=vllm_ | wc -l' 2>/dev/null | tr -d " ") containers"; done
