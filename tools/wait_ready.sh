#!/usr/bin/env bash
# wait_ready.sh -- poll the head until /v1/models answers (or a rank dies), printing the head's log tail on failure.
J="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"; HEAD="$USER_@$PREFIX.10"; PORT="${PORT:-8888}"
T0=$(date +%s); LIMIT="${LIMIT:-3600}"
while :; do
  if $J "$HEAD" "curl -sf -m 5 http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "READY after $(( ($(date +%s)-T0)/60 )) min: $($J "$HEAD" "curl -s -m 5 http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin)["data"][0];print(d["id"],"max_model_len",d.get("max_model_len"))')"
    $J "$HEAD" "docker logs vllm_dsv41 2>&1 | grep -E 'KV cache|Maximum concurrency|Engram DISK mode|cudagraph|Capturing' | tail -8"; exit 0
  fi
  dead=$($J "$HEAD" "docker ps -a --filter name=vllm_dsv41 --format '{{.State}}'" 2>/dev/null)
  [ "$dead" = "exited" ] && { echo "HEAD EXITED"; $J "$HEAD" "docker logs --tail 60 vllm_dsv41 2>&1"; exit 1; }
  [ $(( $(date +%s)-T0 )) -gt "$LIMIT" ] && { echo "TIMEOUT"; exit 2; }
  echo "$(date +%H:%M:%S) waiting ($(( ($(date +%s)-T0)/60 )) min): $($J "$HEAD" "docker logs --tail 1 vllm_dsv41 2>&1 | cut -c1-140")"; sleep 60
done
