#!/usr/bin/env bash
# status.sh -- one line per node: MemAvailable, vllm containers, model shards on disk, image present.
J="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
for r in 0 1 2 3 4 5 6 7; do
  printf "rank %d  %s\n" "$r" "$($J "$USER_@$PREFIX.$((10 + r))" 'printf "avail=%3dG  " $(( $(grep MemAvailable /proc/meminfo | awk "{print \$2}") / 1048576 )); printf "ctr=%-22s " "$(docker ps --format "{{.Names}}" | tr "\n" "," | sed "s/,$//")"; printf "shards=%2d  " "$(ls /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/*.safetensors 2>/dev/null | wc -l)"; printf "overlay5=%s" "$(docker image inspect vllm-dsv41:overlay5 --format "{{.Id}}" 2>/dev/null | cut -c8-19 || echo no)"' 2>&1)"
done
