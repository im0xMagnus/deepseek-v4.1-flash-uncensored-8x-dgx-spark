#!/usr/bin/env bash
# safety_net.sh -- run from the operator machine (the only host with ssh to all 8): if the V4.1 API on the head is not
# healthy, stop whatever is up and relaunch the GLM 512K job so the morning has a model. Schedule for 05:30 on the
# decision night, e.g.:  nohup bash -c 'sleep $(( $(date -d "05:30" +%s 2>/dev/null || echo 0) - $(date +%s) )); bash tools/safety_net.sh' &
# (macOS: use a launchd StartCalendarInterval or an in-session cron; there is no at(1).)
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"; HEAD="$USER_@$PREFIX.10"
GLM_BOOT="${GLM_BOOT:-$(dirname "$0")/../../boot_previous_job.sh}"   # point GLM_BOOT at the boot script of whatever job you fall back to
if $J "$HEAD" 'curl -sf -m 10 http://127.0.0.1:8888/v1/models' 2>/dev/null | grep -q deepseek-v4.1-flash; then
  echo "$(date +%FT%T) V4.1 healthy on the head; nothing to do"; exit 0
fi
echo "$(date +%FT%T) V4.1 NOT healthy: stopping vllm_dsv41 everywhere and relaunching GLM 512K via $GLM_BOOT"
bash "$(dirname "$0")/stop_all.sh"
[ -f "$GLM_BOOT" ] || { echo "GLM boot script missing at $GLM_BOOT"; exit 1; }
bash "$GLM_BOOT"
