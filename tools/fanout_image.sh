#!/usr/bin/env bash
# fanout_image.sh -- copy vllm-dsv41:overlay5 from the build node to the other seven over the fabric.
# docker save -> one tar on the build node's /data, served over HTTP on the fabric IP, each target curls + docker loads.
# Run from the operator machine while nothing is serving. ~35 GB x 7 over 200G RoCE: a few minutes each.
set -u
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
SRC_RANK="${SRC_RANK:-2}"; SRC_IP="$PREFIX.$((10 + SRC_RANK))"; HTTP_PORT=8891
IMG="${IMG:-vllm-dsv41:overlay5}"; TAR="/data/models/.images/$(echo "$IMG" | tr ':/' '__').tar"
TARGETS="${TARGETS:-$(for r in 0 1 2 3 4 5 6 7; do [ "$r" != "$SRC_RANK" ] && printf "%s " "$r"; done)}"   # every rank but the source
echo "== source rank $SRC_RANK: docker save $IMG -> $TAR"
$J "$USER_@$SRC_IP" "mkdir -p /data/models/.images; docker save -o $TAR $IMG; ls -la $TAR; sha256sum $TAR | cut -c1-64 > $TAR.sha256; cat $TAR.sha256"
$J "$USER_@$SRC_IP" "systemctl --user stop dsv41-http 2>/dev/null; systemctl --user reset-failed dsv41-http 2>/dev/null; systemd-run --user --unit dsv41-http --collect -p MemoryMax=1G python3 -m http.server $HTTP_PORT --bind $SRC_IP --directory /data/models/.images >/dev/null && echo '  serving on http://$SRC_IP:$HTTP_PORT'"
sleep 2
B=$(basename "$TAR")
for r in $TARGETS; do
  h="$USER_@$PREFIX.$((10 + r))"
  echo "== rank $r: pull + docker load"
  $J "$h" "set -e; mkdir -p /data/models/.images; cd /data/models/.images; rm -f $B $B.sha256; curl -sSf -o $B.sha256 http://$SRC_IP:$HTTP_PORT/$B.sha256; curl -sSf --retry 5 -o $B http://$SRC_IP:$HTTP_PORT/$B; echo \"\$(cat $B.sha256)  $B\" | sha256sum -c - && docker load -i $B | tail -1 && docker images $IMG --format '  loaded {{.Repository}}:{{.Tag}} {{.ID}}'" &
done
wait
$J "$USER_@$SRC_IP" "systemctl --user stop dsv41-http" && echo "source HTTP server stopped"
echo "== image ids on all eight (must be identical)"; for r in 0 1 2 3 4 5 6 7; do echo "  rank $r: $($J "$USER_@$PREFIX.$((10 + r))" "docker image inspect $IMG --format '{{.Id}}' 2>/dev/null | cut -c8-19")"; done
