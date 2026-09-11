#!/usr/bin/env bash
# fanout_weights.sh -- distribute the 475 GiB checkpoint from the download node to the other 7 over the 200G fabric.
# No node-to-node ssh keys and no sudo needed: the source serves /data/models over plain HTTP on its fabric IP
# (private RoCE subnet), each target pulls with 4 resumable curl streams, then sizes are checked against the manifest.
# Run from the operator machine ONLY while nothing is serving (rule 9: bulk I/O on a serving rank starves it).
set -u
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
SRC_RANK="${SRC_RANK:-2}"; SRC_IP="$PREFIX.$((10 + SRC_RANK))"; HTTP_PORT=8890
MODEL_DIR="DeepSeek-V4.1-Flash-UNCENSORED-FP8"; D="/data/models/$MODEL_DIR"
TARGETS="${TARGETS:-$(for r in 0 1 2 3 4 5 6 7; do [ "$r" != "$SRC_RANK" ] && printf "%s " "$r"; done)}"   # every rank but the source
echo "== source rank $SRC_RANK ($SRC_IP): start HTTP server on the fabric =="
$J "$USER_@$SRC_IP" "systemctl --user stop dsv41-http 2>/dev/null; systemctl --user reset-failed dsv41-http 2>/dev/null; systemd-run --user --unit dsv41-http --collect -p MemoryMax=1G python3 -m http.server $HTTP_PORT --bind $SRC_IP --directory /data/models >/dev/null && echo '  serving /data/models on http://$SRC_IP:$HTTP_PORT'"
sleep 2
$J "$USER_@$SRC_IP" "curl -sf -m 5 -o /dev/null http://$SRC_IP:$HTTP_PORT/$MODEL_DIR/.manifest && echo '  manifest reachable'" || { echo "source HTTP not reachable"; exit 1; }
# the per-target pull script (runs under a memory cap on the target too)
PULL=$(mktemp -t dsv41-pull.XXXXXX); cat > "$PULL" <<'EOF'
set -u
SRC="$1"; MODEL_DIR="$2"; D="/data/models/$MODEL_DIR"; mkdir -p "$D"
curl -sSf -o "$D/.manifest" "http://$SRC/$MODEL_DIR/.manifest"
fetch(){ local p="$1" want="$2" have=0; [ -f "$D/$p" ] && have=$(stat -c %s "$D/$p"); [ "$have" = "$want" ] && { echo "ok(cached) $p"; return 0; }
  [ "$have" != 0 ] && rm -f "$D/$p"   # partial file: python http.server cannot resume, refetch whole
  curl -sSL --retry 6 --retry-delay 10 --retry-all-errors -o "$D/$p" "http://$SRC/$MODEL_DIR/$p"
  have=$(stat -c %s "$D/$p" 2>/dev/null || echo 0); [ "$have" = "$want" ] && echo "ok $p" || { echo "SIZE-MISMATCH $p have=$have want=$want"; return 1; }; }
export -f fetch; export D SRC MODEL_DIR
grep -v $'\.safetensors\t' "$D/.manifest" | cut -f1,2 | while IFS=$'\t' read -r p s; do fetch "$p" "$s"; done
grep $'\.safetensors\t' "$D/.manifest" | cut -f1,2 | tr '\t' ' ' | xargs -P 4 -n 2 bash -c 'fetch "$0" "$1"'
echo "DONE $(hostname): $(ls $D/*.safetensors | wc -l)/48 shards on disk"
EOF
for r in $TARGETS; do
  h="$USER_@$PREFIX.$((10 + r))"
  scp -q -o BatchMode=yes -o ControlPath=none "$PULL" "$h:dsv41-pull.sh"
  $J "$h" "systemctl --user stop dsv41-pull 2>/dev/null; systemctl --user reset-failed dsv41-pull 2>/dev/null; systemd-run --user --unit dsv41-pull --collect -p MemoryMax=2G -p Nice=10 bash -c 'bash ~/dsv41-pull.sh $SRC_IP:$HTTP_PORT $MODEL_DIR > ~/dsv41-pull.log 2>&1'" && echo "  rank $r: pull started"
done
echo "== progress (Ctrl-C safe; pulls keep running as user services) =="
while :; do
  sleep 30; done_n=0; line=""
  for r in $TARGETS; do
    h="$USER_@$PREFIX.$((10 + r))"; s=$($J "$h" "ls $D/*.safetensors 2>/dev/null | wc -l | tr -d ' '; systemctl --user is-active dsv41-pull 2>/dev/null" 2>/dev/null | tr '\n' ' ')
    line="$line r$r=${s% } "; case "$s" in *inactive*|*failed*) done_n=$((done_n+1));; esac
  done
  echo "$(date +%H:%M:%S) $line"; [ "$done_n" -ge "$(echo $TARGETS | wc -w)" ] && break
done
echo "== results =="; for r in $TARGETS; do $J "$USER_@$PREFIX.$((10 + r))" "grep -E 'DONE|SIZE-MISMATCH' ~/dsv41-pull.log | tail -2 | sed 's/^/  rank $r: /'"; done
$J "$USER_@$SRC_IP" "systemctl --user stop dsv41-http" && echo "source HTTP server stopped"
