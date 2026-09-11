#!/usr/bin/env bash
# verify_weights.sh -- on a node: every file's size, then every shard's sha256, against the pinned-revision manifest
# (.manifest = "path<TAB>size<TAB>lfs-sha256"). Runs sha256 four at a time at low priority; ~475 GiB takes ~10 min.
set -u
D="${1:-/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8}"; cd "$D" || exit 2
bad=0
while IFS=$'\t' read -r p s h; do
  [ -f "$p" ] || { echo "MISSING $p"; bad=$((bad+1)); continue; }
  have=$(stat -c %s "$p"); [ "$have" = "$s" ] || { echo "SIZE $p have=$have want=$s"; bad=$((bad+1)); }
done < .manifest
echo "size check: $bad problems"
[ "${SKIP_SHA:-0}" = "1" ] && exit $bad
grep $'\.safetensors\t' .manifest | cut -f1,3 | tr '\t' ' ' | xargs -P 4 -n 2 sh -c 'h=$(nice -n 19 sha256sum "$0" | cut -c1-64); [ "$h" = "$1" ] && echo "sha ok $0" || echo "SHA-MISMATCH $0"' | tee .verify.log
m=$(grep -c "SHA-MISMATCH" .verify.log); echo "sha256: $(grep -c 'sha ok' .verify.log) ok, $m mismatches"; exit $((bad + m))
