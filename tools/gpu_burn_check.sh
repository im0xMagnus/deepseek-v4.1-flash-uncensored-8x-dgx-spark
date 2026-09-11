#!/usr/bin/env bash
# gpu_burn_check.sh -- 15 s fp16 burn on all 8 GPUs at once (from tonyd2wild tools/recover.sh): a GB10 can latch
# below 1 GHz with no visible cause and only a power-cycle clears it. Healthy: ~2.2-2.4 GHz, 80 W+, 75-90 TFLOPS.
# Latched: ~715 MHz / ~18 W. Run with the GPUs free (nothing serving), before boot A.
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none"
USER_="${SPARK_USER:-USER_PLACEHOLDER}"; PREFIX="${NODE_PREFIX:-NODE_PREFIX_PLACEHOLDER}"
IMG="${IMG:-vllm-dsv41:overlay5}"; FALLBACK="ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision"
read -r -d '' BURN <<'R'
IMG="$1"; docker image inspect "$IMG" >/dev/null 2>&1 || IMG="$2"
( docker run --rm --gpus all --network none --entrypoint python3 "$IMG" -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 15:
    c = a @ b; n += 1
torch.cuda.synchronize(); print(f'fp16 {2*4096**3*n/(time.time()-t0)/1e12:.1f} TFLOPS')
" 2>&1 | grep TFLOPS ) &
sleep 12; s=$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader | tr '\n' ' '); wait
echo "under load: $s"
R
for r in 0 1 2 3 4 5 6 7; do
  ( echo "rank $r: $($J "$USER_@$PREFIX.$((10 + r))" "bash -s -- $IMG $FALLBACK" <<< "$BURN" 2>&1 | tr '\n' ' ')" ) &
done
wait
echo "healthy = ~2000+ MHz and 80 W+ under load on every rank; a ~715 MHz / ~18 W rank needs a power-cycle before boot"
