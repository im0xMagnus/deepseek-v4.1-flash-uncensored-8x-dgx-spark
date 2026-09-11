# DeepSeek-V4.1-Flash (dealignai UNCENSORED-FP8) on 8x NVIDIA DGX Spark, TP8 with vLLM

Eight GB10 nodes on a 200G RoCE fabric, tensor parallel 8, DSpark speculative decoding, CUDA graphs, vision and tool
calling on, 300K then 1M context. Companion to
[glm-5.3-uncensored-8x-dgx-spark](https://github.com/im0xMagnus/glm-5.3-uncensored-8x-dgx-spark) (the job this
cluster runs today).

STATUS 2026-09-11: PRE-BOOT. Nothing in this tree has been booted at TP8 yet. Every number below that is
not marked "measured (Tony, TP4)" is an estimate, to be replaced by the profile line and the benches on boot A.

This is tonyd2wild's four-node recipe (Tech2Wild / Kai, boot 10, repo pinned 592540c6) carried to eight nodes.
Only the eight-rank deltas changed; the model-side fixes (seven bind-mounted patches, Engram-on-disk, DSpark k=5,
exact-size CUDA graphs, block size 128) are his, unmodified. Read his RECIPE.md and README boot log first.

## Placeholders

Every script is real and runs as-is on our cluster with two substitutions, done once with sed:

- `NODE_PREFIX_PLACEHOLDER` -- the first three octets of the RoCE fabric subnet (rank N lives at `PREFIX.1N`, head at `PREFIX.10`).
- `USER_PLACEHOLDER` -- the unix user with BatchMode ssh to every node from the operator machine (or export `SPARK_USER`).

The fallback job (what `tools/safety_net.sh` relaunches and `tools/stop_all.sh --glm` stops) is ours; point
`GLM_BOOT` and `PREV_JOB` at yours or ignore them.

## Layout (staged at ~/dsv41-tp8 on every node; the tools/ scripts run from the operator machine)

| path | what |
|---|---|
| launch/dsv41-tp8.sh RANK | one rank. Derived from his launch/dsv41-tp4.sh; deltas listed in the header. Defaults = boot 10. |
| launch/boot_dsv41_tp8.sh | fan out one launch: ranks 7..1 then head 0, identical knobs (his boot_dsv41.sh) |
| launch/bootA-go.sh | first boot: 300K context, everything else boot 10 |
| launch/bootB-go.sh | second boot: 1,048,576 context, nothing else changed |
| build/build_all.sh | builds vllm-dsv41:overlay5 on one node from his build/ chain (stage 1 setup reconstructed) |
| tools/stop_all.sh [--glm] | stop V4.1 (and GLM) containers on all 8, head first, logs saved |
| tools/fanout_weights.sh | 475 GiB from the download node to the other 7 over the fabric (HTTP, 4 streams per target) |
| tools/verify_weights.sh | sizes + sha256 against the pinned-revision manifest (.manifest: path, size, lfs sha) |
| tools/fanout_image.sh | docker save on the build node, HTTP over the fabric, docker load on the other 7 |
| tools/gpu_burn_check.sh | 15 s fp16 burn on all 8: catches a clock-latched GB10 before it costs a boot |
| tools/wait_ready.sh | poll the head until /v1/models answers; prints KV pool and concurrency lines |
| tools/status.sh | one line per node: MemAvailable, containers, shards on disk, image id |
| tools/safety_net.sh | if V4.1 is not healthy, relaunch the GLM 512K job (for 05:30 on the decision night) |

Pins: model dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 @ 81ffd1ef (48 shards, 475 GiB, native FP8 dense + MXFP4
experts); recipe tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark @ 592540c6 cloned to ~/dsv41-recipe on every node;
vLLM base vllm/vllm-openai:nightly-8a728663 (arm64 digest in build_all.sh) + branch dsv41-feat (sha recorded at
build time in /data/v41build/src/.built-from); FlashInfer 0.7.0rc1 @ 07869c61. Patch md5s (patch/mounts.txt order):
sparse_swa cc419353, attention da9ef196, flashinfer_sparse af0f8447, engram c0329107, weight_utils 7e1027f1,
model_state 0a14bee6, sparse_attn_indexer a9b73756.

## What differs from Tony's TP4 (and nothing else)

- 8 ranks: rank N = NODE_PREFIX_PLACEHOLDER.1N; head .10 serves the API on :8888 (same port the GLM job used, so the client
  config only changes the model id); rendezvous port 29551.
- NCCL: both HCAs (rocep1s0f0,roceP2p1s0f0), RoCEv2 GID index probed per node, NCCL_CROSS_NIC=1, 16 MiB buffers,
  addr range NODE_PREFIX_PLACEHOLDER.0/24. Same env the GLM TP8 job runs on.
- Weights node-local on every rank (no NFS), so ENGRAM_LOCAL=0: engram.py reads this rank's rows straight from
  shards 47/48 on local NVMe. His tools/engram_local.py step is not needed.
- --tensor-parallel-size 8 --nnodes 8; EP=1 knob adds --enable-expert-parallel (gate for the 2304/8 MoE question).
- sudo -n for the page-cache drop (our sudoers rule), preflight refuses to boot with fewer than 48 shards.

## Service window (start when the GLM job can go down; ~4.5 h to the 1M decision)

Preconditions, no downtime: download complete on .12 and tools/verify_weights.sh clean there; ~/dsv41-recipe on all 8;
this tree on all 8; nothing else scheduled to touch the cluster overnight.

| T+ | step | command (operator machine unless noted) | gate |
|---|---|---|---|
| 0:00 | GLM down on all 8 | bash tools/stop_all.sh --glm | 0 containers, MemAvailable > 100 GiB everywhere |
| 0:02 | image build on .12 | on .12: systemd-run --user --unit dsv41-build --collect bash ~/dsv41-tp8/build/build_all.sh ; journalctl --user -u dsv41-build -f | 2-3 h; if not done by T+3:00 relaunch GLM and finish next window |
| 0:03 | weights to the other 7 | bash tools/fanout_weights.sh (SRC_RANK=2) | DONE 48/48 on every target, ~30-40 min |
| 0:45 | size check on every node | for r in 0 1 3 4 5 6 7: ssh rank "SKIP_SHA=1 bash ~/dsv41-tp8/tools/verify_weights.sh" | 0 problems |
| ~2:30 | image to the other 7 | bash tools/fanout_image.sh | identical image id on all 8 |
| ~2:45 | GPU clock check | bash tools/gpu_burn_check.sh | every rank ~2 GHz+ / 80 W+ under load |
| ~2:50 | boot A (300K) | bash launch/bootA-go.sh ; bash tools/wait_ready.sh | serves within ~25 min; KV pool and "Maximum concurrency" lines |
| ~3:15 | first numbers | smoke, bench_decode_real, bench_concurrency C1/C2/C4/C6, needle 200K, /metrics DSpark acceptance | decode and needle sane |
| ~4:00 | boot B (1M) | bash tools/stop_all.sh ; bash launch/bootB-go.sh ; wait_ready | needle 500K and 900K pass; else back to boot A |
| ~4:45 | decision | leave V4.1 up and repoint DSH, or the fallback job's boot script (ours: GLM 512K, 17 min) | |
| 05:30 | safety net | bash tools/safety_net.sh (scheduled) | a model is serving in the morning |

Rollback at any gate: bash tools/stop_all.sh then the GLM boot script (ours: the GLM 512K job from the companion repo: ranks 7..1 then 0,
the previous job's per-node launcher). The GLM image and weights stay on every node; nothing here deletes them.

## Knobs (export before a go-script; the go-scripts set the boot-10 values)

GMU 0.80 (host-headroom rule; do not chase), MAXLEN 300000 / 1048576, SEQS 8 (raise to 12-16 once stable),
MAX_BATCHED 8192, SPEC dspark SPEC_K 5, SPEC_ADAPT false (FlashInfer #5015), EAGER 0 with FULL_AND_PIECEWISE graphs
at sizes = multiples of 5 and 6 up to 48, ENGRAM_DISK 1, ENGRAM_LOCAL 0, TEXT_ONLY 0 (vision on, 4 images/prompt),
PARSERS 1 (deepseek_v41 tool + reasoning parsers), THINKING false (per-request chat_template_kwargs turns it on),
EP 0, --block-size 128 (required). Container: --memory 112g --memory-swap 112g --shm-size 32g, MAX_JOBS 2.

## Expected (derived from Tony's TP4 measurements; replace on boot A)

Per rank: ~59 GiB weights as shipped (~41 GiB with Engram left on disk), KV ~33 GiB at gmu 0.80 (~6M tokens of pool
at ~5 KB/token/rank, so 1M context with 5-6 concurrent 1M sequences is the arithmetic, not the promise).
Decode: code 90-110 tok/s at C1 (TP4 measured 73.8), ~200 aggregate at C6-C8 (TP4 measured 131.9 at C6), prose 30-35.
Prefill 1.5-2.5K tok/s (TP4 measured 0.9-1.5K). DSpark acceptance ~3.5 tokens per step (TP4 measured 3.57).

## Speed levers, checked 2026-09-11

- No DFlash, DFlash2 or EAGLE draft exists for DeepSeek-V4.1-Flash (HF search: 33 repos, the only draft-like ones are
  MLX/GGUF MTP variants). The vLLM dsv41-feat branch exposes DSpark only (the checkpoint's own draft layers, k=5 is the
  trained block size). DSpark is already on in boot 10; it is the lever.
- Adaptive verification (bigger effective k on easy tokens) is blocked by FlashInfer #5015 (padded rows hang SM120
  sparse MLA); still open. Re-check before each rebuild.
- After boot A, one per boot: EP=1 (EP8 gives each rank 48 whole experts; try if the MoE at 2304/8 trips or decode is
  below estimate), SEQS 12-16, MAX_BATCHED 16384 for prefill, head-node host headroom under 4 sessions.

## What this repo will add once it has booted

Measured TP8 numbers (decode per stream by category at C1, aggregate at C2-C8, TTFT, prefill at 32K-1M, needle at
200K/500K/900K, DSpark acceptance from /metrics, host headroom under load), the profile lines from boot A and boot B,
and every failure mode the eight-rank step adds to Tony's boot log. Until then this is a staged plan, not a result.

## Credits

- **tonyd2wild / Tech2Wild (Kai)** ([@tonyd2wild](https://github.com/tonyd2wild), [huggingface.co/Tech2wild](https://huggingface.co/Tech2wild), [@Tech2Wild on X](https://x.com/Tech2Wild)) -- the whole recipe: [DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark). The seven patches (Engram on disk with the rank-offset fix, CUDA-graph prestage, SM12x page sizes, the top-k kernel choice), the five-stage image chain, the exact-size CUDA graph rule for DSpark, block size 128, the bench and needle tools, and ten boots of failure analysis. This repo changes only what eight nodes need.
- **dealignai** ([huggingface.co/dealignai](https://huggingface.co/dealignai)) -- the [UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8) edit of the checkpoint, ablated at the FP8 quant level, day-0.
- **DeepSeek** ([deepseek-ai](https://huggingface.co/deepseek-ai)) -- V4.1 Flash: 552B backbone plus 196B Engram, 16B active, native FP8 dense and MXFP4 experts, DSpark draft layers in the checkpoint, 1M context.
- **vllm-project** -- the `dsv41-feat` branch (day-0 support) and the nightly this image chain starts from; **flashinfer-ai** -- 0.7.0rc1 with the SM120 sparse-MLA decode kernels.
