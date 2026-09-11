# Boot A, attempt 6: DeepSeek-V4.1-Flash UNCENSORED-FP8, TP8 on 8x DGX Spark, 300K context (2026-09-11 19:35 AEST)

Image: vllm-dsv41:overlay5 id 2a6e20b3b4b4, built from vLLM dsv41-feat @ e47aa780b (Tony's exact commit), FlashInfer 0.7.0rc1 07869c61.
Config: boot 10 knobs + TP8/nnodes 8, gmu 0.77, EP=0, DSpark k=5, FULL_AND_PIECEWISE graphs [5..48], block 128, vision+tools on, thinking off.
Launcher deltas that were needed at eight ranks: --ulimit nofile=1048576 (NCCL "Too many open files"), no NCCL_BUFFSIZE (16 MiB buffers cost ~14 GiB/rank at
init), gmu 0.77 (8-way NCCL leaves 96-97 GiB free at vLLM's startup check; 0.80 = 97.35 refused).

Profile: weights 49.55 GiB loaded (58.7 GiB consumed incl. graphs 1.38+0.57 GiB), KV 31.95 GiB = 6,819,463 tokens, 22.73x concurrency at 300K,
init 207 s, launch-to-serving ~8 min, head idle MemAvailable 6.5 GiB (all ranks 6-9 GiB).

Smoke (Tony's dsv41_smoke.py, C1, best of 2, thinking off): count 71.7 tok/s (1..100 correct), prose 32.1, code 54.1; corruption 0.000% junk, 0 repeated trigrams.
Decode by category (bench_decode_real, C1, 700 tok): counting 98.2, code write 57.2, code refactor 45.8, prose reasoning 37.0 tok/s (realistic median 45.8).
Concurrency (bench_concurrency, 500 tok each, warm): C1 50.6/stream; C2 45.5/stream 81 aggregate; C4 28.1/stream 100 aggregate; C8 16.7/stream 126 aggregate.
DSpark: 1002 drafts, 5010 draft tokens, 1939 accepted = 2.94 tokens/step (Tony TP4 original checkpoint: 3.57).
Needle 200K (v41needle depth 0.5): PASS, prompt 198,737 tokens, TTFT 150.4 s, prefill 1,321 tok/s.
Vision: red 64x64 PNG -> "Red" (201 prompt tokens, 0.7 s). Tools: get_weather("Sydney") called, finish_reason tool_calls. Thinking per request: 17*23 -> 391, 55 reasoning chars.

Reference: Tony TP4 (original DeepSeek checkpoint, boot 10): code 73.8 c1, 131.9 aggregate c6, prefill 0.9-1.5K, KV 1,032,963 tokens at 300K.
Read: eight ranks buy KV pool (6.6x) and prefill, not single-stream decode (8-way all-reduce latency; lower DSpark acceptance on the ablated checkpoint).
