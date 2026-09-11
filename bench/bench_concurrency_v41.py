#!/usr/bin/env python3
"""Concurrency scaling for the 8-node TP=8 GLM-5.3-FP8 deployment.

Reports BOTH numbers that matter and are usually conflated:
  per-stream tok/s  - what one user feels (latency)
  aggregate tok/s   - what the cluster delivers (throughput)

Uses a realistic coding prompt (not a predictable one) so MTP draft acceptance
reflects real work. Distinct prompts per worker so prefix caching cannot
flatter the result.
"""
import json, time, urllib.request, statistics, sys
from concurrent.futures import ThreadPoolExecutor

URL = "http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
MAXTOK = 500

TASKS = [
    "Write a Python function that parses an nginx access log line into a dict. "
    "Handle quoted fields with spaces and escaped quotes. Type hints + 3 unit tests.",
    "Implement a thread-safe LRU cache in Go with generics, O(1) get and put, "
    "and a benchmark. Explain the eviction invariant.",
    "Write a Rust function that merges N sorted iterators into one sorted stream "
    "using a binary heap. Include error handling and tests.",
    "Implement rate limiting via token bucket in TypeScript, with burst support "
    "and monotonic clock. Include tests for clock skew.",
    "Write a SQL query and supporting indexes to find the 95th percentile request "
    "latency per endpoint per hour, and explain the query plan.",
    "Implement binary search over a rotated sorted array in C, handling duplicates, "
    "with a proof sketch of the loop invariant.",
    "Write a Python asyncio worker pool with backpressure, graceful shutdown, "
    "and per-task timeout. Include tests.",
    "Implement Dijkstra with a Fibonacci heap in Java. Explain the amortised bounds.",
]


def one(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXTOK, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt


def run_level(n):
    prompts = TASKS[:n]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        out = list(ex.map(one, prompts))
    wall = time.time() - t0
    toks = sum(c for c, _ in out)
    per = [c / dt for c, dt in out]
    return {"n": n, "wall_s": round(wall, 1), "total_tokens": toks,
            "aggregate_tok_s": round(toks / wall, 2),
            "per_stream_median": round(statistics.median(per), 2),
            "per_stream_min": round(min(per), 2),
            "per_stream_max": round(max(per), 2)}


levels = [int(x) for x in (sys.argv[1:] or ["1", "2", "4", "8"])]
print(f"{'conc':>5}{'wall_s':>9}{'tokens':>9}{'aggregate':>11}{'per-stream (med)':>19}{'min':>8}{'max':>8}")
print("-" * 69)
rows = []
for n in levels:
    r = run_level(n)
    rows.append(r)
    print(f"{r['n']:>5}{r['wall_s']:>9}{r['total_tokens']:>9}"
          f"{r['aggregate_tok_s']:>11}{r['per_stream_median']:>19}"
          f"{r['per_stream_min']:>8}{r['per_stream_max']:>8}")
    time.sleep(5)

base = rows[0]["aggregate_tok_s"]
print()
for r in rows:
    print(f"  conc={r['n']}: aggregate {r['aggregate_tok_s']:>6} tok/s "
          f"({r['aggregate_tok_s']/base:.2f}x vs single)  "
          f"per-stream {r['per_stream_median']:.2f} tok/s")
json.dump(rows, open("/tmp/bench-concurrency.json", "w"), indent=2)
