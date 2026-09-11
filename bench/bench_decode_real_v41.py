#!/usr/bin/env python3
"""Honest decode benchmark: MTP acceptance depends heavily on how predictable
the output is. "Count from 1 to 300" is a degenerate best case. Measure across
a spread of realistic workloads instead, and report each separately.
"""
import json, time, urllib.request, statistics, sys

URL = "http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"

PROMPTS = {
    "counting (degenerate best case)":
        "Count from 1 to 300, one number per line.",
    "code: write a real function":
        "Write a Python function that parses an nginx access log line into a dict "
        "(remote_addr, time_local, method, path, status, bytes_sent, referer, "
        "user_agent). Handle quoted fields containing spaces and escaped quotes. "
        "Include docstring, type hints, and three unit tests.",
    "code: refactor/explain":
        "Explain how a Bloom filter works, derive the false-positive rate formula, "
        "then implement one in Rust with add/contains and a sizing helper.",
    "prose: open-ended reasoning":
        "Explain the tradeoffs between tensor parallelism and pipeline parallelism "
        "for LLM inference on a cluster of small unified-memory nodes. Be specific "
        "about which is better for decode latency versus aggregate throughput.",
}


def run(prompt, max_tokens=700):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    c = d["usage"]["completion_tokens"]
    return c, dt, c / dt if dt else 0


print(f"{'workload':<36}{'tok':>6}{'sec':>8}{'tok/s':>9}")
print("-" * 59)
results = {}
for label, p in PROMPTS.items():
    rates = []
    for _ in range(2):
        c, dt, r = run(p)
        rates.append(r)
    med = statistics.median(rates)
    results[label] = round(med, 2)
    print(f"{label:<36}{c:>6}{dt:>8.1f}{med:>9.2f}")

print()
vals = list(results.values())
real = [v for k, v in results.items() if not k.startswith("counting")]
print(f"degenerate best case : {results['counting (degenerate best case)']:.2f} tok/s")
print(f"realistic workloads  : {min(real):.2f} - {max(real):.2f} tok/s "
      f"(median {statistics.median(real):.2f})")
json.dump(results, open("/tmp/bench-decode-real.json", "w"), indent=2)
