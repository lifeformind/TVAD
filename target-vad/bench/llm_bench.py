#!/usr/bin/env python3
"""Benchmark a GGUF model served by llama_cpp.server: TTFT + generation tok/s.

Launches its own server on a dedicated port, runs a few representative
voice-assistant prompts with streaming, measures time-to-first-token and
generation throughput, then tears the server down.

Usage:
  python3 bench/llm_bench.py --model /path/to.gguf --label "Gemma 3 12B" --chat-format gemma
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

SYSTEM_PROMPT = (
    "You are a concise voice assistant. Replies should be 1-3 sentences, "
    "natural-sounding, and avoid lists, code blocks, or markdown."
)
PROMPTS = [
    "What's a good way to stay focused while working from home?",
    "Can you suggest a quick healthy breakfast?",
    "Briefly, why is the sky blue?",
]


def wait_ready(base, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(1.0)
    return False


def measure(base, user_msg, max_tokens):
    """One streaming request. Returns (ttft_s, gen_tok_s, completion_tokens, total_s, text)."""
    payload = {
        "model": "bench",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    n_chunks = 0
    completion_tokens = None
    pieces = []
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            break
        obj = json.loads(body)
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
                pieces.append(piece)
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens"):
            completion_tokens = usage["completion_tokens"]
    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else float("nan")
    gen_window = (t_end - t_first) if t_first else float("nan")
    toks = completion_tokens if completion_tokens else n_chunks
    gen_tok_s = (toks / gen_window) if gen_window and gen_window > 0 else float("nan")
    return ttft, gen_tok_s, toks, (t_end - t_start), "".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--chat-format", default=None, help="omit to use the GGUF embedded template")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ready-timeout", type=int, default=180)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--n_ctx", str(args.n_ctx),
        "--n_gpu_layers", str(args.n_gpu_layers),
    ]
    if args.chat_format:
        cmd += ["--chat_format", args.chat_format]

    print(f"\n=== {args.label} ===", flush=True)
    print(f"model: {args.model}", flush=True)
    log = open(f"/tmp/bench_{args.port}.log", "w")
    t_load0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(base, args.ready_timeout):
            print("ERROR: server never became ready; see log", flush=True)
            return
        load_s = time.perf_counter() - t_load0
        print(f"model load + ready: {load_s:.1f}s", flush=True)

        # warmup (not measured)
        try:
            measure(base, "Say hello in one short sentence.", 32)
        except Exception as e:
            print(f"warmup failed: {e}", flush=True)

        ttfts, rates = [], []
        for p in PROMPTS:
            ttft, rate, toks, total, text = measure(base, p, args.max_tokens)
            ttfts.append(ttft)
            rates.append(rate)
            print(f"  TTFT {ttft*1000:6.0f} ms | {rate:5.1f} tok/s | {toks:3d} tok | {total:4.1f}s | "
                  f"{text[:60].replace(chr(10),' ')!r}", flush=True)

        def mean(xs):
            xs = [x for x in xs if x == x]  # drop NaN
            return sum(xs) / len(xs) if xs else float("nan")

        print(f"  -> mean TTFT {mean(ttfts)*1000:.0f} ms | mean {mean(rates):.1f} tok/s", flush=True)
        # machine-readable summary line
        print(f"SUMMARY\t{args.label}\t{load_s:.1f}\t{mean(ttfts)*1000:.0f}\t{mean(rates):.1f}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
