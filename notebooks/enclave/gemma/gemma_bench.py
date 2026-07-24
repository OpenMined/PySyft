#!/usr/bin/env python3
"""gemma_bench.py — real-weights benchmark for the unoptimized vs optimized Gemma engines.

Downloads actual Gemma 3 weights from Kaggle and runs a set of prompts through:
  --mode unopt : gemma_inference_restrict.py       — one generate() per prompt (serial, current path)
  --mode opt   : gemma_inference_restrict_opt.py   — one generate_batch() over all prompts (jit + batch)
  --mode both  : runs both, prints the speedup, AND checks the optimized engine reproduces the
                 unoptimized one (greedy) — i.e. validates correctness as well as speed.

Assumes gemma_inference_restrict.py and gemma_inference_restrict_opt.py sit in THIS directory.

Requirements:  pip install "jax[cpu]" flax orbax-checkpoint sentencepiece kagglehub
Run:           python gemma_bench.py --size 270m --n-prompts 5 --mode both
               python gemma_bench.py --size 1b --n-prompts 32 --mode opt --batch-size 16
"""

from __future__ import annotations

import argparse
import csv as csvmod
import resource
import sys
import time
from pathlib import Path


def _rss_gb():
    """Current resident memory (GB) from /proc — Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024  # KB -> GB
    except Exception:
        return 0.0


def _peak_gb():
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    )  # ru_maxrss is KB on Linux


def _n_params(cfg):
    V = 262144
    D, F = cfg["embed_dim"], cfg["hidden_dim"]
    H, KVH, hd, L = (
        cfg["num_heads"],
        cfg["num_kv_heads"],
        cfg["head_dim"],
        cfg["num_layers"],
    )
    return V * D + L * (2 * D * H * hd + 2 * D * KVH * hd + 3 * D * F)


# Both engines live next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_inference_restrict as unopt  # noqa: E402
import gemma_inference_restrict_opt as opt  # noqa: E402


# A small pool of short, generic benchmark prompts (~15-25 tokens). Cycled to --n-prompts.
PROMPT_POOL = [
    "Explain how a rainbow forms in a few sentences.",
    "What are three benefits of regular exercise?",
    "Summarize the plot of Romeo and Juliet briefly.",
    "How does a bill become a law in a democracy?",
    "Give me a simple recipe for a vegetable soup.",
    "What is the difference between weather and climate?",
    "Describe how photosynthesis works at a high level.",
    "List four tips for writing clear emails.",
    "Why is the sky blue during the day?",
    "Explain the concept of supply and demand simply.",
    "What causes the seasons to change on Earth?",
    "Give three reasons people learn a second language.",
    "How do vaccines help the body fight disease?",
    "Describe the water cycle in a few steps.",
    "What is compound interest and why does it matter?",
    "Explain what an algorithm is to a beginner.",
    "How does a refrigerator keep food cold?",
    "Give a short overview of how the internet works.",
    "What are the main functions of the human heart?",
    "Explain the difference between a virus and a bacterium.",
]


def load_prompts(args, n):
    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csvmod.DictReader(f))
        prompts = [r[args.csv_col] for r in rows][:n]
        if len(prompts) < n:
            print(
                f"note: CSV has {len(prompts)} rows < requested {n}; using {len(prompts)}"
            )
        return prompts
    return [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(n)]


def load_shared(size, weights_dir):
    """Load weights + tokenizer ONCE (param trees are identical across the two engines)."""
    cfg = unopt.MODEL_CONFIGS[size]
    params = unopt.load_params(weights_dir, cfg)
    tok = unopt.load_tokenizer(weights_dir)
    return cfg, params, tok


def run_unopt(model, params, tok, prompts, max_new_tokens, temperature, top_k):
    per_prompt, completions = [], []
    t_all = time.perf_counter()
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        comp, _ = unopt.generate(
            model,
            params,
            tok,
            p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        dt = time.perf_counter() - t0
        per_prompt.append(dt)
        completions.append(comp)
        print(f"    [{i + 1}/{len(prompts)}] {dt:6.2f}s")
    return completions, per_prompt, time.perf_counter() - t_all


def run_opt(model, params, tok, prompts, max_new_tokens, batch_size):
    completions, first_stats = [], None
    t_all = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        t0 = time.perf_counter()
        comps, stats = opt.generate_batch(
            model, params, tok, chunk, max_new_tokens=max_new_tokens
        )
        first_stats = first_stats or stats
        completions.extend(comps)
        print(
            f"    batch {i // batch_size + 1} ({len(chunk)} prompts): {time.perf_counter() - t0:6.2f}s"
        )
    return completions, time.perf_counter() - t_all, first_stats


def report(label, total, n, max_new_tokens, per_prompt=None):
    print(f"\n=== {label} ===")
    print(f"  total            : {total:8.2f} s  for {n} prompts")
    print(
        f"  per prompt (avg) : {total / n:8.2f} s"
        + ("" if per_prompt else "   (amortized: total ÷ N)")
    )
    if per_prompt:
        print(
            f"  per prompt min/max: {min(per_prompt):.2f} / {max(per_prompt):.2f} s"
            f"   (first prompt includes any compile cost)"
        )
    print(
        f"  throughput       : {n * max_new_tokens / total:8.1f} tok/s  (N × max_new_tokens ÷ total)"
    )
    return total / n


def correctness_check(umodel, omodel, params, tok, prompt, k=24):
    """Greedy on both — a correct optimized engine should reproduce the unoptimized output."""
    print("\n=== correctness check (greedy, same prompt) ===")
    a, _ = unopt.generate(
        umodel, params, tok, prompt, max_new_tokens=k, temperature=0
    )  # temperature=0 -> greedy
    b_list, _ = opt.generate_batch(
        omodel, params, tok, [prompt], max_new_tokens=k
    )  # greedy
    b = b_list[0]
    if a.strip() == b.strip():
        print("  PASS — byte-identical")
        return "PASS (byte-identical)"
    # find common prefix length (chars) as a signal of when/if they diverge
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    print(f"  DIFFER after ~{common} chars.")
    print(f"    unopt: {a[:160]!r}")
    print(f"    opt  : {b[:160]!r}")
    print(
        "  (small late divergence is usually float rounding in the static-cache path;"
    )
    print(
        "   a wildly different or incoherent 'opt' output means a real bug — eyeball above.)"
    )
    return f"DIFFER after ~{common} chars (eyeball above)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="270m", choices=list(unopt.MODEL_CONFIGS))
    ap.add_argument(
        "--n-prompts",
        default="5",
        help="prompt count(s): a single int (5) or a comma list to sweep (1,5,10,20)",
    )
    ap.add_argument("--mode", default="both", choices=["unopt", "opt", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="opt batch size; 0 = all prompts in one batch",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.8, help="unopt sampling temperature"
    )
    ap.add_argument("--top-k", type=int, default=40, help="unopt top-k")
    ap.add_argument(
        "--csv",
        default=None,
        help="optional CSV of prompts instead of the built-in pool",
    )
    ap.add_argument("--csv-col", default="prompt_text")
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="skip the correctness check in --mode both",
    )
    ap.add_argument(
        "--mem",
        action="store_true",
        help="memory probe: report peak RAM + bytes/param + extrapolation, then exit",
    )
    args = ap.parse_args()

    import kagglehub

    cfg = unopt.MODEL_CONFIGS[args.size]

    print("Logging in to Kaggle (paste your API token when prompted)...")
    kagglehub.login()
    print(f"Downloading {cfg['kaggle_handle']} ...")
    weights_dir = kagglehub.model_download(cfg["kaggle_handle"])
    wd = Path(weights_dir)
    if not (wd / cfg["ckpt_subdir"]).exists() or not (wd / "tokenizer.model").exists():
        print(
            f"WARNING: expected '{cfg['ckpt_subdir']}/' and 'tokenizer.model' inside {wd}"
        )
        print(f"  actual contents: {[p.name for p in wd.iterdir()]}")
    print(f"weights: {wd}")

    n_values = [int(x) for x in str(args.n_prompts).split(",")]
    prompts_all = load_prompts(args, max(n_values))

    print("\nloading weights (shared across engines)...")
    _cfg, params, tok = load_shared(args.size, weights_dir)
    umodel = unopt.Transformer(cfg=cfg) if args.mode in ("unopt", "both") else None
    omodel = opt.Transformer(cfg=cfg) if args.mode in ("opt", "both") else None

    if args.mem:
        if omodel is None:
            omodel = opt.Transformer(cfg=cfg)
        print(f"\nMEMORY PROBE — size={args.size} (small forward: 2 prompts, 8 tokens)")
        print(f"  RSS after weights load : {_rss_gb():6.1f} GB")
        opt.generate_batch(
            omodel, params, tok, prompts_all[:2], max_new_tokens=8
        )  # triggers jit + peak
        print(f"  RSS after jit + forward: {_rss_gb():6.1f} GB")
        peak, npar = _peak_gb(), _n_params(cfg)
        bpp = peak * 1e9 / npar
        print(f"  PEAK RSS               : {peak:6.1f} GB")
        print(
            f"  params                 : {npar / 1e6:5.0f}M   (bf16 {npar * 2 / 1e9:.0f} GB, fp32 {npar * 4 / 1e9:.0f} GB)"
        )
        print(f"  bytes/param at peak    : {bpp:6.1f}")
        print("\n  extrapolated peak RSS at this bytes/param:")
        for s, c in opt.MODEL_CONFIGS.items():
            print(f"    {s:>5}: ~{_n_params(c) * bpp / 1e9:4.0f} GB")
        print(
            "\n  Run this for 4b AND 12b; a straight line through those two peaks predicts 27b\n"
            "  accurately (captures both per-param cost and fixed compile/overhead)."
        )
        return

    stats = []  # one dict per N
    for N in n_values:
        prompts = prompts_all[:N]
        batch_size = args.batch_size or len(prompts)
        print(
            f"\n{'#' * 60}\n# N={len(prompts)}  model={args.size}  max_new_tokens={args.max_new_tokens}"
            f"  mode={args.mode}"
            + (f"  batch_size={batch_size}" if args.mode in ("opt", "both") else "")
            + f"\n{'#' * 60}"
        )

        row = {"N": len(prompts)}
        if args.mode in ("unopt", "both"):
            print("running UNOPT (serial, gemma_inference_restrict.py)...")
            _, per, ut = run_unopt(
                umodel,
                params,
                tok,
                prompts,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
            report(
                "UNOPT — serial, sampled",
                ut,
                len(prompts),
                args.max_new_tokens,
                per_prompt=per,
            )
            row.update(unopt_total=ut, unopt_min=min(per), unopt_max=max(per))
        if args.mode in ("opt", "both"):
            print("running OPT (batched, gemma_inference_restrict_opt.py)...")
            _, ot, ostats = run_opt(
                omodel, params, tok, prompts, args.max_new_tokens, batch_size
            )
            report("OPT — batched, greedy", ot, len(prompts), args.max_new_tokens)
            row.update(
                opt_total=ot,
                opt_ttft=ostats.get("ttft"),
                opt_decode_tps=ostats.get("decode_tps"),
            )
        if args.mode == "both":
            row["speedup"] = row["unopt_total"] / row["opt_total"]
            print(
                f"\n  >>> SPEEDUP at N={row['N']}: {row['speedup']:.1f}x "
                f"({row['unopt_total']:.1f}s -> {row['opt_total']:.1f}s)"
            )
        stats.append(row)

    verdict = None
    if args.mode == "both" and not args.no_check:
        verdict = correctness_check(umodel, omodel, params, tok, prompts_all[0])

    # ── complete statistics ────────────────────────────────────────────────────────────────
    mnt = args.max_new_tokens
    print(f"\n{'=' * 74}\nCOMPLETE STATISTICS")
    print(
        f"  model={args.size}  max_new_tokens={mnt}  mode={args.mode}  "
        f"unopt=sampled(t={args.temperature},k={args.top_k})  opt=greedy"
    )
    print(f"{'-' * 74}")
    hdr = f"  {'N':>4}"
    if args.mode in ("unopt", "both"):
        hdr += f" {'unopt_s':>9} {'u_tok/s':>8}"
    if args.mode in ("opt", "both"):
        hdr += f" {'opt_s':>8} {'o_tok/s':>8} {'ttft_s':>7}"
    if args.mode == "both":
        hdr += f" {'speedup':>8}"
    print(hdr)
    for r in stats:
        N = r["N"]
        line = f"  {N:>4}"
        if args.mode in ("unopt", "both"):
            line += f" {r['unopt_total']:>9.1f} {N * mnt / r['unopt_total']:>8.1f}"
        if args.mode in ("opt", "both"):
            line += f" {r['opt_total']:>8.1f} {N * mnt / r['opt_total']:>8.1f} {(r['opt_ttft'] or 0):>7.1f}"
        if args.mode == "both":
            line += f" {r['speedup']:>7.1f}x"
        print(line)
    if verdict is not None:
        print(f"{'-' * 74}")
        print(f"  correctness (opt vs unopt, greedy): {verdict}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
