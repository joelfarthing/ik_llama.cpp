#!/usr/bin/env python3
"""Phase 0 KV-cache extractor for the calibrated-PCA experiment.

Runs a HuggingFace causal LM over a calibration corpus and dumps, per layer, a
reservoir sample of the post-RoPE key vectors and the value vectors that the model
caches. These are exactly what ik_llama.cpp stores in its K/V cache (K is post-RoPE,
V is not), so the anisotropy they exhibit is the same regardless of whether inference
runs in PyTorch or in ik_llama's kernels.

Output: one .npz per layer under <output-dir>, each with arrays `k` and `v` of shape
[n_sample, head_dim] (float16), plus a meta.json describing the run.

Example (on halfeagle):
    python extract_kv.py --model meta-llama/Llama-3.2-3B \
        --calib-file calib.txt --seq-len 2048 --output-dir dumps/llama32-3b
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


# A tiny, diverse built-in corpus used only if --calib-file is not given. For a real
# go/no-go you should pass representative (ideally long-context) text via --calib-file.
_BUILTIN_CORPUS = [
    "The history of computing stretches from the abacus to modern accelerators. "
    "Each era reframed what it meant to calculate, store, and retrieve information.",
    "In a quiet harbor town, the tide kept a schedule older than any clock, and the "
    "fishermen read the water the way scholars read a familiar page.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    p = xs[len(xs)//2]\n"
    "    lo = [x for x in xs if x < p]\n    hi = [x for x in xs if x > p]\n"
    "    return quicksort(lo) + [x for x in xs if x == p] + quicksort(hi)",
    "The treaty's third clause, often overlooked, redrew the boundary along the river "
    "rather than the ridge, with consequences that historians still debate.",
    "Photosynthesis converts light energy into chemical energy, storing it in the bonds "
    "of glucose; the oxygen we breathe is, in a sense, a byproduct of that ledger.",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--calib-file", default=None,
                    help="UTF-8 text file; split into --seq-len chunks. "
                         "Falls back to a small built-in corpus if omitted.")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="tokens per forward sequence")
    ap.add_argument("--max-seqs", type=int, default=64,
                    help="cap on number of sequences processed")
    ap.add_argument("--n-sample", type=int, default=20000,
                    help="reservoir sample size per (layer, K/V)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_chunks(calib_file, tokenizer, seq_len, max_seqs):
    """Return a list of 1-D LongTensors, each up to seq_len tokens."""
    if calib_file:
        with open(calib_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        sources = [text]
    else:
        print("[extract] no --calib-file; using built-in corpus (small, for smoke "
              "tests only — pass real text for a meaningful go/no-go).", file=sys.stderr)
        sources = _BUILTIN_CORPUS

    ids = []
    for s in sources:
        ids.extend(tokenizer(s, add_special_tokens=False)["input_ids"])
    chunks = []
    for i in range(0, len(ids), seq_len):
        chunk = ids[i:i + seq_len]
        if len(chunk) >= 16:  # skip tiny tails
            chunks.append(torch.tensor(chunk, dtype=torch.long))
        if len(chunks) >= max_seqs:
            break
    if not chunks:
        raise SystemExit("[extract] no usable calibration chunks produced")
    return chunks


def to_layer_kv(past_key_values):
    """Normalize various transformers cache formats to a list of (k, v) per layer.

    Each k/v is a tensor [batch, n_kv_heads, seq, head_dim].
    """
    # Newer transformers: DynamicCache with to_legacy_cache(); or .key_cache lists.
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [(k, v) for (k, v) in legacy]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    # Legacy tuple-of-tuples.
    return [(k, v) for (k, v) in past_key_values]


class Reservoir:
    """Per-layer reservoir sampler over rows of shape [*, head_dim]."""

    def __init__(self, n_sample, head_dim, rng):
        self.n = n_sample
        self.buf = np.empty((n_sample, head_dim), dtype=np.float32)
        self.seen = 0
        self.filled = 0
        self.rng = rng

    def add(self, rows):  # rows: np.ndarray [m, head_dim]
        for r in rows:
            if self.filled < self.n:
                self.buf[self.filled] = r
                self.filled += 1
            else:
                j = self.rng.integers(0, self.seen + 1)
                if j < self.n:
                    self.buf[j] = r
            self.seen += 1

    def result(self):
        return self.buf[:self.filled]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"[extract] loading {args.model} ({args.dtype}) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device)
    model.eval()

    chunks = load_chunks(args.calib_file, tok, args.seq_len, args.max_seqs)
    print(f"[extract] {len(chunks)} sequences, up to {args.seq_len} tokens each",
          file=sys.stderr)

    k_res = None
    v_res = None
    head_dim = None
    n_layers = None

    with torch.no_grad():
        for si, chunk in enumerate(chunks):
            inp = chunk.unsqueeze(0).to(args.device)
            out = model(inp, use_cache=True, return_dict=True)
            layers = to_layer_kv(out.past_key_values)
            if k_res is None:
                n_layers = len(layers)
                head_dim = layers[0][0].shape[-1]
                k_res = [Reservoir(args.n_sample, head_dim, rng) for _ in range(n_layers)]
                v_res = [Reservoir(args.n_sample, head_dim, rng) for _ in range(n_layers)]
                print(f"[extract] {n_layers} layers, head_dim={head_dim}",
                      file=sys.stderr)
            for li, (k, v) in enumerate(layers):
                # [b, n_kv_heads, seq, head_dim] -> [rows, head_dim]
                k2 = k.reshape(-1, head_dim).float().cpu().numpy()
                v2 = v.reshape(-1, head_dim).float().cpu().numpy()
                k_res[li].add(k2)
                v_res[li].add(v2)
            if (si + 1) % 8 == 0:
                print(f"[extract] {si + 1}/{len(chunks)} sequences", file=sys.stderr)

    for li in range(n_layers):
        np.savez_compressed(
            os.path.join(args.output_dir, f"layer_{li:03d}.npz"),
            k=k_res[li].result().astype(np.float16),
            v=v_res[li].result().astype(np.float16),
        )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "n_layers": int(n_layers),
        "head_dim": int(head_dim),
        "n_sample": int(args.n_sample),
        "seq_len": int(args.seq_len),
        "n_seqs": len(chunks),
        "rows_seen_layer0_k": int(k_res[0].seen),
        "note": "k = post-RoPE keys, v = value vectors; rows are [n_sample, head_dim]",
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] wrote {n_layers} layer dumps + meta.json to {args.output_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
