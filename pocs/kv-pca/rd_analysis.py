#!/usr/bin/env python3
"""Phase 0 rate-distortion analysis for the calibrated-PCA KV-cache experiment.

For each layer dump produced by extract_kv.py, and for K and V separately, this:
  1. builds three orthonormal transforms of the head-dim axis:
       - identity (plain block quantization, the baseline ik_llama uses with no transform)
       - Hadamard (the fixed transform ik_llama uses today)
       - PCA      (the proposed per-layer learned basis)
  2. quantizes the transformed vectors block-wise (groups of 32, symmetric, n-bit) at a
     sweep of bit depths, the way ik_llama's Q*_0 cache types do;
  3. measures reconstruction MSE. Because all three transforms are orthonormal, MSE is
     rotation-invariant, so error measured in the transformed domain equals the error
     attention would see after the inverse rotation in the original domain.

It then prints a GO/NO-GO verdict and (if matplotlib is available) rate-distortion plots.

The decision: PCA wins iff sorting variance along the head-dim axis lets block-wise
quantization fit each block's scale better than the raw or Hadamard layout.

Example:
    python rd_analysis.py --dumps dumps/llama32-3b --out results/llama32-3b
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dir of layer_*.npz from extract_kv.py")
    ap.add_argument("--out", required=True, help="output dir for CSV/JSON/plots")
    ap.add_argument("--block", type=int, default=32, help="quant block size (Q*_0 uses 32)")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--decision-bits", type=int, default=4,
                    help="bit depth at which the GO/NO-GO gate is evaluated")
    ap.add_argument("--win-margin", type=float, default=0.15,
                    help="required fractional MSE reduction (PCA vs Hadamard) to count "
                         "a layer as a PCA win at --decision-bits")
    ap.add_argument("--center", action="store_true",
                    help="also subtract a per-layer mean before transforming (softmax-safe "
                         "upside; off by default to match current ik_llama pipeline)")
    return ap.parse_args()


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(d):
    """Normalized (orthonormal) Sylvester-Hadamard matrix, d must be a power of two."""
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


def pca_basis(x):
    """Columns are eigenvectors of the covariance, sorted by descending variance.

    Returns R [d, d] orthonormal; transform is x @ R.
    """
    c = (x.T @ x) / x.shape[0]
    evals, evecs = np.linalg.eigh(c)          # ascending
    order = np.argsort(evals)[::-1]
    return evecs[:, order]


def _block_err(blk, bits):
    """Sum of squared error for symmetric n-bit per-block quantization of `blk`.

    bits==0 means the block is dropped (reconstructed as zero).
    """
    if bits <= 0:
        return float(np.sum(blk ** 2))
    qmax = max((1 << (bits - 1)) - 1, 1)       # bits=4 -> 7; bits=1 -> sign quantizer
    amax = np.max(np.abs(blk), axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    q = np.clip(np.round(blk / scale), -qmax, qmax)
    return float(np.sum((blk - q * scale) ** 2))


def block_quant_mse(xt, bits, block):
    """Uniform fixed-rate: every block of `block` features gets the same `bits`.

    Mirrors ik_llama Q*_0 cache types (one fp scale per block of 32, symmetric).
    """
    d = xt.shape[1]
    err = sum(_block_err(xt[:, s:s + block], bits) for s in range(0, d, block))
    return err / xt.size


def allocate_bits(variances, avg_bits, bmin=0, bmax=8):
    """High-rate optimal bit allocation across subbands at a fixed average rate.

    b_i = R + 0.5*log2(var_i / GM(var)), water-filled under [bmin,bmax], rounded to
    integers while preserving the total budget. This is the lever KVTC uses and that
    ik_llama's fixed per-tensor cache type does NOT currently have.
    """
    v = np.maximum(np.asarray(variances, dtype=np.float64), 1e-12)
    nb = len(v)
    total = avg_bits * nb
    b = avg_bits + 0.5 * np.log2(v) - np.mean(0.5 * np.log2(v))
    b = np.clip(b, bmin, bmax)
    for _ in range(100):
        free = (b > bmin) & (b < bmax)
        deficit = total - b.sum()
        if abs(deficit) < 1e-9 or not free.any():
            break
        b[free] += deficit / free.sum()
        b = np.clip(b, bmin, bmax)
    bi = np.floor(b).astype(int)
    frac = b - bi
    rem = int(round(total - bi.sum()))
    order = np.argsort(frac)[::-1]
    i = 0
    guard = 8 * nb + 8
    while rem > 0 and i < guard:
        idx = order[i % nb]
        if bi[idx] < bmax:
            bi[idx] += 1
            rem -= 1
        i += 1
    i = 0
    while rem < 0 and i < guard:
        idx = order[nb - 1 - (i % nb)]
        if bi[idx] > bmin:
            bi[idx] -= 1
            rem += 1
        i += 1
    return bi


def block_quant_mse_alloc(xt, avg_bits, block):
    """Variable-rate at ik_llama's block granularity: bits per block-of-`block`
    allocated proportional to block variance, average held at `avg_bits`. Uses a
    per-row dynamic scale (like Q*_0). Shows how much allocation buys WITHIN the
    existing block-32 format."""
    d = xt.shape[1]
    starts = list(range(0, d, block))
    var = [float(np.mean(xt[:, s:s + block] ** 2)) for s in starts]
    bits = allocate_bits(var, avg_bits)
    err = sum(_block_err(xt[:, s:s + block], int(bits[i]))
              for i, s in enumerate(starts))
    return err / xt.size


def col_quant_mse_alloc(xt, avg_bits):
    """Per-dimension calibrated transform coding — the KLT+allocation ceiling.

    Each coefficient (column) is its own subband: a calibration-fixed per-column scale
    (static quantization) plus per-column bit allocation. This is what a real transform
    coder (JPEG/KVTC) does. It needs a per-dimension scale + variable bits per column,
    i.e. a different cache format than ik_llama's per-row block-of-32. Reported as the
    upper bound on what the PCA basis can deliver, to separate 'is the lever real?' from
    'can the current format capture it?'."""
    var = np.mean(xt ** 2, axis=0)
    bits = allocate_bits(var, avg_bits)
    err = 0.0
    for j in range(xt.shape[1]):
        b = int(bits[j])
        col = xt[:, j]
        if b <= 0:
            err += float(np.sum(col ** 2))
            continue
        qmax = max((1 << (b - 1)) - 1, 1)
        amax = float(np.max(np.abs(col)))          # one calibrated scale per column
        scale = amax / qmax if amax > 0 else 1.0
        q = np.clip(np.round(col / scale), -qmax, qmax)
        err += float(np.sum((col - q * scale) ** 2))
    return err / xt.size


def analyze_tensor(x, bits_list, block, center):
    """Return {method: {bits: mse}} for one [n, d] sample.

    Methods:
      plain      - identity rotation, uniform quant (ik_llama with no transform)
      hadamard   - fixed Hadamard, uniform quant (ik_llama today)
      pca        - learned PCA, uniform quant (rotation-only drop-in; expected ~wash)
      pca+alloc  - learned PCA + adaptive bit allocation at matched avg rate (KVTC lever)
    """
    x = x.astype(np.float32)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    d = x.shape[1]

    R_pca = pca_basis(x)
    transforms = {"plain": np.eye(d, dtype=np.float32), "pca": R_pca}
    if is_pow2(d):
        transforms["hadamard"] = hadamard_matrix(d).astype(np.float32)

    results = {}
    for name, R in transforms.items():
        xt = x @ R
        results[name] = {b: block_quant_mse(xt, b, block) for b in bits_list}

    xt_pca = x @ R_pca
    results["pca+alloc"] = {
        b: block_quant_mse_alloc(xt_pca, b, block) for b in bits_list}
    results["pca+alloc-perdim"] = {
        b: col_quant_mse_alloc(xt_pca, b) for b in bits_list}
    # Ablation: the SAME adaptive allocation on the Hadamard basis ik_llama already
    # uses. Isolates "is the win the allocation lever (basis-free, small build) or the
    # PCA basis (needs a calibrated sidecar)?" — the gap pca+alloc - hadamard+alloc is
    # the marginal value of the learned basis.
    if "hadamard" in transforms:
        xt_had = x @ transforms["hadamard"]
        results["hadamard+alloc"] = {
            b: block_quant_mse_alloc(xt_had, b, block) for b in bits_list}
    return results


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta_path = os.path.join(args.dumps, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    files = sorted(glob.glob(os.path.join(args.dumps, "layer_*.npz")))
    if not files:
        raise SystemExit(f"no layer_*.npz found in {args.dumps}")

    # per_layer[kv][method][bits] -> list over layers
    per_layer = {"k": {}, "v": {}}
    rows_csv = ["kv,layer,method,bits,mse"]

    for li, fp in enumerate(files):
        z = np.load(fp)
        for kv in ("k", "v"):
            res = analyze_tensor(z[kv], args.bits, args.block, args.center)
            for method, by_bits in res.items():
                per_layer[kv].setdefault(method, {})
                for b, mse in by_bits.items():
                    per_layer[kv][method].setdefault(b, []).append(mse)
                    rows_csv.append(f"{kv},{li},{method},{b},{mse:.8e}")
        print(f"[rd] layer {li + 1}/{len(files)} done", file=sys.stderr)

    with open(os.path.join(args.out, "rd_results.csv"), "w") as f:
        f.write("\n".join(rows_csv) + "\n")

    # Aggregate means across layers.
    summary = {"meta": meta, "block": args.block, "center": args.center, "agg": {}}
    for kv in ("k", "v"):
        summary["agg"][kv] = {}
        for method, by_bits in per_layer[kv].items():
            summary["agg"][kv][method] = {
                str(b): float(np.mean(v)) for b, v in sorted(by_bits.items())}

    # GO/NO-GO gate: the real lever is PCA + adaptive bit allocation vs the Hadamard +
    # uniform quant ik_llama uses today, on K at decision-bits, per layer. (Rotation
    # alone — bare "pca" — is expected to be a wash/loss against Hadamard under uniform
    # quant, because Hadamard flattens the distribution and uniform quant likes that;
    # the win comes from the allocation, which implies a variable-rate storage format.)
    db = args.decision_bits
    verdict = {"decision_bits": db, "win_margin": args.win_margin,
               "compares": "pca+alloc vs hadamard (uniform), K-cache"}
    have = (db in per_layer["k"].get("pca+alloc", {})
            and db in per_layer["k"].get("hadamard", {}))
    if have:
        cand = np.array(per_layer["k"]["pca+alloc"][db])
        base = np.array(per_layer["k"]["hadamard"][db])
        reduction = np.where(base > 0, 1.0 - cand / base, 0.0)
        wins = int(np.sum(reduction >= args.win_margin))
        # Diagnostic: does the rotation alone do anything vs Hadamard?
        def mean_red(method):
            if db not in per_layer["k"].get(method, {}):
                return None
            arr = np.array(per_layer["k"][method][db])
            return float(np.mean(np.where(base > 0, 1.0 - arr / base, 0.0)))

        verdict.update({
            "n_layers": int(len(cand)),
            "mean_mse_reduction_pca_alloc_vs_hadamard_k": float(np.mean(reduction)),
            "median_mse_reduction": float(np.median(reduction)),
            "rotation_only_mean_reduction_vs_hadamard": mean_red("pca"),
            "hadamard_alloc_mean_reduction_vs_hadamard": mean_red("hadamard+alloc"),
            "perdim_ceiling_mean_reduction_vs_hadamard": mean_red("pca+alloc-perdim"),
            "layers_meeting_margin": wins,
            "fraction_meeting_margin": float(wins / len(cand)),
            "go": bool(wins / len(cand) > 0.5),
        })
    else:
        verdict["go"] = None
        verdict["note"] = ("hadamard baseline unavailable (head_dim not a power of two) "
                           "or decision_bits not in sweep")
    summary["verdict"] = verdict

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _maybe_plot(per_layer, args)

    # Human-readable verdict.
    print("\n==================== Phase 0 verdict ====================")
    model = meta.get("model", "?")
    print(f"model: {model}   layers: {len(files)}   "
          f"head_dim: {meta.get('head_dim', '?')}   centered: {args.center}")
    if verdict.get("go") is None:
        print(verdict.get("note", "inconclusive"))
    else:
        print(f"K-cache, {db}-bit avg — PCA+alloc vs Hadamard(uniform):")
        print(f"  mean MSE reduction:   {verdict['mean_mse_reduction_pca_alloc_vs_hadamard_k']:+.1%}")
        print(f"  median MSE reduction: {verdict['median_mse_reduction']:+.1%}")
        print(f"  layers >= {args.win_margin:.0%} better: "
              f"{verdict['layers_meeting_margin']}/{verdict['n_layers']} "
              f"({verdict['fraction_meeting_margin']:.0%})")
        rot = verdict.get("rotation_only_mean_reduction_vs_hadamard")
        had_alloc = verdict.get("hadamard_alloc_mean_reduction_vs_hadamard")
        ceil = verdict.get("perdim_ceiling_mean_reduction_vs_hadamard")
        if rot is not None:
            print(f"  rotation-only PCA vs Hadamard:   {rot:+.1%}  (expected ~0 or negative)")
        if had_alloc is not None:
            print(f"  Hadamard+alloc vs Hadamard:      {had_alloc:+.1%}  (allocation only, no PCA — basis-free)")
            pa = verdict.get("mean_mse_reduction_pca_alloc_vs_hadamard_k")
            if pa is not None:
                print(f"  -> PCA basis marginal value:     {pa - had_alloc:+.1%}  (pca+alloc minus hadamard+alloc)")
        if ceil is not None:
            print(f"  per-dim transform-coding ceiling: {ceil:+.1%}  (upper bound; needs a new cache format)")
        print(f"\n  >>> {'GO' if verdict['go'] else 'NO-GO'} <<<  "
              f"({'adaptive allocation clears the bar' if verdict['go'] else 'no win even with allocation — cache near-isotropic for this model'})")
    print(f"\nfull tables: {os.path.join(args.out, 'summary.json')} / rd_results.csv")
    print("=========================================================")


def _maybe_plot(per_layer, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[rd] matplotlib unavailable ({e}); skipping plots", file=sys.stderr)
        return
    for kv in ("k", "v"):
        plt.figure(figsize=(6, 4))
        for method, by_bits in sorted(per_layer[kv].items()):
            bits = sorted(by_bits.keys())
            mse = [np.mean(by_bits[b]) for b in bits]
            plt.plot(bits, np.log10(mse), marker="o", label=method)
        plt.xlabel("bits per element")
        plt.ylabel("log10(mean reconstruction MSE)")
        plt.title(f"{kv.upper()}-cache rate-distortion (lower is better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = os.path.join(args.out, f"rd_{kv}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[rd] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
