# Phase 0: calibrated-PCA / transform-coding KV-cache de-risk

Out-of-tree experiment. Answers one question before any C++ is written: on real KV
caches, does KVTC-style **transform coding** (PCA + adaptive bit allocation) beat the
fixed **Hadamard + uniform quant** scheme ik_llama uses today, at matched bitwidth?

**Important (read `PLAN.md` first):** the original "swap Hadamard for a PCA rotation"
idea is *dead* — rotation-only PCA loses to Hadamard under uniform quant, by theory and
by this harness's controls. The real lever is **adaptive bit allocation**, which is a
different cache format, not a drop-in. This harness measures whether that lever pays off
on a given model. Intended to run on `halfeagle`.

## Results so far (2026-06-29, halfeagle)

Ran on four models. **The QK-norm hypothesis inverted:** QK-norm models GO, non-QK-norm
NO-GO (K@4-bit `pca+alloc` vs Hadamard: Qwen3-1.7B +39%, Qwen3-4B +21% | Qwen2.5-3B +11%
sub-gate, Llama-3.2-3B −2%). Qwen2.5-3B vs Qwen3 (same lineage, QK-norm the difference)
isolates QK-norm as the cause. The win is **K-only, low-bit**, and needs the PCA basis +
allocation together (`hadamard+alloc` = +0%). All MSE — logit-KL confirmation
(`eval_kv_quant.py`, Phase 0.5) is the next gate. Full table + detail in `PLAN.md`.

## Setup

```bash
cd pocs/kv-pca
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch/CUDA likely already present on halfeagle
```

## Run

**0. Build a calibration corpus.** A meaningful result needs representative, ideally
long-context text (the built-in corpus in `extract_kv.py` is a smoke test only).

```bash
python prepare_calib.py --out calib.txt                 # wikitext-103 (~1.2M chars)
python prepare_calib.py --out calib.txt --config wikitext-2-raw-v1   # lighter/faster
python prepare_calib.py --out calib.txt --from-files mycode.py mydocs.md  # your domain text
```

`--from-files` needs no `datasets` dependency and is the better choice if your real
workload is code or a specific domain — calibrate on text that looks like production.

**1. Extract KV samples** (uses the 4070). Pass the corpus via `--calib-file`.

```bash
# Primary target: clean go/no-go (no QK-norm)
python extract_kv.py --model meta-llama/Llama-3.2-3B \
    --calib-file calib.txt --seq-len 2048 --max-seqs 64 \
    --output-dir dumps/llama32-3b

# Contrast target: does QK-norm already pre-whiten the cache?
python extract_kv.py --model Qwen/Qwen3-4B \
    --calib-file calib.txt --seq-len 2048 --max-seqs 64 \
    --output-dir dumps/qwen3-4b
```

**Model access notes:**
- `meta-llama/Llama-3.2-3B` is **gated** — accept the license on its HF page and
  `huggingface-cli login` once on this box. `Qwen/Qwen3-4B` is open.
- Ungated fallback if the Llama gate is a hassle: an architecture-identical mirror such
  as `unsloth/Llama-3.2-3B` (same no-QK-norm GQA, head_dim 128). Mirror availability can
  change; any genuine Llama-3 base without QK-norm preserves the contrast vs Qwen3.
- VRAM: Qwen3-4B at 2048 tokens is tight on 12 GB. If it OOMs, drop `--seq-len 1024`,
  or load in 8-bit, or (wall time is free) let it spill to CPU. Llama-3.2-3B has room.

**2. Rate-distortion analysis** (CPU/numpy):

```bash
python rd_analysis.py --dumps dumps/llama32-3b --out results/llama32-3b
python rd_analysis.py --dumps dumps/qwen3-4b  --out results/qwen3-4b

# Optional upside: how much extra a softmax-safe per-layer mean would buy
python rd_analysis.py --dumps dumps/llama32-3b --out results/llama32-3b-centered --center
```

Each run prints a **GO / NO-GO** verdict and writes `summary.json`, `rd_results.csv`,
and `rd_k.png` / `rd_v.png` (rate-distortion curves) to `--out`.

## Reading the result

The verdict reports three numbers vs `hadamard` (uniform) on the K cache at 4 bits:
- **rotation-only PCA** — expected ≤0 (confirms the drop-in is dead).
- **`pca+alloc`** — PCA + adaptive bits within a block-32 format. This drives the gate.
- **per-dim ceiling** — the per-coefficient transform-coding upper bound.

- **GO** = `pca+alloc` mean MSE is >15% below Hadamard across a majority of layers
  (`--win-margin`). A realizable variable-rate format would help → consider Phase 1.
- **NO-GO** = no win even with allocation; the cache is effectively isotropic under a
  realizable quantizer → not worth building for that model.

Mind the gap between `pca+alloc` and the per-dim ceiling: if only the ceiling wins, the
gain needs a per-coefficient format (a much bigger build) — see PLAN.md.

The interesting science is the **contrast**: a GO on Llama-3.2-3B alongside a NO-GO on
Qwen3-4B would mean QK-norm already removes the allocation headroom — telling us which
architectures are worth building for.

## Notes

- MSE is measured in the transformed domain; since all transforms are orthonormal this
  equals the original-domain error attention sees after inverse rotation.
- Uniform-quant schemes mirror ik_llama's `Q*_0` cache types: per-token block of 32,
  symmetric, n-bit. The per-dim ceiling uses a calibrated per-coefficient scale instead
  (static), which is why it can be *hurt* by per-token magnitude outliers — exactly the
  effect that makes real-data results non-obvious.
- `dumps/`, `results/`, `*.npz`, and `calib.txt` are git-ignored (see `.gitignore`).
