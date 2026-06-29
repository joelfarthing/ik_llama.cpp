# Calibrated PCA KV-cache transform for ik_llama.cpp

Status: **Phase 0 (de-risk / go-no-go)** — exploratory, out of tree, not wired into the build.

## TL;DR — and a course correction from Phase 0

The original pitch was: *replace ik_llama's fixed Hadamard KV-cache rotation with a
per-layer calibrated PCA basis, as a cheap drop-in quality win.* **Phase 0 disproved
that premise — both in theory and in the synthetic harness — before any GPU time was
spent.** Keep reading; the corrected conclusion is more useful than the original idea.

ik_llama.cpp applies a *fixed* orthonormal Hadamard rotation to the K/V caches before
quantization (and undoes it on the attention output). An orthonormal `R` applied to
both Q and K leaves the score unchanged (`(Rq)·(Rk) = qᵀRᵀRk = qᵀk`), so the cache can
live in a rotated basis that quantizes better. The question was *which* rotation.

**Why PCA-as-a-drop-in fails.** ik_llama quantizes with a *uniform, fixed-rate*,
per-token, block-of-32 scheme (`Q*_0`). For uniform quantization, the transform you want
is one that **flattens** the distribution so every block has a similar dynamic range —
which is exactly what a Hadamard/random rotation does (the QuaRot/SpinQuant "incoherence"
trick, and the reason ik_llama uses it). PCA does the **opposite**: it *concentrates*
variance into the leading dimensions. Concentration only pays off if you then spend bits
proportional to variance — i.e. **adaptive bit allocation**, the very piece of KVTC the
original pitch dismissed. Take the allocation away and PCA-rotation-alone is a wash-to-
loss against the Hadamard.

**Phase 0 synthetic results (K-cache, 4-bit, vs Hadamard+uniform):**

| regime | rotation-only PCA | PCA + block-32 alloc | PCA + per-dim alloc (ceiling) |
| --- | --- | --- | --- |
| isotropic (white noise) | ~0% | ~0% | (worse — static-scale tax) |
| anisotropic (rotated Gaussian) | −5% | −9% | −32% |
| axis-aligned, bounded channels | −19% | **+98%** | **+98%** |

Reading: rotation-only PCA is always a wash or a *loss*. The big win (+98% MSE
reduction) appears only with **adaptive bit allocation** *and* only when the data has
axis-aligned per-channel structure with bounded per-token magnitude. That is the KVTC
recipe — and it is a **different cache storage format** (per-coefficient calibrated
quantizers + variable bits + entropy coding), **not a rotation swap**.

So the real, honest question Phase 0 hands to the GPU run on `halfeagle` is:
**do real LLM KV caches have the structure where adaptive-allocation transform coding
beats ik_llama's Hadamard+uniform scheme — net of the per-token-outlier tax that
punishes static per-coefficient scales (the reason KVTC keeps recent + sink tokens
uncompressed and uses entropy coding)?** That is empirical, and `rd_analysis.py`
answers it per model.

## Phase 0 results — RUN on halfeagle (2026-06-29)

Ran `extract_kv.py` + `rd_analysis.py` on four models (wikitext-103 calib, K-cache, 4-bit
decision gate). **The result inverts this doc's QK-norm hypothesis** — which predicted
QK-norm would pre-whiten the cache and *remove* the headroom (Qwen NO-GO, Llama GO):

| Model | QK-norm | K@4-bit pca+alloc vs Hadamard | layers ≥15% | verdict |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | yes | +39.1% | 26/28 (93%) | **GO** (strong) |
| Qwen3-4B | yes | +20.9% | 20/36 (56%) | **GO** |
| Qwen2.5-3B | no | +10.9% | 11/36 (31%) | NO-GO |
| Llama-3.2-3B | no | −1.9% | 2/28 (7%) | NO-GO |

- **QK-norm is the causal variable, not model family.** Qwen2.5-3B shares Qwen3's
  tokenizer/lineage/head_dim=128 and differs mainly in QK-norm; it is NO-GO while both
  Qwen3 sizes are GO. So QK-norm *adds* exploitable K-cache headroom — the opposite of the
  prior. (Llama-3.2-3B run via the ungated, arch-identical `unsloth/Llama-3.2-3B`;
  meta-llama is license-gated.)
- **The win needs the PCA basis AND allocation together.** A `hadamard+alloc` ablation
  (added to `rd_analysis.py`) is **+0.0%** on every model — Hadamard flattens variance, so
  allocation has nothing to exploit — and `pca`-rotation-only is catastrophic (−120 to
  −285%). The whole gain is the calibrated basis paired with allocation, *not* a cheap
  allocation-only tweak on the existing format.
- **Scope: K-cache only, low bit-depth.** PCA hurts V (−18 to −30% @4-bit) → transform K,
  leave V. The win lives at 2–4 bit (Qwen3-4B K: +74% @2-bit, +21% @4-bit); at 8-bit the
  allocation saturates and it reverts to the PCA-uniform loss.
- **Robustness:** Qwen3-4B re-run on a code corpus (ik_llama source) → GO +22.4% (vs
  +20.9% wikitext) — not distribution-specific.

**Caveat / next gate:** all of the above is reconstruction MSE — a proxy. Whether the
K-MSE win becomes real memory-at-iso-quality must be confirmed with **logit-KL / perplexity
vs the F16 cache** before any C++ build. That is Phase 0.5 (`eval_kv_quant.py`, not yet
written): quantize the K-cache inline in an HF forward (PCA+alloc vs Hadamard+uniform at
matched bits, V fp) and measure ppl + logit-KL on held-out text — no kernel work, decisive
either way.

## Cost reality (revised)

The original doc claimed this was "cheap to ship" because a PCA rotation is just a
`ggml_mul_mat` (true, and still hardware-agnostic for free) and needs no GGUF
reconversion (also still true — the basis/quantizer is a sidecar like imatrix). **But
the rotation was never the lever.** The lever is adaptive allocation + per-coefficient
quantization, which means a new variable-rate KV-cache format — a substantial change to
cache allocation, the quant path, and possibly the attention kernels. This is NOT a
drop-in. Scope it as a research format, gated on a positive Phase 0 signal on real data.

Still true regardless: **no GGUF reconversion.** The base model GGUF is never touched;
any calibrated artifact ships as a sidecar loaded at runtime, like an imatrix file.

## Verified integration points (for Phase 1+, not Phase 0)

| What | Location |
| --- | --- |
| K rotation (q_cur + k_cur), pre cache-write | `src/llama-build-context.cpp:1844-1853` |
| V rotation, pre cache-write | `src/llama-build-context.cpp:1854-1858` |
| V un-rotation, post flash-attn output | `src/llama-build-context.cpp:1675-1680` |
| Existing transform flag (bool, to become enum) | `src/llama-cparams.h:42-43` |
| Per-layer Hadamard block size | `src/llama-model.h:524-541` |
| `ggml_hadamard` op (the thing we replace) | `ggml/src/ggml.c:6253`, decl `ggml/include/ggml.h:1119` |
| Per-layer cache type selection | `src/llama.cpp:1052-1073` |
| KV cache type registry | `common/common.cpp:4042-4098` |
| Calibration-tool template | `examples/imatrix/imatrix.cpp` |
| Sidecar GGUF write API | `gguf_add_tensor` etc., `ggml/include/ggml.h` |

What is cached, and therefore what each basis is computed on:
- **K cache** stores RoPE'd keys → PCA basis computed on **post-RoPE** keys.
- **V cache** stores value vectors (no RoPE) → PCA basis computed on **value** vectors.

## Phases

### Phase 0 — Offline de-risk  ← done; harness validated

Goal: decide whether the KVTC-style lever (transform + **adaptive bit allocation**)
beats ik_llama's Hadamard+uniform scheme on real KV caches — before writing any C++.

**Decision to dump from PyTorch/HF, not from a C++ hook into ik_llama.** The cache
statistics are a property of the model weights + RoPE, identical in both
implementations. HF's `past_key_values` exposes exactly the post-RoPE keys and value
vectors ik_llama would cache, with a trivial extraction and no rebuild. We only pay for
invasive C++ work once the signal is positive.

Deliverables (this directory, all working + smoke-tested):
- `extract_kv.py` — run a model over a calibration corpus, dump per-layer post-RoPE K
  and value samples to `.npz` (reservoir-sampled).
- `rd_analysis.py` — numpy rate-distortion harness. Per layer, per K/V, it builds the
  PCA and Hadamard bases and measures reconstruction MSE (orthonormal-invariant, so it
  equals original-domain error) under five schemes at 2–8 bits:
  `plain` (no transform), `hadamard` (today), `pca` (rotation-only drop-in),
  `pca+alloc` (PCA + adaptive bits within the block-32 format), and
  `pca+alloc-perdim` (per-coefficient calibrated transform-coding ceiling). Prints a
  GO/NO-GO verdict + rate-distortion plots.

**Harness validated on synthetic positive/negative controls** (see TL;DR table):
isotropic → all schemes ~equal (correctly NO-GO); rotation-only PCA → always wash/loss;
axis-aligned bounded channels → `pca+alloc` correctly detects a +98% MSE win. The
harness can both see a real win and refuse a non-win.

**Targets (in order):**
1. **Llama-3.2-3B** — no QK-norm, standard GQA, head_dim 128; a KVTC reference model,
   so a non-zero allocation gain is *expected*. Clean go/no-go.
2. **Qwen3-4B** — has QK-norm (per-head RMSNorm on Q/K before RoPE), which partially
   pre-whitens the cache. Run as the *contrast*: does QK-norm already remove the
   allocation headroom? That decides which architectures are worth building for.

Avoid MLA models (DeepSeek-V2/V3) for Phase 0 — they cache a low-rank latent, so there
is no full head-dim cache to transform; different problem.

**Decision gate (revised — gates on the real lever, not the rotation):** GO if, on K at
4 bits, `pca+alloc` mean reconstruction MSE is >15% below `hadamard` across a majority
of layers. The verdict also reports rotation-only (expected ≤0, confirming the
drop-in is dead) and the per-dim ceiling (upper bound that would need a new format).
NO-GO means the cache is effectively isotropic *under a realizable quantizer* and the
whole direction isn't worth pursuing for that model.

> Interpreting a GO carefully: a win at the `pca+alloc-perdim` ceiling but NOT at
> `pca+alloc` (block-32) means the gain exists only with a per-coefficient format
> (closer to full KVTC), which is a much larger build. A win at `pca+alloc` itself
> means a variable-rate scheme is achievable within something close to the existing
> block structure. The gap between those two numbers sizes the engineering effort.

### Phase 1 — only if Phase 0 says GO: prototype a variable-rate KV format

This is the honest scope correction: there is **no cheap drop-in**. A real win requires
a variable-rate cache (per-block or per-coefficient bit allocation + calibrated
quantizers, optionally entropy coding), which touches cache allocation, the quant/
dequant path, and possibly the attention kernels. Steps, roughly:
- Calibration tool (fork `examples/imatrix/imatrix.cpp`): accumulate per-layer K/V
  covariance, derive the PCA basis `R_l` *and* the per-subband bit allocation + scales,
  write a sidecar `model.kvtc.gguf` (named tensors + metadata). One-time, big-hardware.
- Decide granularity from Phase 0's `pca+alloc` vs `pca+alloc-perdim` gap: block-32
  variable-rate (smaller change) vs per-coefficient (larger, closer to KVTC).
- Handle the per-token-outlier tax KVTC handles: keep attention-sink + recent-window
  tokens uncompressed (ik_llama already has sink-token plumbing, `src/llama-model.h`).

### Phase 2 — runtime integration (scope depends on Phase 1 format)

- Generalize `k_cache_hadamard`/`v_cache_hadamard` (`src/llama-cparams.h:42-43`) into a
  `kv_transform` enum. The PCA rotation itself is a `ggml_mul_mat(ctx, R[il], …)` at the
  existing call sites (`src/llama-build-context.cpp:1844-1858`, `:1675-1680`); apply the
  same `R_k[il]` to Q and K, and `R_v[il]` at V store + FA output (mind the two V
  layouts, `:589-604`). The harder part is the variable-rate (de)quant in the cache
  read/write and kernels — not the rotation.
- Guard: refuse to start on layer-count / head_dim / arch-hash mismatch with the
  sidecar; a wrong basis silently corrupts attention.

### Phase 3 — measurement

Rate-distortion curve (effective bits/element incl. scale + allocation overhead vs
quality), series {plain, Hadamard, the new format}.
- **Primary quality metric:** KL-divergence vs F16 logits (`llama-perplexity` supports
  it; far more sensitive than raw PPL for KV degradation).
- Cross-check: perplexity on held-out wikitext.
- Long-context stress: RULER / needle-in-haystack (where KV damage actually shows up).
- Intrinsic: per-layer cache reconstruction MSE (Phase 0's metric, for debugging).
- **Speed:** `llama-bench` TG/PP tok/s. The rotation is cheap; watch the variable-rate
  (de)quant cost, which is the real overhead risk. Report fixed-quality→memory-saved and
  fixed-memory→quality-gained.

### Phase 4 — refinements (only if Phase 3 justifies)

Per-head bases; mean-centering (softmax-safe for K and V with a stored per-layer mean —
see note below); entropy coding for the off-GPU multi-turn *storage* use case (full
KVTC territory).

## Notes / risks

- **Distribution dependence.** PCA is optimal for the calibration distribution; a
  Hadamard is distribution-free (never optimal, never wrong). Calibrate on
  representative, long-context traffic.
- **Model-specific artifact.** One sidecar per model; regenerate after fine-tuning.
- **Compute.** Rotation is O(d²) vs Hadamard O(d log d) — ~18× more MACs at d=128, still
  trivial vs attention.
- **Centering is free if we want it later.** Subtracting a per-layer key-mean μ from K
  shifts every score by `q·μ`, constant across keys for a given query, so softmax is
  unchanged. For V, `Σ wᵢ(vᵢ−μ) = (Σ wᵢvᵢ) − μ` (softmax weights sum to 1), recovered by
  adding μ back. Both need only a stored per-layer mean vector. Not in v1; quantified by
  `rd_analysis.py --center` as upside.

## Handoff (cloud → halfeagle)

This was authored in a Claude Code on the web (cloud) session. The GPU runs belong on
`halfeagle` (4070 + 64 GB). Routes to continue on that box:
- **Mac UI + halfeagle GPU + this context:** Desktop app → *Add SSH connection* to
  halfeagle (UI on Mac, execution on the box), then `/teleport` this web session into it.
- **Terminal-only:** `ssh halfeagle` → `claude --teleport` → pick this session
  (teleport lands wherever you run it, so run it *on* halfeagle).
- **Code only:** `git pull` this branch via your usual VS Code Remote-SSH flow.

See `README.md` for how to run Phase 0.
