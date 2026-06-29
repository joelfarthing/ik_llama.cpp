#!/usr/bin/env python3
"""Phase 0.5: does the K-cache MSE win translate to LOGIT QUALITY?

rd_analysis.py measures reconstruction MSE — a proxy. This harness closes the gap to a
real metric without any C++: it quantizes the K-cache *inline* during a held-out forward
pass and reports perplexity + logit-KL vs the full-precision cache.

For each layer it derives, from the calibration dumps produced by extract_kv.py:
  - the PCA basis R_l (post-RoPE K), and a static per-block (32-wide) bit allocation
  - the fixed Hadamard basis (ik_llama's current transform)
then registers a custom attention function that, per layer, rotates K -> block-quantizes
(symmetric n-bit, per-row block-of-32 scale, like ik_llama Q*_0) -> dequantizes -> rotates
back, for two schemes at matched average bits:
  - hadamard : Hadamard rotate + uniform bits   (what ik_llama does today)
  - pca      : PCA rotate + adaptive allocation  (the proposed lever)
V is left full precision in both arms (PCA helps K, hurts V — Phase 0 finding), so this
isolates the K-cache transform-coding question. Compares both to the fp cache.

A GO survives here iff, at matched bits, pca gives LOWER perplexity AND lower logit-KL
than hadamard. If the MSE win evaporates in logit space, the C++ build is not justified.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch


# ---------------- quant primitives (mirror rd_analysis.py semantics) ----------------
def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(d):
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


def pca_basis(x):
    c = (x.T @ x) / x.shape[0]
    evals, evecs = np.linalg.eigh(c)
    return evecs[:, np.argsort(evals)[::-1]]


def allocate_bits(variances, avg_bits, bmin=0, bmax=8):
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


@torch.no_grad()
def block_qdq(x, bits_per_block, block):
    """Quantize-dequantize x [..., d] in blocks of `block` over the last dim.
    Symmetric n-bit, per-row (per leading-index) dynamic scale = amax/qmax. fp32 in/out."""
    d = x.shape[-1]
    out = x.clone()
    for i, s in enumerate(range(0, d, block)):
        b = int(bits_per_block[i])
        sl = x[..., s:s + block]
        if b <= 0:
            out[..., s:s + block] = 0
            continue
        qmax = max((1 << (b - 1)) - 1, 1)
        amax = sl.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        q = torch.clamp(torch.round(sl / scale), -qmax, qmax)
        out[..., s:s + block] = q * scale
    return out


def repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


# ---------------- inline-K-quant attention (custom interface fn) ----------------
STATE = {"mode": "fp", "block": 32, "layers": None}  # layers[li][mode] = (R_fp32, bits)


@torch.no_grad()
def kquant_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    mode = STATE["mode"]
    if mode != "fp":
        R, bits = STATE["layers"][module.layer_idx][mode]
        kf = key.float()
        kt = kf @ R
        kt = block_qdq(kt, bits, STATE["block"])
        key = (kt @ R.transpose(0, 1)).to(key.dtype)
    ks = repeat_kv(key, module.num_key_value_groups)
    vs = repeat_kv(value, module.num_key_value_groups)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if isinstance(attention_mask, torch.Tensor):
        aw = aw + attention_mask[..., : ks.shape[-2]]
    else:
        L = aw.shape[-1]
        aw = aw + torch.triu(torch.full((L, L), float("-inf"), device=aw.device, dtype=aw.dtype), 1)
    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, None


# ---------------- params from calibration dumps ----------------
def build_layer_params(dumps_dir, device):
    files = sorted(glob.glob(os.path.join(dumps_dir, "layer_*.npz")))
    if not files:
        raise SystemExit(f"no layer_*.npz in {dumps_dir}")
    R_pca, block_var, d = [], [], None
    for f in files:
        k = np.load(f)["k"].astype(np.float32)
        d = k.shape[1]
        Rp = pca_basis(k)
        kt = k @ Rp
        block_var.append([float(np.mean(kt[:, s:s + 32] ** 2)) for s in range(0, d, 32)])
        R_pca.append(torch.tensor(Rp, device=device, dtype=torch.float32))
    H = torch.tensor(hadamard_matrix(d), device=device, dtype=torch.float32) if is_pow2(d) else None
    return R_pca, block_var, H, d, len(files)


def layers_for_bits(R_pca, block_var, H, avg_bits):
    nb = len(block_var[0])
    out = []
    for li in range(len(R_pca)):
        ent = {"pca": (R_pca[li], list(allocate_bits(block_var[li], avg_bits)))}
        if H is not None:
            ent["hadamard"] = (H, [avg_bits] * nb)
        out.append(ent)
    return out


# ---------------- eval data ----------------
def eval_windows(tok, n_windows, seqlen):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t.strip() for t in ds["text"] if len(t.strip()) >= 32)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    wins = []
    for i in range(0, len(ids) - seqlen, seqlen):
        wins.append(torch.tensor(ids[i:i + seqlen], dtype=torch.long))
        if len(wins) >= n_windows:
            break
    return wins


# ---------------- metrics ----------------
def nll_sum(logits, ids):
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[1:]
    nll = -lp[torch.arange(len(tgt), device=lp.device), tgt].sum()
    return float(nll), int(len(tgt))


def kl_fp_q(logits_fp, logits_q):
    p = torch.softmax(logits_fp[:-1].float(), dim=-1)
    lq = torch.log_softmax(logits_q[:-1].float(), dim=-1)
    kl = (p * (torch.log(p + 1e-12) - lq)).sum(dim=-1)
    return float(kl.mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["kquant"] = kquant_attention

    dev = "cuda"
    print(f"[eval] loading {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()
    model.config._attn_implementation = "kquant"

    R_pca, block_var, H, d, nlayers = build_layer_params(args.dumps, dev)
    print(f"[eval] {nlayers} layers, head_dim={d}, hadamard={'yes' if H is not None else 'no'}",
          file=sys.stderr)
    wins = eval_windows(tok, args.windows, args.seqlen)
    print(f"[eval] {len(wins)} eval windows x {args.seqlen} tok", file=sys.stderr)

    schemes = ["hadamard", "pca"] if H is not None else ["pca"]
    acc = {"fp": {"nll": 0.0, "ntok": 0}}
    for b in args.bits:
        for sc in schemes:
            acc[f"{sc}@{b}"] = {"nll": 0.0, "ntok": 0, "kl": 0.0, "nwin": 0}

    for wi, ids in enumerate(wins):
        ids = ids.to(dev)
        STATE["mode"] = "fp"
        STATE["layers"] = None
        logits_fp = model(ids.unsqueeze(0), use_cache=False).logits[0]
        n, t = nll_sum(logits_fp, ids)
        acc["fp"]["nll"] += n
        acc["fp"]["ntok"] += t
        for b in args.bits:
            STATE["layers"] = layers_for_bits(R_pca, block_var, H, b)
            for sc in schemes:
                STATE["mode"] = sc
                logits = model(ids.unsqueeze(0), use_cache=False).logits[0]
                n, t = nll_sum(logits, ids)
                kl = kl_fp_q(logits_fp, logits)
                key = f"{sc}@{b}"
                acc[key]["nll"] += n
                acc[key]["ntok"] += t
                acc[key]["kl"] += kl
                acc[key]["nwin"] += 1
                del logits
        STATE["mode"] = "fp"
        STATE["layers"] = None
        del logits_fp
        torch.cuda.empty_cache()
        print(f"[eval] window {wi + 1}/{len(wins)} done", file=sys.stderr)

    res = {"model": args.model, "dumps": args.dumps, "seqlen": args.seqlen,
           "windows": len(wins), "bits": args.bits, "metrics": {}}
    ppl_fp = math.exp(acc["fp"]["nll"] / acc["fp"]["ntok"])
    res["metrics"]["fp"] = {"ppl": ppl_fp}
    for b in args.bits:
        for sc in schemes:
            k = f"{sc}@{b}"
            a = acc[k]
            res["metrics"][k] = {"ppl": math.exp(a["nll"] / a["ntok"]),
                                 "kl_mean": a["kl"] / a["nwin"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n==================== Phase 0.5: logit-quality verdict ====================")
    print(f"model: {args.model}   eval: wikitext-2 test, {len(wins)}x{args.seqlen} tok   "
          f"(K-cache quantized inline, V fp)")
    print(f"  fp cache ppl: {ppl_fp:.4f}")
    for b in args.bits:
        print(f"  -- {b}-bit avg --")
        row = {sc: res["metrics"][f"{sc}@{b}"] for sc in schemes}
        for sc in schemes:
            m = row[sc]
            print(f"     {sc:9s}  ppl {m['ppl']:.4f} (dfp {m['ppl']-ppl_fp:+.4f})   "
                  f"KL(fp||q) {m['kl_mean']:.5f}")
        if "hadamard" in row and "pca" in row:
            dppl = row["hadamard"]["ppl"] - row["pca"]["ppl"]
            dkl = row["hadamard"]["kl_mean"] - row["pca"]["kl_mean"]
            better = dppl > 0 and dkl > 0
            print(f"     -> PCA vs Hadamard: ppl {dppl:+.4f}, KL {dkl:+.5f}  "
                  f"==> {'PCA WINS (MSE win is real)' if better else 'no PCA win in logit space'}")
    print(f"\n  json: {args.out}")
    print("=========================================================================")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
