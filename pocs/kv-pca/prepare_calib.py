#!/usr/bin/env python3
"""Build a calibration corpus (calib.txt) for extract_kv.py.

A meaningful go/no-go wants representative, ideally long-context text — the KV-cache
statistics at position 2000 differ from those at position 20, and the built-in corpus
in extract_kv.py is a smoke test only. This pulls prose from wikitext via the
`datasets` library and concatenates it to a target size, or assembles a corpus from
local files (no dataset dependency).

Examples:
    python prepare_calib.py --out calib.txt                       # wikitext-103, ~1.2M chars
    python prepare_calib.py --out calib.txt --config wikitext-2-raw-v1
    python prepare_calib.py --out calib.txt --target-chars 3000000
    python prepare_calib.py --out calib.txt --from-files src.py notes.md  # your own domain text
"""
import argparse
import sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="calib.txt")
    ap.add_argument("--config", default="wikitext-103-raw-v1",
                    help="HF wikitext config (use wikitext-2-raw-v1 for a quick/light run)")
    ap.add_argument("--target-chars", type=int, default=1_200_000,
                    help="approx corpus size; ~1.2M chars ~= 300k tokens ~= 64x2048-token seqs")
    ap.add_argument("--from-files", nargs="+", default=None,
                    help="build the corpus from these local text files instead of wikitext "
                         "(no `datasets` dependency; good for workload-representative text)")
    ap.add_argument("--min-line", type=int, default=32,
                    help="skip wikitext lines shorter than this (headers/blanks)")
    return ap.parse_args()


def from_dataset(config, target_chars, min_line):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[calib] `datasets` not installed. Either `pip install datasets`, or "
                 "supply your own text with --from-files <file...>.")
    print(f"[calib] streaming wikitext:{config} up to ~{target_chars:,} chars ...",
          file=sys.stderr)
    ds = load_dataset("wikitext", config, split="train", streaming=True)
    parts, n = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < min_line:        # drop section headers, blank/short lines
            continue
        parts.append(t)
        n += len(t) + 1
        if n >= target_chars:
            break
    if not parts:
        sys.exit("[calib] produced no text — check the dataset config name")
    return "\n".join(parts)


def from_files(paths):
    parts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            parts.append(f.read())
    text = "\n".join(parts)
    if not text.strip():
        sys.exit("[calib] --from-files produced empty text")
    return text


def main():
    args = parse_args()
    if args.from_files:
        text = from_files(args.from_files)
    else:
        text = from_dataset(args.config, args.target_chars, args.min_line)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    nchars = len(text)
    approx_tokens = nchars // 4       # rough English heuristic
    print(f"[calib] wrote {args.out}: {nchars:,} chars (~{approx_tokens:,} tokens, "
          f"~{approx_tokens // 2048} sequences of 2048 tokens)", file=sys.stderr)
    if approx_tokens < 2048 * 16:
        print("[calib] WARNING: small corpus; consider a larger --target-chars or more "
              "--from-files for a meaningful result.", file=sys.stderr)


if __name__ == "__main__":
    main()
