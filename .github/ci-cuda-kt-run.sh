#!/bin/bash
# Quantize RalphQwen with the v3 recipe shape twice, without and with --cuda-quantize; print the fallback lines and hashes.
set -uo pipefail
exe=$(find build \( -name llama-quantize -o -name llama-quantize.exe \) -type f | head -1)
echo "exe=$exe"
"$exe" --help 2>&1 | grep -- '--cuda-quantize' || true
args=(--token-embedding-type q8_0 --custom-q 'ffn_down_exps=iq3_kt,ffn_down_shexp=iq3_kt' ralphqwen.gguf)
"$exe" "${args[@]}" plain.gguf iq4_kt 4 > plain.log 2>&1; echo "plain rc=$?"
"$exe" --cuda-quantize "${args[@]}" flag.gguf iq4_kt 4 > flag.log 2>&1; echo "flag rc=$?"
grep -H -E 'cuda-quantize|KT encoder|CUDA|quantize time' plain.log flag.log
(sha256sum plain.gguf flag.gguf 2>/dev/null || shasum -a 256 plain.gguf flag.gguf) | awk '{print substr($1,1,16), $2}'
cmp plain.gguf flag.gguf && echo "plain and flag IDENTICAL"
