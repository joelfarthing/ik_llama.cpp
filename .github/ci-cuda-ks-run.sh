#!/bin/bash
# Quantize RalphQwen to each KS type twice, without and with --cuda-quantize; print the CUDA lines and compare the files.
set -uo pipefail
exe=$(find build \( -name llama-quantize -o -name llama-quantize.exe \) -type f | head -1)
echo "exe=$exe"
"$exe" --help 2>&1 | grep -- '--cuda-quantize' || true
fail=0
for t in iq3_ks iq4_ks iq5_ks iq4_ks_r4 iq4_ks_r16 iq5_ks_r4; do
  "$exe" ralphqwen.gguf plain-$t.gguf $t 4 > plain-$t.log 2>&1; echo "$t plain rc=$?"
  "$exe" --cuda-quantize ralphqwen.gguf flag-$t.gguf $t 4 > flag-$t.log 2>&1; echo "$t flag rc=$?"
  grep -H -E 'cuda-quantize|CUDA encoder|CUDA backend' flag-$t.log | head -3
  (sha256sum plain-$t.gguf flag-$t.gguf 2>/dev/null || shasum -a 256 plain-$t.gguf flag-$t.gguf) | awk '{print substr($1,1,16), $2}'
  if cmp plain-$t.gguf flag-$t.gguf; then echo "$t plain and flag IDENTICAL"; else fail=1; fi
  rm -f plain-$t.gguf flag-$t.gguf
done
exit $fail
