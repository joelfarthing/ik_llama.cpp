# MTP Shared Context Maintainer Discussion Draft

I investigated `TAG_SERVER_SPEC_REWORK` in a public experimental branch:

`https://github.com/joelfarthing/ik_llama.cpp/tree/filament/mtp-shared-context-experiment-20260530`

The branch is intentionally not a PR yet. It is a prototype to answer the
architecture question: how should ik_llama structure ownership for a shared
self-MTP draft/companion context before the work is split into reviewable
patches?

What changed:

- `server_context` owns a shared self-MTP companion context for the env-gated
  shared path.
- Slots keep per-slot speculative state: seq id, draft tokens, validation
  indices, recurrent checkpoint state, hidden-state validity, sampler state, and
  counters.
- Shared companion cleanup is explicit across release, cancellation, prompt
  reuse, prompt-cache restore, slot restore, context shift, KV clear, and rewind
  paths.
- An env-gated batched draft prototype batches one shared MTP draft step across
  `-np 2` slots when the input shape is narrow enough.
- An env-gated range checkpoint path defers recurrent checkpoint saves to
  isolated slot ranges and uses `gpu-fallback` for shared parallel MTP.

What was measured:

- Final MoE Qwen3.6 shared/batched/range smoke on commit `bea9b4ff` returned
  `200/200` with no CUDA/assert/invalid-logit/mixed-sequence failure:
  `20260530_084233-filament-qwen3.6-35b-a3b-ud-iq2_m.log`.
- Final dense Qwen3.6 no-flash-attention smoke returned `200/200` with no
  correctness failure, but restore/redecode still dominated wall time:
  `20260530_084356-filament-qwen3.6-27b-iq4_xs.log`.
- MoE range-checkpoint repeats showed the strongest result: median max request
  time went from `2.20s` to `1.65s`, and checkpoint save from `630.872ms` to
  `28.425ms`.
- Dense range-checkpoint repeats reduced checkpoint save, but did not show a
  stable wall-time win because restore/redecode grew.
- Single-slot MTP already selects `per-step` and should stay on that path.
- Shared parallel `per-step` is not yet safe; a scout hit CUDA `invalid
  argument` in `ggml_backend_cuda_cpy_tensor_async`.
- Dense Qwen3Next flash-attention parallel-slot crash is separate and the
  shared-MTP evidence uses `--no-flash-attn`.

What seems worth splitting first:

1. Shared self-MTP context ownership/lifecycle, without batching or range
   checkpoint policy.
2. Range-scheduled checkpoint save, because it has the cleanest measurement and
   preserves normal single-slot per-step behavior.
3. Batched shared companion draft, after the ownership model is agreed.

Open questions for Iwan:

- Should shared self-MTP context ownership live directly in `server_context`, or
  should it be wrapped in a small companion-context owner object?
- Is the right slot contract "slot owns state, server owns context" for MTP
  self-draft, matching upstream's shared draft-context direction while keeping
  ik_llama's MTP hidden-state and checkpoint plumbing?
- Should range checkpointing be reviewed as a stacked follow-up after shared
  ownership, or does it need to be part of the first shared-context patch to
  keep shared parallel MTP safe?
- Is `gpu-fallback` the acceptable shared parallel fallback until per-step
  checkpoint buffers are split per slot?
- Should the profiler buckets stay as a temporary review aid, or should they be
  kept only in Joel's experimental branches and removed from the final review
  series?
- Should external draft-model sharing be explicitly left out of this series
  until the self-MTP ownership model is settled?
