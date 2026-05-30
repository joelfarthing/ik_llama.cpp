# MTP Shared Context PR Splitting Plan

## Executive summary

This branch is useful, but it is not reviewable as one PR. It mixes profiler
instrumentation, shared self-MTP context ownership, batched draft execution,
range-scheduled checkpointing, and lifecycle hardening in one 1.6k-line diff.
The right next step is to preserve the branch as a forensic reference and
extract smaller branches with measurement-backed claims.

The maintainer-facing story should be:

- Shared self-MTP context ownership is the architecture question.
- Range-scheduled checkpoint save is the cleanest measured win.
- Batched companion draft works, but is the highest-complexity shard and should
  come after the ownership model is accepted.
- Dense Qwen3.6 must remain a guardrail, not a speedup claim.
- Normal single-slot MTP already has the good per-step checkpoint path and
  should remain untouched.

## Branch inspected

- Worktree: `/home/joel/Projects/ik_llama.cpp-filament`
- Local branch: `filament/mtp-batched-draft-hardening-20260530`
- Public reference branch: `joel/filament/mtp-shared-context-experiment-20260530`
- HEAD inspected: `bea9b4ffa6bd`
- Merge base with `origin/main`: `8960c5ba5ee9`
- `origin/main` at inspection: `3f40e73c367a`
- Relationship to `origin/main`: ahead 7, behind 1
- Worktree at inspection: clean

Important: `origin/main` now includes `3f40e73c` (`expand np guardrail for all
mtp types (#1901)`) after this branch base. Expect conflicts in
`examples/server/server-context.cpp` when replaying onto fresh `origin/main`.

## Current diff map

`git diff --stat origin/main...HEAD`:

| File | Delta | Primary shard |
| --- | ---: | --- |
| `common/speculative.cpp` | 560 insertions, 12 deletions | A, B, C |
| `common/speculative.h` | 18 insertions, 1 deletion | B, C |
| `examples/server/server-context.cpp` | 797 insertions, 121 deletions | A, B, C, D, E |
| `examples/server/server-context.h` | 16 insertions, 3 deletions | B, E |
| `src/graphs/build_glm4.cpp` | 1 insertion, 1 deletion | C |
| `src/graphs/build_qwen35.cpp` | 2 insertions, 2 deletions | C |
| `src/llama-spec-features.h` | 3 insertions, 1 deletion | A |
| `src/llama.cpp` | 230 insertions, 1 deletion | A |

No obviously unrelated file was found. The graph-builder changes are tied to
batched hidden-state rows for shared MTP drafting. The `llama.cpp` and
`llama-spec-features.h` changes are decode profiling.

## Proposed PR shards

| Shard | Suggested branch | Current commits | Risk | Summary |
| --- | --- | --- | --- | --- |
| A. Env-gated profiler instrumentation | `filament/mtp-profiler-instrumentation` | `fc90dda8`, `63d97f0e` | Low to medium | Adds server and decode profile buckets behind `IK_MTP_SERVER_PROFILE`, `IK_MTP_PROFILE`, and `IK_MTP_DECODE_PROFILE`. |
| B. Shared companion context ownership/lifecycle | `filament/mtp-shared-context-lifecycle` | `34ba0e2d` plus selected hunks from `35b7374d`, `bea9b4ff` | High | Makes `server_context` own `ctx_mtp_shared`; slots keep per-slot speculative state. |
| C. Batched shared companion draft | `filament/mtp-batched-draft` | `b7975509` | High | Adds `common_speculative_draft_mtp_batch` and server batching for one shared MTP step across slots. |
| D. Range-scheduled checkpoint save | `filament/mtp-range-checkpoint` | `1f0d5518` | Medium | Defers recurrent checkpoint saves to isolated slot ranges and forces `gpu-fallback` for shared parallel MTP. |
| E. Cleanup/hardening | fold into B | `35b7374d`, `bea9b4ff` | Medium to high | Symmetric cleanup for release, cancellation, prompt-cache restore, slot restore, context shift, rewind, and KV clear. |
| F. Accidental/unrelated changes | none found | none | N/A | Nothing obvious to split out or discard at the file level. |

## Recommended order

For discussion, show the evidence note and the range-checkpoint result first.
It has the cleanest measurement and the narrowest performance claim.

For code extraction, use this order:

1. Profiler instrumentation, if Iwan is willing to review temporary
   env-gated diagnostics. This gives later shards their measurement surface.
2. Shared context ownership and lifecycle. This is the architecture decision
   required by `TAG_SERVER_SPEC_REWORK`.
3. Range-scheduled checkpoint save. It depends on the shared parallel shape and
   has the strongest MoE result.
4. Batched shared companion draft. This is real, but it is the easiest shard to
   derail in review because it touches batching, sequence IDs, graph shapes, and
   shared context state all at once.

Do not lead with the full branch. Do not lead with batched draft. Do not claim a
general dense speedup.

## Shard details

### A. Env-gated profiler instrumentation

What it does:

- Adds MTP server profile buckets in `examples/server/server-context.cpp`.
- Adds speculative profile buckets in `common/speculative.cpp`.
- Adds decode profile probes in `src/llama.cpp`.
- Exposes `llama_mtp_decode_profile_print()` through
  `src/llama-spec-features.h`.

Must not include:

- Behavioral changes to speculative acceptance.
- Shared context ownership changes.
- Range checkpoint selection changes.

Validation required:

- `git diff --check`
- `cmake --build build-cuda --target llama-server`
- One short run with profile env off to confirm no visible log churn.
- One short run with profile env on to confirm bucket output.

Likely maintainer objection:

- "Temporary env-gated profiling does not belong in final code."

Recommended response:

- Agree. This shard can be kept out of final PRs if needed. It exists to make
  the experiment auditable and to avoid making performance claims from wall
  time alone.

### B. Shared companion context ownership/lifecycle

What it does:

- Replaces per-slot draft context ownership in the shared self-MTP path with
  `server_context::ctx_mtp_shared`.
- Leaves slot-local state per slot: seq id, speculative tokens, validation
  indices, sampler state, recurrent checkpoint state, hidden-state validity,
  and counters.
- Routes companion lookup through `companion_ctx_for_slot()`.
- Adds explicit companion cleanup on release, cancellation, prompt-cache
  target-only restore, slot restore, slot erase, context shift, prompt reuse,
  rewind, and KV clear.

Must not include:

- Batched drafting.
- Range checkpoint scheduling.
- External draft-model sharing, unless explicitly split and tested.
- Dense restore/redecode optimization.

Validation required:

- `git diff --check`
- `cmake --build build-cuda --target llama-server`
- Dense Qwen3.6, `-np 2`, `--no-flash-attn`, shared self-MTP gate on.
- MoE Qwen3.6, `-np 2`, `--no-flash-attn`, shared self-MTP gate on.
- Same-server slot reuse smoke.
- Prompt-cache or target-only restore smoke.
- Cancellation smoke if practical.
- Explicit check for no invalid logits, mixed-sequence logs, CUDA errors, or
  stale continuation between slots.

Likely maintainer objection:

- "A shared draft context is dangerous unless cleanup is obviously symmetric."

Recommended response:

- Agree. Make the ownership diff small, route every target-only invalidation
  through named helpers, and require slot-reuse and prompt-cache tests before
  review.

### C. Batched shared companion draft

What it does:

- Adds `common_speculative_mtp_draft_input`.
- Adds `common_speculative_draft_mtp_batch()`.
- Batches one MTP draft step across multiple slots when all inputs are the
  narrow shared self-MTP shape.
- Adjusts Qwen/GLM graph builders so hidden-state tensors are 2D when
  `n_tokens > 1`.

Must not include:

- Checkpoint policy changes.
- Profiler-only churn beyond metrics necessary to prove the path executes.
- Any claim that arbitrary draft models are batched.

Validation required:

- Dense and MoE Qwen3.6 `-np 2` smokes with `IK_MTP_BATCH_DRAFT=1`.
- Verify `spec_draft_batched` buckets are nonzero.
- Verify fallback to normal per-slot draft when inputs are not the narrow MTP
  shared-context shape.
- Verify no repeated-sequence or mixed-sequence warnings.
- Verify hidden-state rows map to the intended slot/sequence.

Likely maintainer objection:

- "This is too much batching logic for one speculative decoder patch."

Recommended response:

- Keep it behind a separate env gate and review it only after shared ownership
  is accepted. The narrow fallback contract is the important part: if the input
  shape is not pure shared self-MTP, it should decline and use the existing path.

### D. Range-scheduled checkpoint save

What it does:

- Adds range-level speculative checkpoint save for shared parallel self-MTP.
- Saves recurrent checkpoints only when a batch range maps to a single
  speculative slot.
- Forces `gpu-fallback` instead of `per-step` for shared parallel MTP, because
  the current per-step buffers are context-global and failed under shared
  parallel use.

Must not include:

- A claim that dense Qwen3.6 wall time is fixed.
- Changes to normal single-slot `auto` or `per-step` checkpoint behavior.
- CPU fallback as the preferred shared parallel path.

Validation required:

- Same-binary A/B with `IK_MTP_RANGE_CKPT` off/on.
- MoE Qwen3.6 `-np 2`, `--no-flash-attn`.
- Dense Qwen3.6 `-np 2`, `--no-flash-attn`.
- Single-slot MTP auto/per-step control proving the normal path still selects
  per-step.
- Verify selected checkpoint mode logs say `gpu-fallback` in shared parallel
  range mode.

Likely maintainer objection:

- "Why is per-step disabled in shared parallel mode?"

Recommended response:

- The failed scout hit CUDA `invalid argument` in
  `ggml_backend_cuda_cpy_tensor_async` when shared parallel MTP auto-selected
  per-step. Range isolation alone was not enough. Until per-step state is truly
  split per slot, `gpu-fallback` is the safer fallback.

### E. Cleanup/hardening

What it does:

- Centralizes `release_slot()`.
- Adds `clear_slot_companion_state()`, `trim_slot_companion_state()`, and
  `shift_slot_companion_state()`.
- Makes failures clear or invalidate companion state explicitly instead of
  silently proceeding with stale shared MTP state.

Must not include:

- Feature expansion.
- New performance claims.
- Broad server refactors outside shared companion invalidation.

Validation required:

- Slot reuse.
- Cancellation.
- Prompt-cache load/save.
- Context shift.
- String-ban rewind.
- KV clear.

Likely maintainer objection:

- "This looks like unrelated server churn."

Recommended response:

- Fold it into the shared-context lifecycle PR and show each call site as a
  target-state invalidation that must also invalidate the companion state.

## Suggested clean branch commands

Use the current experimental branch as the forensic source. Start from fresh
`origin/main` for review branches, but expect conflicts from `3f40e73c`.

```bash
cd /home/joel/Projects/ik_llama.cpp-filament
git fetch --prune origin joel
git branch filament/mtp-shared-context-forensics bea9b4ffa6bd
```

Profiler instrumentation:

```bash
git switch --detach origin/main
git switch -c filament/mtp-profiler-instrumentation
git cherry-pick -x fc90dda8 63d97f0e
git diff --check
cmake --build build-cuda --target llama-server
```

Shared context lifecycle:

```bash
git switch --detach origin/main
git switch -c filament/mtp-shared-context-lifecycle
git cherry-pick -x -n 34ba0e2d 35b7374d bea9b4ff
# Resolve server-context.cpp conflicts from origin/main.
# Drop batched-draft and range-checkpoint hunks if they appear through overlap.
git diff --check
cmake --build build-cuda --target llama-server
git commit -m "server: add shared self-MTP companion context lifecycle"
```

Range checkpoint:

```bash
git switch filament/mtp-shared-context-lifecycle
git switch -c filament/mtp-range-checkpoint
git cherry-pick -x 1f0d5518
git diff --check
cmake --build build-cuda --target llama-server
```

If a branch must literally start from `origin/main`, recreate the lifecycle
prerequisite first and then apply `1f0d5518`. Reviewing range checkpoint on top
of the lifecycle branch will be much cleaner.

Batched draft:

```bash
git switch filament/mtp-shared-context-lifecycle
git switch -c filament/mtp-batched-draft
git cherry-pick -x b7975509
# Keep the graph-builder hidden-state shape changes in this branch.
# Keep range-checkpoint changes out unless this branch is explicitly stacked.
git diff --check
cmake --build build-cuda --target llama-server
```

The current commit stack does not map perfectly to final PRs. `35b7374d` and
`bea9b4ff` are cleanup commits that overlap with earlier architecture and later
range/batch call sites. Manual extraction is safer than blindly preserving the
exact experimental commits.
