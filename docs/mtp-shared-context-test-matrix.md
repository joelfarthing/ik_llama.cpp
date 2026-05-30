# MTP Shared Context Test Matrix

This matrix is for splitting and reviewing the shared self-MTP experiment. It
uses command shapes, not a frozen benchmark harness. Capture the exact command,
commit, binary hash, model, log path, HTTP status/timing, accept rate, and MTP
profile buckets for every run.

## Common setup

Build:

```bash
cd /home/joel/Projects/ik_llama.cpp-filament
git diff --check
cmake --build build-cuda --target llama-server
```

Preferred runner shape:

```bash
/home/joel/Projects/Admin/inference/scripts/try-local-gguf.sh start <profile> \
  --filament \
  --env IK_MTP_SERVER_PROFILE=1 \
  --env IK_MTP_DECODE_PROFILE=1 \
  --env GGML_CUDA_NO_PINNED=1 \
  -- <llama-server args>
```

The helper supports `--filament` and repeated `--env KEY=VALUE`. If wrapper
syntax changes, use the same env and args directly against
`build-cuda/bin/llama-server`.

Common shared parallel args:

```text
-np 2 -mtp --draft-max 2 --draft-p-min 0.75
--fit --fit-margin 2400
--cache-type-k q4_0 --cache-type-v f16
-b 512 -ub 512 --attention-max-batch 512
--cache-ram 0 --cont-batching --no-flash-attn
```

Dense guardrail model:

```text
/mnt/gguf/models/Qwen3.6-27B-IQ4_XS.gguf
```

MoE guardrail model:

```text
/mnt/gguf/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
```

For parallel-slot tests, send two simultaneous OpenAI-compatible chat requests
against the server. Record both HTTP statuses and both request times.

## Matrix

| Test | Purpose | Model | Flags/env | Expected result | Metrics to collect | Failure signatures |
| --- | --- | --- | --- | --- | --- | --- |
| Single-slot no MTP | Baseline server health with no speculative path. | Dense and MoE if time; at least the active model. | `-np 1`, no `-mtp`, no shared MTP env. Keep profiling env off for the pure baseline. | One request returns `200`; no MTP accept or draft buckets should be required. | HTTP time, tokens/sec, server stderr, CUDA errors. | Non-200, crash, invalid logits, unexpected MTP profile output. |
| Single-slot MTP auto/per-step | Prove normal MTP remains on the existing fast path. | Dense and MoE Qwen3.6. | `-np 1 -mtp --draft-max 2 --draft-p-min 0.75 --recurrent-ckpt-mode auto`; profile env on; no shared context env. | Selected checkpoint mode is `per-step`; re-decode is `0.000ms` or near-zero; request returns `200`. | Accept rate, `ckpt_save`, `restore`, `redecode`, selected checkpoint mode. | Auto selects `cpu` or `gpu-fallback` unexpectedly, re-decode fires on accepted-prefix restore, request fails. |
| Single-slot MTP explicit per-step | Control for auto mode. | Dense Qwen3.6 first; MoE if time. | Same as above with `--recurrent-ckpt-mode per-step`. | Matches auto within noise; request returns `200`. | Accept rate, save/restore/redecode, selected checkpoint mode. | Per-step allocation failure, non-200, CUDA copy error. |
| Single-slot MTP gpu-fallback | Prove fallback remains safe and quantify cost. | Dense and MoE Qwen3.6. | `-np 1 -mtp ... --recurrent-ckpt-mode gpu-fallback`; profile env on. | Request returns `200`; restore/redecode cost appears; slower than per-step but much safer than CPU when per-step is unavailable. | Accept rate, `ckpt_save`, `restore`, `fallback_call`, `redecode`, HTTP time. | Non-200, stale hidden state after restore, invalid logits, mixed-sequence logs. |
| Single-slot MTP CPU checkpoint | Show why CPU serialized checkpoints should not be the shared parallel fallback. | Dense and MoE Qwen3.6. | `-np 1 -mtp ... --recurrent-ckpt-mode cpu`; profile env on. | Request returns `200`, but `ckpt_save` is large. This is a cost demonstration, not a target mode. | HTTP time, `ckpt_save`, `restore`, `redecode`, accept rate. | CPU path corrupts state, crash, misleadingly selected as shared parallel fallback. |
| Shared parallel self-MTP without range checkpoint | Guard the old shared parallel fallback behavior and compare with range. | Dense and MoE Qwen3.6. | Common shared args, `IK_MTP_SHARED_CONTEXT=1`, `IK_MTP_BATCH_DRAFT=1`, profile env on, no `IK_MTP_RANGE_CKPT`. | Two concurrent requests return `200/200`. Checkpoint mode should not silently use unsafe per-step in shared parallel mode. | Both HTTP times, accept rates, `spec_draft_batched`, `ckpt_save`, `restore`, `redecode`, selected checkpoint mode. | Empty reply, CUDA `invalid argument`, invalid logits, mixed-sequence logs, stale slot reuse. |
| Shared parallel self-MTP with range checkpoint | Main range-checkpoint evidence path. | Dense and MoE Qwen3.6. | Common shared args plus `IK_MTP_SHARED_CONTEXT=1`, `IK_MTP_BATCH_DRAFT=1`, `IK_MTP_RANGE_CKPT=1`, profile env on. | Two concurrent requests return `200/200`; range checkpoint save occurs only for isolated slot ranges; shared parallel mode uses `gpu-fallback`, not per-step. | Both HTTP times, accept rates, `spec_draft_batched`, `ckpt_save`, `restore`, `fallback_call`, `redecode`, selected mode, range checkpoint log lines. | Per-step selected in shared parallel mode, CUDA copy error, no range checkpoints, checkpoint save still CPU-sized, mixed-sequence logs. |
| Dense Qwen3.6 no-flash-attn guardrail | Prevent dense regressions and keep claims honest. | Dense Qwen3.6 27B IQ4_XS. | Common shared args with `--no-flash-attn`, shared/batch/range env on. | `200/200`; no CUDA/assert/invalid-logit/mixed-sequence failure. Do not require wall-time win. | HTTP times, accept rates, `ckpt_save`, `restore`, `redecode`, `spec_draft_batched`. | Any crash, invalid logits, mixed sequences, dense wall-time claim without matching restore/redecode evidence. |
| MoE Qwen3.6 no-flash-attn guardrail | Preserve strongest positive evidence. | Qwen3.6 35B A3B UD IQ2_M. | Common shared args with `--no-flash-attn`, shared/batch/range env on. | `200/200`; checkpoint save is low under range; wall time should remain in the same rough band as the final MoE smoke unless acceptance shape changes heavily. | HTTP times, accept rates, `ckpt_save`, `restore`, `redecode`, `spec_draft_batched`. | Checkpoint save regresses toward CPU-sized values, mixed-sequence logs, invalid logits, CUDA error. |
| Slot reuse | Catch stale shared companion KV or hidden state after request completion. | MoE first, dense second if time. | Run the shared/batch/range server, then send a second pair of short requests without restarting the server. | Both reuse requests return `200/200`; no stale continuation, no target checkpoint restore/reset contamination. | HTTP times, output sanity, accept rates if any, cleanup logs from `release_slot()` and companion clear helpers. | Previous prompt bleeds into next request, invalid logits, mixed sequence, companion KV not cleared for released slot. |
| Cancellation | Catch cleanup asymmetry when a request is aborted. | MoE first. | Shared/batch/range server; start two long requests, cancel one client mid-generation, allow the other to finish. | Remaining request returns `200`; canceled slot is released; later reuse of that slot is clean. | Server logs around `release_slot(... "task cancellation")`, companion clear logs, HTTP outcome for later reuse. | Canceled slot leaves draft KV/hidden rows, later request fails, mixed sequence, crash during cleanup. |
| Prompt cache | Catch target-only restore/reset invalidating target but not companion state. | Dense if practical because restore/redecode cost is visible; MoE if runtime is tight. | Enable prompt-cache shape used by the server, then restore or reuse a saved prompt with shared/batch/range env on. | Target-only prompt cache restore clears companion state; subsequent generation returns `200`. | Logs around `target-only prompt cache restore`, checkpoint restore/reset, HTTP time, output sanity. | Stale companion KV after target restore, invalid logits, mixed sequence, wrong continuation. |
| Context shift or prompt trim | Exercise companion shift/trim symmetry. | Dense if practical. | Small context or prompt shape that forces context shift/trim; shared/batch/range env on. | Target and companion state shift/trim consistently or companion state is fully cleared. | Logs from `shift_slot_companion_state()` or `trim_slot_companion_state()`, output sanity, HTTP status. | Target shifts but companion does not, invalid logits, stale continuation, mixed sequence. |
| Flash-attention known-failing control | Keep the separate FA crash out of shared-MTP claims. | Dense Qwen3.6 27B IQ4_XS. | `-np 2`, flash attention enabled, no MTP, no shared MTP env. | This may reproduce the known crash and should not be part of pass/fail for shared MTP. Run only as an explicit control. | HTTP statuses, crash signature, server stderr. | Known signature: `iqk_fa_templates.h:1175: GGML_ASSERT(S > 0) failed`; empty reply on one request. |

## Minimum pass set for a review branch

Profiler-only branch:

- Build passes.
- Profile env off is quiet.
- Profile env on prints expected buckets.

Shared-context lifecycle branch:

- Build passes.
- Dense `-np 2`, `--no-flash-attn`, shared context smoke returns `200/200`.
- MoE `-np 2`, `--no-flash-attn`, shared context smoke returns `200/200`.
- Slot reuse returns `200/200`.

Range-checkpoint branch:

- Build passes.
- Same-binary off/on A/B for MoE and dense.
- MoE shows checkpoint-save reduction and wall-time win.
- Dense shows checkpoint-save reduction or at least no correctness regression;
  wall-time win is not required.
- Single-slot MTP auto still selects per-step.

Batched-draft branch:

- Build passes.
- Dense and MoE `-np 2`, `--no-flash-attn` smokes return `200/200`.
- `spec_draft_batched` bucket is nonzero.
- Non-narrow input shape falls back to the regular per-slot draft path.
