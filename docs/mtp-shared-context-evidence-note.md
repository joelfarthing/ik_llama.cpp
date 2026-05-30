# MTP Shared Context Evidence Note

This note summarizes evidence for the shared self-MTP experiment. It is not a
PR description and it should not be read as a general performance claim.

## Sources

- Branch inspected: `filament/mtp-batched-draft-hardening-20260530`
- Public reference branch:
  `https://github.com/joelfarthing/ik_llama.cpp/tree/filament/mtp-shared-context-experiment-20260530`
- HEAD: `bea9b4ffa6bd`
- Binary hash in final smokes: `df068e674cb3f0cf`
- Autopsy note:
  `/home/joel/Sync/vault/inference/archive/2026/mtp-profiler-autopsy-2026-05-24.md`
- Current experiment note:
  `/home/joel/Sync/vault/inference/_current-experiment.md`
- Logs live under:
  `/home/joel/Projects/Admin/inference/logs/vagrant-gguf/`

## Proved by current logs

### Shared self-MTP context is viable under a narrow env-gated shape

The latest branch served concurrent `-np 2` requests on both MoE and dense
Qwen3.6 with:

- `IK_MTP_SHARED_CONTEXT=1`
- `IK_MTP_BATCH_DRAFT=1`
- `IK_MTP_RANGE_CKPT=1`
- `IK_MTP_SERVER_PROFILE=1`
- `IK_MTP_DECODE_PROFILE=1`
- `GGML_CUDA_NO_PINNED=1`
- `--no-flash-attn`
- `--fit --fit-margin 2400`
- `--cache-type-k q4_0 --cache-type-v f16`
- `-b 512 -ub 512 --attention-max-batch 512`
- `--cache-ram 0`
- `--cont-batching`

Final MoE smoke:

- Log:
  `20260530_084233-filament-qwen3.6-35b-a3b-ud-iq2_m.log`
- Model: Qwen3.6 MoE 35B A3B UD IQ2_M
- HTTP: `200/200`
- Request times: `2.907s`, `3.027s`
- Accept rates: `44/68 = 0.647`, `40/68 = 0.588`
- Profile buckets:
  `spec_draft_batched=50/100/73.253ms`,
  `ckpt_save=105/105/60.439ms`,
  `restore=46/46/627.153ms`,
  `redecode=46/46/527.851ms`
- Same-server slot reuse after the main smoke returned `200/200`
  (`0.341s`, `0.864s`) and exercised target checkpoint restore/reset paths.
- No CUDA/assert/invalid-logit/mixed-sequence failure was recorded.

Final dense smoke:

- Log:
  `20260530_084356-filament-qwen3.6-27b-iq4_xs.log`
- Model: dense Qwen3.6 27B IQ4_XS
- HTTP: `200/200`
- Request times: `15.234s`, `19.978s`
- Accept rates: `14/28 = 0.500`, `51/63 = 0.810`
- Profile buckets:
  `spec_draft_batched=24/48/55.255ms`,
  `ckpt_save=68/68/644.869ms`,
  `restore=22/24/4362.895ms`,
  `redecode=22/24/4080.757ms`
- No CUDA/assert/invalid-logit/mixed-sequence failure was recorded.

Read: the shared self-MTP path is viable in this narrow no-flash-attention,
env-gated, `-np 2` shape. This does not prove it is ready as a broad default.

### Batched companion drafting works for `-np 2` self-MTP

The batched path executed in both dense and MoE guardrails, not merely fallback:

| Shape | Log | HTTP | Key bucket | Accept rates |
| --- | --- | --- | --- | --- |
| dense no-FA batched guardrail | `20260530_064238-filament-qwen3.6-27b-iq4_xs.log` | `200/200` (`3.25s`, `9.97s`) | `spec_draft_batched=3/6/16.707ms` | one logged rate `24/32 = 0.75000` |
| MoE no-FA batched guardrail | `20260530_065134-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `200/200` (`2.11s`, `2.23s`) | `spec_draft_batched=22/44/33.182ms` | `25/33 = 0.75758`, `22/30 = 0.73333` |
| final dense lifecycle/range | `20260530_084356-filament-qwen3.6-27b-iq4_xs.log` | `200/200` | `spec_draft_batched=24/48/55.255ms` | `0.500`, `0.810` |
| final MoE lifecycle/range | `20260530_084233-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `200/200` | `spec_draft_batched=50/100/73.253ms` | `0.647`, `0.588` |

Read: batched companion drafting works for the narrow shared self-MTP `-np 2`
case. It is still the highest-complexity shard because it touches batching,
sequence IDs, hidden-state row shape, and shared companion context state.

### Range-scheduled checkpoint save is the cleanest measured improvement

Initial same-binary A/B:

| Model | Gate | Log | HTTP times | Mode | Accept rates | Checkpoint save | Restore | Re-decode |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| dense Qwen3.6 | off | `20260530_073435-filament-qwen3.6-27b-iq4_xs.log` | `3.23s`, `9.90s` | `cpu` | `4/5`, `24/32` | `1418.589ms` | `1833.661ms` | `1657.933ms` |
| dense Qwen3.6 | `IK_MTP_RANGE_CKPT=1` | `20260530_073504-filament-qwen3.6-27b-iq4_xs.log` | `3.06s`, `8.44s` | `gpu-fallback` | `4/5`, `23/29` | `245.495ms` | `1382.289ms` | `1293.236ms` |
| MoE Qwen3.6 | off | `20260530_073530-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `2.52s`, `2.56s` | `cpu` | `20/32`, `19/38` | `714.723ms` | `466.332ms` | `275.928ms` |
| MoE Qwen3.6 | `IK_MTP_RANGE_CKPT=1` | `20260530_073554-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `1.73s`, `1.77s` | `gpu-fallback` | `20/35`, `19/34` | `30.682ms` | `363.282ms` | `307.691ms` |

Read:

- Dense initial A/B checkpoint save dropped from `1418.589ms` to `245.495ms`
  and the longer request dropped from `9.90s` to `8.44s`.
- MoE initial A/B checkpoint save dropped from `714.723ms` to `30.682ms` and
  max paired wall time dropped from `2.56s` to `1.77s`.

Post-hardening confidence repeats:

| Model | Gate | Logs | Median max HTTP | Checkpoint save | Restore | Re-decode | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| dense Qwen3.6 | off | `20260530_074358`, `20260530_074439`, `20260530_074513` | `10.90s` | `813.449ms` | `2118.414ms` | `1987.385ms` | CPU save tax visible |
| dense Qwen3.6 | `IK_MTP_RANGE_CKPT=1` | `20260530_074419`, `20260530_074453`, `20260530_074526` | `16.20s` | `461.575ms` | `4228.036ms` | `3955.380ms` | save lower, wall not better |
| MoE Qwen3.6 | off | `20260530_074544`, `20260530_074553`, `20260530_074602` | `2.20s` | `630.872ms` | `379.637ms` | `200.854ms` | CPU save tax visible |
| MoE Qwen3.6 | `IK_MTP_RANGE_CKPT=1` | `20260530_074548`, `20260530_074558`, `20260530_074607` | `1.65s` | `28.425ms` | `279.752ms` | `236.073ms` | robust wall win |

Read:

- MoE Qwen3.6 shows a robust wall-time win: median max request went from
  `2.20s` to `1.65s`, while checkpoint save went from `630.872ms` to
  `28.425ms`.
- Dense Qwen3.6 shows checkpoint-save reduction, but wall time is not stable.
  Restore/redecode dominates in the range-on repeats.

### Normal single-slot MTP already uses per-step checkpointing

Single-slot checkpoint mode scout:

| Model | Mode | Log | HTTP | Selected mode | Accept | Save | Restore | Re-decode |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| dense Qwen3.6 | `auto` | `20260530_072824-filament-qwen3.6-27b-iq4_xs.log` | `3.33s` | `per-step` | `17/18` | `0.101ms` | `8.361ms` | `0.000ms` |
| dense Qwen3.6 | explicit `per-step` | `20260530_072940-filament-qwen3.6-27b-iq4_xs.log` | `3.33s` | `per-step` | `17/18` | `0.095ms` | `8.357ms` | `0.000ms` |
| dense Qwen3.6 | `gpu-fallback` | `20260530_072850-filament-qwen3.6-27b-iq4_xs.log` | `3.74s` | `gpu-fallback` | `17/19` | `61.598ms` | `396.657ms` | `380.360ms` |
| dense Qwen3.6 | `cpu` | `20260530_072916-filament-qwen3.6-27b-iq4_xs.log` | `4.48s` | `cpu` | `17/19` | `779.880ms` | `421.337ms` | `377.714ms` |
| MoE Qwen3.6 | `auto` | `20260530_073007-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `0.67s` | `per-step` | `22/27` | `0.157ms` | `9.466ms` | `0.000ms` |
| MoE Qwen3.6 | `gpu-fallback` | `20260530_073028-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `0.69s` | `gpu-fallback` | `23/26` | `6.108ms` | `47.813ms` | `42.320ms` |
| MoE Qwen3.6 | `cpu` | `20260530_073048-filament-qwen3.6-35b-a3b-ud-iq2_m.log` | `1.35s` | `cpu` | `23/26` | `604.533ms` | `82.043ms` | `42.200ms` |

Read:

- Normal single-slot MTP already selects `per-step` and should not be moved to
  range scheduling or CPU serialization.
- CPU serialized checkpoints are toxic when forced.
- `gpu-fallback` is a better fallback than CPU when `per-step` is unsafe.

### Dense Qwen3Next flash-attention crash is separate

Known failing control:

- Log:
  `20260530_064159-filament-qwen3.6-27b-iq4_xs.log`
- Shape: dense Qwen3.6, `-np 2`, no MTP, flash attention enabled
- Result: one request returned `200`; the other got an empty reply; server
  crashed.
- Signature:
  `iqk_fa_templates.h:1175: GGML_ASSERT(S > 0) failed`

Read: this is a dense Qwen3Next parallel-slot flash-attention crash, not a
shared-MTP proof or disproof. Keep `--no-flash-attn` in the shared-MTP evidence
until that separate issue is handled.

### Shared parallel per-step checkpointing is not proved safe

Failed scout:

- Log:
  `20260530_071203-filament-qwen3.6-27b-iq4_xs.log`
- Shape: dense Qwen3.6 with `IK_MTP_RANGE_CKPT=1` before forcing
  `gpu-fallback`, so `auto` selected `per-step`.
- Result: both concurrent requests got empty replies.
- Signature: CUDA `invalid argument` in
  `ggml_backend_cuda_cpy_tensor_async`.

Read: range isolation alone is not enough to make the current per-step
checkpoint buffers safe for shared parallel MTP. Keep `per-step` out of shared
parallel mode until the checkpoint state is truly split per slot.

## Hypotheses, not proved

- A future split-state per-step implementation could beat `gpu-fallback` in
  shared parallel MTP. The current branch does not prove it.
- Dense Qwen3.6 may improve if restore/redecode is reduced, but the current
  branch does not prove stable dense wall-time improvement.
- External draft-model sharing may be architecturally similar to self-MTP
  ownership, but this branch does not validate external draft behavior.

## Do not claim

- Do not claim a general speedup.
- Do not claim dense Qwen3.6 is faster.
- Do not claim flash-attention is fixed.
- Do not claim shared parallel `per-step` checkpointing is safe.
- Do not claim external draft-model behavior is covered.
- Do not claim this is ready as one PR.
