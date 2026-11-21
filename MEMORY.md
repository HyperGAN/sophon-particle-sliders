# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| CPU-only current thinker response training | 0 | 0 | 1420 | CPU BF16 base/FP32 branches; batch 2; ≤ 4096 tokens; no checkpointing; text-only current Source |
| CPU thinker short-context planning profile | 0 | 0 | 110 | CPU BF16 base/FP32 branches; batch 2; ≤ 256 tokens; eager; no checkpointing |
| Original speech teacher/preparation or finished audio: explicit 32/32 | 2 | 56/66 | 108 | two GPUs; full 64-layer BF16 base; FP32 talker/E2 upper; total context≤ 4096 positions and≤ 512nativeframes per example; no CPU offload |
| Cached acoustic reg/FM response training only | 2 | 0/4 | 8 | 2-device owner topology; no 64-layer trunk/E2; complete frozen FP32 reg+FM+whitening; ≤ 512 cached5120-wide rows; batch 2 |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.11.1 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **AdventistsAI/Sophon-Realtime-v1**, revision **`c6e664d8ddca6522ef905fa170d9ca6cfa5dd568`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| CPU-only current thinker response training | frozen_file_byte_upper: 61.4015; frozen_tag_upper: 1.0003; trainable_upper: 6.7452; activation_envelope: 1334.0000; workspace: 16.0000 |
| CPU thinker short-context planning profile | frozen_file_byte_upper: 61.4015; tag_upper: 1.0003; trainable_transient: 6.7452; saved_activations: 23.3750; workspace: 16.0000 |
| Original speech teacher/preparation or finished audio: explicit 32/32 | frozen_trunk_each_upper: 30.7007; GPU0_decoder_FP32_upper: 6.0667; GPU1_talker_FP32_upper: 17.2523; KV_each: 0.5000; prefill_each_upper: 9.0938; workspace_each: 8.0000 |
| Cached acoustic reg/FM response training only | frozen_reg_FM_whitening: 1.0382; trainable_transient: 0.0151; GPU_workspace: 2.0000; CPU_load_export_workspace: 4.0000 |

Detailed arithmetic and assumptions:

- **original base shard bytes upper**: `65929340192`.
- **tag checkpoint bytes upper**: `537005392`.
- **talker checkpoint bytes upper**: `9262259406`.
- **E2 HiFT checkpoint bytes upper**: `3257040728`.
- **FP32 expansion rule**: `2*stored file bytes is conservative even if every stored tensor wereBF16; alreadyFP32 tensors overcount, not undercount`.
- **LoRA upper elements**: `110100480`.
- **cached reg parameters**: `26229760`.
- **cached FM parameters**: `226223104`.
- **cached FM buffer elements**: `26229761`.
- **cached frozen FP32 bytes**: `1114730500`.
- **trainable total upper**: `113166177`.
- **host loading**: `complete base checkpoint +two talker checkpoints for load/FP32 cast +decoder +two largest base shards +16 GiB +exports`.

Scope and limits:

- The qualified current thinker Source is CPU-only; the explicit speech Source is a separately admitted two-GPU32/32profile. These numbers are calculations, not a new32B/GPU qualification or a promise that either profile will fit a particular card.
- The exact original model_files.json lock supplies complete shard/file sizes. Its1758-byte base/config.json (SHA2561c5ce47e7c6ca1f79e921153852173d4ae4ea5c44c906173cfbc34e19861c808) confirms thinkerH5120/L64/Q64/KV8/head128/FF25600,vocab151936 and18-layer audio encoder896/FF3584/14 heads. The top-level wrapperH4096 is not the thinker width. Remote pinned metadata returns404 to this account; the verified local metadata and original lock are used. File-byte upper bounds avoid inventing an exact parameter count.
- Native EOS audio has no universal duration maximum. The speech row is conditional on≤ 4096 cached positions and≤ 512nativeframes, maintained by the caller; longer sessions need recalculation. Voice-prefix KV caches for allthree languages and decoder activations are included in the8 GiB/card workspace reserve; unbounded sessions are outside it.
- The separate CPU thinker loads the frozen rank32tag PEFT adapter, included in itsCPU ledger. The original speech loader uses base tensors and rank256talker LoRA only (native.py111–126); it verifies the tagfile identity but does not load its tensor weights, so no fictitious second tag allocation is added to either speechGPU.
- Cached acoustic readout training omits the full thinker and E2 by design. It is not a full-native-generation budget and does not establish that a desired audible effect has succeeded. All RAW/no EMA and existing LR results stay unchanged.

Source facts (read as text; no model execution):

- [src/sophon_speech95/model_files.json](src/sophon_speech95/model_files.json)
- [src/sophon_speech95/native.py](src/sophon_speech95/native.py)
- [src/sophon_speech95/placement.py](src/sophon_speech95/placement.py)
- [src/sophon_speech95/contract.py](src/sophon_speech95/contract.py)
- [src/sophon_speech95/acoustic_response.py](src/sophon_speech95/acoustic_response.py)
- [src/sophon_particle_sliders/slider.py](src/sophon_particle_sliders/slider.py)
- [src/sophon_particle_sliders/current_backend.py](src/sophon_particle_sliders/current_backend.py)
- [examples/current-config.json](examples/current-config.json)
- [src/sophon_speech95/finished_audio.py](src/sophon_speech95/finished_audio.py)
- [Pinned model metadata](https://huggingface.co/AdventistsAI/Sophon-Realtime-v1/tree/c6e664d8ddca6522ef905fa170d9ca6cfa5dd568) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.1 release `d5dec2fe005305082acf382f0939a4bdb51c7131`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
