# Explicit owned two-GPU Sophon cohort

`sophon095_twogpu` / `sophon_twogpu95` is a new Source/cohort. Original `sophon`, CPU-qualified `sophon095`/175edee, their fourteen/nine CPU receipts, locks, states and CUDA refusals stay unchanged. Do not feed old FULL/RAW packets to this profile. This implementation is not yet CPU/GPU/pretrained qualified; the focused tests below are authored, not executed by the Source author.

The immutable official thinker/audio/tag implementation remains67fe6cf; weights remain original full local `base/` plus frozen active/unmerged `adapters/tag`. No ASR, hidden projection, quantization, offload, reduced response, automatic/balanced device map, EMA or changed PG penalty is introduced.

The exact map places native audio tower/projector, token embedding, rotary state and layers0–31 on logicalcuda0; layers32–63, final norm and LM head on logicalcuda1. CUDA requires the released64-layer/full5120 geometry. All fresh FP32 routed branches/router/bank and critic are on cuda1. The unchanged core `TinyParticleProjection` differentiably copies x to its branch and delta back to the original host device. Full native final-token5120 response copies to cuda1; PG retains its continuous profile and original formula. One original128×4 bank starts from the host's declared std0.1 policy; fresh QR/R2 keeps zero-UP/pretrained/tag weights. Exact resume and RAW inference never call the fresh initializer.

## Actual owner, not a configuration-only permission claim

The product does not create leases/admissions or promise fit. Root's execution envelope must acquire/retain both physicalGPU0/1 and its host/GPU caps, parent lifetime and cleanup, then supply its real `check_live` callback and actual owner receipt FullRecord. `CUDA_VISIBLE_DEVICES` must be exactly `0,1`. The callback is checked before CUDA allocation and before each native training/response/generation call. This interface depends on the external owner being truthful; a Python callback/receipt shape is not independent proof of kernel leases. Volatile PIDs/births stay in actual execution records, not scientific packet identity, so a genuinely new owner can resume the exact same scientific config.

Use the product entry functions for CUDA. Registry preparation without its owner context refuses before model allocation. CPU single-device fixtures need no CUDA admission. Both owned CUDA devices appear explicitly under `placement.owned_devices[].device` so the shared factory/inference RNG guards capture both, and generation scopes/seeds only those two owned generators.

```python
from copy import deepcopy
from particle_sliders.lifecycle import SessionConfig
from sophon_twogpu95.backend import open_training, open_inference
from sophon_twogpu95.contract import SOURCE_PROFILE, INITIALIZATION_POLICY
from sophon_twogpu95.placement import PLAN, OwnedTwoGPUAdmission

# Supply actual Root execution-owner values, never fabricated example authority:
owner = OwnedTwoGPUAdmission(deepcopy(PLAN), actual_owner_receipt, actual_live_owner_check)
config = SessionConfig('sophon095_twogpu', 'unique-new-fresh-cohort',
    model_dir='/exact/verified/original/model', source_dir='/exact/67fe/official/source',
    training_file='/exact/matched-native-audio-text.jsonl', device='cuda:1', placement=deepcopy(PLAN),
    base_precision='bfloat16', response_width=5120, rank=8, particles=128, particle_dim=4,
    pg_version='0.9.5', pg_source_profile='release-0.9.5', penalty_input_domain='continuous',
    source_profile=SOURCE_PROFILE, initialization_policy=INITIALIZATION_POLICY,
    ema_decay=None, steps=2, batch=2, seed=7,
    lr=.012, d_lr_mult=1.5, prior_lr_mult=2.5)
with open_training(config, admission=owner) as session:
    # Explicit G/D/bank literals are starting recipe values, not optimized rates.
    session.train()
    full = session.checkpoint()
    raw = session.export_inference(selection='raw')
# Exact same config and completed FULL; no old-cohort migration/replay/reset:
with open_training(config, admission=owner, resume=full) as resumed:
    pass
with open_inference(raw, admission=owner,
        options=dict(model_dir=config.options['model_dir'], source_dir=config.options['source_dir'],
                     device='cuda:1', placement=deepcopy(PLAN)), operation='render') as inference:
    # See existing CURRENT-PG095 for exact packed prompt/ChatML audio inputs.
    output = inference.render(messages=actual_tagged_messages, audios=actual_audio_paths,
                              max_new_tokens=64, strength=1., seed=7)
    output.save('/fresh/finished.txt')
```

Generation uses the actual official tagged wrapper and native audio merge. Input/embedding/audio live on cuda0; explicit decoder hooks route intermediate tensors, and final text logits follow native dispatch. `use_cache=False` avoids the pinned HF/native prefill-flag mismatch and stale cross-device KV: every full prefix receives the same original native audio. Direct greedy kwargs (`do_sample=False`, `use_model_defaults=False`) override model defaults. No text-only substitute, prompt truncation, vocabulary projection or quality verdict is claimed. Tags/EOS are retained.

## Focused checks for Root

`tests/test_explicit_twogpu_profile.py` adds six checks: literal full map/admission refusal without CUDA; external owner context/reset; real new CPU slider gradient path/frozen tag/zero-UP/full width; exact synthetic FULL split-resume/RAW/no reinit/common rates; a real differentiable Accelerate-hook preservation/refusal check; and real tiny official audio-conditioned text through the new path with observed audio merge and RNG/state checks. They do not load pretrained weights, call Hub or initialize CUDA. Root must execute them under an owned CPU admission using the exact qualified dependency prefix read-only and this new Source, with bytecode/cache writes disabled. This is a new-source check, not a rerun of the old nine/61 checks.

CPU fixtures do not establish real two-GPU dispatch, full32B fit, pretrained update/resume/render correctness, catastrophic eligibility, effect/LR measurement or released weights. Those remain separate genuine Root-owned work after complete model staging and fresh two-GPU admission. No initializer reset or repetition of completed historical training is requested.
