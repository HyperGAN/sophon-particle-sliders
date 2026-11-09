# Current fresh PG0.9.5 + tagged native text profile

This Source adds `sophon095`; the old `sophon` implementation, original
`core.lock.json`, example and CPU evidence remain historical and unchanged.
On 2026-11-09, source commit `175edeeade49f1e4044dc76f6a7f156cb1928482`
passed **nine relevant local CPU checks**, with zero failures/errors/skips,
pip check exit 0, no CUDA initialization and no external pretrained/network
attempts. See the [current CPU receipt and exact scope](evidence/current-pg095-cpu-qualification-20261109.md).
The pretrained 32B model has not been loaded or qualified. This is not a
pretrained-quality, GPU, realtime, audio-synthesis or LR-result qualification.
No existing checkpoint, LR search or RAW export is reset, migrated or re-labelled.

Install public core41f6066f2d2644902670d5eb736de052e1a70553 (distribution0.14.1)
and this product in a new isolated environment. `core-current.lock.json` and the
packaged `current_core.json` pin eight actual public API Source files; new factory
entry refuses differing same-version Source before loading weights. The official
Sophon source67fe6cfc02616cec5a5a883b063bdefd7b31fc2b, offline base/tag loader,
frozen unmerged tag and full LLM coordinates are unchanged. The new route explicitly
declares `penalty_input_domain="continuous"`, including the full5120-wide thinker;
it does not substitute projector coordinates or reduce the vocabulary/response.

## Pinned environment and CPU checks

Use a new Python 3.12 environment. The checked combination used Torch 2.9.1,
ParticleGAN 0.9.5, NumPy 2.2.6, Transformers 4.57.6, PEFT 0.20.0 and Accelerate
1.15.0. A version number alone is insufficient for the shared core: install its
exact commit above. From this product checkout, the following uses the actual
declared public dependencies and tested API versions:

```sh
python -m pip install 'setuptools==84.2.1'
python -m pip install \
  'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@41f6066f2d2644902670d5eb736de052e1a70553#subdirectory=packages/particle-sliders-core' \
  'hypergan[train] @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7' \
  'torch==2.9.1' 'particlegan==0.9.5' 'numpy==2.2.6' \
  'transformers==4.57.6' 'peft==0.20.0' 'accelerate==1.15.0' \
  'librosa==0.11.0' 'soundfile==0.14.0' 'soxr==1.1.0' 'audioread==3.1.0' \
  'safetensors==0.8.0' 'tokenizers==0.22.2' 'hndl==0.10.0' 'pytest==8.4.2' '.[native,test]'
git clone https://github.com/adventists-ai/sophon-realtime sophon-source
git -C sophon-source checkout 67fe6cfc02616cec5a5a883b063bdefd7b31fc2b
```

The qualification used a separately admitted private prefix and read-only
dependencies, not a global/captured-environment upgrade. Its complete actual
version summary is in the [portable evidence JSON](evidence/current-pg095-cpu-qualification-20261109.json).
To reproduce just the nine checks on tiny random models, no pretrained weights
are needed:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  SOPHON_SOURCE="$PWD/sophon-source" SOPHON_NATIVE_LOADER=1 \
  python -m pytest -q \
  tests/test_current_profile.py tests/test_current_native_text.py \
  tests/test_native_small.py::test_official_audio_merge_hidden_equivalence_and_native_gradients \
  tests/test_native_small.py::test_official_processor_chunk_counts \
  tests/test_native_loader.py
```

The real base/tag loader tests create their own tiny random files. The 32B base
and tag weights for real use must be staged separately in the original offline
layout (`base/` and `adapters/tag/`); this package never downloads them.

## Fresh training and exact resume

Use `examples/current-config.json`, choose a new cohort and local paths. Optional
literal `lr`, `d_lr_mult`, `prior_lr_mult` go to the actual shared recipe overrides:
G=`lr`, D=`lr*d_lr_mult`, bank=`lr*prior_lr_mult`. Values must be finite positive
non-Boolean numbers. Omitted values keep shared defaults; no Sophon LR optimum
is claimed. Nested `recipe_overrides` is refused in this new public profile.
For future LR selection, use one equal-case mean of intended-effect intensity
ON minus matched OFF in finished text, retaining negatives, zeros and all ties.
Teacher/student SSE is diagnostic only; generic beauty, preservation drift and
training loss do not rank candidates. Reuse completed work rather than adding
confirmation runs. The CPU receipt supplies no effect scores or LR winner.

Fresh initialization is explicit through source/policy fields. The product scopes
only newly added adapter/router/MLP and critic leaves. Official deterministic
orthogonal initialization retains their declared Kaiming/default-Linear RMS,
preserves all constructor zeros (including zero-up) and frozen base/audio/tag,
and applies R2Normal(mean0,std0.1) to the already-present four-coordinate bank.
The existing default bank is128×4; no new bank is inserted. Down remains nonzero,
so zero-up remains a trainable neutral adapter. This is a declared new cohort,
not numerical equivalence to historical random initialization. Named seeds and
actual PG/helper/product identities, method/gain/distributions and before/after
hashes are recorded in the shared initialization receipt.

`open_training(config,resume=...)` authenticates the new packet/config/receipt and
supplies fresh-versus-restore intent **before** model/critic optimizers. A fresh
call initializes once before constructing the game. Resume only validates and
restores its original receipt/state; inference constructs shape then loads RAW,
without fresh initialization. The marker and supplied intent cannot discover an
arbitrary external load hidden by a caller. Direct generic training without this
intent is refused; use this product dispatcher rather than assuming a universal
initialization CLI flag:

```sh
sophon-slider train --config current-config.json --out first --stop-after 2 --selection raw
sophon-slider train --config current-config.json --resume first/training.pt --out resumed --selection raw
```

The shared core owns actual updates, both optimizers, bank, all RNG and exact
FULL resume. All fresh configs use `ema_decay:null`; weight EMA and non-RAW
selection are refused. Changing options/cohort/source requires new work, not
retagging old packets. The original declared total horizon is retained on resume.

## Finished text from a RAW artifact

The new inference factory supports numerical response and finished **text**.
It calls the actual pinned `DuplexJevForConditionalGeneration.generate`; official
prefill merges native audio and the frozen tag stays active while the routed
slider spans generation. It does not use an ASR transcript, hidden-state decoder
approximation or waveform generator. Greedy `do_sample=False` and
`use_model_defaults=False` are direct kwargs. `use_cache=False` retains native
audio on every full-prefix call: the official67fe wrapper expects a prefill flag
that Transformers4.57.6 does not supply for the cached path. The official Source
is unchanged; no text-only fallback is permitted. Seed/RNG and slider scales are
scoped. Sound tags and native EOS tokens are preserved in decoded text. A legal
EOS or length-bound truncation is not classified as catastrophic here.

```python
from particle_sliders.inference import open_inference
runtime = {"model_dir": "/models/Sophon-Realtime-v1", "source_dir": "/src/sophon-realtime", "device": "cpu"}
messages = [
    {"role":"system", "content":"Answer naturally. When it fits, you may use sound tags for laughter, coughs and emphasis."},
    {"role":"user", "content":"<audio>"},
]
with open_inference("resumed/adapter-v2.pt", options=runtime,
                    operation="render", output_type="text") as session:
    result = session.render(messages=messages, audios=["/data/question.wav"],
                            max_new_tokens=64, strength=0.5, seed=7)
    result.save("finished.txt")
```

For the common CLI, put the runtime dictionary above in `runtime.json`. Put
`output_type:"text"`, `messages`, `audios`, `max_tokens` and `max_new_tokens` in
`text-request.json`; strength and seed are CLI arguments:

```sh
particle-slider infer --artifact resumed/adapter-v2.pt --runtime runtime.json \
  --mode render --native text-request.json --strength 0.5 --seed 7 --out finished.txt
```

`.save()` writes text plus a `.native.json` token/generation receipt; the common
CLI adds its normal artifact receipt. Output paths must be fresh. This API path
is implemented and CPU-tested with tiny random official models, not a published
pretrained sample or a trained Sophon LR recommendation.

CPU-only admission guards remain. Full32B/two-GPU placement and execution need
separate explicit ownership, measured fit and qualification; no auto device map
or removed CPU guard is supplied here. The nine-check receipt covers actual
source pins; literal rates/refusals; fresh/full4 versus split2+resume2 on the
synthetic contract host; zero-up/down/frozen ownership and RAW; no initializer
on restore/inference; intent refusal before allocation; official tiny native
gradients/processor/generation; and local-random tag loading in both precisions.
The old 14-check cohort is retained separately and is not relabelled as this run.

Primary generation source: [official pinned thinker](https://github.com/adventists-ai/sophon-realtime/blob/67fe6cfc02616cec5a5a883b063bdefd7b31fc2b/finetune/sophon_swift/modeling_duplexjev.py).
