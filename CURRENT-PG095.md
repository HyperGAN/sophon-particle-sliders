# Prospective fresh PG0.9.5 + tagged native text profile

This Source adds `sophon095`; the old `sophon` implementation, original
`core.lock.json`, example and CPU evidence remain historical and unchanged.
The new profile has **not** run its native CPU tests or loaded the pretrained32B
model. It is not a pretrained-quality, realtime or audio-synthesis qualification.
No existing checkpoint, LR search or RAW export is reset, migrated or re-labelled.

Install public core41f6066f2d2644902670d5eb736de052e1a70553 (distribution0.14.1)
and this product in a new isolated environment. `core-current.lock.json` and the
packaged `current_core.json` pin eight actual public API Source files; new factory
entry refuses differing same-version Source before loading weights. The official
Sophon source67fe6cfc02616cec5a5a883b063bdefd7b31fc2b, offline base/tag loader,
frozen unmerged tag and full LLM coordinates are unchanged. The new route explicitly
declares `penalty_input_domain="continuous"`, including the full5120-wide thinker;
it does not substitute projector coordinates or reduce the vocabulary/response.

Use `examples/current-config.json`, choose a new cohort and local paths. Optional
literal `lr`, `d_lr_mult`, `prior_lr_mult` go to the actual shared recipe overrides:
G=`lr`, D=`lr*d_lr_mult`, bank=`lr*prior_lr_mult`. Values must be finite positive
non-Boolean numbers. Omitted values keep shared defaults; no Sophon LR optimum
is claimed. Nested `recipe_overrides` is refused in this new public profile.

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

The actual common `particle-slider infer --mode render` also works with the new
artifact, a runtime JSON and native JSON containing `output_type:"text"`,
`messages`, `audios`, `max_tokens` and `max_new_tokens`. Its `.save()` output is
text plus a `.native.json` token/generation receipt; the common CLI adds its
normal artifact receipt. Output paths must be fresh.

CPU-only admission guards remain. Full32B/two-GPU placement and execution need
separate explicit ownership, measured fit and qualification; no auto device map
or removed CPU guard is supplied here. New tests are authored, not executed:
actual source pins; literal rates/refusals; fresh/full4 versus split2+resume2;
zero-up/down/frozen ownership and RAW; no initializer on restore/inference;
intent refusal before allocation; and official tiny audio-conditioned native
text generation/RNG. Old14 CPU checks do not automatically qualify these changes.

Primary generation source: [official pinned thinker](https://github.com/adventists-ai/sophon-realtime/blob/67fe6cfc02616cec5a5a883b063bdefd7b31fc2b/finetune/sophon_swift/modeling_duplexjev.py).
