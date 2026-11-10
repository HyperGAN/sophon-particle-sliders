# Native Sophon speech: owner-supplied Python API

The `sophon_speech95` backend produces original Sophon speech through its full
64-layer BF16 base, frozen tag adapter and original E2/HiFT decoder. It trains
the added reg branch, critic and 128×4 bank on full 5120-coordinate native acoustic
responses using ParticleGAN 0.9.5/core 0.14.1. All model-weight EMA is disabled.
The older tagged-text backends and their checkpoints remain separate. See the
[speech attribution and historical notice scope](NOTICE-SPEECH.md).

This interface needs an execution owner supplied by your runner. The library
does not acquire devices, reserve memory, download models or create admission.
The `sophon-speech-slider` console entry deliberately refuses without an owner;
your runner can call `sophon_speech95.cli.main(argv, admission=owner)`. A receipt
string, configuration flag or generic training command is insufficient.

## Dependencies and original assets

Install the package's `native-speech` extra in your owned environment. This
captured profile additionally checks exact core API bytes from
[`41f6066f2d2644902670d5eb736de052e1a70553`](https://github.com/HyperGAN/particle-sliders/tree/41f6066f2d2644902670d5eb736de052e1a70553),
not just a compatible version label. Its locks are
[current_core.json](src/sophon_speech95/current_core.json),
[upstream.json](src/sophon_speech95/upstream.json) and
[model_files.json](src/sophon_speech95/model_files.json).

Stage the original source at
[`67fe6cfc02616cec5a5a883b063bdefd7b31fc2b`](https://github.com/adventists-ai/sophon-realtime/tree/67fe6cfc02616cec5a5a883b063bdefd7b31fc2b)
and the original model at
[`c6e664d8ddca6522ef905fa170d9ca6cfa5dd568`](https://huggingface.co/AdventistsAI/Sophon-Realtime-v1/tree/c6e664d8ddca6522ef905fa170d9ca6cfa5dd568).
Provide the source, base/tag and speech roots through `source_dir`,
`base_model_dir` and `speech_model_dir`. The lock lists 35 original files; the
library validates their whole sizes/hashes. No weights are bundled here.

The runtime requires Torch 2.9.1, Transformers 4.57.6, Torchaudio 2.9.1,
x-transformers 2.31.7, einops 0.8.1, safetensors 0.8.0 and NumPy 2.2.6. It also
records the installed SciPy, librosa and soundfile versions. RAW identity includes
the actual runtime metadata, so an environment satisfying the extra is not by
itself proof that an existing artifact is compatible. Preserve its exact runtime
and locks; there is no implicit source or checkpoint migration.

## Supply a real owner

Your runner supplies an `OwnedTwoGPUAdmission` from
[placement.py](src/sophon_speech95/placement.py), backed by its actual ownership
receipt and live check. Its plan must equal `PLAN`, and visible devices must be
the explicitly owned physical 0/1 pair (`CUDA_VISIBLE_DEVICES=0,1`). The live check
must enforce that runner's caps, memory floors and lifetime. The default plan
uses 32 original trunk layers on each GPU, PG/heads on GPU1 and decoder on GPU0.
It makes no automatic placement or fit promise.
The runner must authorize framework imports and establish ownership before it
imports `backend`/`native` or enters a cache, train or render call. The library
owner is a live ownership hook, not a replacement for that runner lifecycle.

These functions accept that already-created owner; none manufactures one:

```python
from sophon_speech95.cli import cache, train, render

def capture_missing_cache(owner, native_options, declared_cases, new_cache_dir):
    return cache(native_options, declared_cases, new_cache_dir, admission=owner)

def train_missing_updates(owner, config, new_output_dir, *, full=None, stop_after=2):
    return train(config, new_output_dir, admission=owner,
                 resume=full, stop_after=stop_after)

def render_raw(owner, raw, local_options, new_output_dir, *, text, seed=7):
    return render(raw, local_options, new_output_dir, text=text,
                  language="en", seed=seed, strength=1.0, admission=owner)
```

`local_options` contains only the three local roots, `device="cuda:1"` and
`placement=PLAN`. RAW supplies the immutable architecture/source/init profile.
Strength 0 is OFF and strength 1 is ON; the captured public factory supports only
the closed interval [0,1]. An exploratory strength 2 consumer is a separate
execution profile and is not supplied by this interface.

## Cache once, then fresh or exact FULL restore

The cache builder accepts cases with exactly `neutral`, `positive`, `language`.
Use original supported EV conditioning while keeping identical cleaned speech
text. It captures neutral/positive full 5120 acoustic targets at a naturally
matched first-KF event and persists each returned teacher call. It refuses a
forced event or mismatched text/history. Training requires at least two genuine
matched rows. Reuse a completed cache; an incomplete or unknown call is not
permission to execute it again.

Build a `SessionConfig` with backend `sophon_speech95`, your explicit cohort and
these option members in addition to the local roots/placement and actual saved
cache FileRecord (`cache_file`, `cache_bytes`, `cache_sha256`):

```json
{
  "source_profile": "sophon-original-speech-reg5120-pg095-v1",
  "initialization_policy": "sophon-speech-fresh-Kaiming-QR-zeroUP-bankR2std01-v1",
  "pg_version": "0.9.5",
  "pg_source_profile": "release-0.9.5",
  "penalty_input_domain": "continuous",
  "ema_decay": null,
  "rank": 8,
  "particles": 128,
  "particle_dim": 4,
  "response_width": 5120,
  "base_precision": "bfloat16",
  "steps": 128,
  "batch": 2,
  "seed": 7,
  "lr": 0.012,
  "d_lr_mult": 1.5,
  "prior_lr_mult": 2.5
}
```

Use `sophon_speech95.backend.open_training(config, admission=owner, resume=full)`
or the `train` wrapper above. That wrapper declares fresh/restore intent before
allocation and optimizer construction; direct generic backend dispatch lacks
this required construction context. Fresh mode scopes only the owned Parameters,
uses the declared host laws/QR/R2 bank std 0.1 and preserves zero-UP/frozen weights.
It saves an initialization receipt. A genuine FULL restore validates that receipt
and restores game, optimizers, bank, cursors and RNG without initializing or
calling teachers. Head-only cached training retains all 5120 coordinates and the
official critic penalty; it does not load the full trunk just to replay teachers.

The train wrapper durably saves FULL/RAW/COMMIT for its starting state and every
new update. Keep the frozen `steps=128` horizon when stopping at 32 or restoring
later; `stop_after` is an explicit new stopping budget. RAW is inference-only.
FULL resume requires its exact configuration, including saved paths/cache;
relocating the package does not migrate those configuration paths. Never rerun
completed updates, searches, cache rows or initializers to recreate an example.

The literal rate tuple above is the **provisional exploratory** measured G0.012
choice, not an ordinary default. D=`lr*d_lr_mult` is 0.018000000000000002 and the
bank rate=`lr*prior_lr_mult` is 0.03. The completed matched RAW32 one-case laughter
comparison scored G0.012 gain +0.25 versus G0.001 gain 0.0; the latter used the
literal tuple 0.001/18.0/30.0 with the same actual D/bank. Trait qualification is
false and catastrophic eligibility unknown. Later 128 updates/strength changes
are separate budgets, not another LR comparison. Retain ties, zeros and original
observations; there is no global optimum, released weight or automatic promotion.

## Explicit mixed-device RAW inference

The separately packaged `sophon_speech_execution.mixed_execution` companion
retains the captured factory files and strict RAW identity. It places original
layers 0–19 on GPU0, 20–43 on CPU and 44–63 on GPU1, all BF16; PG/heads/embedding/norm
remain GPU1 and E2/HiFT remains GPU0. It records the actual placement separately
from the captured 32/32 declaration. Use CPU BF16 autocast during native sampling;
there is no bitwise CUDA equivalence or fit claim.

```python
def render_mixed(owner, raw, local_options, new_output_dir, *, text, seed=7):
    owner.check()  # Runner has already authorized imports and acquired ownership.
    from sophon_speech95 import backend, native
    from sophon_speech_execution import mixed_execution as mixed
    request = dict(text=text, language="en")
    # The callback is active only during original factory construction/RAW load.
    with mixed.construction(native) as construction:
        inference = backend.open_inference(
            raw, options=local_options, admission=owner, operation="render",
            output_type="audio",
            native_request=dict(request, durable_directory=str(new_output_dir)))
    with inference:
        placement = mixed.describe(inference, construction, owner.check)
        with mixed.cpu_sampling() as autocast:
            result = inference.render(
                **request, seed=seed, strength=1.0, output_type="audio",
                durable_directory=str(new_output_dir))
    # Persist result, placement and restored autocast with your runner's receipts.
    return result, placement, autocast
```

`new_output_dir` must be unused. Generation preserves complete original native
frames and decoder inputs before native decoding, then saves its native 24k PCM16
WAV and JSON receipt. Native EOS may legitimately contain zero samples; that is
recorded as an empty native result, not described as playable speech. The library
does not normalize waveforms, substitute ASR/hidden vectors, rerate outputs or
issue a categorical guard/semantic qualification. Your runner owns durable
completion accounting and cleanup; an unknown render claim must not auto-retry.
