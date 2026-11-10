# Sophon: recorded RAW32 laughter rates and API boundaries

The completed saved32-update comparison favors **G0.012, gain+0.25**, over
**G0.001, gain0.0**, in one matched native speech case. It holds actual
D0.018000000000000002 and bank0.03 fixed. The measurement is a frozen provisional
Qwen2-Audio model instrument; `trait_qualified:false`. Catastrophe status and
current eligibility are **unmeasured**, with no own FULL32 categorical PASS or
FAIL established by this effect-only comparison. G0.012 was chosen for the
user-requested exploratory longer training. This is no global optimum, official
default, released weight, ordinary Generate or automatic promotion.

| G | Shared OFF | ON | Equal-case mean ON−OFF |
| --- | --- | --- | --- |
| **0.012** | 0.50 | 0.75 | **+0.25** |
| 0.001 | 0.50 | 0.50 | 0.0 |

The units are **ordinal/4 intensity differences**, not probabilities: the frozen
five classes describe no laughter, faint/brief chuckle, occasional clear laughter,
frequent/sustained laughter, and laughter dominating the utterance. The native
observer chooses the highest logit among literal0–4 tokens, then divides by4.
The three genuine first observations share one OFF and form two gains with case
weight1. The spoken word “laugh,” ordinary cheerful speech, transcript content,
pitch, volume, noise or distortion alone are excluded by the rubric. This does
not establish semantic calibration, human agreement or discrimination beyond the
retained observations. Earlier CUDA RAW2 observations, including their different
OFF0.25, are separate and are not pooled or rerated.

The matched request is English `The garden gate is open. It is a lovely day.`,
seed7, OFF strength0 and ON strength1, original full native EOS, mono24k PCM16
through the original E2/HiFT decoder. Both arms used RAW/no EMA,32 updates within
horizon128, batch2, rank8, one128×4 bank, full5120 acoustic response and the same
prepared native cache. The G0.001 fresh arm matched the original initial added
network/critic/bank values before update1; it has its own fresh optimizer/cohort,
not a relabeled restore of the G0.012 packet. No teacher, completed update,
finished output or old first was replayed to make this comparison.

## Public routing boundary

The captured public package at commit `c7bc7cb08bd3bbc78e1ae25d2813b67b887b0f6d`
has these entry points:

| Public entry | Supported recorded scope | Speech-result compatibility |
| --- | --- | --- |
| `sophon` | Historical thinker hidden response | Not this native speech cohort |
| `sophon095` / `sophon-slider` | Current CPU tagged, audio-conditioned text; see [CURRENT-PG095](../CURRENT-PG095.md) | No waveform renderer |
| `sophon095_twogpu` / `sophon_twogpu95` | Explicit owned two-GPU tagged text; see [EXPLICIT-TWO-GPU](../EXPLICIT-TWO-GPU.md) | Text outputs only |

The measured speech implementation is the separate captured source
`fcb157549099515b7269d2f56119eadf0266eeaf`, namespace `sophon_speech95`, source
profile `sophon-original-speech-reg5120-pg095-v1`, on captured core0.14.1/PG0.9.5
at `41f6066f2d2644902670d5eb736de052e1a70553`. Its full acoustic response is
`fm.from_z(reg(y).float())` at5120 coordinates; native inference applies the
same owned branch to the original reg output. It uses the original full64-layer
BF16 Sophon model/tag and original E2/HiFT speech components. This is a different
factory/RAW identity from the public tagged-text profiles.

This public package lacks that speech namespace, its `native-speech` dependency
extra, `sophon-speech-slider` entry point, original speech source/weights and
private mixed-placement execution companions. A documentation patch does not add
these components. No command in this guide promises to install or run the private
speech artifact through a public text backend. Reproducing the captured result
requires its exact speech source/runtime, model/cache/artifact identities and
an actual bounded execution owner; these files are not bundled or public weight
downloads. The portable JSON records scalar arithmetic and retained identities,
not an executable checkpoint conversion or source migration.

## Exact rate options in the captured speech source

For that source profile only, the saved G0.012 option members are:

```json
{"lr": 0.012, "d_lr_mult": 1.5, "prior_lr_mult": 2.5, "ema_decay": null}
```

The compared G0.001 members are:

```json
{"lr": 0.001, "d_lr_mult": 18.0, "prior_lr_mult": 30.0, "ema_decay": null}
```

G is `lr`; actual D is `lr*d_lr_mult`; actual bank rate is
`lr*prior_lr_mult`. These literal products give the same D0.018000000000000002
and bank0.03. Preserve the saved binary literals instead of reconstructing
multipliers from rounded labels. The portable summary leaves automatic `rate_overrides` null and records these
under `provisional_rate_overrides`, with explicit exploration scope. No rate in
this document is a recommendation for a different public text or hidden-response
task.

The captured speech source exposes
`open_training(config, resume=None, admission=actual_owner)` and
`open_inference(artifact, options=local_paths_and_captured_placement,
admission=actual_owner, ...)`. Its command wrapper refuses to run without an
actual supplied owner; a JSON flag or receipt string does not create admission.
No universal speech CLI, auto device map or unqualified GPU permission is
invented here.

Fresh speech construction authenticates a completed prepared native cache and
applies its declared scoped initializer before optimizers, preserving original
Parameters, host laws, zero-UP and frozen pretrained weights. A genuine FULL
resume authenticates and restores the complete game, optimizers, bank, RNG,
exposure counters and original initialization receipt. It never initializes or
prepares teachers again. RAW is inference-only. `steps=128` is the original
horizon;32 was an earlier stopping budget, not a horizon reset. The saved32 and
later128 updates are complete: reuse them, do not repeat the examples as a new
search or replay a prefix.

## Separate RAW128 continuation and execution profile

The selected G0.012 FULL32 subsequently resumed exactly, committed only updates
33–128, and saved FULL128/RAW128 with genuine closed0/resource0. This was cached
head/critic/bank training, without a second full native teacher pass or a new
initialization. It preserves the original128 horizon and all completed state.
It is a later training budget, not another rate comparison or a replacement for
the RAW32 scores above.

The matched finished speech profile is explicitly private **CPU24/GPU20+20**:
original layers20–43 on CPU,0–19 on GPU0 and44–63 on GPU1, BF16 with CPU autocast;
E2/HiFT stays GPU0 and heads/PG/embedding/norm stay GPU1. All64 original layers
and their language KV routing remain. Captured32/32 factory identity and actual
mixed construction are recorded separately. No dtype/quantization/model/geometry
shortcut, bitwise-CUDA or placement-independent equivalence is claimed. The
older CUDA OFF and its judgments do not transfer; the new mixed OFF was sampled
once and is shared by the compared arms and later128 preview.

Both strength1 and strength2 RAW128 recordings completed native EOS (3.84s each)
and were published for inspection. Two new observations with the same frozen
laughter instrument reused the saved OFF0.50 first: strength1 ON0.50/gain0.0;
strength2 ON0.75/gain+0.25. The best RAW32 strength1 example also scored+0.25.
These separate budgets therefore show no improvement over that earlier best on
this one-case metric. Preserve all three first observations and the tie; do not
substitute beauty, preservation or completion as a tiebreak.

Strength2 used an explicit single-owner consumer-range extension after strict
RAW128 load; the original consumer accepts0–1. The complete recording was saved
before a new cleanup metadata check failed. That check incorrectly required
unchanged Parameter version counters, although the original core restores state
through in-place copies that advance those counters. The witness separately
reports all four guard attributes restored, but its coarse check remains false
and no independent post-call numerical-identity proof exists. The actual job
remains closed with exit1 and no normal result or saved CPU-autocast receipt.
Root separately adopted the complete saved recording for inspection and its
first missing effect measurement; no recording was generated again and no
successful-job or categorical PASS was fabricated. This is not public
`--strength 2` support. Neither preview changes the RAW32 LR comparison, old
firsts or package defaults.

## One selector, reuse and failure scope

Rank by equally weighted finished intended-laughter ON−OFF only. Preserve
negative values, zero gains and exact ties; unavailable judgments remain missing.
Teacher/student SSE, beauty, preservation, confidence and completion are not
ranking terms. An established qualified own catastrophic failure stops that
candidate separately. Weak laughter is not catastrophic failure; unqualified or
unknown checks do not manufacture a PASS, FAIL or eligibility. This comparison
provides no universal speech-accuracy or categorical guarantee.

Reuse the completed32/128 updates, saved cache/FULL/RAW and first observations.
No repeated search, additional confirmation, semantic-anchor loop, old rerating
or release/default change is required to report this scoped provisional result.
