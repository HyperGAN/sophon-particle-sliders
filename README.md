# Sophon Particle Sliders

**Calculated memory:** Native original speech 32/32 planning budget: **56/66 GiB GPU / 108 GiB host** for the bounded4096-context profile. CPU thinker and cached acoustic training have separate estimates. [Full memory ledger and limits](MEMORY.md). These are planning estimates, not measured peaks.

## Recorded native speech result: a separate research profile

A completed, provisional one-case **RAW32 laughter** comparison gives **G0.012
+0.25 versus G0.001 0.0**, at fixed actual D0.018000000000000002 and bank0.03.
The frozen model instrument is unqualified for the trait; catastrophic eligibility
is unknown. G0.012 was chosen only for the requested exploratory continuation,
which completed the96 missing updates to the original128-step horizon.

[Values, scope and API boundaries](docs/finished-output-learning-rates.md) and
[portable scalar/provenance summary](https://github.com/HyperGAN/sophon-particle-sliders/blob/9c6ca508c8fb66cac9fc07263e87199d4a98e844/docs/finished-output-rate.json) distinguish
the historical native E2/HiFT speech run from the current source. The
[native speech API](NATIVE-SPEECH.md) retains the strength range [0,1] and
requires a live scheduler owner. The captured speech result predates this
source migration.

Install this checkout with `python -m pip install '.[native,test]'` for the
current text profile. Core 0.15.2 and ParticleGAN 0.11.1 are resolved from
project metadata. [Scoped initialization and tagged native text](CURRENT.md)
describes the current profile. The 32B pretrained model, GPU execution,
output quality and LR tuning remain unqualified.

The `sophon` backend provides a CPU thinker hidden-response session. The
`sophon_scoped` backend adds native tagged text generation with explicit
scoped initialization. The speech backend and explicit two-device text
backend use their respective [speech](NATIVE-SPEECH.md) and
[placement](EXPLICIT-TWO-GPU.md) contracts.

The [earlier CPU cohort and installation guide](https://github.com/HyperGAN/sophon-particle-sliders/blob/9c6ca508c8fb66cac9fc07263e87199d4a98e844/README.md)
has a separate [qualification receipt](https://github.com/HyperGAN/sophon-particle-sliders/blob/9c6ca508c8fb66cac9fc07263e87199d4a98e844/evidence/verification.json).

Code is Apache-2.0. See [NOTICE](NOTICE) for upstream and shared component licenses.
