# Current Sophon095 CPU qualification — 2026-11-09

Source commit `175edeeade49f1e4044dc76f6a7f156cb1928482` passed **9 tests,
0 failures, 0 errors, 0 skipped** in **11.105 seconds**. This was an actual local
CPU run in a separate admitted environment, not a GitHub Actions result. Pip
check returned 0. The worker and outer job exited 0 and were genuinely reaped;
the exact owned CPU8 role was released after original-process absence checks.

The [portable metadata summary](current-pg095-cpu-qualification-20261109.json)
records exact source/version identities, test names and original receipt hashes.
The original detailed execution receipts are retained by the operator and are
not bundled in this public repository. This summary omits private machine paths
and process identities; their absence does not make these public CI artifacts.

| Checked behavior | Actual scope |
|---|---|
| Current core/source identity and common rate validation | Exact core `41f6066…`, ParticleGAN 0.9.5 and literal G/D/bank overrides/refusals |
| Scoped fresh initialization, FULL resume and RAW export | Synthetic contract host: full4 equals split2+resume2; zero-up/nonzero-down and frozen ownership; no fresh initializer on restore/inference |
| Construction intent and packet refusal | Refusal before native allocation when intent is missing or legacy options are supplied |
| Audio-conditioned finished text | Real tiny random official Sophon model; actual audio merge on generation calls, direct greedy/no-cache kwargs, RNG/frozen-state checks and fresh text save |
| Native gradients and preprocessing | Real tiny official architecture and processor chunk packing |
| Offline base/tag loader | Fresh local random base and PEFT tag, FP32 and BF16; actual audio loading/resampling/native response |

No CUDA was initialized. No network or external pretrained-model attempt was
made during the test phase. The loader checks did use `from_pretrained` on
**their own freshly generated tiny local fixtures** and converted tag files;
this is not a load of the released 32B checkpoint.

The seven official Python files matched the existing upstream lock at
`67fe6cfc02616cec5a5a883b063bdefd7b31fc2b`. The tested source uses the distinct
`sophon095` profile, explicit continuous penalty domain, the original zero-up
host law, the existing 128×4/std0.1 bank law for default architecture and raw
weights with `ema_decay=None`. The tiny fixtures reduce model geometry only for
this declared CPU check; they do not qualify full-width 32B execution.

Still unqualified: the released 32B base/tag load, full native-width training,
owned GPU placement/memory, pretrained finished-text/tag quality, audio
synthesis, realtime latency/duplex behavior, catastrophic-detector behavior and
intended-effect LR scores. CUDA remains refused in this product; no automatic
device map or admission bypass was introduced. The historical
[core0.14.0 CPU receipt](verification.json) remains a separate unchanged cohort.
