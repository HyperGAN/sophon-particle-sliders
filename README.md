# Sophon Particle Sliders

**Current prospective Source:** [fresh PG0.9.5 initialization and real tagged native text](CURRENT-PG095.md) uses the distinct `sophon095` backend. Native CPU/pretrained qualification is pending. Historical `sophon` Source and evidence below remain unchanged; their CPU result does not qualify the new profile.

## Historical core0.14.0 CPU cohort

**Experimental thinker-response integration, not a validated realtime voice service.**

- **Validated:** 14 installed-wheel CPU tests passed, including synthetic and tiny randomly initialized official architecture checks, real PEFT tag loading in FP32/BF16, gradients, native audio packing, and exact checkpoint/resume.
- **Not validated:** the full 32B pretrained model, audio synthesis, realtime duplex/barge-in behavior, latency, or speech quality.
- **Execution boundary:** CPU only; CUDA is refused pending a Sophon scheduler admission integration. No model weights are included.

See [validation evidence](evidence/verification.json). Automated CI has not run in this repository: the publication credential permits source publication but lacks the workflow scope required to add GitHub Actions workflows. The reported tests are local CPU checks.

Local Sophon Realtime v1 **thinker response** integration using
`particle-sliders-core==0.14.0` and its unchanged ParticleGAN 0.9.5 game.
This is a new source cohort. It supplies native model/audio packing and a routed
LLM residual; the shared library owns training, raw checkpoints, exact resume,
evaluation and inference exports. No trainer or optimizer is copied here.

This release exposes numerical final-token hidden responses. It does not serve
websockets, synthesize audio, change turn-taking, or claim pretrained quality.
The nonlinear routed particle branch cannot be exported as an ordinary PEFT
LoRA or loaded into stock vLLM. Keep the shared v2 artifact and this native host.

## Installation

Use an isolated Python 3.11+ environment (official serving requires Linux/CUDA
and Python 3.12). Install the shared distribution from reviewed commit
`8a696eb33c1a9d34472afad3a3199bd4776a797f`:

```sh
git clone https://github.com/HyperGAN/particle-sliders shared
git -C shared checkout 8a696eb33c1a9d34472afad3a3199bd4776a797f
python -m pip install ./shared/packages/particle-sliders-core
python -m pip install '.[native,test]'
git clone https://github.com/adventists-ai/sophon-realtime sophon-source
git -C sophon-source checkout 67fe6cfc02616cec5a5a883b063bdefd7b31fc2b
```

Stage model files separately from `adventists-ai/Sophon-Realtime-v1`:
`base/` and `adapters/tag/` are required. This package never downloads weights,
installs dependencies, requests capacity, or starts a server. It hashes all
staged base/tokenizer/tag files before allocation, and verifies the official
Python source against the packaged `upstream.json` lock.

The official `base/` thinker is required, **not** the FP8 Tag/Flash serving
checkpoint. The bundled rank-16/alpha-32 tag LoRA is converted using the official
helper, loaded frozen and kept active. The separate slider is rank 8 with one
shared 128×4 bank by default. Only LLM attention/MLP projections are selected;
audio, projector, tag parameters and base weights are frozen. OFF bypasses the
routed branch exactly while preserving the original tag behavior. No merging,
weight averaging or EMA is used.

## Matched data and shared lifecycle

Each JSONL record has `neutral_messages`, `positive_messages` and `audios`:

```json
{"neutral_messages":[{"role":"system","content":"Answer naturally. When it fits, you may use sound tags for laughter, coughs and emphasis."},{"role":"user","content":"<audio>"}],"positive_messages":[{"role":"system","content":"Answer cheerfully. When it fits, you may use sound tags for laughter, coughs and emphasis."},{"role":"user","content":"<audio>"}],"audios":["question.wav"]}
```

Supply at least two records. Audio paths are relative to the JSONL. Audio is
loaded as float32 mono and resampled to 16 kHz, then the pinned official
processor builds 128-bin mels, ≤30-second chunks and expanded `<|audio|>` tokens.
The same audio tensors condition each matched teacher/student pair. Native
ChatML and assistant history are retained. Positive instructions change text;
teacher targets are frozen positive final-token hidden states. The student
uses neutral instructions and the routed branch. No transcript/ASR substitute,
vocabulary truncation, synthetic target or waveform loss is used by this path.

Copy `examples/config.json`, set absolute local paths and the **full LLM width**
from `base/config.json:text_config.hidden_size` (not the projector width).
Rates are shared defaults, not Sophon-tuned recommendations. CPU execution is
admitted; CUDA is refused before model allocation until this product has an
actual scheduler admission integration. Do not bypass this with a device map.

```sh
particle-slider capabilities --interface response --config config.json
particle-slider train --interface response --config config.json --out run --stop-after 2 --export-format both --selection raw
particle-slider train --interface response --config config.json --resume run/training.pt --out continued --selection raw
```

Resume uses the original total `steps`, same source, files and cohort; it does
not reset the horizon. Full checkpoints include critic, optimizers, bank, RNG,
row order/exposure and actual completed updates through the shared lifecycle.
Inference v2 exports only raw adapter state and native identity. The registered
`sophon` inference factory requires local model/source paths and the artifact,
and constructs no training data, teachers, critic or optimizer:

```python
from particle_sliders.inference import open_inference
runtime = {"model_dir": "/models/Sophon-Realtime-v1", "source_dir": "/src/sophon-source", "device": "cpu"}
messages = [
    {"role": "system", "content": "Answer naturally. When it fits, you may use sound tags for laughter, coughs and emphasis."},
    {"role": "user", "content": "<audio>"},
]
with open_inference("run/adapter-v2.pt", options=runtime, operation="response") as session:
    hidden = session.response(messages=messages, audios=["/data/question.wav"], strength=0.5)
```

The same request dictionary works with the shared CLI's `infer --mode response
--native request.json`. Low-level Python callers may supply a prepacked tensor
dictionary as `prompt` instead of `messages`/`audios`.

## Resources, scope and verification

Upstream recommends one 80 GB GPU for Tag/Flash serving, two for E2E. Its
official thinker LoRA SFT recipe reports about 65 GB peak at batch 1 and ≤2048
tokens. Those are **upstream SFT/serving measurements**, not measurements of
this routed ParticleGAN path. Our critic/regularization and full-state
checkpointing have not been profiled on the 32B weights. No heavy/native GPU
job, scheduler lease, live server, deployment or publication is initiated here.

Run bounded checks with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q`.
Set `SOPHON_SOURCE=/path/to/pinned/source` to include a tiny randomly initialized
model built from the actual official architecture on CPU. In the native extras
environment also set `SOPHON_NATIVE_LOADER=1` to test real PEFT tag loading on tiny
local random weights. Synthetic checks
verify gradient ownership, bank/router gradients, frozen tag/audio/base bytes,
shared updates, exact continuation, raw inference and admission refusals. Tiny
native execution tests audio embedding merge and gradients; neither establishes
pretrained speech quality, latency, realtime barge-in or turn-taking behavior.

Code is Apache-2.0. See `NOTICE` for upstream and shared component licenses.
Official sources: [Sophon repository](https://github.com/adventists-ai/sophon-realtime),
[fine-tuning](https://github.com/adventists-ai/sophon-realtime/tree/67fe6cfc02616cec5a5a883b063bdefd7b31fc2b/finetune),
[model card](https://huggingface.co/adventists-ai/Sophon-Realtime-v1).
