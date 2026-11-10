# Native speech attribution and scope

The historical root NOTICE describes the older thinker/tagged-text backends.
Its statement that speech renderers are not loaded does not describe the added
`sophon_speech95` profile: that profile loads the original E2/HiFT speech stack.
No model weights or execution-owner grants are distributed in this package.

The original Sophon-Realtime source is by Adventists AI, Apache-2.0, at
[`67fe6cfc02616cec5a5a883b063bdefd7b31fc2b`](https://github.com/adventists-ai/sophon-realtime/tree/67fe6cfc02616cec5a5a883b063bdefd7b31fc2b).
The added [native_session.py](src/sophon_speech95/native_session.py) includes
adapted original `Session` methods. Their original engine/method hashes and
device-routing changes are recorded in
[session-lineage.json](src/sophon_speech95/session-lineage.json); the other
official source leaves remain external and are authenticated by
[upstream.json](src/sophon_speech95/upstream.json).

The shared core and ParticleGAN remain separate dependencies under their own
licenses/notices. Consult the upstream model/build notices for the original
base, tag, voice and speech components. This supplement does not change the
historical NOTICE, package license or upstream model terms.
