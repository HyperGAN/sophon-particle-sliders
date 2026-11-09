"""Actual official tagged thinker generation; audio stays native, no ASR proxy."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from sophon_particle_sliders.data import pack, read_waveforms
from sophon_particle_sliders.response import snapshot_prompt


def validate_request(*, prompt=None, messages=None, audios=(), max_tokens=4096,
                     max_new_tokens=64, strength=1., seed=0, output_type='text', images=None):
    if images or output_type != 'text':
        raise ValueError('native Sophon text only; no image or waveform renderer')
    if (prompt is None) == (messages is None):
        raise ValueError('supply exactly one native packed prompt or tagged conversation')
    if messages is not None and (not isinstance(messages, list) or not messages):
        raise ValueError('nonempty tagged native messages required')
    if not isinstance(audios, (list, tuple)) or any(not isinstance(path, str) for path in audios):
        raise ValueError('explicit local audio paths required')
    if prompt is not None and audios:
        raise ValueError('packed native prompts already contain their audio')
    for key, value, limit in (('max_tokens', max_tokens, 32768), ('max_new_tokens', max_new_tokens, 2048)):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError('invalid bounded native ' + key)
    if type(seed) is not int or not 0 <= seed < 2**63-100:
        raise ValueError('explicit valid generation seed required')
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('finite native slider strength in [0,1] required')


@dataclass(frozen=True)
class TextOutput:
    text: str
    native: dict

    def save(self, path):
        """The actual common inference CLI calls save; preserve native tags/EOS."""
        path = Path(path)
        # Refuse overwrite before either output is created; no hidden resampling.
        meta = Path(str(path)+'.native.json')
        if path.exists() or meta.exists():
            raise ValueError('fresh native text and receipt paths required')
        with path.open('x') as stream:
            stream.write(self.text)
        with meta.open('x') as stream:
            json.dump(self.native, stream, indent=2, allow_nan=False);stream.write('\n')


@torch.no_grad()
def generate(native, *, prompt=None, messages=None, audios=(), max_tokens=4096,
             max_new_tokens=64, strength=1., seed=0, output_type='text', images=None):
    validate_request(prompt=prompt, messages=messages, audios=audios, max_tokens=max_tokens,
        max_new_tokens=max_new_tokens, strength=strength, seed=seed, output_type=output_type, images=images)
    if native.processor is None:
        raise ValueError('finished text requires the actual released processor/tokenizer')
    if messages is not None:
        prompt = pack(native.processor, messages, read_waveforms(audios) if audios else [], max_tokens=max_tokens)
    prompt = snapshot_prompt(prompt, native.model.config.audio_token_index)
    length = prompt['input_ids'].shape[1]
    maximum = getattr(native.model.config.text_config, 'max_position_embeddings', None)
    if length > max_tokens or isinstance(maximum, int) and length+max_new_tokens > maximum:
        raise ValueError('native context exceeded; no truncation or vocabulary projection')
    native.check_owner()
    device=native.entry_device
    rng_devices=[]if device.type=='cpu'else [0,1]
    tensors = {key: value.to(device) for key, value in prompt.items()}
    # Direct kwargs win over HF model defaults. The official67fe wrapper carries
    # audio when use_cache=False; HF4.57 does not supply its is_first_iteration
    # flag. Full-prefix calls therefore retain native audio without patching the
    # official implementation or silently producing text-only cached prefill.
    with torch.random.fork_rng(devices=rng_devices):
        torch.default_generator.manual_seed(seed)
        for index in rng_devices:torch.cuda.default_generators[index].manual_seed(seed)
        with native.network.scaled(strength):
            output = native.model.generate(**tensors, max_new_tokens=max_new_tokens,
                do_sample=False, use_model_defaults=False, use_cache=False,
                return_dict_in_generate=True, output_scores=False)
    sequences = output.sequences
    if sequences.ndim != 2 or sequences.shape[0] != 1 or sequences.shape[1] < length:
        raise ValueError('one actual native generated token sequence required')
    if not torch.equal(sequences[:, :length], tensors['input_ids']):
        raise ValueError('native generated sequence must retain exact packed input prefix')
    tokens = sequences[0, length:].detach().cpu().tolist()
    # Keep sound tags and native EOS visible; a legal EOS/truncation is not a
    # failed candidate and this function supplies no quality judgment.
    text = native.processor.tokenizer.decode(tokens, skip_special_tokens=False)
    return TextOutput(text, dict(schema='sophon095-explicit-two-gpu-native-text-generation-v1',
        source_commit=native.model_identity['source_commit'], selection='raw', weight_EMA=None,
        tag_adapter='frozen, active', prompt_tokens=length, audio_chunks=int(prompt.get('audio_lens', torch.empty(0)).numel()),
        new_token_ids=tokens, max_new_tokens=max_new_tokens, strength=strength, seed=seed,
        actual_generate_kwargs=dict(do_sample=False, use_model_defaults=False, use_cache=False),
        frozen_module_map=native.model_identity.get('explicit_frozen_module_map'),pg_device=str(native.pg_device),
        audio_synthesis=False, pretrained_quality_qualified=False))
