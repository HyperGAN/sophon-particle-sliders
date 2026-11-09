import pytest
import torch

from sophon_particle_sliders.data import TAG_SENTENCE, pack
from sophon_particle_sliders.native import LOCK, official_modules


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True)
        return (
            [1]
            + [
                127
                for message in messages
                for _ in range(message["content"].count("<|audio|>"))
            ]
            + [2]
        )


class Processor:
    tokenizer = Tokenizer()
    audio_token_index = 127

    def audio_features(self, waveforms):
        return dict(
            audio_values=torch.zeros(len(waveforms), 128, 32),
            audio_lens=torch.full((len(waveforms),), 32, dtype=torch.long),
            audio_token_len=torch.full((len(waveforms),), 2, dtype=torch.long),
            tokens_per_clip=[2] * len(waveforms),
        )


def test_chatml_audio_expansion_and_no_silent_truncation():
    messages = [
        {"role": "system", "content": TAG_SENTENCE},
        {"role": "user", "content": "<audio>"},
    ]
    p = pack(Processor(), messages, [object()])
    assert p["input_ids"].tolist() == [[1, 127, 127, 2]]
    assert messages[1]["content"] == "<audio>"
    with pytest.raises(ValueError, match="one audio"):
        pack(Processor(), messages, [])
    with pytest.raises(ValueError, match="max_tokens"):
        pack(Processor(), messages, [object()], max_tokens=3)
    with pytest.raises(ValueError, match="sound-tag"):
        pack(Processor(), messages[1:], [object()])


def test_upstream_source_rejected_before_import(tmp_path):
    for path in LOCK["files"]:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("raise RuntimeError('unreviewed source must not execute')")
    with pytest.raises(ValueError, match="reviewed pin"):
        official_modules(tmp_path)
