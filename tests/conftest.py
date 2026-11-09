"""Small synthetic contract host; explicitly not pretrained Sophon."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sophon_particle_sliders.response import SophonResponse, SpeechRow
from sophon_particle_sliders.slider import SophonSlider


class TaggedLinear(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.base_layer = nn.Linear(width, width)
        self.tag = nn.Parameter(torch.randn(width, width) * 0.01)

    def get_base_layer(self):
        return self.base_layer

    def forward(self, x):
        return self.base_layer(x) + x @ self.tag


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = TaggedLinear(width)

    def forward(self, x):
        return x + torch.tanh(self.self_attn.q_proj(x + x.mean(dim=1, keepdim=True)))


class Decoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([Block(width), Block(width)])

    def forward(self, *, inputs_embeds, **kwargs):
        x = inputs_embeds
        for block in self.layers:
            x = block(x)
        return SimpleNamespace(last_hidden_state=x)


class SyntheticThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=32), audio_token_index=127
        )
        self.embed = nn.Embedding(128, 32)
        self.language_model = nn.Module()
        self.language_model.model = Decoder(32)
        self.audio_tower = nn.Module()
        self.audio_tower.q_proj = nn.Linear(128, 32)
        self.multi_modal_projector = nn.Linear(32, 32)

    def _merge_audio(self, ids, inputs_embeds, values, lens, lengths):
        x = self.embed(ids)
        if values is not None:
            features = self.multi_modal_projector(
                self.audio_tower.q_proj(values.mean(-1))
            )
            audio = torch.cat(
                [f[None].expand(int(n), -1) for f, n in zip(features, lengths)]
            )
            mask = ids == 127
            x = x.masked_scatter(mask[..., None].expand_as(x), audio)
        return x


def prompt(ids=(1, 127, 127, 4), audio=True):
    result = {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
    }
    if audio:
        result.update(
            audio_values=torch.linspace(-0.1, 0.1, 128 * 32).reshape(1, 128, 32),
            audio_lens=torch.tensor([32]),
            audio_token_len=torch.tensor([2]),
        )
    return result


def rows():
    return [
        SpeechRow(prompt((1 + i, 127, 127, 4)), prompt((10 + i, 127, 127, 4)))
        for i in range(2)
    ]


def native(options=None):
    options = options or {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1919)
        model = SyntheticThinker().requires_grad_(False)
    slider = SophonSlider(
        model,
        rank=options.get("rank", 4),
        particles=options.get("particles", 8),
        particle_dim=options.get("particle_dim", 4),
    )
    return SophonResponse(
        model,
        slider,
        model_identity={"fixture": "synthetic contract host, no pretrained weights"},
    )


@pytest.fixture(autouse=True)
def bounded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def backend_fixture(tmp_path, monkeypatch):
    from particle_sliders import response

    from sophon_particle_sliders import backend

    (tmp_path / "base").mkdir()
    (tmp_path / "base/config.json").write_text('{"text_config":{"hidden_size":32}}')
    (tmp_path / "rows.jsonl").write_text("fixture rows")
    monkeypatch.setattr(backend, "load_native", native)
    monkeypatch.setattr(backend, "read_rows", lambda *a, **kw: rows())
    monkeypatch.setattr(response, "response_backend", lambda name: backend.TRAINING)
    return backend


def options(tmp_path):
    return dict(
        model_dir=str(tmp_path),
        source_dir=str(tmp_path),
        training_file=str(tmp_path / "rows.jsonl"),
        base_precision="float32",
        response_width=32,
        rank=4,
        particles=8,
        particle_dim=4,
        steps=4,
        batch=2,
        seed=17,
        ema_decay=None,
        device="cpu",
    )
