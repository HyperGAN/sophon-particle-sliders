"""Actual released architecture, tiny random weights; no pretrained/quality proof."""

import os

import pytest
import torch
from conftest import rows
from particle_sliders import pg09
from particle_sliders.pg09_audio import AudioResponseCritic

from sophon_particle_sliders.native import official_modules
from sophon_particle_sliders.response import SophonResponse, conditions
from sophon_particle_sliders.slider import SophonSlider


@pytest.fixture
def native_small():
    source = os.environ.get("SOPHON_SOURCE")
    if not source:
        pytest.skip(
            "set SOPHON_SOURCE to the reviewed official source checkout for native-small CPU tests"
        )
    modeling, processing, _ = official_modules(source)
    from transformers import Qwen3Config

    config = modeling.DuplexJevConfig(
        text_config=Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attention_dropout=0.0,
            use_cache=False,
        ),
        audio_config=modeling.Qwen3ASREncoderConfig(
            d_model=16,
            downsample_hidden_size=8,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            encoder_layers=1,
        ),
        hidden_size=32,
        audio_token_index=127,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(31)
    model = (
        modeling.DuplexJevForConditionalGeneration(config).requires_grad_(False).eval()
    )
    n = SophonResponse(
        model,
        SophonSlider(model, rank=4, particles=8, particle_dim=4),
        model_identity={
            "qualification": "actual official architecture; tiny random weights, no pretrained tag adapter"
        },
    )
    yield n, processing
    n.close()


def test_official_audio_merge_hidden_equivalence_and_native_gradients(native_small):
    n, _ = native_small
    row = rows()[0]
    p = n.prepare(row)
    with torch.no_grad():
        official = n.model(
            **row.neutral, output_hidden_states=True, return_dict=True, use_cache=False
        )
        assert torch.equal(p.neutral, official.hidden_states[-1][:, -1])
        for projection in n.slider.projections:
            projection.lora_up.weight.normal_(std=0.01)
    response = n.generated(p)
    (response - p.target).square().mean().backward()
    assert (
        n.network.particles.grad is not None
        and n.network.particles.grad.abs().sum() > 0
    )
    assert all(p.grad is None for p in n.model.parameters())
    assert all(
        torch.isfinite(p.grad).all()
        for p in n.network.parameters()
        if p.grad is not None
    )
    assert torch.equal(n.sample(p, scale=0), p.neutral)
    assert not torch.equal(n.sample(p, scale=1), p.neutral)
    assert not torch.cuda.is_initialized()


def test_native_small_shared_game_update_and_exact_resume(native_small):
    n, _ = native_small
    prepared = [n.prepare(row) for row in rows()]
    cond = conditions(prepared)
    torch.manual_seed(9)
    game = pg09.ConditionalPG09(
        n,
        prepared,
        AudioResponseCritic(32, cond.shape[1]),
        conditions=cond,
        particles=n.network.particles,
        response_domain="continuous_vector",
        total_steps=2,
        batch_size=2,
        seed=11,
    )
    game.step()
    saved = game.state_dict()
    game.step()
    expected = game.state_dict()
    game.load_state_dict(saved)
    game.step()
    from test_lifecycle import equal

    equal(expected, game.state_dict())
    assert game.completed_steps == 2
    assert not torch.cuda.is_initialized()


def test_official_processor_chunk_counts(native_small):
    _, processing = native_small
    import numpy as np
    from transformers import WhisperFeatureExtractor

    processor = processing.SophonRealtimeProcessor(
        None,
        WhisperFeatureExtractor(
            feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
        ),
    )
    features = processor.audio_features(
        [np.zeros(1600, dtype=np.float32), np.zeros(16000 * 31, dtype=np.float32)]
    )
    assert features["audio_values"].shape[1] == 128
    assert len(features["audio_lens"]) == 3
    assert features["audio_lens"].max() <= 3000
    assert features["tokens_per_clip"] == [1, 195]  # ceil(3000/16) + ceil(104/16)
    assert int(features["audio_token_len"].sum()) == sum(features["tokens_per_clip"])
