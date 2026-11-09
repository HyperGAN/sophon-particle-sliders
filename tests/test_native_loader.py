"""Opt-in loader round trip using tiny local random weights and real PEFT tag."""

import os

import pytest
import torch
from conftest import rows

from sophon_particle_sliders.native import load_native, official_modules


@pytest.mark.skipif(
    os.environ.get("SOPHON_NATIVE_LOADER") != "1",
    reason="opt in with SOPHON_NATIVE_LOADER=1 in the isolated PEFT 0.20 native environment",
)
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_offline_native_base_tag_loader(tmp_path, precision):
    source = os.environ["SOPHON_SOURCE"]
    modeling, _, _ = official_modules(source)
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3Config,
        WhisperFeatureExtractor,
    )

    torch.manual_seed(13)
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
    model = modeling.DuplexJevForConditionalGeneration(config).eval()
    base = tmp_path / "base"
    model.save_pretrained(base)
    vocab = {"t" + str(i): i for i in range(124)}
    vocab.update(
        {"<|im_start|>": 124, "<|im_end|>": 125, "<unk>": 126, "<|audio|>": 127}
    )
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|audio|>"],
    )
    fast.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    fast.save_pretrained(base)
    WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
    ).save_pretrained(base)
    from sophon_particle_sliders.slider import TARGET

    peft = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=TARGET.pattern,
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.01)
    peft.save_pretrained(tmp_path / "adapters/tag")
    opts = dict(
        model_dir=str(tmp_path),
        source_dir=source,
        base_precision=precision,
        device="cpu",
        rank=4,
        particles=8,
        particle_dim=4,
    )
    native = load_native(opts)
    try:
        assert native.processor.tokenizer.convert_tokens_to_ids("<|audio|>") == 127
        tag = {
            name: p.detach().clone()
            for name, p in native.model.named_parameters()
            if "lora_" in name
        }
        assert tag and any(
            "lora_B.tag" in name and torch.count_nonzero(p) for name, p in tag.items()
        )
        assert all(not p.requires_grad for p in native.model.parameters())
        q = native.model.language_model.model.layers[0].self_attn.q_proj
        x = torch.randn(1, 2, 32, dtype=q.base_layer.weight.dtype)
        with torch.no_grad(), native.slider.scaled(0):
            assert not torch.equal(q(x), q.base_layer(x))
        prepared = native.prepare(rows()[0])
        assert torch.equal(
            native.sample(prepared, scale=0), native.sample(prepared, scale=1)
        )
        with torch.no_grad():
            for projection in native.slider.projections:
                projection.lora_up.weight.normal_(std=0.01)
        native.generated(prepared).square().mean().backward()
        assert native.network.particles.grad.abs().sum() > 0
        assert all(
            torch.equal(p, tag[name]) and p.grad is None
            for name, p in native.model.named_parameters()
            if name in tag
        )
        assert native.model_identity["files"]["base/config.json"]
        assert native.model_identity["files"]["adapters/tag/adapter_model.safetensors"]
        from sophon_particle_sliders.data import TAG_SENTENCE

        messages = [
            {"role": "system", "content": TAG_SENTENCE},
            {"role": "user", "content": "<audio>"},
        ]
        # Actual WAV loading/downmix/resampling, native mel packing and forward.
        import numpy as np
        import soundfile as sf

        wav = tmp_path / "question.wav"
        mono = 0.02 * np.sin(np.arange(800, dtype=np.float32) * 0.1)
        sf.write(wav, np.stack([mono, mono], axis=1), 8000, subtype="FLOAT")
        assert native.response(
            messages=messages, audios=[str(wav)], strength=0.5
        ).shape == (1, 32)
    finally:
        native.close()
    assert not torch.cuda.is_initialized()
