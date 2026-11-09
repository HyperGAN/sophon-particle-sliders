from copy import deepcopy

import pytest
import torch
from conftest import SyntheticThinker, native, prompt, rows

from sophon_particle_sliders.response import SpeechRow
from sophon_particle_sliders.slider import SophonSlider


def test_real_autograd_reaches_only_routed_branch_and_off_keeps_tag():
    torch.manual_seed(5)
    n = native()
    try:
        saved = {k: p.detach().clone() for k, p in n.model.named_parameters()}
        prepared = n.prepare(rows()[0])
        assert torch.equal(n.sample(prepared, scale=0), prepared.neutral)
        assert torch.equal(
            n.sample(prepared, scale=1), prepared.neutral
        )  # zero-up init
        with torch.no_grad():
            for projection in n.slider.projections:
                projection.lora_up.weight.normal_(std=0.02)
        value = n.generated(prepared)
        loss = (value - prepared.target).square().mean()
        loss.backward()
        assert n.network.particles.grad is not None
        assert n.network.particles.grad.abs().sum() > 0
        for projection in n.slider.projections:
            assert projection.lora_down.weight.grad.abs().sum() > 0
            assert projection.lora_up.weight.grad.abs().sum() > 0
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in projection.bridge.parameters()
            )
        assert all(
            not p.requires_grad and p.grad is None and torch.equal(p, saved[k])
            for k, p in n.model.named_parameters()
        )
        assert torch.equal(n.sample(prepared, scale=0), prepared.neutral)
        assert not torch.equal(n.sample(prepared, scale=1), prepared.neutral)
        assert all(
            name.startswith("language_model.model.layers.")
            for name in n.slider.target_names
        )
        assert not hasattr(n.model.audio_tower.q_proj, "_sophon_slider_owner")
    finally:
        n.close()
    assert not torch.cuda.is_initialized()


def test_attachment_scopes_cleanup_and_duplicate_refusal():
    model = SyntheticThinker()
    flags = [(p, p.requires_grad) for p in model.parameters()]
    forwards = [(m, m.forward) for m in model.modules()]
    slider = SophonSlider(model)
    with pytest.raises(ValueError, match="already"):
        SophonSlider(model)
    with slider.scaled(0.5):
        assert all(p.multiplier == 0.5 for p in slider.projections)
        with slider.scaled(0):
            assert all(p.multiplier == 0 for p in slider.projections)
        assert all(p.multiplier == 0.5 for p in slider.projections)
        with pytest.raises(RuntimeError, match="active"):
            slider.close()
    slider.close()
    slider.close()
    assert all(p.requires_grad == flag for p, flag in flags)
    assert all(m.forward == forward for m, forward in forwards)


def test_partial_attachment_failure_rolls_back_native_host_and_rng(monkeypatch):
    from sophon_particle_sliders import slider as module

    model = SyntheticThinker()
    forwards = [(m, m.forward) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    original = module.TinyParticleProjection
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected partial attachment failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "TinyParticleProjection", fail_second)
    with pytest.raises(RuntimeError, match="partial attachment"):
        SophonSlider(model)
    assert all(m.forward == forward for m, forward in forwards)
    assert all(not hasattr(m, "_sophon_slider_owner") for m, _ in forwards)
    assert all(p.requires_grad == flag for p, flag in flags)
    assert torch.equal(rng, torch.get_rng_state())


def test_audio_teacher_identity_and_mutation_isolation():
    n = native()
    try:
        row = rows()[0]
        prepared = n.prepare(row)
        before = n.sample(prepared, scale=0)
        row.neutral["audio_values"].add_(100)
        assert torch.equal(before, n.sample(prepared, scale=0))
        with pytest.raises(ValueError, match="identical audio"):
            n.prepare(row)
        bad = deepcopy(rows()[0])
        bad.neutral["audio_token_len"][0] = 1
        with pytest.raises(ValueError, match="audio token"):
            n.prepare(bad)
        with pytest.raises(ValueError, match="distinct"):
            n.prepare(SpeechRow(prompt(), prompt()))
    finally:
        n.close()
