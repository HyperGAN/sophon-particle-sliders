from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from sophon_speech.backend import CurrentResponse
from sophon_speech.scheduled_audio import render


def test_frame_strengths_apply_to_acoustic_events_and_repeat_the_final_gain(tmp_path):
    applied = []
    active = 0.0

    class AcousticReg(torch.nn.Module):
        def forward(self, value):
            applied.append(active)
            return value

    @contextmanager
    def scaled(value):
        nonlocal active
        previous = active
        active = value
        try:
            yield
        finally:
            active = previous

    reg = AcousticReg()
    current = CurrentResponse.__new__(CurrentResponse)
    current.response_adapter = SimpleNamespace(reg=reg, scaled=scaled)
    current.runtime = SimpleNamespace(owner=SimpleNamespace(check=lambda: None))

    def generate(*, strength, **request):
        with scaled(strength):
            for _ in range(3):
                reg(torch.zeros(1, 5120))
        return {"native_frames": 3, "native_EOS": True, "output": {"path": "fixture.wav"}}

    inference = SimpleNamespace(binding=SimpleNamespace(response=current.response), selection="raw",
                                render=generate)
    receipt = render(inference, strength=0.5, frame_gains=[1.5, 0.5], durable_directory=tmp_path)
    assert applied == receipt["effective_frame_strengths"] == [0.75, 0.25, 0.25]
    assert json.loads((tmp_path / "scheduled-strength.json").read_text())["effective_frame_strengths"] == applied
    assert active == 0.0 and not reg._forward_pre_hooks and not reg._forward_hooks
    with pytest.raises(ValueError, match="effective strengths"):
        render(inference, strength=1.0, frame_gains=[1.5], durable_directory=tmp_path)
