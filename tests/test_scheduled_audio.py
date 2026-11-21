from contextlib import contextmanager
import json
from threading import Lock
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from sophon_speech.backend import CurrentResponse
from sophon_speech import native_session
from sophon_speech.scheduled_audio import render, render_adaptive
from sophon_particle_sliders.activation_gate import GatePlan, SPEECH_SIGNAL, TAGGED_TEXT_SIGNAL


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


def test_gate_tracks_live_kf_frames_and_native_eos(tmp_path, monkeypatch, steering_wheel):
    # Run the original event selection and feedback loop with small synthetic
    # components at the actual 5120-wide frame boundary, without a native model.
    width = 5120
    keys = SimpleNamespace(KA=0, KF=1, KS=2, STALL=0)
    monkeypatch.setattr(native_session, "L", keys, raising=False)
    monkeypatch.setattr(native_session, "PL", 0, raising=False)
    monkeypatch.setattr(native_session, "dev", torch.device("cpu"), raising=False)
    monkeypatch.setattr(native_session, "T32_EV", False)
    active = 0.0
    seen = {}

    class AcousticReg(torch.nn.Module):
        def forward(self, value):
            return -value + active / 4

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

    def generate(*, strength, durable_directory, **request):
        durable_directory.mkdir()
        native = native_session.Session.__new__(native_session.Session)
        model = SimpleNamespace(
            trunk=lambda value, *args, **kwargs: (value, []),
            dec=lambda value: value.new_tensor([[0, 1, 0] if native.nf < 3 else [0, 0, 1]]),
            reg=reg, fm=SimpleNamespace(from_z=lambda value: value),
        )
        native.c = SimpleNamespace(m=model, IM=7, lock=Lock())
        native.ids, native.r, native.final = [7], 1, True
        native.x, native.nf, native.p, native.run_ = torch.ones(1, 1, width), 0, 0, 0
        native.g, native.mel_out, native.Lc = None, 8, 1
        native.Kc, native.Vc, native.cache, native.frames = [], [], [], []
        native.done, native.cancelled = False, False
        native.stats = dict(t0=perf_counter(), ms_talker=0)
        with scaled(strength):
            native._steps(6)
        assert native.done and native.p == 4  # three KF frames and one EOS
        seen[durable_directory.name] = torch.stack(native.frames)
        return {"native_frames": native.nf, "native_EOS": native.done, "output": {"path": "fixture.wav"}}

    inference = SimpleNamespace(binding=SimpleNamespace(response=current.response), selection="raw",
                                render=generate)
    direction = torch.zeros(width)
    direction[0] = 1
    plan = GatePlan(core_wheel=steering_wheel, signal=SPEECH_SIGNAL, method="cfp", parameter=0,
                    calibration_sha256="a" * 64, direction=direction)
    gated = render_adaptive(inference, strength=0.5, gate_plan=plan,
                            durable_directory=tmp_path / "adaptive")
    replay = render(inference, strength=0.5, frame_gains=[0, 1, 0],
                    durable_directory=tmp_path / "fixed")
    off = render_adaptive(inference, strength=0, gate_plan=plan, durable_directory=tmp_path / "zero")
    render(inference, strength=0, frame_gains=[1], durable_directory=tmp_path / "off")
    assert gated["effective_frame_strengths"] == replay["effective_frame_strengths"] == [0, 0.5, 0]
    assert torch.equal(seen["adaptive"], seen["fixed"])
    assert torch.equal(seen["zero"], seen["off"]) and off["activation_gate"]["eligible"] == 0
    decision = gated["activation_gate"]
    assert [item["position"] for item in decision["decisions"]] == [0, 1, 2]
    assert [item["alignment"] for item in decision["decisions"]] == [1, -1, 1.125]
    assert decision["eligible"] == 3 and decision["firing_rate"] == 1 / 3
    assert json.loads((tmp_path / "adaptive/scheduled-strength.json").read_text()) == {
        key: gated[key] for key in ("activation_gate", "effective_frame_strengths",
                                   "nominal_strength", "native_EOS", "output")
    }

    control = GatePlan(core_wheel=steering_wheel, signal=SPEECH_SIGNAL, method="random", parameter=0,
                       seed=7, calibration_sha256="b" * 64)
    random_off = render_adaptive(inference, strength=0.5, gate_plan=control,
                                 durable_directory=tmp_path / "random")
    assert random_off["activation_gate"]["eligible"] == 3
    assert torch.equal(seen["off"], seen["random"])

    clabs = GatePlan(
        core_wheel=steering_wheel, signal=SPEECH_SIGNAL, method="clabs", parameter=0.5, seed=7,
        calibration_sha256="c" * 64, direction=direction,
        clabs=dict(mean=0, std=1, epsilon=0.01, setpoint=0, proportional_gain=0.1,
                   integral_gain=0.1, beta=0.5),
    )
    first = render_adaptive(inference, strength=0.5, gate_plan=clabs,
                            durable_directory=tmp_path / "clabs-first")
    second = render_adaptive(inference, strength=0.5, gate_plan=clabs,
                             durable_directory=tmp_path / "clabs-second")
    assert first["activation_gate"] == second["activation_gate"]
    assert first["activation_gate"]["decisions"][0]["probability"] != 0.5
    assert torch.equal(seen["clabs-first"], seen["clabs-second"])

    wrong_signal = GatePlan(core_wheel=steering_wheel, signal=TAGGED_TEXT_SIGNAL,
                           method="cfp", parameter=0, calibration_sha256="a" * 64, direction=direction)
    with pytest.raises(ValueError, match="different native"):
        render_adaptive(inference, strength=0.5, gate_plan=wrong_signal,
                        durable_directory=tmp_path / "wrong")

    def fail(_value):
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(reg, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        render_adaptive(inference, strength=0.5, gate_plan=plan,
                        durable_directory=tmp_path / "failure")
    assert active == 0 and not reg._forward_pre_hooks and not reg._forward_hooks


def test_gate_rejects_changed_shared_equations(tmp_path, steering_wheel):
    module_path = "particle_sliders/adaptive_steering.py"
    with ZipFile(steering_wheel) as archive:
        verified = archive.read(module_path)
    modified = tmp_path / "changed.whl"
    with ZipFile(modified, "w") as archive:
        archive.writestr(module_path, b"#" + verified[1:])
    with pytest.raises(ValueError, match="exact shared steering"):
        GatePlan(core_wheel=modified, signal=SPEECH_SIGNAL, method="random", seed=7,
                 parameter=0, calibration_sha256="a" * 64)
