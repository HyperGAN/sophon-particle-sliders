from copy import deepcopy

import pytest
import torch
from conftest import options, prompt
from particle_sliders import response
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, _hash


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_actual_pg_updates_exact_resume_raw_export_and_no_training_in_inference(
    tmp_path, backend_fixture, monkeypatch
):
    config = SessionConfig("sophon", "cpu-synthetic-contract-v1", **options(tmp_path))
    with response.open_response(config) as full:
        initial = [p.detach().clone() for p in full.game.network.parameters()]
        base = {
            name: p.detach().clone()
            for name, p in full.game.backend.model.named_parameters()
        }
        full.train()
        expected = full.checkpoint()
        assert full.game.completed_steps == 4
        assert any(
            not torch.equal(p, old)
            for p, old in zip(full.game.network.parameters(), initial)
        )
        assert all(
            torch.equal(p, base[k])
            for k, p in full.game.backend.model.named_parameters()
        )
        assert full.ema.values is None
    with response.open_response(config) as split:
        split.train(stop_after=2)
        split.checkpoint(tmp_path / "training.pt")
    with response.open_response(config, resume=tmp_path / "training.pt") as resumed:
        assert resumed.game.completed_steps == 2
        before, rng = resumed.checkpoint(), torch.get_rng_state().clone()
        for strength in (0.0, 0.5, 1.0):
            assert torch.equal(
                resumed.evaluate(strength=strength), resumed.evaluate(strength=strength)
            )
        equal(before, resumed.checkpoint())
        assert torch.equal(rng, torch.get_rng_state())
        resumed.train()
        equal(expected, resumed.checkpoint())
        artifact = resumed.export_inference(tmp_path / "adapter-v2.pt")
        expected_response = resumed.game.backend.response(prompt=prompt(), strength=0.5)
        bad = deepcopy(before)
        bad["manifest"]["config"]["cohort"] = "other"
        bad["sha256"] = _hash({k: v for k, v in bad.items() if k != "sha256"})
        with pytest.raises(CheckpointError):
            resumed.restore(bad)
    (tmp_path / "rows.jsonl").unlink()
    monkeypatch.setattr(
        backend_fixture,
        "prepare_training",
        lambda *a: pytest.fail("inference allocated training"),
    )
    monkeypatch.setattr(
        backend_fixture,
        "read_rows",
        lambda *a, **kw: pytest.fail("inference read teacher data"),
    )
    runtime = dict(model_dir=str(tmp_path), source_dir=str(tmp_path), device="cpu")
    with open_inference(
        artifact,
        options=runtime,
        backend=backend_fixture.INFERENCE,
        operation="response",
    ) as inference:
        assert torch.equal(
            expected_response, inference.response(prompt=prompt(), strength=0.5)
        )
    assert not torch.cuda.is_initialized()


def test_changed_native_audio_rejects_resume(tmp_path, backend_fixture, monkeypatch):
    from conftest import rows

    config = SessionConfig("sophon", "cpu-synthetic-contract-v1", **options(tmp_path))
    with response.open_response(config) as session:
        session.step()
        session.checkpoint(tmp_path / "training.pt")
    changed = rows()
    for row in changed:
        row.neutral["audio_values"].add_(0.1)
        row.positive["audio_values"].add_(0.1)
    monkeypatch.setattr(backend_fixture, "read_rows", lambda *a, **kw: changed)
    with pytest.raises(CheckpointError, match="source, capabilities or cohort"):
        response.open_response(config, resume=tmp_path / "training.pt")


def test_unsupported_gpu_and_wrong_width_refused_before_allocation(
    tmp_path, backend_fixture, monkeypatch
):
    def forbidden(*a, **kw):
        pytest.fail("native allocation before admission")

    monkeypatch.setattr(backend_fixture, "load_native", forbidden)
    opts = options(tmp_path)
    opts["device"] = "cuda:0"
    with pytest.raises(ValueError, match="admission"):
        backend_fixture.TRAINING.open(SessionConfig("sophon", "blocked", **opts))
    opts["device"] = "cpu"
    opts["response_width"] = 16
    with pytest.raises(ValueError, match="full native"):
        backend_fixture.TRAINING.open(SessionConfig("sophon", "blocked", **opts))
    opts["ema_decay"] = 0.9
    with pytest.raises(ValueError, match="EMA"):
        SessionConfig("sophon", "blocked", **opts)
