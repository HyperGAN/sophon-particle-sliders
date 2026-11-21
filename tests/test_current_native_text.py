"""Actual tiny official generation; no pretrained model or audio synthesis."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from conftest import prompt
from test_native_small import native_small
from sophon_particle_sliders.finished_text import generate
from sophon_particle_sliders.native import LOCK


def test_actual_official_tiny_audio_conditioned_greedy_text(native_small,tmp_path,monkeypatch):
    native,_=native_small
    # This tests exact official architecture, no pretrained or tag quality claim.
    native.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:' '.join(map(str,ids))))
    native.model_identity['source_commit']=LOCK['commit']
    merged=[]
    original_merge=native.model._merge_audio
    def observe_merge(ids,embeds,values,lens,lengths):
        merged.append(dict(input_tokens=ids.shape[1],
            audio_values=None if values is None else values.detach().clone(),
            audio_lens=None if lens is None else lens.detach().clone(),
            audio_token_len=None if lengths is None else lengths.detach().clone()))
        return original_merge(ids,embeds,values,lens,lengths)
    monkeypatch.setattr(native.model,'_merge_audio',observe_merge)
    before=deepcopy(native.model.state_dict());rng=torch.get_rng_state().clone()
    first=generate(native,prompt=prompt(),strength=0,max_new_tokens=2,seed=7)
    second=generate(native,prompt=prompt(),strength=1,max_new_tokens=2,seed=7)
    assert first.text==second.text  # Actual zero-up host initially bypasses exactly.
    assert first.native['audio_chunks']==1 and first.native['max_new_tokens']==2
    assert first.native['actual_generate_kwargs']['do_sample']is False
    assert first.native['actual_generate_kwargs']['use_cache']is False
    # Observe the REAL official wrapper during generation, not receipt counts:
    # cached-prefill bug would leave values=None or never merge the audio.
    expected=prompt()
    assert merged and merged[0]['input_tokens']==expected['input_ids'].shape[1]
    for call in merged:
        for key in ('audio_values','audio_lens','audio_token_len'):
            assert call[key] is not None and torch.equal(call[key],expected[key])
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,before[name])for name,value in native.model.state_dict().items())
    path=tmp_path/'finished.txt';first.save(path)
    assert path.read_text()==first.text
    with pytest.raises(ValueError,match='fresh'):first.save(path)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("explicit_two_gpu", [False, True])
def test_native_raw_text_schedules_preserve_audio_and_full_prefix(
    native_small, tmp_path, monkeypatch, explicit_two_gpu,
):
    from conftest import rows
    from particle_sliders.inference import open_inference
    from particle_sliders.lifecycle import SessionConfig
    from sophon_particle_sliders.scheduled_text import render

    if explicit_two_gpu:
        from sophon_twogpu import backend
        from sophon_twogpu.response import SophonResponse
        from sophon_twogpu.slider import SophonSlider
        from test_explicit_twogpu_profile import config_for
        config = config_for(tmp_path)
        slider_options = dict(pg_device="cpu")
    else:
        from sophon_particle_sliders import current_backend as backend
        from sophon_particle_sliders.response import SophonResponse
        from sophon_particle_sliders.slider import SophonSlider
        from test_current_profile import current_options
        config = SessionConfig("sophon_scoped", "tiny-tagged-schedule", **current_options(tmp_path))
        slider_options = {}

    captured, _ = native_small
    captured.close()
    loaded = []

    def load_native(options):
        model = deepcopy(captured.model)
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        slider = SophonSlider(model, rank=options["rank"], particles=options["particles"],
                              particle_dim=4, **slider_options)
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        native = SophonResponse(model, slider, model_identity=dict(source_commit=LOCK["commit"]),
                                processor=SimpleNamespace(tokenizer=tokenizer))
        loaded.append(native)
        return native

    monkeypatch.setattr(backend, "load_native", load_native)
    monkeypatch.setattr(backend, "read_rows", lambda *args, **kwargs: rows())
    with backend.open_training(config) as training:
        training.train(stop_after=1)
        artifact = training.export_inference(selection="raw")
    options = dict(model_dir=str(tmp_path), source_dir=str(tmp_path), device="cpu")
    request = prompt()
    steps = 3
    with open_inference(artifact, backend=backend.INFERENCE, options=options, output_type="text",
                        native_request=dict(prompt=request, max_new_tokens=steps)) as session:
        native = loaded[-1]
        model = native.model
        native_merge = model._merge_audio
        audio_widths = []

        def observe_merge(ids, embeds, values, lengths, token_lengths):
            assert torch.equal(values, request["audio_values"])
            assert torch.equal(lengths, request["audio_lens"])
            assert torch.equal(token_lengths, request["audio_token_len"])
            audio_widths.append(ids.shape[1])
            return native_merge(ids, embeds, values, lengths, token_lengths)

        def force_token(_head, _inputs, logits):
            result = logits.clone()
            result[..., 5] += 1_000
            return result

        monkeypatch.setattr(model, "_merge_audio", observe_merge)
        prefix = request["input_ids"].shape[1]
        host, projection, attached = native.slider._hosts[0]
        residuals = []

        def measure_residual(_host, args, output):
            residuals.append(output - projection.org_forward(args[0]))

        with model.language_model.lm_head.register_forward_hook(force_token):
            scalar = session.render(prompt=request, strength=0.5, max_new_tokens=steps)
            constant = render(session, prompt=request, strength=0.5, max_new_tokens=steps, step_gains=[1] * steps)
            with host.register_forward_hook(measure_residual):
                varied = render(session, prompt=request, strength=0.5, max_new_tokens=steps, step_gains=[1, 0, 2])
            replay = session.render(prompt=request, strength=0.5, max_new_tokens=steps)
        assert scalar == replay
        assert scalar.text == constant.text and scalar.native["new_token_ids"] == [5] * steps
        assert varied.native["strength_schedule"] == dict(prefix_tokens=prefix, effective_strengths=[0.5, 0.0, 1.0])
        assert audio_widths == [prefix, prefix + 1, prefix + 2] * 4
        assert len(residuals) == steps
        assert torch.count_nonzero(residuals[1][:, :prefix]) > 0
        assert torch.count_nonzero(residuals[1][:, prefix]) == 0
        assert torch.count_nonzero(residuals[2][:, prefix]) == 0
        assert torch.count_nonzero(residuals[2][:, prefix + 1]) > 0
        assert host.forward == attached
    assert not torch.cuda.is_initialized()
