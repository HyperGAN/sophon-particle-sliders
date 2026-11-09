"""New profile only; authored for a future admitted current-core CPU run."""
from copy import deepcopy
import math
import pytest
import torch
from conftest import options, native, rows
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.inference import open_inference
from test_lifecycle import equal
from sophon_particle_sliders import current_backend as current
from sophon_particle_sliders.current_contract import (
    INITIALIZATION_POLICY, SOURCE_PROFILE, rate_overrides, validate_installed_core)


def current_options(tmp_path):
    (tmp_path/'base').mkdir(exist_ok=True)
    (tmp_path/'base/config.json').write_text('{"text_config":{"hidden_size":32}}')
    (tmp_path/'rows.jsonl').write_text('matched fixture rows')
    return dict(options(tmp_path), source_profile=SOURCE_PROFILE,
        initialization_policy=INITIALIZATION_POLICY, pg_version='0.9.5',
        pg_source_profile='release-0.9.5', penalty_input_domain='continuous')


def test_actual_current_core_source_identity():
    validate_installed_core()


def test_exact_common_literals_and_refusals():
    literal = dict(lr=.012, d_lr_mult=1.5000000000000002, prior_lr_mult=2.4999999999999996)
    assert rate_overrides(literal) == literal and rate_overrides({}) == {}
    for value in (True, False, 0, -.1, math.nan, math.inf, '0.012'):
        for key in literal:
            with pytest.raises(ValueError):rate_overrides({key: value})


def test_new_fresh_full_resume_raw_and_initializer_boundary(tmp_path, monkeypatch):
    opts = current_options(tmp_path)
    opts.update(lr=.012, d_lr_mult=.5, prior_lr_mult=.75)
    config = SessionConfig('sophon095', 'new-synthetic-fresh-v1', **opts)
    monkeypatch.setattr(current, 'load_native', native)
    monkeypatch.setattr(current, 'read_rows', lambda *a, **kw: rows())
    with current.open_training(config) as full:
        n=full.game.backend
        assert full.game.signature['penalty_input_domain']=='continuous'
        assert all(torch.count_nonzero(p.lora_up.weight)==0 and torch.count_nonzero(p.lora_down.weight)>0 for p in n.slider.projections)
        assert all(not p.requires_grad and p.grad is None for p in n.model.parameters())
        receipt=deepcopy(n.fresh_initialization)
        assert receipt['completed_steps']==0 and receipt['restored_from']is None
        assert receipt['components']['adapter.bank']['parameters']['particles']['host_declaration']['std']==.1
        assert [group['lr']for group in full.game.opt_g.param_groups]==[opts['lr'],opts['lr']*opts['prior_lr_mult']]
        assert all(group['lr']==.006 for group in full.game.opt_d.param_groups)
        full.train();expected=full.checkpoint()
        assert full.ema.values is None
    with current.open_training(config) as split:
        split.train(stop_after=2);packet=split.checkpoint()
    import sophon_particle_sliders.fresh_initialization as initialization
    monkeypatch.setattr(initialization,'initialize_fresh_components',lambda *a,**kw: pytest.fail('fresh initializer during restore/inference'))
    with current.open_training(config,resume=packet) as resumed:
        assert resumed.game.completed_steps==2
        resumed.train();equal(expected,resumed.checkpoint())
        artifact=resumed.export_inference()
    monkeypatch.setattr(current,'read_rows',lambda *a,**kw: pytest.fail('inference read teacher rows'))
    with open_inference(artifact,backend=current.INFERENCE,options=dict(model_dir=str(tmp_path),source_dir=str(tmp_path),device='cpu'),operation='response') as session:
        assert session.selection=='raw'
    assert not torch.cuda.is_initialized()


def test_missing_intent_and_legacy_packet_refuse_before_allocation(tmp_path,monkeypatch):
    config=SessionConfig('sophon095','new-fresh',**current_options(tmp_path))
    monkeypatch.setattr(current,'load_native',lambda *a:pytest.fail('native allocation before intent refusal'))
    with pytest.raises(ValueError,match='intent'):current.TRAINING.open(config)
    bad=deepcopy(config.as_dict());bad['options']['recipe_overrides']={'lr':.01}
    with pytest.raises(ValueError,match='literal'):current.open_training(SessionConfig.from_dict(bad))
