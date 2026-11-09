"""New profile's real CPU fixtures; author does not execute Torch/tests."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from particle_sliders.lifecycle import SessionConfig
from conftest import SyntheticThinker,options,prompt,rows
from test_lifecycle import equal
from test_native_small import native_small
from sophon_twogpu95 import backend as current
from sophon_twogpu95.contract import SOURCE_PROFILE,INITIALIZATION_POLICY
from sophon_twogpu95.placement import PLAN,OwnedTwoGPUAdmission,admitted,execution,module_map
from sophon_twogpu95.response import SophonResponse
from sophon_twogpu95.slider import SophonSlider


def config_for(path):
    (path/'base').mkdir(exist_ok=True)
    (path/'base/config.json').write_text('{"text_config":{"hidden_size":32}}')
    (path/'rows.jsonl').write_text('matched fixture rows')
    opts=dict(options(path),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,
        pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous',
        lr=.012,d_lr_mult=.5,prior_lr_mult=.75)
    return SessionConfig('sophon095_twogpu','new-two-gpu-source-CPU-fixture',**opts)


def native(options):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1919);model=SyntheticThinker().requires_grad_(False)
    # Real new slider/response classes; synthetic model exposes the same embedding API.
    model.get_input_embeddings=lambda:model.embed
    slider=SophonSlider(model,rank=options['rank'],particles=options['particles'],particle_dim=4,pg_device='cpu')
    return SophonResponse(model,slider,model_identity={'fixture':'new explicit placement CPU host'})


def test_literal_map_and_missing_admission_refuse_without_cuda(monkeypatch):
    mapping=module_map({'text_config':{'hidden_size':5120,'num_hidden_layers':64}})
    assert len(mapping)==70 and mapping['language_model.model.layers.31']=='cuda:0'
    assert mapping['language_model.model.layers.32']=='cuda:1'
    assert mapping['audio_tower']==mapping['multi_modal_projector']=='cuda:0'
    assert mapping['language_model.model.norm']==mapping['language_model.lm_head']=='cuda:1'
    for name in ('auto','balanced','cpu','disk'):assert name not in mapping.values()
    for width,layers in ((4096,64),(5120,32)):
        with pytest.raises(ValueError):module_map({'text_config':{'hidden_size':width,'num_hidden_layers':layers}})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA probe from structural validation'))
    with pytest.raises(ValueError,match='Admission'):execution(dict(device='cuda:1',placement=PLAN))
    assert execution({'device':'cpu'})is None


def test_owner_context_is_explicit_live_and_does_not_leak(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1');calls=[]
    owner=OwnedTwoGPUAdmission(deepcopy(PLAN),dict(path='/Root/actual-owner.json',bytes=1,sha256='a'*64),lambda:calls.append('live'))
    with admitted(owner):assert execution(dict(device='cuda:1',placement=PLAN))is owner
    assert calls==['live']
    with pytest.raises(ValueError,match='Admission'):execution(dict(device='cuda:1',placement=PLAN))
    owner=OwnedTwoGPUAdmission(deepcopy(PLAN),dict(path='/Root/actual-owner.json',bytes=1,sha256='a'*64),lambda:False)
    with admitted(owner),pytest.raises(RuntimeError):execution(dict(device='cuda:1',placement=PLAN))
    assert not torch.cuda.is_initialized()


def test_native_gradient_path_frozen_tag_zero_up_and_bank_hostlaw():
    n=native(dict(rank=4,particles=8));before=deepcopy(n.model.state_dict())
    try:
        prepared=n.prepare(rows()[0]);off=n.sample(prepared,scale=0);zero=n.sample(prepared,scale=1)
        assert torch.equal(off,zero)and n.generated(prepared).shape==(1,32)
        for projection in n.slider.projections:
            assert torch.count_nonzero(projection.lora_down.weight)>0
            assert torch.count_nonzero(projection.lora_up.weight)==0
            projection.lora_up.weight.data.fill_(.01)
        value=n.generated(prepared);value.square().mean().backward()
        assert n.slider.particles.grad is not None and torch.isfinite(n.slider.particles.grad).all()
        assert all(not p.requires_grad and p.grad is None for p in n.model.parameters())
        assert all(p.device.type=='cpu'and p.dtype==torch.float32 for p in n.network.parameters())
        assert all(torch.equal(v,before[k])for k,v in n.model.state_dict().items())
    finally:n.close()


def test_exact_fresh_full_resume_raw_and_no_reinitialization(tmp_path,monkeypatch):
    config=config_for(tmp_path);monkeypatch.setattr(current,'load_native',native);monkeypatch.setattr(current,'read_rows',lambda *a,**k:rows())
    with current.open_training(config)as full:
        assert full.ema.values is None
        assert full.game.backend.runtime.pg_device.type=='cpu'
        receipt=full.game.backend.fresh_initialization
        assert receipt['components']['adapter.bank']['parameters']['particles']['host_declaration']['std']==.1
        assert [g['lr']for g in full.game.opt_g.param_groups]==[.012,.009]
        assert all(g['lr']==.006 for g in full.game.opt_d.param_groups)
        full.train();expected=full.checkpoint()
    with current.open_training(config)as first:
        first.train(stop_after=2);saved=first.checkpoint()
    import sophon_twogpu95.initialization as init
    monkeypatch.setattr(init,'initialize_fresh_components',lambda *a,**k:pytest.fail('initialization during exact resume/inference'))
    with current.open_training(config,resume=saved)as resumed:
        assert resumed.game.completed_steps==2;resumed.train();equal(expected,resumed.checkpoint());artifact=resumed.export_inference()
    with current.open_inference(artifact,options=dict(model_dir=str(tmp_path),source_dir=str(tmp_path),device='cpu'),operation='response')as session:
        assert session.selection=='raw'
    assert not torch.cuda.is_initialized()


def test_real_tiny_official_audio_text_uses_new_path(native_small,monkeypatch):
    old,_=native_small;old.close()
    n=SophonResponse(old.model,SophonSlider(old.model,rank=4,particles=8,particle_dim=4,pg_device='cpu'),
        model_identity=old.model_identity,processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**k:' '.join(map(str,ids)))))
    from sophon_twogpu95.finished_text import generate
    merged=[];original=n.model._merge_audio
    def observe(ids,embeds,values,lens,lengths):
        merged.append((values,lens,lengths));return original(ids,embeds,values,lens,lengths)
    monkeypatch.setattr(n.model,'_merge_audio',observe)
    rng=torch.get_rng_state().clone();before=deepcopy(n.model.state_dict())
    try:
        off=generate(n,prompt=prompt(),strength=0,max_new_tokens=2,seed=7)
        on=generate(n,prompt=prompt(),strength=1,max_new_tokens=2,seed=7)
        assert off.text==on.text and merged and all(all(v is not None for v in parts)for parts in merged)
        assert on.native['actual_generate_kwargs']==dict(do_sample=False,use_model_defaults=False,use_cache=False)
        assert torch.equal(rng,torch.get_rng_state())and all(torch.equal(v,before[k])for k,v in n.model.state_dict().items())
    finally:n.close()


def test_real_differentiable_dispatch_hook_preserved_through_attachment():
    from accelerate.hooks import AlignDevicesHook,add_hook_to_module,remove_hook_from_module
    from sophon_twogpu95.placement import validate_dispatch_hooks
    model=SyntheticThinker().requires_grad_(False);model.get_input_embeddings=lambda:model.embed
    host=model.language_model.model.layers[0].self_attn.q_proj.base_layer
    add_hook_to_module(host,AlignDevicesHook(execution_device='cpu',offload=False))
    attached=host.forward;validate_dispatch_hooks(model)
    slider=SophonSlider(model,rank=4,particles=8,particle_dim=4,pg_device='cpu')
    n=SophonResponse(model,slider,model_identity={'fixture':'actual differentiable Accelerate hook'})
    try:
        for projection in slider.projections:projection.lora_up.weight.data.fill_(.01)
        n.generated(n.prepare(rows()[0])).square().mean().backward()
        assert slider.projections[0].lora_up.weight.grad is not None
    finally:n.close()
    assert host.forward==attached
    host._hf_hook.no_grad=True
    with pytest.raises(ValueError,match='autograd'):validate_dispatch_hooks(model)
    host._hf_hook.no_grad=False;remove_hook_from_module(host)
