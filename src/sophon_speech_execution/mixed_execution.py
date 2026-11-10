"""Explicit full-original mixed construction and BF16 CPU sampling contexts.

Captured32/32 RAW identity is retained as capture provenance, not claimed as
actual execution placement. No captured Source/function or Parameter is edited.
"""
from contextlib import contextmanager
from copy import deepcopy
import threading

PROFILE='original-Sophon64-BF16-CPU20to43-GPU0first20-GPU1last20-v1'
DEVICES=['cuda:0']*20+['cpu']*24+['cuda:1']*20
_LOCK=threading.RLock()


def device(index,count=64):
    if count!=64 or type(index)is not int or not 0<=index<count:raise ValueError('Full original64 layers required')
    return DEVICES[index]


@contextmanager
def construction(native):
    from sophon_speech95 import placement
    with _LOCK:
        original=native.layer_device
        if original is not placement.layer_device:raise RuntimeError('Exact original constructor device callback required')
        witness=dict(profile=PROFILE,callback_active_during_original_factory_construction=True,
            original_callback_module=original.__module__,original_callback_name=original.__qualname__,
            original_callback_source_file=original.__code__.co_filename,
            actual_layer_devices=list(DEVICES),original_callback_restored=False,
            captured_factory_Source_files_changed=False,strict_RAW_guards_relaxed=False)
        native.layer_device=device
        try:yield witness
        finally:
            same=native.layer_device is device
            native.layer_device=original
            witness['original_callback_restored']=native.layer_device is original
            if not same:raise RuntimeError('Run-local constructor callback changed unexpectedly; original restored')


def describe(inference,construction_witness,check):
    import torch
    from sophon_speech95 import native as module,placement
    from sophon_speech95.backend import CurrentResponse
    current=getattr(inference.binding.close,'__self__',None)
    if type(current)is not CurrentResponse or type(current.runtime)is not module.NativeRuntime:
        raise RuntimeError('Original strict shared RAW factory only')
    native=current.runtime;check()
    if native.gs is not None or any(getattr(native.session_module,name)is not False for name in ('SCHED','M2','PAR_E2')):
        raise RuntimeError('Frozen synchronous native Session required for thread-local CPU autocast')
    if not construction_witness['original_callback_restored']or module.layer_device is not placement.layer_device:
        raise RuntimeError('Original callback must be restored before any output or identity receipt')
    layers=native.m.layers
    if len(layers)!=64 or any(a is not b for a,b in zip(layers,native.TH.m.layers)):
        raise RuntimeError('Complete original shared64 layers required')
    if [str(next(layer.parameters()).device)for layer in layers]!=DEVICES:raise RuntimeError('Actual original mixed placement differs')
    for index,layer in enumerate(layers):
        if any(str(p.device)!=DEVICES[index]or p.dtype!=torch.bfloat16 for p in layer.parameters()):
            raise RuntimeError('Every original base/LoRA layer Parameter must retain BF16 on its declared device')
    for lang,(K,V)in native.Kp_l.items():
        if len(K)!=64 or len(V)!=64 or any(str(K[i].device)!=DEVICES[i]or str(V[i].device)!=DEVICES[i]for i in range(64)):
            raise RuntimeError('Original voice-prefix K/V must be generated directly on each declared layer device:'+lang)
    if native.Kp is not native.Kp_l['zh'][0]or native.Vp is not native.Kp_l['zh'][1]:raise RuntimeError('Original zh aliases required')
    if str(native.pg_device)!='cuda:1'or str(native.decoder_device)!='cuda:0'or native.options['placement']!=placement.PLAN:
        raise RuntimeError('Original PG/heads/decoder and captured RAW construction declaration required')
    identity=deepcopy(native.model_identity)
    if current.model_identity!=identity:raise RuntimeError('Captured native model identity differs')
    return dict(schema='Sophon-original-mixed-CPU24-native-construction-witness-v1',profile=PROFILE,
        constructor=construction_witness,captured_model_identity=identity,
        captured_original_layer_devices=['cuda:0']*32+['cuda:1']*32,actual_layer_devices=list(DEVICES),
        original_layer_count=64,original_cpu_layer_indices=list(range(20,44)),
        original_PG_heads_embed_norm_device='cuda:1',original_E2_HiFT_device='cuda:0',
        original_factory_and_RAW_network_checks_passed=True,callback_active_during_construction_before_RAW_network_load=True,
        original_callback_restored_before_any_render=True,captured_identity_rebound=False,
        original_layer_functions_order_geometry_and_BF16_weights_retained=True,
        original_prefix_KV_generated_on_actual_devices=True,CPU_BF16_autocast_required_during_native_render=True,
        bitwise_CUDA_or_LR_only_causal_equivalence_claimed=False,fit_guaranteed=False)


@contextmanager
def cpu_sampling():
    import torch
    before=dict(enabled=torch.is_autocast_enabled('cpu'),dtype=str(torch.get_autocast_dtype('cpu')),cache_enabled=torch.is_autocast_cache_enabled())
    witness=dict(schema='Sophon-original-native-CPU-BF16-autocast-execution-v1',requested_device='cpu',requested_dtype='torch.bfloat16',
        original_CPU_autocast_state=before,restored=False,original_CUDA_autocast_scope_unchanged=True)
    try:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            if not torch.is_autocast_enabled('cpu')or torch.get_autocast_dtype('cpu')!=torch.bfloat16:raise RuntimeError('Actual CPU BF16 autocast required')
            witness['actual_enabled']=True;witness['actual_dtype']=str(torch.get_autocast_dtype('cpu'));yield witness
    finally:
        after=dict(enabled=torch.is_autocast_enabled('cpu'),dtype=str(torch.get_autocast_dtype('cpu')),cache_enabled=torch.is_autocast_cache_enabled())
        witness['restored']=after==before
        if after!=before:raise RuntimeError('Original CPU autocast state was not restored')
