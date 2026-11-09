"""Offline original full base/audio/tag with a literal frozen-module CUDA map."""
import tempfile
from pathlib import Path
from importlib.metadata import version
import torch
from sophon_particle_sliders.native import LOCK,official_modules,model_files
from .placement import execution,module_map,validate_dispatch_hooks
from .slider import SophonSlider
from .response import SophonResponse

def load_native(options):
    owner=execution(options);root=Path(options['model_dir']).resolve();files=model_files(root)
    if not version('peft').startswith('0.20.'):raise ImportError('exact qualified PEFT0.20 native conversion required')
    modeling,processing,adapters=official_modules(options['source_dir'])
    from peft import PeftModel
    dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[options.get('base_precision','bfloat16')]
    import json
    config=json.loads((root/'base/config.json').read_text())
    mapping=module_map(config)if owner is not None else None
    kwargs=dict(local_files_only=True,torch_dtype=dtype,attn_implementation='eager')
    if mapping is not None:kwargs.update(device_map=mapping,low_cpu_mem_usage=True)
    if owner is not None:owner.check()
    model=modeling.DuplexJevForConditionalGeneration.from_pretrained(root/'base',**kwargs)
    if owner is None:model.to('cpu')
    with tempfile.TemporaryDirectory(prefix='sophon-twogpu-tag-')as temporary:
        adapters.to_swift(str(root/'adapters/tag'),temporary)
        wrapped=PeftModel.from_pretrained(model,temporary,adapter_name='tag',is_trainable=False,local_files_only=True)
    model=wrapped.get_base_model();model.gradient_checkpointing_disable();model.requires_grad_(False)
    if mapping is not None:
        validate_dispatch_hooks(model)
        if getattr(model,'hf_device_map',None)!=mapping:raise ValueError('actual frozen placement differs from literal declared map')
        for name,p in model.named_parameters():
            prefixes=[key for key in mapping if name==key or name.startswith(key+'.')]
            if not prefixes or str(p.device)!=mapping[max(prefixes,key=len)]:raise ValueError('unmapped/misplaced original frozen parameter:'+name)
        owner.check()
    processor=processing.SophonRealtimeProcessor.from_pretrained(root/'base',model.config)
    identity=dict(model='adventists-ai/Sophon-Realtime-v1',files=files,source_commit=LOCK['commit'],source_files=LOCK['files'],
        peft=version('peft'),transformers=version('transformers'),tag='frozen, unmerged',explicit_frozen_module_map=mapping)
    slider=SophonSlider(model,pg_device='cuda:1'if owner is not None else 'cpu',**{k:options[k]for k in('rank','particles','particle_dim')if k in options})
    try:return SophonResponse(model,slider,model_identity=identity,processor=processor,admission=owner)
    except BaseException:slider.close();raise
