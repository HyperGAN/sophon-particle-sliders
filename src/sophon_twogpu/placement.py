"""Literal placement and external live ownership; never an automatic scheduler."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json,os
from typing import Callable

PLAN={'kind':'sophon-frozen-base-explicit-two-gpu-v1','physical_gpu_ids':[0,1],
      'visible_devices':'0,1','layer_split':32,'pg_device':'cuda:1',
      'owned_devices':[{'device':'cuda:0'},{'device':'cuda:1'}]}
_CURRENT=ContextVar('sophon_two_gpu_admission',default=None)

@dataclass(frozen=True)
class OwnedTwoGPUAdmission:
    """Execution-owner callback, not a claim that configuration proves leases.

    Root supplies its real live lease/cap/parent-lifetime check and receipt.
    Neither volatile PID/birth nor the callback enters scientific checkpoint identity.
    """
    plan: dict
    owner_receipt: dict
    check_live: Callable[[],None]

    def check(self):
        if self.plan!=PLAN or not callable(self.check_live):raise ValueError('exact owned two-GPU plan/check required')
        if (set(self.owner_receipt)!={'path','bytes','sha256'} or not isinstance(self.owner_receipt['path'],str)
                or type(self.owner_receipt['bytes'])is not int or self.owner_receipt['bytes']<=0
                or not isinstance(self.owner_receipt['sha256'],str)or len(self.owner_receipt['sha256'])!=64
                or any(x not in '0123456789abcdef'for x in self.owner_receipt['sha256'])
                or not self.owner_receipt['path'].startswith('/')):
            raise ValueError('actual external execution-owner receipt FullRecord required')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1':raise ValueError('owned physicalGPU0/1 must be explicit logicalcuda0/1')
        if self.check_live()is False:raise RuntimeError('actual owner refused two-GPU execution')

@contextmanager
def admitted(admission):
    token=_CURRENT.set(admission)
    try:yield
    finally:_CURRENT.reset(token)

def execution(options):
    if options.get('device','cpu')=='cpu':
        if options.get('placement')is not None:raise ValueError('CPU fixture has no CUDA placement')
        return None
    if options.get('device')!='cuda:1' or options.get('placement')!=PLAN:
        raise ValueError('literal full two-GPU plan with PGdevice cuda:1 required; no auto/balanced/offload')
    owner=_CURRENT.get()
    if not isinstance(owner,OwnedTwoGPUAdmission):raise ValueError('explicit live OwnedTwoGPUAdmission required before allocation')
    owner.check();return owner

def module_map(config):
    text=config['text_config']if isinstance(config,dict)else config.text_config
    layers=text['num_hidden_layers']if isinstance(text,dict)else text.num_hidden_layers
    width=text['hidden_size']if isinstance(text,dict)else text.hidden_size
    if layers!=64 or width!=5120:raise ValueError('released64-layer/5120-width native geometry required on CUDA')
    result={'audio_tower':'cuda:0','multi_modal_projector':'cuda:0',
            'language_model.model.embed_tokens':'cuda:0','language_model.model.rotary_emb':'cuda:0',
            'language_model.model.norm':'cuda:1','language_model.lm_head':'cuda:1'}
    result.update({'language_model.model.layers.'+str(n):'cuda:'+str(n>=32 and 1 or 0)for n in range(64)})
    return result

def validate_local(options,architecture,*,training):
    from pathlib import Path
    for key in ('model_dir','source_dir'):
        if not isinstance(options.get(key),str)or not Path(options[key]).is_dir():raise ValueError('local '+key+' required')
    config=json.loads((Path(options['model_dir'])/'base/config.json').read_text())
    if config.get('text_config',{}).get('hidden_size')!=architecture['response_width']:raise ValueError('full native response width required')
    if options.get('device','cpu')!='cpu':
        module_map(config)
        if architecture['rank']!=8 or architecture['particles']!=128 or architecture['particle_dim']!=4:
            raise ValueError('native CUDA profile retains rank8 and full128x4 bank')
    execution(options)
    if training and (not isinstance(options.get('training_file'),str)or not Path(options['training_file']).is_file()):raise ValueError('local matched training_file required')


def validate_dispatch_hooks(model):
    """Accept only differentiable explicit placement hooks; never offload/no_grad."""
    from accelerate.hooks import AlignDevicesHook,SequentialHook
    def check(hook):
        if type(hook)is SequentialHook:
            for child in hook.hooks:check(child)
        elif type(hook)is AlignDevicesHook:
            if hook.offload or hook.no_grad:raise ValueError('frozen dispatch must keep autograd through inputs; no offload/no_grad')
        else:raise ValueError('unreviewed native dispatch hook')
    for module in model.modules():
        hook=getattr(module,'_hf_hook',None)
        if hook is not None:check(hook)
