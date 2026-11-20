"""One complete original trunk, literal two-GPU owner. No implicit admission."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
import os
PLAN=dict(kind='sophon-original-speech-full64-layer32-32-v1',physical_gpu_ids=[0,1],visible_devices='0,1',layer_split=32,pg_device='cuda:1',decoder_device='cuda:0',heads_device='cuda:1')
_CURRENT=ContextVar('sophon_speech_owner',default=None)
@dataclass(frozen=True)
class OwnedTwoGPUAdmission:
    plan:dict
    owner_receipt:dict
    check_live:object
    def check(self):
        p=self.owner_receipt
        if self.plan!=PLAN or not callable(self.check_live)or set(p)!={'path','bytes','sha256'}or not isinstance(p['path'],str)or not p['path'].startswith('/')or type(p['bytes'])is not int or p['bytes']<=0 or len(p['sha256'])!=64 or any(c not in '0123456789abcdef'for c in p['sha256']):raise ValueError('actual exact twoGPU owner receipt/check required')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1':raise ValueError('explicit owned physicalGPU0/1 visibility required')
        if self.check_live()is False:raise RuntimeError('owner refused')
@contextmanager
def admitted(owner):
    token=_CURRENT.set(owner)
    try:yield
    finally:_CURRENT.reset(token)
def execution(options):
    if options.get('device')!='cuda:1'or options.get('placement')!=PLAN:raise ValueError('full native speech requires literal declared twoGPU profile; no auto maps/quantization')
    owner=_CURRENT.get()
    if not isinstance(owner,OwnedTwoGPUAdmission):raise ValueError('live external owner required before allocation')
    owner.check();return owner
def layer_device(index,count=64):
    if count!=64 or type(index)is not int or not 0<=index<count:raise ValueError('full original64 layers required')
    return 'cuda:0'if index<32 else'cuda:1'
