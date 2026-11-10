"""Original full Sophon talker/E2 owners, explicit physical placement and strict pins.

No engine.Core call: that constructor forces one CUDA device and may repack
LoRA Parameters. This product uses its exact original constructors/loaders,
retains every original Parameter and routes original per-layer functions.
"""
from pathlib import Path
from contextlib import contextmanager
import importlib,importlib.util,json,os,sys,threading,types,uuid
import torch
from .contract import file_hash,DEFAULTS,validate_profile,validate_options
from .placement import execution,layer_device
from .runtime import identity as runtime_identity

FLAGS=dict(STALL='10',FM_K='1024',LORA_BF16='1',PARA_INIT='zero',EV_GP0='8',EV_GS0='1',
    T32_EV='1',EV_GAIN='1',T32_FP8='0',T32_FUSED='0',T32_M2='0',T32_GRAPH='0',T32_CTXGRAPH='0',T32_SCHED='0',T32_E2B='0',T32_LORA16='0',E2V2='0',SPK_CTX='1',E2_PROMPT='1',T32_PAR_E2='0',T32_AHEAD='6',T32_SERIAL_BUDGET='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
_ENV_LOCK=threading.RLock()
@contextmanager
def frozen_environment():
    with _ENV_LOCK:
        old={k:os.environ.get(k)for k in FLAGS};os.environ.update(FLAGS)
        try:yield
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

def verify_sources(root):
    root=Path(root);lock=json.loads(Path(__file__).with_name('upstream.json').read_text())
    for name,expected in lock['files'].items():
        p=root/name
        if p.is_symlink()or not p.is_file()or file_hash(p)!=expected:raise ValueError('exact official speech Source required:'+name)
    return lock

def verify_model_roots(base,speech):
    """Actual admitted loader verifies every whole published identity, no tensors."""
    import hashlib
    expected=json.loads(Path(__file__).with_name('model_files.json').read_text())
    for key,root in (('base',Path(base)),('speech',Path(speech))):
        for name,item in expected[key].items():
            p=root/name
            if any(q.is_symlink()for q in (p,*p.parents))or not p.is_file()or p.stat().st_size!=item['bytes']:raise ValueError('literal full original model file required:'+name)
            digest=hashlib.sha256()
            with p.open('rb')as stream:
                while block:=stream.read(2**20):digest.update(block)
            if digest.hexdigest()!=item['sha256']:raise ValueError('original whole model identity differs:'+name)
    config=json.loads((Path(base)/'base/config.json').read_text());tc=config['text_config']
    if tc['hidden_size']!=5120 or tc['num_hidden_layers']!=64:raise ValueError('full original64-layer/5120-width trunk required')
    talk=json.loads((Path(speech)/'talker/config.json').read_text());e2=json.loads((Path(speech)/'e2/config.json').read_text())
    if talk!={'version':'talker-evf4','lora_r':256,'lora_alpha':512,'base':'base/'}or e2['version']!='e2-emo6'or (e2['din'],e2['d'],e2['La'],e2['Lf'],e2['soft_g'])!=(5120,1280,20,2,0.):raise ValueError('original talker/E2 configuration required; no upgraded E2')
    return expected

def source_modules(source_dir,base_dir,speech_dir):
    """Private package preserves original relative imports and split physical roots."""
    verify_sources(source_dir);prefix='_sophon_speech_original_'+uuid.uuid4().hex
    package=types.ModuleType(prefix);package.__path__=[str(Path(source_dir)/'sophon_expressive')];sys.modules[prefix]=package
    paths=types.ModuleType(prefix+'.paths')
    paths.model_dir=lambda:str(speech_dir)
    paths.base_dir=lambda:str(Path(base_dir)/'base')
    paths.talker_weights=lambda:str(Path(speech_dir)/'talker/model.safetensors')
    paths.e2_dir=lambda:str(Path(speech_dir)/'e2')
    paths.voice_file=lambda name='chelsie':str(Path(speech_dir)/'voices'/(name+'.safetensors'))
    sys.modules[paths.__name__]=paths
    try:
        with frozen_environment():
            L=importlib.import_module(prefix+'.talker_model');MK=importlib.import_module(prefix+'.ev_tags');E2=importlib.import_module(prefix+'.e2_model')
        if (L.D,L.NL,L.LORA_R,L.LORA_A,L.HID_LAYER,L.SPK_MODE)!=(5120,64,256,512.,-1,'ctx'):raise ValueError('unexpected original acoustic runtime constants')
        return prefix,L,MK,E2
    except BaseException:
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+'.'):sys.modules.pop(name,None)
        raise

def _move_original(module,device):
    original=tuple(module.parameters());module.to(device)
    if tuple(map(id,module.parameters()))!=tuple(map(id,original)):raise RuntimeError('placement replaced an original Parameter')

def route_original_trunk(thinker,talker,L,check):
    """Only differentiable device transfers around original layer functions."""
    def kv(self,ids,valid):
        check();B,N=ids.shape[:2];x=self.E[ids]if ids.dtype==torch.long else ids
        pos=torch.arange(N,device=x.device)[None].expand(B,-1)
        cm=torch.tril(torch.ones(N,N,dtype=torch.bool,device=x.device))[None,None]&valid[:,None,None,:]
        cm=cm|torch.eye(N,dtype=torch.bool,device=x.device)[None,None]
        Ks=[];Vs=[];hl=None
        for index,layer in enumerate(self.m.layers):
            check();d=next(layer.parameters()).device;x=x.to(d);c,sn=L.rope_cs(pos.to(d),d)
            x,k,v=L.layer_fwd(layer,x,c,sn,None,None,cm.to(d));Ks.append(k);Vs.append(v)
            if index==L.HID_LAYER:hl=x
        return Ks,Vs,(hl if hl is not None else self.m.norm(x.to(self.E.device)))
    def trunk(self,x,pos,Kc,Vc,mask,cache=None):
        check();new=[]
        for index,layer in enumerate(self.layers):
            check();d=next(layer.parameters()).device;x=x.to(d);c,sn=L.rope_cs(pos.to(d),d)
            kc=Kc[index]if cache is None else torch.cat([Kc[index],cache[index][0]],2)if cache[index]is not None else Kc[index]
            vc=Vc[index]if cache is None else torch.cat([Vc[index],cache[index][1]],2)if cache[index]is not None else Vc[index]
            x,k,v=L.layer_fwd_t(layer,x,c,sn,kc,vc,mask.to(d));new.append((k,v))
        return self.norm(x.to(next(self.norm.parameters()).device)),new
    thinker.kv=types.MethodType(kv,thinker);talker.trunk=types.MethodType(trunk,talker)

class NativeRuntime:
    def __init__(self,options):
        validate_options(options,training=False)
        self.owner=execution(options);runtime_identity();self.owner.check();self.closed=False;self.prefix=None;self.response=None
        self.source_identity=verify_sources(options['source_dir']);self.model_files=verify_model_roots(options['base_model_dir'],options['speech_model_dir'])
        self.pg_device=torch.device('cuda:1');self.decoder_device=torch.device('cuda:0')
        self.options=dict(options)
        try:
            self.prefix,self.L,self.MK,self.E2=source_modules(options['source_dir'],options['base_model_dir'],options['speech_model_dir']);L=self.L
            self.owner.check()
            # Official loader constructs the complete original Qwen3 from meta
            # and all original base tensors, BF16; no new initial model values.
            with frozen_environment():
                base=L.qwen3_model(torch.bfloat16);base.eval();base.requires_grad_(False);L.add_lora(base,L.LORA_R,L.LORA_A)
            # Place complete original layers before allocating/loading talker heads,
            # avoiding simultaneous full CPU trunk plus duplicate CPU heads/state.
            for index,layer in enumerate(base.layers):self.owner.check();_move_original(layer,layer_device(index))
            _move_original(base.embed_tokens,'cuda:1');_move_original(base.norm,'cuda:1')
            self.TH=types.SimpleNamespace(m=base,E=base.embed_tokens.weight,dev='cuda:1')
            # Retain the native LM head even though acoustic argmax uses dec.
            self.TH.LMH=L._sd('language_model.lm_head.')['weight'].to('cuda:1',torch.bfloat16)
            self.m=L.TalkerLT(self.TH).eval()
            from safetensors.torch import load_file
            sd=load_file(str(Path(options['speech_model_dir'])/'talker/model.safetensors'))
            missing=self.m.load_state_dict({k:v.float()if v.is_floating_point()else v for k,v in sd.items()},strict=False)
            if missing.unexpected_keys or any(not k.startswith('layers.')or '.lora_'in k for k in missing.missing_keys):raise ValueError('missing/unexpected original speech parameters:'+str(missing))
            del sd
            self.m.requires_grad_(False)
            self.TH.E=base.embed_tokens.weight
            for name,child in self.m.named_children():
                if name!='layers':_move_original(child,'cuda:1')
            # Direct Parameter leaves (bos/EV gains) have no child module.
            for name,p in self.m.named_parameters(recurse=False):
                before=id(p);p.data=p.data.to('cuda:1')
                if id(dict(self.m.named_parameters(recurse=False))[name])!=before:raise RuntimeError('original direct Parameter replaced')
            route_original_trunk(self.TH,self.m,L,self.owner.check)
            self.tok=L.TOKZ;self.IM=self.tok.convert_tokens_to_ids('<|im_end|>')
            self.voice_data=load_file(str(Path(options['speech_model_dir'])/'voices/chelsie.safetensors'))
            self.lock=threading.RLock();self.e2lock=threading.Lock();self.gs=None;self.pool=[]
            self.Kp_l={};lengths=[]
            with torch.no_grad():
                for lang in ('zh','en','ja'):
                    v=self.voice_data['ctx.'+lang].to(self.pg_device,self.TH.E.dtype)
                    ce=torch.cat([self.TH.E[torch.tensor(L.TURN_HEAD['assistant'],device=self.pg_device)],v,self.TH.E[torch.tensor(L.TURN_END,device=self.pg_device)],self.TH.E[torch.tensor(L.PROMPT_IDS,device=self.pg_device)]])[None]
                    K,V,_=self.TH.kv(ce,torch.ones(ce.shape[:2],dtype=torch.bool,device=self.pg_device));self.Kp_l[lang]=(K,V);lengths.append(ce.shape[1])
                if len(set(lengths))!=1:raise ValueError('native language prefix lengths differ; do not invent one PL')
                self.PL=lengths[0];self.Kp,self.Vp=self.Kp_l['zh']
                self.zb=self.m.ztok_up(torch.zeros(8,device=self.pg_device)).detach()
                self.x0=(self.m.bos+self.m.zutt_up(torch.zeros(1,32,device=self.pg_device))+self.m.spk_up(torch.zeros(1,192,device=self.pg_device)))[:,None].detach()
            self.owner.check()
            self.m2,self.dec,self.aff,self.hift,self.e2config=self.E2.load_all(str(self.decoder_device))
            for module in (self.m2,self.dec,self.aff,self.hift):module.requires_grad_(False)
            self.voices={'brand_'+lg:self.voice_data['spk.'+lg].to(self.decoder_device)for lg in ('zh','en','ja')}
            self.prompts={lg:(self.voice_data['prompt_mu.'+lg].to(self.decoder_device),self.voice_data['prompt_mel.'+lg].to(self.decoder_device))for lg in ('zh','en','ja')}
            self.e2stream=torch.cuda.Stream(device=self.decoder_device)
            self.session_module=self._session_module()
            self.model_identity=model_identity(options,self.source_identity,self.model_files)
            self.owner.check()
        except BaseException:self.close();raise
    def _session_module(self):
        path=Path(__file__).with_name('native_session.py');name=self.prefix+'.owned_native_session';mod=types.ModuleType(name)
        mod.__dict__.update(L=self.L,MK=self.MK,dev=str(self.pg_device),decoder_device=str(self.decoder_device),PL=self.PL)
        exec(compile(path.read_bytes(),str(path),'exec'),mod.__dict__);sys.modules[name]=mod
        original=mod.Session._render;owner=self.owner
        def render(session):owner.check();return original(session)
        mod.Session._render=render;return mod
    def ctx_extend(self,Kc,Vc,ids_new,L0):
        self.owner.check();L=self.L;N=len(ids_new);x=self.TH.E[torch.tensor(ids_new,device=self.pg_device)][None];pos=torch.arange(L0,L0+N,device=self.pg_device)[None]
        mask=torch.ones(1,1,N,L0+N,dtype=torch.bool,device=self.pg_device);mask[0,0,:,L0:]=torch.tril(torch.ones(N,N,dtype=torch.bool,device=self.pg_device));nK=[];nV=[]
        for index,layer in enumerate(self.TH.m.layers):
            self.owner.check();d=next(layer.parameters()).device;x=x.to(d);c,sn=L.rope_cs(pos.to(d),d)
            x,k,v=L.layer_fwd(layer,x,c,sn,Kc[index],Vc[index],mask.to(d));nK.append(torch.cat([Kc[index],k],2));nV.append(torch.cat([Vc[index],v],2))
        return nK,nV
    def new_session(self,lang='en'):
        if self.closed:raise RuntimeError('closed native owner')
        self.owner.check()
        if lang not in self.Kp_l:raise ValueError('original voice language en/zh/ja only')
        return self.session_module.Session(self,lang,'brand_'+lang,lambda block:None)
    def attach(self,**architecture):
        if self.response is not None:raise RuntimeError('only one native reg owner')
        from .acoustic_response import AcousticResponse
        self.response=AcousticResponse(self.m.reg,self.m.fm,model_identity=self.model_identity,device=self.pg_device,**{k:architecture[k]for k in ('rank','particles','particle_dim')if k in architecture});return self.response
    def close(self):
        if getattr(self,'closed',False):return
        if self.response is not None:self.response.close();self.response=None
        self.closed=True
        for name in ('m','TH','m2','dec','aff','hift','Kp','Vp','Kp_l','voice_data','voices','prompts','session_module'):
            if hasattr(self,name):setattr(self,name,None)
        if self.prefix:
            for name in list(sys.modules):
                if name==self.prefix or name.startswith(self.prefix+'.'):sys.modules.pop(name,None)

def model_identity(options,source_identity,model_files):
    names=('contract.py','placement.py','native.py','native_session.py','cache.py','backend.py','initialization.py','acoustic_response.py','finished_audio.py','runtime.py','upstream.json','current_core.json','model_files.json')
    return dict(model=model_files['model'],model_revision=model_files['revision'],original_model_files=model_files,official_source=source_identity,placement=options['placement'],profile='original-eager-E2-noFP8-no-fused-no-graph-no-LoRA-repack',whole_native_acoustic_width=5120,finished_audio_sample_rate=24000,runtime=runtime_identity(),product_sources={n:file_hash(Path(__file__).with_name(n))for n in names})

class HeadRuntime:
    """Training uses saved native inputs/targets plus original full reg/from_z.

    No trunk, prompt/KV forward, positive teacher, or decoder is called again.
    All FM/from_z original tensors are loaded, including full whitening state.
    """
    def __init__(self,options):
        validate_options(options,training=False)
        self.owner=execution(options);runtime_identity();self.closed=False;self.response=None;self.prefix=None;self.pg_device=torch.device('cuda:1')
        try:
            self.source_identity=verify_sources(options['source_dir']);self.model_files=verify_model_roots(options['base_model_dir'],options['speech_model_dir'])
            self.prefix,L,_,_=source_modules(options['source_dir'],options['base_model_dir'],options['speech_model_dir'])
            self.owner.check()
            self.reg=torch.nn.Sequential(torch.nn.LayerNorm(L.D),torch.nn.Linear(L.D,L.D));self.fm=L.FMHead()
            from safetensors import safe_open
            state={}
            with safe_open(str(Path(options['speech_model_dir'])/'talker/model.safetensors'),framework='pt',device='cpu')as source:
                for key in source.keys():
                    if key.startswith(('reg.','fm.')):state[key]=source.get_tensor(key)
            self.reg.load_state_dict({k[4:]:v for k,v in state.items()if k.startswith('reg.')},strict=True)
            self.fm.load_state_dict({k[3:]:v for k,v in state.items()if k.startswith('fm.')},strict=True);del state
            for m in (self.reg,self.fm):m.float();_move_original(m,self.pg_device);m.eval();m.requires_grad_(False)
            self.model_identity=model_identity(options,self.source_identity,self.model_files);self.owner.check()
        except BaseException:self.close();raise
    def attach(self,**architecture):
        if self.response is not None:raise RuntimeError('one original reg owner')
        from .acoustic_response import AcousticResponse
        self.response=AcousticResponse(self.reg,self.fm,model_identity=self.model_identity,device=self.pg_device,**{k:architecture[k]for k in ('rank','particles','particle_dim')if k in architecture});return self.response
    def close(self):
        if self.closed:return
        if self.response is not None:self.response.close();self.response=None
        self.closed=True;self.reg=None;self.fm=None
        if self.prefix:
            for n in list(sys.modules):
                if n==self.prefix or n.startswith(self.prefix+'.'):sys.modules.pop(n,None)
