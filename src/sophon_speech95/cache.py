"""Actual native-KF counterfactual EV alignment; all full frame coordinates retained.

A positive teacher is a one-step fork of the neutral native event, not an
independent trajectory forced to have matching history. Both must naturally
select KF under the original decision masks. No teacher call occurs in train.
"""
from pathlib import Path
import copy,hashlib,json,os,random,types
import numpy as np
import torch
from particle_sliders.pg09_audio import PreparedAudioResponse
from .acoustic_response import AcousticRow
from .native import frozen_environment

class UnmatchedNativeEvent(ValueError):pass

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else [])
def save(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError('retained first native record exists:'+str(path))
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb')as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.link(tmp,path);tmp.unlink();fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def record(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();path=Path(path)
    with path.with_name(path.name+'.partial').open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(path.with_name(path.name+'.partial'),path);path.with_name(path.name+'.partial').unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def file_record(path):
    path=Path(path);digest=hashlib.sha256();count=0
    with path.open('rb')as stream:
        while chunk:=stream.read(2**20):digest.update(chunk);count+=len(chunk)
    return dict(path=str(path.resolve()),bytes=count,sha256=digest.hexdigest())

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu').clone()
    if isinstance(value,tuple):return tuple(cpu(x)for x in value)
    if isinstance(value,list):return [cpu(x)for x in value]
    if isinstance(value,dict):return {k:cpu(v)for k,v in value.items()}
    return copy.deepcopy(value)

class OnceJournal:
    """Exclusive call claim + complete inputs/RNG then returned output/RNG.

    Unknown or completed claims are never automatically called again. All
    returned tensors are durable for a separately explicit technical recovery.
    """
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False);self.calls=[]
    def call(self,label,inputs,invoke):
        index=len(self.calls);stem=f'{index:04d}-{label}';before=self.root/(stem+'-input.pt')
        save(before,dict(inputs=cpu(inputs),rng=rng_state()))
        claim=self.root/(stem+'-claim.json');record(claim,dict(schema='Sophon-native-teacher-call-claim-v1',input=file_record(before),label=label,completed=False,replay_authorized=False))
        self.calls.append(dict(claim=file_record(claim),output=None))
        value=invoke();after=self.root/(stem+'-output.pt');save(after,dict(value=cpu(value),rng=rng_state()))
        self.calls[-1]['output']=file_record(after);record(self.root/(stem+'-return.json'),self.calls[-1]);return value

def native_action(runtime,session,y):
    """Literal original _steps masked argmax, using its pre-increment state."""
    L=runtime.L;n=len(session.ids);fin=session.final and n and session.ids[-1]==runtime.IM
    # Observer runs inside the trunk's autocast context, whereas original
    # Session executes decision/reg/EV heads outside that context.
    with torch.autocast(device_type=y.device.type,enabled=False):lg=runtime.m.dec(y)[0].float()
    mf=int(n*4+20)
    if session.r>=n:lg[L.KA]=-1e9
    if session.r<n or session.nf==0 or not fin:lg[L.KS]=-1e9
    cap=session.nf>=mf or(L.STALL>0 and session.run_>=L.STALL and session.r<n)
    if cap:lg[L.KF]=-1e9
    if cap and session.r>=n and fin:lg[L.KS]=1e9
    return int(lg.argmax())

def collect_pair(runtime,*,neutral_text,positive_text,lang,journal):
    if runtime.response is not None:raise ValueError('teacher capture precedes added adapter; no live slider teacher')
    se=runtime.new_session(lang);se.add_text(neutral_text);se.finish()
    positive=runtime.new_session(lang);positive.add_text(positive_text);positive.finish()
    with frozen_environment(),torch.no_grad():
        neutral_clean=se._ev_clean();positive_clean=positive._ev_clean()
        if neutral_clean!=positive_clean or not neutral_clean:raise ValueError('native EV syntax must retain identical nonempty cleaned text; no token/history forcing')
        se._commit();positive._ev_build(positive_clean)
        positive_ids=runtime.tok(positive_clean,add_special_tokens=False).input_ids+[runtime.IM]
        if positive_ids!=se.ids:raise UnmatchedNativeEvent('native cleaned token IDs differ')
        if torch.equal(se._evv,positive._evv)and ((se._evl is None and positive._evl is None)or(se._evl is not None and positive._evl is not None and torch.equal(se._evl,positive._evl))):raise ValueError('distinct actual native EV conditioning required')
        old_ev=se._ev_add;original=runtime.m.trunk;pending={};actions=[]
        def ev(self):
            self._speech_pre_EV=self.x.detach().clone();return old_ev()
        se._ev_add=types.MethodType(ev,se)
        def observed(_m,*args,**kwargs):
            state=dict(r=se.r,p=se.p,nf=se.nf,run_=se.run_,ids=list(se.ids),final=se.final,prior_actions=list(actions))
            inputs=dict(args=args,kwargs=kwargs,native_state=state,model_identity=runtime.model_identity)
            value=journal.call('neutral-native-trunk',inputs,lambda:original(*args,**kwargs));y=value[0][:,-1].float();action=native_action(runtime,se,y)
            if action==runtime.L.KF and not pending:
                if se.nf!=0 or not hasattr(se,'_speech_pre_EV'):raise UnmatchedNativeEvent('first native KF context/EV origin missing')
                # Shallow shell retains exactly the same original prior KV/cache,
                # native state/position and voice; only current EV is changed.
                fork=object.__new__(type(se));fork.__dict__=dict(se.__dict__);fork.x=se._speech_pre_EV.detach().clone();fork._evv=positive._evv;fork._evl=positive._evl
                with torch.autocast(device_type=fork.x.device.type,enabled=False):type(se)._ev_add(fork)
                pos_args=(fork.x.float(),*args[1:])
                pos_inputs=dict(args=pos_args,kwargs=kwargs,native_state=state,model_identity=runtime.model_identity,changed_only='current native EV injection',positive_EV=cpu(dict(values=positive._evv,lead=positive._evl)))
                pv=journal.call('positive-EV-same-prior-trunk',pos_inputs,lambda:original(*pos_args,**kwargs));py=pv[0][:,-1].float()
                with torch.autocast(device_type=y.device.type,enabled=False):
                    neutral_frame=runtime.m.fm.from_z(runtime.m.reg(y).float()).detach().clone();positive_frame=runtime.m.fm.from_z(runtime.m.reg(py).float()).detach().clone()
                frames=dict(neutral_y=y,positive_y=py,neutral=neutral_frame,target=positive_frame,positive_action=native_action(runtime,se,py))
                save(journal.root/'first-KF-returned-pair.pt',cpu(frames))
                if frames['positive_action']!=runtime.L.KF:raise UnmatchedNativeEvent('positive counterfactual independently selected a non-KF action; no forced alignment')
                pending.update(frames=frames,state=state)
            actions.append(action);return value
        runtime.m.trunk=types.MethodType(observed,runtime.m)
        try:
            while not pending:
                runtime.owner.check()
                if se.done or se.cancelled:raise UnmatchedNativeEvent('native EOS/cancel before a matched KF; no padded/forced frame')
                if not se._steps(1):raise UnmatchedNativeEvent('native could not advance to a matched KF')
            f=pending['frames']
            if len(se.frames)!=1 or not torch.equal(se.frames[0],f['neutral'][0]):raise UnmatchedNativeEvent('original Session did not commit the observed neutral KF')
            context=dict(schema='Sophon-native-KF-same-prior-counterfactual-EV-context-v1',native_state=pending['state'],clean_text=neutral_clean,clean_ids=se.ids,neutral_text=neutral_text,positive_text=positive_text,language=lang,voice='chelsie',same_prior_KV_and_action_state=True,previous_acoustic_frames=0,positive_action_independently_KF=True,positive_is_counterfactual_one_step_not_independent_history=True,model_identity=runtime.model_identity,actual_teacher_calls=journal.calls)
            row=AcousticRow(f['neutral_y'].detach().clone(),f['positive_y'].detach().clone(),context)
            return PreparedAudioResponse(row,f['target'],f['neutral'])
        finally:runtime.m.trunk=original;se.cancel();positive.cancel()

def build_cache(runtime,cases,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    for index,case in enumerate(cases):
        if set(case)!={'neutral','positive','language'}:raise ValueError('literal matched native speech case fields required')
        journal=OnceJournal(output/f'case-{index:03d}')
        row=collect_pair(runtime,neutral_text=case['neutral'],positive_text=case['positive'],lang=case['language'],journal=journal)
        rows.append(cpu(dict(neutral_y=row.row.neutral_y,positive_y=row.row.positive_y,context=row.row.context,target=row.target,neutral=row.neutral)))
        save(output/f'row-{index:03d}.pt',rows[-1])
    path=output/'native-KF-cache.pt';save(path,dict(schema='Sophon-original-native-KF-cache-v1',model_identity=runtime.model_identity,rows=rows))
    record(output/'native-KF-cache.json',dict(schema='Sophon-original-native-KF-cache-commit-v1',cache=file_record(path),rows=len(rows),old_text_cache_reused=False,teacher_replay=False));return file_record(path)

def load_cache(path,identity,device,expected):
    if file_record(path)!=expected:raise ValueError('exact completed native cache bytes/SHA required')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['schema']!='Sophon-original-native-KF-cache-v1'or value['model_identity']!=identity or len(value['rows'])<2:raise ValueError('exact new native-KF cache/identity with at least2 matched rows required')
    result=[]
    for item in value['rows']:
        c=item['context']
        if c.get('model_identity')!=identity or c.get('same_prior_KV_and_action_state')is not True or c.get('positive_action_independently_KF')is not True or c.get('previous_acoustic_frames')!=0:raise ValueError('genuine matched native first-KF context required')
        arrays={k:item[k].to(device=device,dtype=torch.float32)for k in ('neutral_y','positive_y','target','neutral')}
        if any(v.shape!=(1,5120)or not torch.isfinite(v).all()for v in arrays.values()):raise ValueError('full finite native5120 cached frame required')
        result.append(PreparedAudioResponse(AcousticRow(arrays['neutral_y'],arrays['positive_y'],c),arrays['target'],arrays['neutral']))
    return result
