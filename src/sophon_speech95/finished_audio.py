"""Original complete Session.say_all -> original E2/HiFT -> native24k PCM16."""
from pathlib import Path
import math,os,wave
import numpy as np
import torch
from .native import frozen_environment
from .cache import save,file_record,record,rng_state

def validate_request(*,text,language='en',strength=1.,seed=7,output_type='audio',images=None):
    if images is not None or output_type!='audio'or not isinstance(text,str)or not text.strip()or language not in ('en','zh','ja')or type(seed)is not int or seed<0 or type(strength)not in (int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('native nonempty speech text/en-zh-ja/seed/strength[0,1]/audio required')

def generate(runtime,*,text,language='en',strength=1.,seed=7,output_type='audio',durable_directory=None,images=None):
    validate_request(text=text,language=language,strength=strength,seed=seed,output_type=output_type,images=images)
    if runtime.response is None:raise ValueError('same RAW-loaded original acoustic reg branch must be attached')
    runtime.owner.check();session=runtime.new_session(language)
    # The complete actual generated frames and every native render input are
    # durable BEFORE their original decoder call; no partial hidden stream is
    # labelled a finished utterance. Unknown decode calls cannot auto-retry.
    directory=Path(durable_directory)if durable_directory is not None else None
    if directory is None:raise ValueError('new durable output directory required before native generation')
    directory.mkdir(parents=True,exist_ok=False);original=runtime.session_module.Session._render;counter=[0]
    def render_state(se):
        return dict(frames=[f.detach().cpu().clone()for f in se.frames],done=se.done,language=language,model_identity=runtime.model_identity,native_state=dict(win_start=se.win_start,mel_out=se.mel_out,speech_off=se.speech_off,tail=None if se.tail is None else se.tail.detach().cpu().clone(),mel_cum=None if se.mel_cum is None else se.mel_cum.detach().cpu().clone(),pending=se.pending.copy(),started=se.started,stats=dict(se.stats)),rng=rng_state())
    def durable_render(se):
        index=counter[0];counter[0]+=1;before=directory/f'render-{index:05d}-input.pt';save(before,render_state(se));emit=se.emit;chunks=[]
        def durable_emit(block):
            path=directory/f'render-{index:05d}-chunk-{len(chunks):04d}.pcm16'
            with path.open('xb')as stream:stream.write(block);stream.flush();os.fsync(stream.fileno())
            chunks.append(file_record(path));emit(block)
        se.emit=durable_emit
        try:
            result=original(se);after=directory/f'render-{index:05d}-return.pt';save(after,render_state(se))
            record(directory/f'render-{index:05d}-commit.json',dict(input=file_record(before),returned=file_record(after),returned_render=bool(result),actual_PCM16_chunks=chunks,unknown_decode_replay_authorized=False));return result
        finally:se.emit=emit
    # Per-instance override only; another runtime/session Source is not patched.
    import types
    session._render=types.MethodType(durable_render,session)
    with torch.random.fork_rng(devices=[0,1]),frozen_environment(),torch.no_grad(),runtime.response.scaled(strength):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        save(directory/'native-generation-input.pt',dict(text=text,language=language,strength=strength,seed=seed,rng=rng_state(),model_identity=runtime.model_identity))
        samples=session.say_all(text)
        if session.cancelled or not session.done or not session.finished:raise RuntimeError('native EOS utterance did not complete; retain all partials')
        save(directory/'complete-native-frames.pt',dict(frames=[x.detach().cpu().clone()for x in session.frames],rng=rng_state(),native_stats=session.stats,done=session.done))
    if samples.ndim!=1 or not np.isfinite(samples).all():raise ValueError('actual complete finite native mono waveform required')
    # say_all already uses original clip*32767->int16 and returns /32767 FP32.
    pcm=np.rint(samples*32767).astype(np.int16);path=directory/'native24k.wav'
    with path.open('xb')as stream:
        with wave.open(stream,'wb')as audio:audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(24000);audio.writeframes(pcm.tobytes())
        stream.flush();os.fsync(stream.fileno())
    receipt=dict(schema='Sophon-original-native-EOS24k-audio-v1',output=file_record(path),native_sample_rate=24000,samples=len(pcm),native_frames=len(session.frames),native_EOS=True,native_zero_outcome=len(pcm)==0,strength=strength,seed=seed,language=language,model_identity=runtime.model_identity,decoder='original e2-emo6/HiFT',audio_effect_or_catastrophe_qualified=False)
    record(directory/'native-audio.json',receipt);return receipt
