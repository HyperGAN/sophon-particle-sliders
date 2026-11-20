"""Root-owned native speech cache/train/render entry; no implicit CUDA admission."""
from pathlib import Path
import argparse,json

def train(config,output,*,admission,resume=None,stop_after=2):
    from .backend import open_training
    from .cache import record,file_record
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with open_training(config,resume=resume,admission=admission)as session:
        def commit():
            completed=session.game.completed_steps
            full=output/f'FULL-{completed:06d}.pt';raw=output/f'RAW-{completed:06d}.pt'
            session.checkpoint(full);session.export_inference(raw,selection='raw')
            record(output/f'COMMIT-{completed:06d}.json',dict(schema='Sophon-speech-native-exact-state-commit-v1',completed_steps=completed,FULL=file_record(full),RAW=file_record(raw),raw_selection='raw',ema_decay=None,cache=dict(path=config.options['cache_file'],bytes=config.options['cache_bytes'],sha256=config.options['cache_sha256']),initialization=session.game.backend.fresh_initialization))
        # Includes authentic initialized FULL0/RAW0 and all core RNG/optimizer0
        # before update1; resume saves its restored starting state without init.
        commit()
        if type(stop_after)is not int or not session.game.completed_steps<stop_after<=config.options['steps']:raise ValueError('explicit new stop within frozen horizon required')
        session.train(stop_after=stop_after,on_step=lambda metrics:commit())
        return dict(completed_steps=session.game.completed_steps,automatic_continuation=False,effect_or_quality_qualified=False)

def cache(options,cases,output,*,admission):
    from .native import NativeRuntime
    from .cache import build_cache
    from .placement import admitted
    with admitted(admission):
        runtime=NativeRuntime(options)
        try:return build_cache(runtime,cases,output)
        finally:runtime.close()

def render(artifact,options,output,*,text,language='en',seed=7,strength=1.,admission):
    from .backend import open_inference
    with open_inference(artifact,options=options,admission=admission,operation='render',output_type='audio',native_request=dict(text=text,language=language,durable_directory=str(output)))as session:
        if session.selection!='raw':raise ValueError('RAW inference only')
        return session.render(text=text,language=language,strength=strength,seed=seed,output_type='audio',durable_directory=str(output))

def main(argv=None,*,admission=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('cache','train','render'));p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--raw',type=Path);p.add_argument('--stop-after',type=int,default=2);p.add_argument('--text');p.add_argument('--language',default='en');p.add_argument('--seed',type=int,default=7);p.add_argument('--strength',type=float,default=1.);a=p.parse_args(argv)
    if admission is None:p.error('Root execution owner must call main(argv, admission=actual_owned_twoGPU_admission); a receipt string alone is not live authority')
    value=json.loads(a.config.read_text());options=value['options']if set(value)=={'backend','cohort','options'}else value
    if a.mode=='cache':
        if a.cases is None:p.error('literal native cases file required')
        result=cache(options,json.loads(a.cases.read_text()),a.output,admission=admission)
    elif a.mode=='train':
        from particle_sliders.lifecycle import SessionConfig
        result=train(SessionConfig.from_dict(value),a.output,admission=admission,resume=a.resume,stop_after=a.stop_after)
    else:
        if a.raw is None or a.text is None:p.error('actual RAW artifact and literal text required')
        local={k:options[k]for k in ('source_dir','base_model_dir','speech_model_dir','device','placement')}
        result=render(a.raw,local,a.output,text=a.text,language=a.language,seed=a.seed,strength=a.strength,admission=admission)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
