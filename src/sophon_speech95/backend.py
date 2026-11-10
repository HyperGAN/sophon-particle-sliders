"""Strict new speech FULL/RAW lifecycle. Cached teachers never execute here."""
from copy import deepcopy
import torch
from particle_sliders.lifecycle import Binding,Capabilities,read_packet
from particle_sliders.response import ResponseBackend,ResponseCapabilities
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from particle_sliders.pg095 import ConditionalPG095,engine_options
from particle_sliders.pg09_audio import AudioResponseCritic
from .contract import BACKEND,SOURCE_PROFILE,PROTOCOL,INITIALIZATION_POLICY,WEIGHT_POLICY,DEFAULTS,architecture,digest,rate_overrides,validate_options,validate_installed_core
from .initialization import Construction,construction_for,initialize_owned,validate_receipt
from .native import NativeRuntime,HeadRuntime
from .acoustic_response import conditions

class CurrentResponse:
    family='sophon_original_full_native_acoustic_reg5120'
    width=5120
    def __init__(self,native):
        self.runtime=native;self.response_adapter=native.attach(**DEFAULTS);self.slider=self.response_adapter.slider;self.network=self.response_adapter.network;self.model_identity=deepcopy(native.model_identity);self.fresh_initialization=None;self.cache_identity=None
    def generated(self,row):self.runtime.owner.check();return self.response_adapter.generated(row)
    def response(self,*,seed=0,**request):
        if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('valid observation seed required')
        self.runtime.owner.check();return self.response_adapter.response(**request)
    def sample(self,row,*,scale=1.):self.runtime.owner.check();return self.response_adapter.sample(row,scale=scale)
    def metadata(self):return dict(self.response_adapter.metadata(),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,fresh_initialization=self.fresh_initialization,weight_policy=WEIGHT_POLICY,penalty_input_domain='continuous',native_cache_identity=self.cache_identity)
    def close(self):self.runtime.close()

def capabilities(options):return ResponseCapabilities('routed_particle_gan','speech','sophon_original_native_KF_acoustic_frame','continuous_vector',(1,5120),'0.9.5','release-0.9.5',penalty_input_domain='continuous',base_precisions=('bfloat16',),qualification='New speech software unexecuted; full original acoustic response, no speech quality/effect qualification')
AUDIO=Capabilities('routed_particle_gan',render=True,render_outputs=('audio',),operations=('response','render'),qualification='Original full E2/HiFT native24k EOS audio Source, not executed qualification')
def validate_training(options):validate_options(options,training=True);validate_installed_core()
def preflight(options,request,staged):
    construction=staged.get('construction')
    if set(staged)!={'construction'}or not isinstance(construction,Construction)or construction.options_sha256!=digest(options)or construction.mode not in ('fresh','resume')or request['native']:raise ValueError('explicit product fresh/restore before allocation/optimizers; cached native training only')
def prepare_training(options,*,construction=None):
    validate_training(options)
    if not isinstance(construction,Construction)or construction.options_sha256!=digest(options):raise ValueError('explicit fresh/restore intent required')
    native=HeadRuntime(options)
    try:
        current=CurrentResponse(native)
        from .cache import load_cache
        current.cache_identity=dict(path=options['cache_file'],bytes=options['cache_bytes'],sha256=options['cache_sha256'])
        rows=load_cache(options['cache_file'],current.model_identity,native.pg_device,current.cache_identity);cond=conditions(rows)
        with torch.random.fork_rng(devices=[]if native.pg_device.type=='cpu'else[0,1]):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),[]if native.pg_device.type=='cpu'else[0,1]);critic=AudioResponseCritic(5120,5120).float().to(native.pg_device)
            receipt=initialize_owned(current.slider,critic,construction,options);current.fresh_initialization=receipt.as_dict()
        if any(p.device!=native.pg_device or p.dtype!=torch.float32 for root in (current.slider,critic)for p in root.parameters()if p.requires_grad):raise ValueError('all PG trainables must remain original FP32 objects on PGdevice')
        game=ConditionalPG095(current,rows,critic,conditions=cond,particles=current.network.particles,response_domain='continuous_vector',total_steps=options.get('steps',128),batch_size=options.get('batch',2),seed=options.get('seed',7),recipe_overrides=rate_overrides(options),**engine_options(options))
        spec=INFERENCE.specification(current,model_identity=current.model_identity,protocol=PROTOCOL,configuration=dict(architecture=architecture(options),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,weight_policy=WEIGHT_POLICY,initialization=receipt.as_dict(),placement=options['placement']))
        return Binding(game,current.close,inference=spec,before_update=lambda state:native.owner.check())
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.close],primary=primary);raise

def validate_inference(options,spec):
    validate_installed_core();c=spec.configuration
    if set(c)!={'architecture','source_profile','initialization_policy','weight_policy','initialization','placement'}or c['source_profile']!=SOURCE_PROFILE or c['initialization_policy']!=INITIALIZATION_POLICY or c['weight_policy']!=WEIGHT_POLICY or c['architecture']!=DEFAULTS or spec.identity['protocol']!=PROTOCOL or options.get('placement')!=c['placement']:raise ValueError('exact speech RAW source/factory/architecture required; no text-packet migration')
    validate_receipt(c['initialization'],cohort=c['initialization']['cohort_id'])
    if set(options)-{'source_dir','base_model_dir','speech_model_dir','device','placement'}:raise ValueError('inference accepts only original local paths and exact owned placement')
    from .placement import execution
    execution(options)
def preflight_inference(options,request,spec):
    if request['operation']=='render':
        from .finished_audio import validate_request
        native={k:v for k,v in request['native'].items()if k!='durable_directory'};validate_request(strength=request['strength'],output_type=request['output_type'],**native)
def prepare_inference(options,spec):
    validate_inference(options,spec)
    # Local callers can select paths/owned placement only. The runtime needs
    # the same full native profile validated for training; derive it from the
    # immutable RAW specification and exact source constants, never local
    # overrides and never a fresh-initialization invocation.
    configuration=spec.configuration
    runtime_options=dict(options,**configuration['architecture'],
        source_profile=configuration['source_profile'],
        initialization_policy=configuration['initialization_policy'],
        pg_version='0.9.5',pg_source_profile='release-0.9.5',
        penalty_input_domain='continuous',ema_decay=None)
    native=NativeRuntime(runtime_options)
    try:
        current=CurrentResponse(native);current.fresh_initialization=spec.configuration['initialization']
        def render(**request):
            from .finished_audio import generate
            return generate(native,**request)
        return InferenceBinding(current.network,native_identity(current,model_identity=current.model_identity,protocol=PROTOCOL),current.close,response=current.response,render=render)
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.close],primary=primary);raise
TRAINING=ResponseBackend(BACKEND,capabilities(DEFAULTS),prepare_training,validate_training,configure_capabilities=capabilities,preflight=preflight)
INFERENCE=InferenceBackend(BACKEND,AUDIO,prepare_inference,validate_inference,preflight=preflight_inference)

def open_training(config,*,resume=None,admission=None,**request):
    from .placement import admitted
    with admitted(admission):
        validate_training(config.options);packet=deepcopy(resume)if isinstance(resume,dict)else read_packet(resume)if resume is not None else None
        construction=construction_for(config,packet);session=TRAINING.open(config,operation='resume'if packet is not None else'train',construction=construction,**request)
        try:
            if packet is not None:session.restore(packet)
            return session
        except BaseException as primary:
            from particle_sliders.restoration import _restore_all
            _restore_all([session.close],primary=primary);raise

def open_inference(artifact,*,options,admission=None,**request):
    from particle_sliders.inference import open_inference as shared_open
    from .placement import admitted
    with admitted(admission):return shared_open(artifact,backend=INFERENCE,options=options,**request)
