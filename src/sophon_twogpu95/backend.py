"""Distinct prospective PG095/fresh-init cohort; old sophon Source is untouched."""
from copy import deepcopy
from pathlib import Path

import torch
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, Capabilities, read_packet
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.pg09_audio import AudioResponseCritic
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from .contract import (BACKEND, DEFAULTS, INITIALIZATION_POLICY, PROTOCOL,
    SOURCE_PROFILE, WEIGHT_POLICY, architecture, digest, rate_overrides, validate_profile,
    validate_training as validate_training_options, validate_installed_core)
from .initialization import Construction, construction_for, initialize_owned, validate_receipt
from .native import load_native
from sophon_particle_sliders.response import conditions
from sophon_particle_sliders.data import read_rows


class CurrentResponse:
    """Own the original native host; add only new-cohort provenance/text callback."""
    family = 'sophon095_explicit_two_gpu_tagged_thinker'

    def __init__(self, native):
        self.runtime = native
        self.model, self.slider, self.network = native.model, native.slider, native.network
        self.model_identity = deepcopy(native.model_identity)
        from .contract import file_hash
        self.model_identity['current_product_sources'] = {
            name: file_hash(Path(__file__).with_name(name))
            for name in ('backend.py','contract.py','initialization.py','finished_text.py','current_core.json',
                         'native.py','placement.py','slider.py','response.py')}
        import sophon_particle_sliders as reused
        reused_root=Path(reused.__file__).parent
        self.model_identity['reused_product_sources']={name:file_hash(reused_root/name)
            for name in ('native.py','response.py','data.py','upstream.json')}
        self.width = native.width
        self.fresh_initialization = None

    def prepare(self, row):return self.runtime.prepare(row)
    def generated(self, row):return self.runtime.generated(row)
    def sample(self, row, *, scale=1.):return self.runtime.sample(row, scale=scale)
    def response(self, **request):return self.runtime.response(**request)
    def close(self):return self.runtime.close()

    def metadata(self):
        return dict(self.runtime.metadata(), source_profile=SOURCE_PROFILE,
            initialization_policy=INITIALIZATION_POLICY, fresh_initialization=self.fresh_initialization,
            weight_policy=WEIGHT_POLICY, penalty_input_domain='continuous')


def capabilities(options):
    return ResponseCapabilities('routed_particle_gan', 'speech', 'sophon_thinker_final_token_hidden',
        'continuous_vector', (1, architecture(options)['response_width']), '0.9.5', 'release-0.9.5',
        penalty_input_domain='continuous', base_precisions=('float32', 'bfloat16'),
        qualification='New PG095/fresh-init Source; native CPU/pretrained/finished-text quality pending')


TEXT_CAPABILITIES = Capabilities('routed_particle_gan', render=True,
    render_outputs=('text',), operations=('response', 'render'),
    qualification='Official tagged audio-conditioned text generation Source; native CPU/pretrained qualification pending')


def validate_training(options):
    validate_training_options(options)
    validate_installed_core()


def preflight(options, request, staged):
    construction = staged.get('construction')
    if (set(staged) != {'construction'} or not isinstance(construction, Construction)
            or construction.options_sha256 != digest(options) or construction.mode not in ('fresh', 'resume')):
        raise ValueError('use product open_training with explicit intent before model/optimizers')
    if request['native']:
        raise ValueError('training accepts matched JSONL only')


def prepare_training(options, *, construction=None):
    validate_training(options)
    if (not isinstance(construction, Construction) or construction.options_sha256 != digest(options)
            or construction.mode not in ('fresh', 'resume')):
        raise ValueError('fresh/resume intent required before native allocation and optimizers')
    native = load_native(options)  # Explicit live admission or the separate CPU fixture path.
    try:
        current = CurrentResponse(native)
        rows = [current.prepare(row) for row in read_rows(options['training_file'], native.processor,
            max_tokens=options.get('max_tokens', 4096))]
        if current.width != architecture(options)['response_width']:
            raise ValueError('native full LLM response width differs')
        cond = conditions(rows)
        with torch.random.fork_rng(devices=[]if native.pg_device.type=='cpu'else [0,1]):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed', 7), []if native.pg_device.type=='cpu'else [0,1])
            critic = AudioResponseCritic(current.width, cond.shape[1]).float().to(next(current.network.parameters()).device)
            receipt = initialize_owned(native.slider, critic, construction, options)
            current.fresh_initialization = receipt.as_dict()
        owned=[p for root in (current.slider,critic)for p in root.parameters()if p.requires_grad]
        if not owned or any(p.device!=native.pg_device or p.dtype!=torch.float32 for p in owned):
            raise ValueError('every PG trainable must be FP32 on the one declared PG device before optimizers')
        game = ConditionalPG095(current, rows, critic, conditions=cond,
            particles=current.network.particles, response_domain='continuous_vector',
            total_steps=options.get('steps', 4), batch_size=options.get('batch', 2), seed=options.get('seed', 7),
            recipe_overrides=rate_overrides(options), **engine_options(options))
        spec = INFERENCE.specification(current, model_identity=current.model_identity, protocol=PROTOCOL,
            configuration=dict(architecture=architecture(options), source_profile=SOURCE_PROFILE,
                initialization_policy=INITIALIZATION_POLICY, weight_policy=WEIGHT_POLICY,
                initialization=receipt.as_dict(),placement=options.get('placement')))
        return Binding(game, current.close, inference=spec)
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.close], primary=primary)
        raise


def validate_inference(options, spec):
    validate_installed_core()
    if set(options) - {'model_dir', 'source_dir', 'device','placement'}:
        raise ValueError('inference accepts local model/source paths and device only')
    c = spec.configuration
    if (set(c) != {'architecture', 'source_profile', 'initialization_policy', 'weight_policy', 'initialization','placement'}
            or c['source_profile'] != SOURCE_PROFILE or c['initialization_policy'] != INITIALIZATION_POLICY
            or c['weight_policy'] != WEIGHT_POLICY or spec.identity['protocol'] != PROTOCOL):
        raise ValueError('requires original new Sophon095 RAW artifact; historical packets are not migrated')
    validate_receipt(c['initialization'], cohort=c['initialization'].get('cohort_id'))
    # Reuse original validation without a teacher-file requirement or init call.
    from .placement import validate_local
    if options.get('placement')!=c['placement']:raise ValueError('RAW explicit placement identity differs')
    validate_local(options,c['architecture'],training=False)


def preflight_inference(options, request, spec):
    if request['operation'] == 'render':
        from .finished_text import validate_request
        validate_request(strength=request['strength'], output_type=request['output_type'], **request['native'])


def prepare_inference(options, spec):
    validate_inference(options, spec)
    native = load_native(dict(options, **spec.configuration['architecture']))
    try:
        current = CurrentResponse(native)
        # Shape construction only. Never run a fresh initializer on restored RAW.
        current.fresh_initialization = spec.configuration['initialization']
        def render(**request):
            from .finished_text import generate
            return generate(native, **request)
        return InferenceBinding(current.network,
            native_identity(current, model_identity=current.model_identity, protocol=PROTOCOL),
            current.close, response=current.response, render=render)
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.close], primary=primary)
        raise


TRAINING = ResponseBackend(BACKEND, capabilities(DEFAULTS), prepare_training, validate_training,
    configure_capabilities=capabilities, preflight=preflight)
INFERENCE = InferenceBackend(BACKEND, TEXT_CAPABILITIES, prepare_inference, validate_inference,
    preflight=preflight_inference)


def _open_training(config, *, resume=None, **request):
    """Explicit construction intent BEFORE prepare; exact same-cohort restore only."""
    validate_training(config.options)
    packet = deepcopy(resume) if isinstance(resume, dict) else read_packet(resume) if resume is not None else None
    construction = construction_for(config, packet)
    session = TRAINING.open(config, operation='resume' if packet is not None else 'train', construction=construction, **request)
    try:
        if packet is not None:session.restore(packet)
        return session
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([session.close], primary=primary)
        raise


def open_training(config,*,resume=None,admission=None,**request):
    """CPU fixture or actual caller-owned two-GPU entry; no implicit admission."""
    from .placement import admitted
    with admitted(admission):return _open_training(config,resume=resume,**request)

def open_inference(artifact,*,options,admission=None,**request):
    from particle_sliders.inference import open_inference as shared_open
    from .placement import admitted
    with admitted(admission):return shared_open(artifact,backend=INFERENCE,options=options,**request)
