"""Exact new speech cohort. No historical text-packet migration or implicit EMA."""
from pathlib import Path
import hashlib,json,math
BACKEND='sophon_speech95'
SOURCE_PROFILE='sophon-original-speech-reg5120-pg095-v1'
PROTOCOL='sophon-original-native-KF-reg5120-E2-HiFT24k-v1'
INITIALIZATION_POLICY='sophon-speech-fresh-Kaiming-QR-zeroUP-bankR2std01-v1'
WEIGHT_POLICY='raw-no-weight-EMA'
DEFAULTS=dict(rank=8,particles=128,particle_dim=4,response_width=5120,base_precision='bfloat16')
CORE_REVISION='41f6066f2d2644902670d5eb736de052e1a70553'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def architecture(options):return {k:options.get(k,v)for k,v in DEFAULTS.items()}
def rate_overrides(options):
    values={k:options[k]for k in ('lr','d_lr_mult','prior_lr_mult')if k in options}
    if any(type(v)not in (int,float)or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('positive literal common rates required')
    return values

def validate_profile(options):
    if any(options.get(k)!=v for k,v in dict(source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous').items()):raise ValueError('explicit original-speech/current095 source profile required')
    if options.get('ema_decay')is not None or options.get('ema')is not None:raise ValueError('all weight EMA refused')
    if architecture(options)!=DEFAULTS:raise ValueError('full native5120, rank8/128x4, original BF16 base/FP32 heads only')
    rate_overrides(options)

def validate_options(options,*,training):
    keys=set(DEFAULTS)|{'source_profile','initialization_policy','pg_version','pg_source_profile','penalty_input_domain','ema_decay','source_dir','base_model_dir','speech_model_dir','device','placement','steps','batch','seed','lr','d_lr_mult','prior_lr_mult','cache_file','cache_bytes','cache_sha256'}
    if set(options)-keys:raise ValueError('unknown speech option')
    validate_profile(options)
    for key in ('source_dir','base_model_dir','speech_model_dir'):
        if not isinstance(options.get(key),str)or not Path(options[key]).is_dir():raise ValueError('local '+key+' required')
    for key,default,minimum in (('steps',128,1),('batch',2,2),('seed',7,0)):
        value=options.get(key,default)
        if type(value)is not int or value<minimum:raise ValueError('invalid '+key)
    if training and(not isinstance(options.get('cache_file'),str)or not Path(options['cache_file']).is_file()):raise ValueError('saved original native-KF matched cache required; no teacher calls in prepare/restore')
    if training and(type(options.get('cache_bytes'))is not int or options['cache_bytes']<=0 or not isinstance(options.get('cache_sha256'),str)or len(options['cache_sha256'])!=64 or any(x not in '0123456789abcdef'for x in options['cache_sha256'])):raise ValueError('exact actual native cache FileRecord required')
    from .placement import execution
    execution(options)

def validate_installed_core():
    from importlib import metadata
    lock=json.loads(Path(__file__).with_name('current_core.json').read_text());dist=metadata.distribution('particle-sliders-core')
    if dist.version!=lock['version']or lock['commit']!=CORE_REVISION:raise ValueError('exact admitted core source required')
    for name,expected in lock['checked_api_files'].items():
        if file_hash(dist.locate_file('particle_sliders/'+name))!=expected:raise ValueError('current core Source differs:'+name)
