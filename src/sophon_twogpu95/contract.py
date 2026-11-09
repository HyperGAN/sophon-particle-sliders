"""Explicit new-cohort contract; historical sophon packets retain their loader."""
import hashlib
import json
import math
from pathlib import Path

CORE_REVISION = '41f6066f2d2644902670d5eb736de052e1a70553'
SOURCE_PROFILE = 'sophon-explicit-two-gpu-current095-fresh-v1'
INITIALIZATION_POLICY = 'sophon-twogpu-kaiming-RMS-QR-zero-up-bank-R2-std01-v1'
PROTOCOL = 'sophon095-explicit-two-gpu-native-audio-text-v1'
BACKEND = 'sophon095_twogpu'
WEIGHT_POLICY = 'raw-no-weight-EMA'
DEFAULTS = dict(rank=8, particles=128, particle_dim=4, base_precision='bfloat16', response_width=5120)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture(options):
    return {key: options.get(key, value) for key, value in DEFAULTS.items()}


def rate_overrides(options):
    values = {key: options[key] for key in ('lr', 'd_lr_mult', 'prior_lr_mult') if key in options}
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive non-Boolean common rate required: ' + key)
    return values


def validate_profile(options):
    if (options.get('source_profile') != SOURCE_PROFILE
            or options.get('initialization_policy') != INITIALIZATION_POLICY
            or options.get('pg_version') != '0.9.5'
            or options.get('pg_source_profile') != 'release-0.9.5'
            or options.get('penalty_input_domain') != 'continuous'):
        raise ValueError('explicit new Sophon PG0.9.5 fresh-init/continuous source profile required')
    if options.get('ema_decay') is not None or options.get('ema') is not None:
        raise ValueError('Sophon095 refuses all model-weight EMA')
    rate_overrides(options)


def validate_training(options):
    allowed = set(DEFAULTS) | {'source_dir', 'model_dir', 'training_file', 'device', 'max_tokens',
        'steps', 'batch', 'seed', 'ema_decay', 'pg_version', 'pg_source_profile',
        'penalty_input_domain', 'source_profile', 'initialization_policy',
        'lr', 'd_lr_mult', 'prior_lr_mult', 'placement'}
    if set(options) - allowed:
        raise ValueError('unknown Sophon095 options; use literal lr/d_lr_mult/prior_lr_mult, no nested recipe')
    validate_profile(options)
    arch = architecture(options)
    for key in ('rank', 'particles', 'particle_dim', 'response_width'):
        if type(arch[key]) is not int or arch[key] < 1:
            raise ValueError('positive integer ' + key + ' required')
    if arch['particle_dim'] != 4:
        raise ValueError('the actual shared routed host uses a four-coordinate bank')
    if arch['base_precision'] not in ('float32', 'bfloat16'):
        raise ValueError('unquantized FP32/BF16 thinker required')
    for key, default, minimum in (('steps', 4, 1), ('batch', 2, 2), ('seed', 7, 0), ('max_tokens', 4096, 1)):
        value = options.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError('invalid ' + key)
    if options.get('batch', 2) > 512 or options.get('seed', 7) >= 2**63-100:
        raise ValueError('shared lifecycle budget/seed limit exceeded')
    from .placement import validate_local
    validate_local(options,arch,training=True)


def validate_installed_core():
    """Exact admitted public source, including same-version core updates."""
    from importlib import metadata
    lock = json.loads(Path(__file__).with_name('current_core.json').read_text())
    installed = metadata.distribution('particle-sliders-core')
    if installed.version != lock['version'] or lock['commit'] != CORE_REVISION:
        raise ValueError('Sophon095 requires its exact current-core source cohort')
    for name, expected in lock['checked_api_files'].items():
        source = installed.locate_file('particle_sliders/' + name)
        if file_hash(source) != expected:
            raise ValueError('Sophon095 current-core Source differs: ' + name)
