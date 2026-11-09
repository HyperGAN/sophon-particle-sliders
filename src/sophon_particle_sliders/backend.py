"""Product factories; the installed core owns every update, resume and export."""

import json
from pathlib import Path

import torch
from particle_sliders import pg09
from particle_sliders.inference import (
    InferenceBackend,
    InferenceBinding,
    native_identity,
)
from particle_sliders.lifecycle import Binding
from particle_sliders.pg09_audio import AudioResponseCritic
from particle_sliders.response import ResponseBackend, ResponseCapabilities

from .data import read_rows
from .native import load_native
from .response import conditions

PROTOCOL = "sophon-thinker-last-hidden-v1"
DEFAULTS = dict(
    rank=8,
    particles=128,
    particle_dim=4,
    base_precision="bfloat16",
    response_width=5120,
)


def architecture(options):
    return {key: options.get(key, value) for key, value in DEFAULTS.items()}


def capabilities(options):
    return ResponseCapabilities(
        "routed_particle_gan",
        "speech",
        "sophon_thinker_final_token_hidden",
        "continuous_vector",
        (1, options.get("response_width", 5120)),
        pg09.PG_VERSION,
        pg09.PG_SOURCE_PROFILE,
        base_precisions=("float32", "bfloat16"),
        qualification="CPU synthetic/native-small execution only; pretrained model, realtime and quality unqualified",
    )


def validate_native(options):
    for key in ("source_dir", "model_dir"):
        if not Path(options[key]).is_dir():
            raise ValueError("local " + key + " required")
    config = json.loads((Path(options["model_dir"]) / "base/config.json").read_text())
    if config.get("text_config", {}).get("hidden_size") != options.get(
        "response_width", 5120
    ):
        raise ValueError(
            "response_width must equal the full native LLM width, not projector width"
        )
    for key in ("rank", "particles", "particle_dim", "response_width"):
        if (
            type(architecture(options)[key]) is not int
            or architecture(options)[key] < 1
        ):
            raise ValueError("positive integer " + key + " required")
    if options.get("base_precision", "bfloat16") not in ("float32", "bfloat16"):
        raise ValueError("unquantized float32/bfloat16 thinker required")
    device = options.get("device", "cpu")
    if not isinstance(device, str) or device != "cpu":
        raise ValueError(
            "GPU execution requires a product scheduler admission integration; this release admits CPU only"
        )


def validate_training(options):
    validate_native(options)
    if not Path(options["training_file"]).is_file():
        raise ValueError("local matched JSONL training_file required")


def prepare_training(options):
    torch.manual_seed(options.get("seed", 7))
    native = load_native(options)
    try:
        rows = [
            native.prepare(row)
            for row in read_rows(
                options["training_file"],
                native.processor,
                max_tokens=options.get("max_tokens", 4096),
            )
        ]
        if native.width != options.get("response_width", 5120):
            raise ValueError("native response width differs")
        cond = conditions(rows)
        critic = AudioResponseCritic(native.width, cond.shape[1]).to(
            next(native.network.parameters()).device
        )
        game = pg09.ConditionalPG09(
            native,
            rows,
            critic,
            conditions=cond,
            particles=native.network.particles,
            response_domain="continuous_vector",
            total_steps=options.get("steps", 4),
            batch_size=options.get("batch", 2),
            seed=options.get("seed", 7),
            recipe_overrides=options.get("recipe_overrides"),
        )
        spec = INFERENCE.specification(
            native,
            model_identity=native.model_identity,
            protocol=PROTOCOL,
            configuration=architecture(options),
        )
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if set(options) - {"model_dir", "source_dir", "device"}:
        raise ValueError(
            "native runtime accepts only local model/source paths and device"
        )
    validate_native(dict(options, **spec.configuration))


def prepare_inference(options, spec):
    native = load_native(dict(options, **spec.configuration))
    try:
        return InferenceBinding(
            native.network,
            native_identity(
                native, model_identity=native.model_identity, protocol=PROTOCOL
            ),
            native.close,
            response=native.response,
        )
    except BaseException:
        native.close()
        raise


TRAINING = ResponseBackend(
    "sophon",
    capabilities(DEFAULTS),
    prepare_training,
    validate_training,
    configure_capabilities=capabilities,
)
INFERENCE = InferenceBackend(
    "sophon",
    capabilities(DEFAULTS),
    prepare_inference,
    validate_inference,
    configure_capabilities=capabilities,
)
