"""Offline-only, source-verified thinker loading. Never downloads weights or serves."""

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch

from .response import SophonResponse
from .slider import SophonSlider

LOCK = json.loads(Path(__file__).with_name("upstream.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def official_modules(source):
    source = Path(source).resolve()
    for relative, expected in LOCK["files"].items():
        if digest(source / relative) != expected:
            raise ValueError(
                "official Sophon source differs from reviewed pin: " + relative
            )
    name = "_sophon_pinned_" + LOCK["commit"]
    if name not in sys.modules:
        package = source / "finetune/sophon_swift"
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return tuple(
        importlib.import_module(name + "." + part)
        for part in ("modeling_duplexjev", "processing", "adapters")
    )


def model_files(root):
    root = Path(root)
    files = sorted(
        p
        for sub in ("base", "adapters/tag")
        for p in (root / sub).rglob("*")
        if p.is_file()
    )
    if not any(p.suffix == ".safetensors" and "base" in p.parts for p in files):
        raise ValueError("local base safetensors required; no download fallback")
    if not (root / "adapters/tag/adapter_model.safetensors").is_file():
        raise ValueError("bundled tag LoRA is required")
    return {str(p.relative_to(root)): digest(p) for p in files}


def load_native(options):
    if options.get("device", "cpu") != "cpu":
        raise ValueError(
            "GPU execution requires scheduler admission; native loader admits CPU only"
        )
    # Hash before allocating any model/device. Checkpoint identity binds all local
    # base/tokenizer/tag bytes, rather than trusting paths, mtimes or filenames.
    root = Path(options["model_dir"]).resolve()
    files = model_files(root)
    if not version("peft").startswith("0.20."):
        raise ImportError(
            "Sophon native loader requires peft>=0.20,<0.21 in an isolated environment"
        )
    modeling, processing, adapters = official_modules(options["source_dir"])
    from peft import PeftModel

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        options.get("base_precision", "bfloat16")
    ]
    model = modeling.DuplexJevForConditionalGeneration.from_pretrained(
        root / "base",
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(options.get("device", "cpu"))
    # Use the official key/target conversion. Do not merge the reply adapter.
    with tempfile.TemporaryDirectory(prefix="sophon-tag-") as temporary:
        adapters.to_swift(str(root / "adapters/tag"), temporary)
        wrapped = PeftModel.from_pretrained(
            model,
            temporary,
            adapter_name="tag",
            is_trainable=False,
            local_files_only=True,
        )
    model = wrapped.get_base_model()
    model.gradient_checkpointing_disable()
    model.requires_grad_(False)
    processor = processing.SophonRealtimeProcessor.from_pretrained(
        root / "base", model.config
    )
    identity = dict(
        model="adventists-ai/Sophon-Realtime-v1",
        files=files,
        source_commit=LOCK["commit"],
        source_files=LOCK["files"],
        peft=version("peft"),
        transformers=version("transformers"),
        tag="frozen, unmerged",
    )
    slider = SophonSlider(
        model,
        **{
            k: options[k] for k in ("rank", "particles", "particle_dim") if k in options
        },
    )
    try:
        return SophonResponse(
            model, slider, model_identity=identity, processor=processor
        )
    except BaseException:
        slider.close()
        raise
