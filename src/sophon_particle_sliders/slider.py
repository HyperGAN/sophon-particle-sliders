"""Select only thinker LLM projections; reuse the shared routed branch."""

import math
import re
from contextlib import contextmanager

import torch
from particle_sliders.adapter_activation import push_activation, restore_activation
from particle_sliders.adapter_attachment import atomic_attachment
from particle_sliders.routed_projection import TinyParticleProjection
from torch import nn

TARGET = re.compile(
    r"language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
)


class SophonSlider(nn.Module):
    """A separate routed residual, leaving the frozen tag LoRA at strength one.

    PEFT tag projections are hooked at their base Linear. The tag delta remains
    active even at slider OFF. No model weights are merged or duplicated.
    """

    @atomic_attachment("model")
    def __init__(self, model, *, rank=8, particles=128, particle_dim=4):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (rank, particles, particle_dim)):
            raise ValueError("positive integer routed dimensions required")
        if any(hasattr(m, "_sophon_slider_owner") for m in model.modules()):
            raise ValueError("Sophon host already has an attached slider")
        plan = []
        for name, module in model.named_modules():
            if TARGET.fullmatch(name):
                host = (
                    module.get_base_layer()
                    if hasattr(module, "get_base_layer")
                    else module
                )
                if not isinstance(host, nn.Linear):
                    raise ValueError(
                        "only unquantized thinker Linear projections are supported"
                    )
                plan.append((name, host))
        if not plan or len({id(m) for _, m in plan}) != len(plan):
            raise ValueError("distinct Sophon thinker LLM projections required")
        devices = {m.weight.device for _, m in plan}
        if len(devices) != 1:
            raise ValueError(
                "single-device thinker required; no implicit device-map migration"
            )
        self._original = [(p, p.requires_grad) for p in model.parameters()]
        for p, _ in self._original:
            p.requires_grad_(False)
        device = next(iter(devices))
        self.particles = nn.Parameter(
            torch.randn(particles, particle_dim, device=device) * 0.1
        )
        self.projections = nn.ModuleList()
        self._hosts = []
        self.target_names = tuple(name for name, _ in plan)
        for name, host in plan:
            projection = (
                TinyParticleProjection(name, host, self.particles, rank, float(rank))
                .float()
                .to(device)
            )
            object.__setattr__(host, "_sophon_slider_owner", self)
            self._hosts.append((host, projection, host.forward))
            self.projections.append(projection)
        self.closed = False

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if self.closed or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("open slider and strength in [0,1] required")
        push_activation(self, self.projections, ("multiplier",))
        try:
            for projection in self.projections:
                projection.multiplier = strength
            yield
        finally:
            restore_activation(self)

    def close(self):
        if self.closed:
            return
        if getattr(self, "_activation_stack", None):
            raise RuntimeError("cannot detach an active slider")
        if any(
            getattr(host, "_sophon_slider_owner", None) is not self
            or host.forward != attached
            for host, _, attached in self._hosts
        ):
            raise RuntimeError("native projection ownership changed; refusing detach")
        for host, projection, _ in reversed(self._hosts):
            host.forward = projection.org_forward
            del host._sophon_slider_owner
        for p, flag in self._original:
            p.requires_grad_(flag)
        self.closed = True
