"""The original full acoustic frame, with one separately owned routed reg branch.

Official Sophon engine._steps/_steps_g uses exactly
``fm.from_z(reg(y).float())`` at an actually selected KF event. The native
loader/cache owns event conditioning and alignment; this module never invents
an event, calls FM.sample, reduces coordinates, or initializes restored state.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math

import torch
from torch import nn
from particle_sliders.adapter_activation import push_activation, restore_activation
from particle_sliders.adapter_attachment import atomic_attachment
from particle_sliders.pg09_audio import PreparedAudioResponse
from particle_sliders.pg09_backends import AdapterParameters
from particle_sliders.routed_projection import TinyParticleProjection


WIDTH = 5120
_OWNER = "_sophon_speech95_reg_owner"


def _frame(value, *, device=None):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.ndim != 2 or value.shape[1] != WIDTH or value.shape[0] < 1
            or not torch.isfinite(value).all()
            or device is not None and value.device != device):
        raise ValueError("complete finite native FP32 Bx5120 acoustic-event input required")
    return value


@dataclass(frozen=True)
class AcousticRow:
    """Matched actual KF hidden inputs; native cache owns context/EV provenance."""

    neutral_y: torch.Tensor
    positive_y: torch.Tensor
    context: dict
    scale: float = 1.0


class _RegAdapter(nn.Module):
    """Own only the new PG Parameters; never register/replace pretrained reg."""

    @atomic_attachment("reg")
    def __init__(self, reg, *, device, rank, particles, particle_dim):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (rank, particles, particle_dim)):
            raise ValueError("positive integer routed dimensions required")
        if (type(reg) is not nn.Sequential or len(reg) != 2
                or type(reg[0]) is not nn.LayerNorm
                or reg[0].normalized_shape != (WIDTH,)
                or type(reg[1]) is not nn.Linear
                or (reg[1].in_features, reg[1].out_features) != (WIDTH, WIDTH)
                or list(reg[1].children()) or hasattr(reg[1], "parametrizations")
                or hasattr(reg[1], _OWNER)):
            raise ValueError("unattached original full LayerNorm5120/Linear5120 reg required")
        device = torch.device(device)
        if device.index is None and device.type == "cuda":
            raise ValueError("explicit PG CUDA device index required")
        if any(p.device != device or p.dtype != torch.float32 or p.requires_grad
               for p in reg.parameters()):
            raise ValueError("original reg must already be frozen FP32 on the declared PG device")
        object.__setattr__(self, "_reg", reg)
        object.__setattr__(self, "_original", tuple(reg.named_parameters()))
        self.particles = nn.Parameter(torch.randn(particles, particle_dim, device=device) * 0.1)
        projection = TinyParticleProjection("reg.1", reg[1], self.particles, rank, float(rank))
        self.projections = nn.ModuleList([projection.float().to(device)])
        object.__setattr__(reg[1], _OWNER, self)
        self.target_names = ("reg.1",)
        self.closed = False

    def check_host(self):
        reg = self._reg
        current = tuple(reg.named_parameters())
        if (self.closed or getattr(reg[1], _OWNER, None) is not self
                or reg[1].forward != self.projections[0].forward
                or len(current) != len(self._original)
                or any(name != old_name or parameter is not old_parameter
                       for (name, parameter), (old_name, old_parameter) in zip(current, self._original))):
            raise RuntimeError("original acoustic reg ownership/Parameter identity changed")
        if any(p.requires_grad or p.dtype != torch.float32 or p.device != self.particles.device
               for _, p in self._original):
            raise RuntimeError("pretrained reg dtype/device/frozen policy changed")

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("acoustic strength in [0,1] required")
        self.check_host()
        push_activation(self, self.projections, ("multiplier",))
        try:
            self.projections[0].multiplier = strength
            yield self
        finally:
            restore_activation(self)

    def close(self):
        if self.closed:
            return
        if getattr(self, "_activation_stack", None):
            raise RuntimeError("cannot detach an active acoustic reg branch")
        self.check_host()
        host = self._reg[1]
        host.forward = self.projections[0].org_forward
        delattr(host, _OWNER)
        self.closed = True


class AcousticResponse(nn.Module):
    """Full5120 response and trainable-only RAW state, sharing native reg hook.

    Construction allocates the new bank/routed leaves with the shared
    constructor laws (including protected zero-UP). The lifecycle alone runs
    the scoped fresh policy before optimizers; RAW/resume merely constructs
    shapes and strictly restores ``network``. No fresh helper is called here.
    """

    family = "sophon_original_native_reg_acoustic_frame"
    width = WIDTH

    def __init__(self, reg, fm, *, model_identity, device, rank=8, particles=128, particle_dim=4):
        super().__init__()
        if not isinstance(fm, nn.Module) or not callable(getattr(fm, "from_z", None)):
            raise ValueError("original native FM module/from_z required")
        identity = json.dumps(model_identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if not isinstance(model_identity, dict) or not model_identity:
            raise ValueError("nonempty source-bound native model identity required")
        device = torch.device(device)
        if any(p.requires_grad or p.device != device for p in fm.parameters()):
            raise ValueError("original FM must already be frozen on the declared PG device")
        for name in ("mu", "sd", "U", "ws", "white"):
            value = getattr(fm, name, None)
            shape = (WIDTH, WIDTH) if name == "U" else () if name == "white" else (WIDTH,)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32 or value.device != device):
                raise ValueError("original complete FP32 from_z whitening buffers required: " + name)
        object.__setattr__(self, "reg", reg)
        object.__setattr__(self, "fm", fm)
        self._model_identity_json = identity
        self.slider = _RegAdapter(reg, device=device, rank=rank, particles=particles, particle_dim=particle_dim)
        self.network = AdapterParameters(self.slider.named_parameters(), self.slider.scaled)

    @property
    def model_identity(self):
        return json.loads(self._model_identity_json)

    @property
    def particles(self):
        return self.network.particles

    def scaled(self, strength):
        return self.network.scaled(strength)

    def from_y(self, y, *, strength=1.0):
        y = _frame(y, device=self.particles.device)
        with self.network.scaled(strength):
            value = self.fm.from_z(self.reg(y).float())
        if value.shape != y.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("original reg/from_z must return the complete finite FP32 acoustic frame")
        return value

    def response(self, rows=None, conditions=None, *, y=None, strength=1.0):
        """Use already native-event-conditioned hidden input, never FM.sample.

        ``conditions`` alone can be the full native hidden tensor. With rows,
        an optional condition tensor must equal those same cached neutral inputs.
        No critic/proxy condition is silently substituted for native coordinates.
        """
        if y is not None:
            if rows is not None or conditions is not None:
                raise ValueError("supply one explicit native hidden-input source")
        elif rows is None:
            y = conditions
        else:
            values = list(rows)
            if not values or any(not isinstance(row, AcousticRow) for row in values):
                raise ValueError("explicit cached AcousticRow sequence required")
            y = torch.cat([_frame(row.neutral_y) for row in values])
            if conditions is not None and not torch.equal(_frame(conditions), y):
                raise ValueError("conditions differ from the actual cached native KF inputs")
        return self.from_y(y, strength=strength)

    def prepare(self, row):
        """Optional once-only teacher preparation; never called by cached restore."""
        if not isinstance(row, AcousticRow) or row.scale != 1.0 or not row.context:
            raise ValueError("source-bound matched KF row with unipolar scale1 required")
        neutral, positive = (_frame(x, device=self.particles.device) for x in (row.neutral_y, row.positive_y))
        if neutral.shape != (1, WIDTH) or positive.shape != neutral.shape:
            raise ValueError("one aligned complete native KF event per teacher row required")
        context = json.loads(json.dumps(row.context, allow_nan=False))
        copied = AcousticRow(neutral.detach().clone(), positive.detach().clone(), context)
        with torch.no_grad():
            return PreparedAudioResponse(copied, self.from_y(copied.positive_y, strength=0).detach().clone(),
                                         self.from_y(copied.neutral_y, strength=0).detach().clone())

    def generated(self, prepared):
        return self.from_y(prepared.row.neutral_y, strength=prepared.row.scale)

    @torch.no_grad()
    def sample(self, prepared, *, scale=1.0):
        return self.from_y(prepared.row.neutral_y, strength=scale).detach().clone()

    def metadata(self):
        return dict(protocol="sophon-original-full5120-native-reg-from-z-v1", model_identity=self.model_identity,
                    response_domain="continuous_vector", response_width=WIDTH, targets=list(self.slider.target_names),
                    native_expression="fm.from_z(reg(y).float())", FM_sample_used=False,
                    event_conditioning="Already applied at the matched actual native KF site by loader/cache",
                    weight_policy="raw/no_weight_EMA", audio_render_qualified=False, effect_qualified=False)

    def close(self):
        self.slider.close()


def conditions(prepared_rows):
    rows = list(prepared_rows)
    if not rows:
        raise ValueError("nonempty prepared acoustic rows required")
    return torch.cat([_frame(prepared.row.neutral_y) for prepared in rows])
