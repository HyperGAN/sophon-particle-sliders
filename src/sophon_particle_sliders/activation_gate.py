"""Explicit, calibrated plans backed by the shared steering equations."""

from dataclasses import asdict
from hashlib import sha256
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
_MODULE = "particle_sliders/adaptive_steering.py"
_MODULE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
TAGGED_TEXT_SIGNAL = "tagged-text-layer0-input"
SPEECH_SIGNAL = "speech-kf-reg-input"


def _shared_steering(wheel):
    # The strict RAW owners keep their installed core. This verified standalone
    # module has no intra-package imports or need for a global module alias.
    with ZipFile(wheel) as archive:
        entry = archive.getinfo(_MODULE)
        if entry.file_size != 10329:
            raise ValueError("Sophon requires the exact shared steering module from core PR #269")
        source = archive.read(entry)
    if sha256(source).hexdigest() != _MODULE_SHA256:
        raise ValueError("Sophon requires the exact shared steering module from core PR #269")
    module = ModuleType(f"{__name__}._core_269")
    exec(compile(source, f"{wheel}/{_MODULE}", "exec"), module.__dict__)
    return module


class GatePlan:
    """One product-calibrated plan. Each render constructs fresh gate state."""

    def __init__(self, *, core_wheel, signal, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if signal not in (TAGGED_TEXT_SIGNAL, SPEECH_SIGNAL):
            raise ValueError("Select the native Sophon signal used for calibration")
        if type(parameter) not in (int, float):
            raise ValueError("Provide the measured threshold, bias or control rate")
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(value not in "0123456789abcdef" for value in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the native calibration record")
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random control does not use a direction")
            self._direction = None
        else:
            if direction is None:
                raise ValueError("Sophon CFP and CLABS require a fitted native direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 1 or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("Provide a finite unit direction in the native signal space")
            self._direction = probe.cpu()
        self._shared = _shared_steering(core_wheel)
        self._signal, self._method = signal, method
        self._parameter, self._seed = float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._clabs = self._shared.CLABSParameters(**clabs) if clabs is not None else None
        self._new_gate()

    def _new_gate(self):
        return self._shared.ActivationGate(
            self._method, self._parameter, seed=self._seed, clabs=self._clabs,
        )

    def _start(self, signal, width, strength):
        if signal != self._signal:
            raise ValueError("This gate plan was calibrated for a different native Sophon signal")
        if self._direction is not None and self._direction.shape != (width,):
            raise ValueError("Direction width differs from the calibrated native Sophon signal")
        return _NativeGate(self, strength)


class _NativeGate:
    def __init__(self, plan, strength):
        self.plan = plan
        self.strength = strength
        self.trace = []
        self.gate = plan._new_gate()

    def strength_at(self, residual, *, position):
        if not self.strength:
            return 0.0
        plan = self.plan
        alignment = None
        if plan._direction is not None:
            alignment = float(plan._shared.activation_projection(
                residual, plan._direction, strength=self.strength,
            ))
        decision = self.gate.step(alignment)
        effective = self.strength if decision.fired else 0.0
        self.trace.append(dict(position=position, alignment=alignment, fired=decision.fired,
                               probability=decision.probability, effective_strength=effective))
        return effective

    def receipt(self):
        plan, usage = self.plan, self.gate.usage
        return dict(
            core_revision=CORE_REVISION, signal=plan._signal,
            calibration_sha256=plan._calibration_sha256, method=plan._method,
            parameter=plan._parameter, gate_seed=plan._seed,
            direction=plan._direction.tolist() if plan._direction is not None else None,
            clabs=asdict(plan._clabs) if plan._clabs is not None else None,
            nominal_strength=self.strength, eligible=usage.eligible, fired=usage.fired,
            firing_rate=usage.rate if usage.eligible else None, decisions=self.trace,
        )
