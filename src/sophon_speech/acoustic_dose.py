"""Match the FP32 coefficient dose of completed native Sophon speech schedules."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from struct import Struct
from sys import float_info
from types import ModuleType
from zipfile import BadZipFile, ZipFile

_CORE_MEMBER = "particle_sliders/strength_schedule.py"
_CORE_SOURCE_SHA256 = "fed34a2d3f70fd793f8b13cb605e80442699912edb3cab3258c1e21f92e50282"
_FP32 = Struct(">f")


@dataclass(frozen=True)
class AcousticDoseProposal:
    """Nearest permitted constant on the assumed native KF frame clock."""

    adaptive_coefficients: tuple[float, ...]
    adaptive_dose: Fraction
    constant_frames: int
    constant_strength: float
    constant_dose: Fraction

    @property
    def residual(self) -> Fraction:
        """Signed constant dose minus adaptive dose."""
        return self.constant_dose - self.adaptive_dose

    @property
    def saturated(self) -> bool:
        """Whether even strength 1 cannot deliver the adaptive dose."""
        return self.adaptive_dose > self.constant_frames


@dataclass(frozen=True)
class AcousticDoseComparison:
    """Actual coefficients, signed residual and target on the observed clock."""

    proposal: AcousticDoseProposal
    actual_clock_proposal: AcousticDoseProposal
    constant_coefficients: tuple[float, ...]
    constant_dose: Fraction

    @property
    def residual(self) -> Fraction:
        return self.constant_dose - self.proposal.adaptive_dose

    @property
    def clock_changed(self) -> bool:
        return len(self.constant_coefficients) != self.proposal.constant_frames

    @property
    def qualified_nearest(self) -> bool:
        """Whether the actual tested constant is nearest on a feasible, unchanged clock."""
        return (
            not self.clock_changed
            and not self.proposal.saturated
            and self.constant_coefficients[0] == self.proposal.constant_strength
        )

    @property
    def matched(self) -> bool:
        """Whether the qualified nearest control has exactly the same dose."""
        return self.qualified_nearest and self.residual == 0


def _shared_schedule(wheel):
    # Use a standalone verified wheel member, as GatePlan does for steering.
    try:
        with ZipFile(wheel) as archive:
            entry = archive.getinfo(_CORE_MEMBER)
            if entry.file_size != 7125:
                raise ValueError("Sophon requires the exact strength scheduler from core PR #338")
            source = archive.read(entry)
    except (BadZipFile, KeyError) as error:
        raise ValueError("Sophon requires a core wheel containing the strength scheduler") from error
    if sha256(source).hexdigest() != _CORE_SOURCE_SHA256:
        raise ValueError("Sophon requires the exact strength scheduler from core PR #338")
    module = ModuleType(f"{__name__}._core_schedule")
    exec(compile(source, f"{wheel}/{_CORE_MEMBER}", "exec"), module.__dict__)  # noqa: S102
    return module


def _native_coefficients(receipt):
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("native_EOS") is not True
        or receipt.get("native_zero_outcome") is not False
    ):
        raise ValueError("Sophon dose matching requires a clean native EOS receipt")
    strengths = receipt.get("effective_frame_strengths")
    frames = receipt.get("native_frames")
    if (
        type(frames) is not int
        or frames < 1
        or not isinstance(strengths, (tuple, list))
        or len(strengths) != frames
    ):
        raise ValueError("Sophon dose matching requires one strength per nonempty native KF frame clock")
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in strengths):
        raise ValueError("Sophon effective strengths must be finite numbers in [0, 1]")
    return tuple(_FP32.unpack(_FP32.pack(value))[0] for value in strengths)


def _proposal(core, coefficients, constant_frames):
    if type(constant_frames) is not int or not 1 <= constant_frames <= 2 ** float_info.mant_dig:
        raise ValueError("Sophon constant frame count must be positive and on the exact binary64 integer grid")
    match = core.match_binary32_interval_dose(
        coefficients, range(len(coefficients) + 1), constant_coordinates=(0, constant_frames)
    )
    strength = min(1.0, match.constant_strength)
    dose = core.interval_dose((strength,), (0, constant_frames), coefficient_format="binary32")
    return AcousticDoseProposal(coefficients, match.adaptive_dose, constant_frames, strength, dose)


def propose_constant_control(adaptive_receipt, *, core_wheel, constant_frames=None) -> AcousticDoseProposal:
    """Propose a native FP32 constant from a completed scheduled speech receipt.

    By default, assume the constant run will have the adaptive run's KF frame
    count. An independently observed constant frame count may replace that
    assumption. A clock too short to deliver the adaptive dose yields strength
    1 and the exact remaining dose difference.
    """
    coefficients = _native_coefficients(adaptive_receipt)
    frames = len(coefficients) if constant_frames is None else constant_frames
    return _proposal(_shared_schedule(core_wheel), coefficients, frames)


def verify_constant_control(
    adaptive_receipt, constant_receipt, *, proposal, core_wheel
) -> AcousticDoseComparison:
    """Compare a completed fixed render on its own native EOS frame clock.

    When the fixed clock changes, use `actual_clock_proposal` to re-evaluate
    the control at the observed length before accepting a match. The residual
    compares the actual FP32 coefficients from the two receipts. The evaluator
    supplies independently paired request, seed, RAW adapter and observer evidence.
    """
    adaptive = _native_coefficients(adaptive_receipt)
    constant = _native_coefficients(constant_receipt)
    if "activation_gate" in constant_receipt or any(value != constant[0] for value in constant[1:]):
        raise ValueError("Sophon dose matching requires a fixed native FP32 constant control")
    core = _shared_schedule(core_wheel)
    if not isinstance(proposal, AcousticDoseProposal) or proposal != _proposal(
        core, adaptive, proposal.constant_frames
    ):
        raise ValueError("Sophon constant proposal does not match the adaptive receipt")
    actual_target = _proposal(core, adaptive, len(constant)) if len(constant) != proposal.constant_frames else proposal
    actual_dose = core.interval_dose(constant, range(len(constant) + 1), coefficient_format="binary32")
    return AcousticDoseComparison(proposal, actual_target, constant, actual_dose)
