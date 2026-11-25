import os
from fractions import Fraction
from zipfile import ZipFile

import pytest
import torch

from sophon_speech.acoustic_dose import (
    propose_constant_control,
    verify_constant_control,
)


@pytest.fixture
def dose_wheel():
    wheel = os.environ.get("SOPHON_DOSE_WHEEL")
    if not wheel:
        pytest.skip("set SOPHON_DOSE_WHEEL to the core wheel built at PR #338 or #339")
    return wheel


def completed(*strengths):
    return {
        "native_EOS": True,
        "native_zero_outcome": False,
        "native_frames": len(strengths),
        "effective_frame_strengths": list(strengths),
    }


def test_one_third_binary64_equality_does_not_match_native_fp32(dose_wheel):
    adaptive = completed(1, 0, 0)
    nominal_constant = 1 / 3
    assert nominal_constant * 3 == 1.0
    applied = (torch.ones((), dtype=torch.float32) * nominal_constant).item()

    proposal = propose_constant_control(adaptive, core_wheel=dose_wheel)
    assert proposal.constant_strength == applied
    assert proposal.adaptive_coefficients == (1.0, 0.0, 0.0)

    result = verify_constant_control(
        adaptive, completed(*([nominal_constant] * 3)), proposal=proposal, core_wheel=dose_wheel
    )
    assert result.constant_coefficients == (applied,) * 3
    assert result.proposal.adaptive_dose == 1
    assert result.constant_dose == proposal.constant_dose == 1 + Fraction(1, 2**25)
    assert result.residual == proposal.residual == Fraction(1, 2**25)
    assert not result.matched
    assert result.qualified_nearest
    assert not result.clock_changed


def test_verification_recomputes_the_target_on_the_actual_constant_eos(dose_wheel):
    adaptive = completed(1, 0)
    proposal = propose_constant_control(adaptive, core_wheel=dose_wheel)
    assert proposal.constant_strength == 0.5

    observed = verify_constant_control(
        adaptive, completed(*([0.5] * 4)), proposal=proposal, core_wheel=dose_wheel
    )
    assert observed.clock_changed
    assert not observed.qualified_nearest
    assert not observed.matched
    assert observed.constant_dose == 2
    assert observed.residual == 1
    assert observed.actual_clock_proposal.constant_frames == 4
    assert observed.actual_clock_proposal.constant_strength == 0.25

    corrected = verify_constant_control(
        adaptive, completed(*([0.25] * 4)), proposal=observed.actual_clock_proposal, core_wheel=dose_wheel
    )
    assert corrected.matched
    assert corrected.qualified_nearest
    assert corrected.residual == 0

    zero = completed(0)
    zero_proposal = propose_constant_control(zero, core_wheel=dose_wheel)
    different_but_equal_dose = verify_constant_control(
        zero, completed(0, 0), proposal=zero_proposal, core_wheel=dose_wheel
    )
    assert different_but_equal_dose.residual == 0
    assert different_but_equal_dose.clock_changed
    assert not different_but_equal_dose.qualified_nearest
    assert not different_but_equal_dose.matched


def test_shorter_clock_caps_the_control_at_the_permitted_strength(dose_wheel):
    adaptive = completed(1, 1)
    proposal = propose_constant_control(adaptive, constant_frames=1, core_wheel=dose_wheel)
    assert proposal.constant_strength == 1
    assert proposal.constant_dose == 1
    assert proposal.residual == -1
    assert proposal.saturated
    observed = verify_constant_control(adaptive, completed(1), proposal=proposal, core_wheel=dose_wheel)
    assert not observed.clock_changed
    assert not observed.qualified_nearest
    assert not observed.matched


def test_rounding_to_full_strength_does_not_saturate_a_feasible_clock(dose_wheel):
    adaptive = completed(1, 1 - 2**-24)
    proposal = propose_constant_control(adaptive, core_wheel=dose_wheel)
    result = verify_constant_control(adaptive, completed(1, 1), proposal=proposal, core_wheel=dose_wheel)
    assert proposal.constant_strength == 1
    assert not proposal.saturated
    assert result.qualified_nearest
    assert result.residual == Fraction(1, 2**24)
    assert not result.matched


@pytest.mark.parametrize("invalid", [
    {"native_EOS": False},
    {"native_EOS": None},
    {"native_zero_outcome": True},
    {"native_zero_outcome": None},
    {"native_frames": 0, "effective_frame_strengths": []},
    {"native_frames": 2},
    {"native_frames": True},
    {"effective_frame_strengths": [float("nan")]},
    {"effective_frame_strengths": [-0.25]},
    {"effective_frame_strengths": [1.25]},
])
def test_incomplete_or_invalid_eos_schedules_are_refused(invalid):
    bad = completed(1) | invalid
    with pytest.raises(ValueError, match="Sophon"):
        propose_constant_control(bad, core_wheel="unneeded.whl")
    with pytest.raises(ValueError, match="Sophon"):
        verify_constant_control(completed(1), bad, proposal=None, core_wheel="unneeded.whl")


def test_fixed_receipt_requires_a_constant_applied_fp32_coefficient(dose_wheel):
    adaptive = completed(1, 0)
    proposal = propose_constant_control(adaptive, core_wheel=dose_wheel)
    for receipt in (completed(0.25, 0.75), completed(0.5, 0.5) | {"activation_gate": {}}):
        with pytest.raises(ValueError, match="fixed native FP32 constant"):
            verify_constant_control(adaptive, receipt, proposal=proposal, core_wheel=dose_wheel)


def test_changed_shared_strength_scheduler_is_refused(dose_wheel, tmp_path):
    member = "particle_sliders/strength_schedule.py"
    with ZipFile(dose_wheel) as archive:
        core = archive.read(member)
    altered = tmp_path / "changed.whl"
    with ZipFile(altered, "w") as archive:
        archive.writestr(member, b" " + core[1:])
    with pytest.raises(ValueError, match="exact strength scheduler"):
        propose_constant_control(completed(1), core_wheel=altered)
