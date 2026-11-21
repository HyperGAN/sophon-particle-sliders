"""Frame-position control for the original Sophon acoustic adapter."""

from contextlib import ExitStack
from pathlib import Path
from sys import float_info

from .backend import CurrentResponse
from .acoustic_response import WIDTH
from .cache import record
from sophon_particle_sliders.activation_gate import GatePlan, SPEECH_SIGNAL


def render(inference, *, frame_gains, strength=1.0, **request):
    """Use the last gain for subsequent acoustic frames through native EOS."""
    return _render(inference, frame_gains, None, strength, request)


def render_adaptive(inference, *, gate_plan, strength=1.0, **request):
    """Gate the saved RAW reg edit once per selected native KF frame."""
    if not isinstance(gate_plan, GatePlan):
        raise ValueError("Provide an explicit calibrated Sophon GatePlan")
    return _render(inference, None, gate_plan, strength, request)


def _render(inference, frame_gains, gate_plan, strength, request):
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Sophon nominal strength must be in [0, 1]")
    if gate_plan is None:
        if not isinstance(frame_gains, (list, tuple)) or not 1 <= len(frame_gains) <= 4096:
            raise ValueError("Sophon requires an explicit acoustic frame envelope")
        if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in frame_gains):
            raise ValueError("Sophon frame gains must be finite and nonnegative")
        values = [strength * gain for gain in frame_gains]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Sophon effective strengths must be in [0, 1]")
        gate = None
    else:
        gate = gate_plan._start(SPEECH_SIGNAL, WIDTH, strength)

    current = getattr(inference.binding.response, "__self__", None)
    if not isinstance(current, CurrentResponse) or inference.selection != "raw":
        raise ValueError("Sophon scheduled audio requires its open RAW speech inference session")
    current.runtime.owner.check()
    scope = ExitStack()
    applied = []

    def before(_model, args):
        position = len(applied)
        if gate is None:
            strength_at_frame = values[min(position, len(values) - 1)]
        else:
            (hidden,) = args
            if hidden.shape != (1, WIDTH):
                raise RuntimeError("Sophon needs one pre-edit native KF input per acoustic frame")
            strength_at_frame = gate.strength_at(hidden[0], position=position)
        scope.enter_context(current.response_adapter.scaled(strength_at_frame))
        applied.append(strength_at_frame)

    def after(_model, _args, _output):
        scope.close()

    reg = current.response_adapter.reg
    with ExitStack() as handles:
        handles.enter_context(current.response_adapter.scaled(strength))
        handles.callback(reg.register_forward_pre_hook(before).remove)
        handles.callback(reg.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        receipt = inference.render(strength=strength, **request)
    if len(applied) != receipt["native_frames"]:
        raise RuntimeError("Sophon acoustic adapter calls differ from the finished native frames")
    schedule = {"nominal_strength": strength, "effective_frame_strengths": applied,
                "native_EOS": receipt["native_EOS"], "output": receipt["output"]}
    if gate is not None:
        schedule["activation_gate"] = gate.receipt()
    record(Path(request["durable_directory"]) / "scheduled-strength.json", schedule)
    return {**receipt, **schedule}
