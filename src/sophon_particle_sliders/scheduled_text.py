"""Strength schedules for the native audio-conditioned full-prefix thinker."""

from contextlib import ExitStack
from sys import float_info

import torch

from .current_backend import CurrentResponse
from .activation_gate import GatePlan, TAGGED_TEXT_SIGNAL
from sophon_twogpu.backend import CurrentResponse as TwoGPUResponse


def render(inference, *, step_gains, strength=1.0, max_new_tokens=64, context_gain=None, **request):
    """Keep each position's strength across the native audio-conditioned replays.

    An explicit context gain leaves the final prompt position to step_gains[0].
    Without it, step_gains[0] covers the entire prompt as before.
    """
    return _render(inference, step_gains, None, strength, max_new_tokens, context_gain, request)


def render_adaptive(inference, *, gate_plan, strength=1.0, max_new_tokens=64, **request):
    """Gate the saved RAW edit using the current pre-edit decoder input."""
    if not isinstance(gate_plan, GatePlan):
        raise ValueError("Provide an explicit calibrated Sophon GatePlan")
    return _render(inference, None, gate_plan, strength, max_new_tokens, 0.0, request)


def _render(inference, step_gains, gate_plan, strength, max_new_tokens, context_gain, request):
    inference._open()
    binding = inference.binding
    owner = getattr(binding.close, "__self__", None)
    if (inference.selection != "raw" or not isinstance(owner, (CurrentResponse, TwoGPUResponse))
            or owner.network is not binding.network or binding.render is None):
        raise ValueError("Sophon needs an open native tagged-text RAW session")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
        raise ValueError("Sophon requires a native token budget in [1,2048]")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Sophon nominal strength must be in [0,1]")
    if context_gain is not None and (type(context_gain) not in (int, float)
            or not 0 <= context_gain <= float_info.max or strength * context_gain > 1):
        raise ValueError("Sophon effective context strength must be in [0,1]")
    if gate_plan is None:
        if (not isinstance(step_gains, (list, tuple)) or len(step_gains) != max_new_tokens
                or any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains)):
            raise ValueError("Sophon requires one finite nonnegative gain per possible output token")
        strengths = [strength * gain for gain in step_gains]
        if any(value > 1 for value in strengths):
            raise ValueError("Sophon effective strengths must be in [0,1]")
        gate = None
    else:
        width = owner.model.config.text_config.hidden_size
        gate = gate_plan._start(TAGGED_TEXT_SIGNAL, width, strength)
        first_layer = owner.model.language_model.model.layers[0]

    applied = []
    prefix_tokens = None
    pending = None
    envelope = None
    device_weights = {}
    scalar_gain = ExitStack()

    def apply(value):
        nonlocal envelope
        applied.append(value)
        context = applied[0] if context_gain is None else strength * context_gain
        gains = (context,) * (prefix_tokens - 1) + tuple(applied)
        if all(value == gains[0] for value in gains):
            scalar_gain.enter_context(owner.network.scaled(gains[0]))
        else:
            envelope = gains

    def before(_model, _args, inputs):
        nonlocal prefix_tokens, pending
        ids = inputs.get("input_ids")
        position = len(applied)
        if (pending is not None or ids is None or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] == 0 or inputs.get("past_key_values") is not None
                or position >= max_new_tokens):
            raise RuntimeError("Sophon requires the original uncached native token sequence")
        if position == 0:
            prefix_tokens = ids.shape[1]
        if ids.shape[1] != prefix_tokens + position:
            raise RuntimeError("Sophon evaluated an unexpected full-prefix length")
        if gate is None:
            apply(strengths[position])
        else:
            pending = ids.shape[1]

    def before_decoder(_layer, args, kwargs):
        nonlocal pending
        hidden = args[0] if args else kwargs.get("hidden_states")
        if pending is None or hidden is None or hidden.shape != (1, pending, width):
            raise RuntimeError("Sophon must read layer 0 once per native full-prefix decision")
        value = gate.strength_at(hidden[0, -1], position=pending - 1)
        pending = None
        apply(value)

    def after(*_hook_args):
        nonlocal envelope
        envelope = None
        device_weights.clear()
        scalar_gain.close()

    def project(x, branch, attached):
        if envelope is None:
            if pending is not None:
                raise RuntimeError("Sophon routed projection preceded the native decoder reading")
            return attached(x)
        if x.ndim != 3 or x.shape[:2] != (1, len(envelope)):
            raise RuntimeError("Sophon projection does not match the current full prefix")
        if x.device not in device_weights:
            gains = torch.tensor(envelope, device=x.device, dtype=torch.float32).view(1, -1, 1)
            device_weights[x.device] = (gains, gains != 0)
        gains, edited = device_weights[x.device]
        # Match the original residual's device and cast before applying the gain.
        features = branch.lora_down(x.to(branch.lora_down.weight))
        routed = branch.bridge(features, branch._particles)
        delta = branch.lora_up(routed).to(x)
        base = branch.org_forward(x)
        return torch.where(edited, base + (delta * gains).to(x), base)

    with ExitStack() as handles:
        handles.enter_context(owner.network.scaled(strength))
        for host, branch, attached in owner.slider._hosts:
            if (host.forward != attached or branch.scale != 1
                    or getattr(host, "_sophon_slider_owner", None) is not owner.slider):
                raise RuntimeError("Sophon requires its original attached routed projections")
            handles.callback(setattr, host, "forward", attached)
            host.forward = lambda x, branch=branch, attached=attached: project(x, branch, attached)
        handles.callback(owner.model.register_forward_pre_hook(before, with_kwargs=True).remove)
        if gate is not None:
            handles.callback(first_layer.register_forward_pre_hook(before_decoder, with_kwargs=True).remove)
        handles.callback(owner.model.register_forward_hook(after, always_call=True).remove)
        handles.callback(after)
        output = inference.render(strength=strength, max_new_tokens=max_new_tokens, **request)
    if (pending is not None or not applied or len(applied) != len(output.native["new_token_ids"])
            or prefix_tokens != output.native["prompt_tokens"]):
        raise RuntimeError("Sophon finished text differs from the applied token schedule")
    output.native["strength_schedule"] = dict(prefix_tokens=prefix_tokens, effective_strengths=applied)
    if context_gain is not None:
        output.native["strength_schedule"]["context_strength"] = strength * context_gain
    if gate is not None:
        output.native["activation_gate"] = gate.receipt()
    return output
