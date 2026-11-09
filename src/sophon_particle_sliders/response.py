"""Sophon native final-token coordinates, including real audio conditioning."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from particle_sliders.pg09_audio import PreparedAudioResponse
from particle_sliders.pg09_backends import AdapterParameters


@dataclass(frozen=True)
class SpeechRow:
    neutral: dict
    positive: dict
    scale: float = 1.0


def snapshot_prompt(prompt, audio_token_index):
    allowed = {
        "input_ids",
        "attention_mask",
        "audio_values",
        "audio_lens",
        "audio_token_len",
    }
    if (
        not isinstance(prompt, dict)
        or set(prompt) - allowed
        or "input_ids" not in prompt
    ):
        raise ValueError(
            "explicit native input_ids and supported audio tensors required"
        )
    result = {k: v.detach().clone() for k, v in prompt.items()}
    ids = result["input_ids"]
    if (
        ids.dtype != torch.long
        or ids.ndim != 2
        or ids.shape[0] != 1
        or not ids.numel()
        or (ids < 0).any()
    ):
        raise ValueError("single nonempty integer token sequence required")
    mask = result.get("attention_mask")
    if mask is not None and (mask.shape != ids.shape or not torch.all(mask == 1)):
        raise ValueError("unpadded single-example attention mask required")
    audio_keys = {"audio_values", "audio_lens", "audio_token_len"}
    present = audio_keys & result.keys()
    count = int((ids == audio_token_index).sum())
    if present:
        if present != audio_keys:
            raise ValueError("all three native audio tensors required")
        values, lens, lengths = (
            result[k] for k in ("audio_values", "audio_lens", "audio_token_len")
        )
        if (
            values.ndim != 3
            or values.shape[1] != 128
            or values.shape[0] < 1
            or lens.shape != (len(values),)
            or lengths.shape != lens.shape
            or lens.dtype != torch.long
            or lengths.dtype != torch.long
            or not values.is_floating_point()
            or not torch.isfinite(values).all()
            or (lens <= 0).any()
            or (lens > values.shape[-1]).any()
            or (lens > 3000).any()
            or (lengths <= 0).any()
            or not torch.equal(lengths, (lens + 15) // 16)
            or count != int(lengths.sum())
        ):
            raise ValueError("invalid native mel chunks or audio token/embedding count")
    elif count:
        raise ValueError("audio tokens require their native audio features")
    return result


class SophonResponse:
    family = "sophon_realtime_v1_thinker"

    def __init__(self, model, slider, *, model_identity, processor=None):
        self.model, self.slider, self.processor = model, slider, processor
        self.model_identity = deepcopy(model_identity)
        if not self.model_identity:
            raise ValueError("verified native or explicit synthetic identity required")
        if any(
            getattr(module, "gradient_checkpointing", False)
            for module in model.modules()
        ):
            raise ValueError(
                "disable native gradient checkpointing before binding adapter scales"
            )
        self._modes = [(module, module.training) for module in model.modules()]
        self.model.eval()
        self.width = int(model.config.text_config.hidden_size)
        self.network = AdapterParameters(slider.named_parameters(), slider.scaled)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError(
                "thinker, audio projector and bundled tag adapter must be frozen"
            )

    def _response(self, prompt, strength):
        device = next(self.model.parameters()).device
        kwargs = {k: v.to(device) for k, v in prompt.items()}
        # Call the released wrapper: it performs the real audio embedding merge.
        with self.network.scaled(strength):
            embeds = self.model._merge_audio(
                kwargs["input_ids"],
                None,
                kwargs.get("audio_values"),
                kwargs.get("audio_lens"),
                kwargs.get("audio_token_len"),
            )
            output = self.model.language_model.model(
                inputs_embeds=embeds,
                attention_mask=kwargs.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
            value = output.last_hidden_state[:, -1].float()
        if value.shape != (1, self.width) or not torch.isfinite(value).all():
            raise ValueError("invalid native final-token hidden state")
        return value

    def prepare(self, row):
        if row.scale != 1.0:
            raise ValueError("unipolar frozen-positive teacher requires scale +1")
        index = self.model.config.audio_token_index
        neutral, positive = (
            snapshot_prompt(p, index) for p in (row.neutral, row.positive)
        )
        # Audio must be exactly matched; the concept instruction changes text only.
        for key in ("audio_values", "audio_lens", "audio_token_len"):
            a, b = neutral.get(key), positive.get(key)
            if (a is None) != (b is None) or a is not None and not torch.equal(a, b):
                raise ValueError("teacher and student must share identical audio")
        if torch.equal(neutral["input_ids"], positive["input_ids"]):
            raise ValueError("distinct neutral and positive text conditioning required")
        copied = SpeechRow(neutral, positive)
        with torch.no_grad():
            return PreparedAudioResponse(
                copied,
                self._response(positive, 0).detach().clone(),
                self._response(neutral, 0).detach().clone(),
            )

    def generated(self, prepared):
        return self._response(prepared.row.neutral, prepared.row.scale)

    @torch.no_grad()
    def sample(self, prepared, *, scale=1.0):
        return self._response(prepared.row.neutral, scale).detach().clone()

    @torch.no_grad()
    def response(
        self,
        *,
        prompt=None,
        messages=None,
        audios=(),
        max_tokens=4096,
        strength=1.0,
        seed=0,
    ):
        if (prompt is None) == (messages is None):
            raise ValueError(
                "supply one packed native prompt or one ChatML conversation"
            )
        if messages is not None:
            from .data import pack, read_waveforms

            if self.processor is None:
                raise ValueError("ChatML requires the native released processor")
            prompt = pack(
                self.processor,
                messages,
                read_waveforms(audios) if audios else [],
                max_tokens=max_tokens,
            )
        elif audios:
            raise ValueError("packed native prompts already contain their audio")
        if prompt["input_ids"].numel() > max_tokens:
            raise ValueError("native context exceeds max_tokens")
        return (
            self._response(
                snapshot_prompt(prompt, self.model.config.audio_token_index), strength
            )
            .detach()
            .clone()
        )

    def metadata(self):
        return dict(
            protocol="sophon-thinker-last-hidden-v1",
            model_identity=self.model_identity,
            targets=list(self.slider.target_names),
            response_domain="continuous_vector",
            response_width=self.width,
            teacher="frozen tagged thinker positive; matched audio",
            student="neutral text; same audio; routed residual",
            tag_adapter="frozen, always active",
            audio_render_qualified=False,
            quality_qualified=False,
        )

    def close(self):
        self.slider.close()
        for module, training in self._modes:
            module.training = training


def conditions(rows):
    return torch.cat(
        [
            torch.cat(
                (
                    p.neutral,
                    p.neutral.new_tensor(
                        [
                            [
                                p.row.neutral["input_ids"].numel(),
                                p.row.positive["input_ids"].numel(),
                                p.row.scale,
                            ]
                        ]
                    ),
                ),
                dim=1,
            )
            for p in rows
        ]
    )
