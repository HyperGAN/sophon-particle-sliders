"""Pack ChatML and native audio with the released tokenizer/processor."""

import json
from copy import deepcopy
from pathlib import Path

import torch

from .response import SpeechRow, snapshot_prompt

TAG_SENTENCE = "When it fits, you may use sound tags for laughter, coughs and emphasis."


def pack(processor, messages, waveforms=(), *, max_tokens=4096):
    if not messages or messages[-1].get("role") not in ("user", "tool"):
        raise ValueError("conversation must end at a user/tool response prefix")
    messages = deepcopy(messages)
    if not any(
        m.get("role") == "system" and TAG_SENTENCE in m.get("content", "")
        for m in messages
    ):
        raise ValueError("keep the bundled tag LoRA's sound-tag system instruction")
    count = 0
    for message in messages:
        if not isinstance(message.get("content"), str):
            raise ValueError("native ChatML content must be text with <audio> markers")
        if "<|audio|>" in message["content"]:
            raise ValueError(
                "use <audio>; reserved expanded audio tokens are not input text"
            )
        count += message["content"].count("<audio>")
        message["content"] = message["content"].replace("<audio>", "<|audio|>")
    if count != len(waveforms):
        raise ValueError(
            "one audio file per <audio> marker required, in conversation order"
        )
    ids = processor.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    features = processor.audio_features(waveforms) if waveforms else {}
    lengths = iter(features.pop("tokens_per_clip", ()))
    expanded = []
    seen = 0
    for token in ids:
        if token == processor.audio_token_index:
            seen += 1
            expanded.extend([token] * next(lengths))
        else:
            expanded.append(token)
    if seen != count:
        raise ValueError("released tokenizer did not preserve the audio markers")
    if type(max_tokens) is not int or max_tokens < 1 or len(expanded) > max_tokens:
        raise ValueError(
            "expanded native context exceeds max_tokens; no silent truncation"
        )
    tokens = torch.tensor([expanded], dtype=torch.long)
    return snapshot_prompt(
        dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), **features),
        processor.audio_token_index,
    )


def read_waveforms(paths, *, root=Path(".")):
    import librosa
    import numpy as np
    import soundfile as sf

    result = []
    for path in paths:
        audio, sr = sf.read(root / path, dtype="float32", always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("nonempty finite audio required")
        mono = audio.mean(axis=1)
        if sr != 16000:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        result.append(mono)
    return result


def read_rows(path, processor, *, max_tokens=4096):
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if set(item) != {"neutral_messages", "positive_messages", "audios"}:
            raise ValueError(
                "row requires neutral_messages, positive_messages and audios"
            )
        neutral, positive = item["neutral_messages"], item["positive_messages"]
        if [m for m in neutral if m.get("role") != "system"] != [
            m for m in positive if m.get("role") != "system"
        ]:
            raise ValueError(
                "matched conversations may differ only in system instructions"
            )
        audio = (
            read_waveforms(item["audios"], root=path.parent) if item["audios"] else []
        )
        rows.append(
            SpeechRow(
                *(
                    pack(processor, item[key], audio, max_tokens=max_tokens)
                    for key in ("neutral_messages", "positive_messages")
                )
            )
        )
    if len(rows) < 2:
        raise ValueError("at least two matched native response rows required")
    return rows
