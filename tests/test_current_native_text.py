"""Actual tiny official generation; no pretrained model or audio synthesis."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from conftest import prompt
from test_native_small import native_small
from sophon_particle_sliders.finished_text import generate
from sophon_particle_sliders.native import LOCK


def test_actual_official_tiny_audio_conditioned_greedy_text(native_small,tmp_path,monkeypatch):
    native,_=native_small
    # This tests exact official architecture, no pretrained or tag quality claim.
    native.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:' '.join(map(str,ids))))
    native.model_identity['source_commit']=LOCK['commit']
    merged=[]
    original_merge=native.model._merge_audio
    def observe_merge(ids,embeds,values,lens,lengths):
        merged.append(dict(input_tokens=ids.shape[1],
            audio_values=None if values is None else values.detach().clone(),
            audio_lens=None if lens is None else lens.detach().clone(),
            audio_token_len=None if lengths is None else lengths.detach().clone()))
        return original_merge(ids,embeds,values,lens,lengths)
    monkeypatch.setattr(native.model,'_merge_audio',observe_merge)
    before=deepcopy(native.model.state_dict());rng=torch.get_rng_state().clone()
    first=generate(native,prompt=prompt(),strength=0,max_new_tokens=2,seed=7)
    second=generate(native,prompt=prompt(),strength=1,max_new_tokens=2,seed=7)
    assert first.text==second.text  # Actual zero-up host initially bypasses exactly.
    assert first.native['audio_chunks']==1 and first.native['max_new_tokens']==2
    assert first.native['actual_generate_kwargs']['do_sample']is False
    assert first.native['actual_generate_kwargs']['use_cache']is False
    # Observe the REAL official wrapper during generation, not receipt counts:
    # cached-prefill bug would leave values=None or never merge the audio.
    expected=prompt()
    assert merged and merged[0]['input_tokens']==expected['input_ids'].shape[1]
    for call in merged:
        for key in ('audio_values','audio_lens','audio_token_len'):
            assert call[key] is not None and torch.equal(call[key],expected[key])
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,before[name])for name,value in native.model.state_dict().items())
    path=tmp_path/'finished.txt';first.save(path)
    assert path.read_text()==first.text
    with pytest.raises(ValueError,match='fresh'):first.save(path)
    assert not torch.cuda.is_initialized()
