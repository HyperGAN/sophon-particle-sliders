"""Actual tiny official generation; no pretrained model or audio synthesis."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from conftest import prompt
from test_native_small import native_small
from sophon_particle_sliders.finished_text import generate
from sophon_particle_sliders.native import LOCK


def test_actual_official_tiny_audio_conditioned_greedy_text(native_small,tmp_path):
    native,_=native_small
    # This tests exact official architecture, no pretrained or tag quality claim.
    native.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:' '.join(map(str,ids))))
    native.model_identity['source_commit']=LOCK['commit']
    before=deepcopy(native.model.state_dict());rng=torch.get_rng_state().clone()
    first=generate(native,prompt=prompt(),strength=0,max_new_tokens=2,seed=7)
    second=generate(native,prompt=prompt(),strength=1,max_new_tokens=2,seed=7)
    assert first.text==second.text  # Actual zero-up host initially bypasses exactly.
    assert first.native['audio_chunks']==1 and first.native['max_new_tokens']==2
    assert first.native['actual_generate_kwargs']['do_sample']is False
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,before[name])for name,value in native.model.state_dict().items())
    path=tmp_path/'finished.txt';first.save(path)
    assert path.read_text()==first.text
    with pytest.raises(ValueError,match='fresh'):first.save(path)
    assert not torch.cuda.is_initialized()
