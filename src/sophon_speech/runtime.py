"""A declared new original-speech runtime candidate, not old text qualification."""
from importlib import metadata
REQUIRED=dict(torch='2.9.1',transformers='4.57.6',torchaudio='2.9.1',**{'x-transformers':'2.31.7'},einops='0.8.1',safetensors='0.8.0',numpy='2.2.6')
def identity():
    result={}
    for name,expected in REQUIRED.items():
        actual=metadata.version(name)
        if actual.split('+')[0]!=expected:raise ImportError('exact new speech runtime candidate required:'+name)
        result[name]=actual
    # Native HiFT classes genuinely import SciPy; decoder must not use stubs.
    for name in ('scipy','librosa','soundfile'):result[name]=metadata.version(name)
    return result
