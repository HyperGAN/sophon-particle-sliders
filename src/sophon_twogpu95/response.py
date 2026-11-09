"""Native audio merge and differentiable full hidden transfer to the PG device."""
import torch
from sophon_particle_sliders.response import SophonResponse as OriginalResponse,conditions

class SophonResponse(OriginalResponse):
    family='sophon095_explicit_two_gpu_native_thinker'
    def __init__(self,*args,admission=None,**kwargs):
        super().__init__(*args,**kwargs);self.admission=admission
        self.entry_device=self.model.get_input_embeddings().weight.device
        self.pg_device=next(self.slider.parameters()).device
    def check_owner(self):
        if self.admission is not None:self.admission.check()
    def _response(self,prompt,strength):
        self.check_owner();kwargs={k:v.to(self.entry_device)for k,v in prompt.items()}
        with self.network.scaled(strength):
            embeds=self.model._merge_audio(kwargs['input_ids'],None,kwargs.get('audio_values'),kwargs.get('audio_lens'),kwargs.get('audio_token_len'))
            output=self.model.language_model.model(inputs_embeds=embeds,attention_mask=kwargs.get('attention_mask'),use_cache=False,return_dict=True)
            value=output.last_hidden_state[:,-1].to(device=self.pg_device,dtype=torch.float32)
        if value.shape!=(1,self.width)or not torch.isfinite(value).all():raise ValueError('full native hidden response required')
        return value
