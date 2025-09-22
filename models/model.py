from typing import Union, List, Any
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from copy import deepcopy
from utils.prompt_emb import PromptEmbedding
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from jaxtyping import Float, Integer
from PIL import Image
import os

def broadcast_trailing_dims(tensor: Float[torch.Tensor, '(c)'], reference: Float[torch.Tensor, '(c) ...']) -> torch.Tensor:
    num_trailing = len(reference.shape) - len(tensor.shape)
    for _ in range(num_trailing):
        tensor = tensor.unsqueeze(-1)
    return tensor

class diffusion:
    def __init__(self,path_to_model:str,device:str='cuda',pipe_kwargs:dict={}):
        device = torch.device(device)   
        self.pipe:StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(path_to_model,**pipe_kwargs).to(device) 
        self.pipe.safety_checker=None
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(device)
        
        set_timesteps_orig = self.pipe.scheduler.set_timesteps
        def set_timesteps_custom(num_inference_steps: int = None, timesteps: torch.Tensor = None, device: Union[str, torch.device] = None):
            if not timesteps is None:
                if isinstance(timesteps, torch.Tensor):
                    self.pipe.scheduler.timesteps = timesteps.to(device)
                else:
                    self.pipe.scheduler.timesteps = torch.from_numpy(timesteps).to(device)
            else:
                return set_timesteps_orig(num_inference_steps=num_inference_steps, device=device)
        self.num_inference_steps=50
        self.pipe.scheduler.set_timesteps = set_timesteps_custom

        self.tokenizer:CLIPTokenizer = self.pipe.tokenizer
        self.text_model:CLIPTextModel = self.pipe.text_encoder
        self.original_prompt:PromptEmbedding = None

    def prompt(self,prompt:str) -> PromptEmbedding:
        self.original_prompt = PromptEmbedding(prompt=prompt,text_model=self.text_model,tokenizers=self.tokenizer)
        res = deepcopy(self.original_prompt)
        return res

    def sample(self,prompt: PromptEmbedding=None, timestep_s:float=0.0,timestep_e:float=1.0,start_latent:Float[torch.Tensor,'n d h w']=None,prompt_embedding=None,**kwargs) -> Union[List[Image.Image],Any]:
        if prompt_embedding is None:
            prompt_embedding = prompt.embedding
        self.pipe.scheduler.set_timesteps(kwargs['num_inference_steps'])
        timesteps = self.pipe.scheduler.timesteps
        timesteps=timesteps[int(timestep_s*self.pipe.scheduler.num_inference_steps):int(timestep_e*self.pipe.scheduler.num_inference_steps)]
        with torch.no_grad():
            samples = self.pipe(**({"timesteps":timesteps,"num_inference_steps":len(timesteps),"prompt_embeds":prompt_embedding,"latents":start_latent}|kwargs)).images
        return samples
        
    def delayed_sample(self,prompt : PromptEmbedding, relative_delay:float,**kwargs) -> Union[List[Image.Image],Any]:
        if relative_delay==0:
            return self.sample(prompt,**kwargs)
        else:
            intermediate_latents = self.sample(self.original_prompt,timestep_s=0.0,timestep_e=relative_delay,**kwargs,output_type='latent')
            return self.sample(prompt,start_latent=intermediate_latents,timestep_s=relative_delay,timestep_e=1.0,**kwargs)
        
    def _get_eps_pred(self, t: Integer[torch.Tensor, 'n'], sample: Float[torch.Tensor, 'n ...'], model_output: Float[torch.Tensor, 'n ...']) -> Float[torch.Tensor, 'n ...']:
        alpha_prod_t = broadcast_trailing_dims(self.pipe.scheduler.alphas_cumprod[t.to(self.pipe.scheduler.alphas_cumprod.device)].to(model_output.device), model_output)
        beta_prod_t = 1 - alpha_prod_t
        if self.pipe.scheduler.config.prediction_type == "epsilon":
            return (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.pipe.scheduler.config.prediction_type == "sample":
            return model_output
        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
            return (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise NotImplementedError(f'Missing implementation for {self.pipe.scheduler.config.prediction_type=}.')
    
    @torch.no_grad
    def embed_prompt(self, prompt: str) -> PromptEmbedding:
        return PromptEmbedding(prompt=prompt,text_model=self.text_model,tokenizers=self.tokenizer)

    def predict_eps(self, embs: List[PromptEmbedding], start_sample: Float[torch.Tensor, 'n c h w'], t_relative: Float[torch.Tensor, 'n']) -> Float[torch.Tensor, 'n c h w']:
        i_t = torch.round(t_relative * (self.num_inference_steps - 1)).to(torch.int64)
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        t = self.pipe.scheduler.timesteps[i_t.to(self.pipe.scheduler.timesteps.device)].to(start_sample.device)
        if isinstance(embs,list):
            if len(embs)>1:
                if isinstance(embs[0],PromptEmbedding):
                    prompt_embedding = torch.cat([embs[i].embedding for i in range(len(embs))],dim=0)
                elif isinstance(embs[0],torch.Tensor):
                    prompt_embedding=torch.cat([embs[i] for i in range(len(embs))],dim=0)
                else:
                    raise ValueError(f'Unexpected type: {type(embs[0])}')
            else:
                if isinstance(embs[0],PromptEmbedding):
                    prompt_embedding=embs[0].embedding
                elif isinstance(embs[0],torch.Tensor):
                    prompt_embedding=embs[0]
                else:
                    raise ValueError(f'Unexpected type: {type(embs[0])}')
            # print(prompt_embedding.dtype)
            return self._get_eps_pred(t, start_sample, self.pipe.unet(start_sample, t, encoder_hidden_states=prompt_embedding).sample)
        else:
            return self._get_eps_pred(t, start_sample, self.pipe.unet(start_sample, t, encoder_hidden_states=embs.embedding).sample)
        