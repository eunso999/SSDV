import torch
from typing import List
from diffusers.models.attention_processor import AttnProcessor, Attention
from utils.prompt_emb import PromptEmbedding

class AttentionStore():
    def __init__(self,
                 additional_context:List[PromptEmbedding],
                 average: bool,
                 batch_size=1,
                 max_resolution=16
                 ):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size
        self.max_size = max_resolution ** 2
        self.additional_context= additional_context
        self.num_additional_context=len(additional_context)
        self.skip_uncond_attn=True
        self.num_attn=-1
        self.cur_attn=0

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def get_additional_context(self,encoder_hidden_states):
        if len(self.additional_context)==0:
            return None
        return torch.cat([emb.embedding.to(encoder_hidden_states.device,dtype=encoder_hidden_states.dtype) for emb in self.additional_context],dim=0)

    def value_handler(self,attention_probs,value,focus_on_unconditional):
        return  torch.bmm(attention_probs, value)
    
    def __call__(self,attention_probs,is_cross,place_in_unet):

        if attention_probs.shape[1] <= self.max_size:
            if self.skip_uncond_attn:
                self.forward(attention_probs, is_cross, place_in_unet)
            else:
                bs = 1
                skip = 1
                self.forward(attention_probs[skip * self.batch_size:, :], is_cross, place_in_unet)

        self.cur_attn += 1
        if self.cur_attn==self.num_attn:
            self.cur_attn = 0
            self.between_steps()

    def forward(self,attention_probs,is_cross,place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attention_probs)
    
    def between_steps(self,store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)
            self.cur_step += 1
        self.step_store = self.get_empty_store()
    
    def get_attention_store(self):
        return self.attention_store
    
    def get_attention(self,step:int):
        if self.average:
            attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        else:
            assert step is not None
            attention = self.attention_store[step]
        return attention

class CCCrossAttnProcessor:
    def __init__(self, attention_store, place_in_unet,focus_on_unconditional=False):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.focus_on_unconditional = focus_on_unconditional

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        assert (not attn.residual_connection)
        assert (attn.spatial_norm is None)
        assert (attn.group_norm is None)
        assert (hidden_states.ndim != 4)
        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None: 
            encoder_hidden_states = hidden_states 
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        if is_cross:
            additional_context=self.attnstore.get_additional_context(encoder_hidden_states)
            if additional_context is not None:
                encoder_hidden_states = torch.cat([encoder_hidden_states,additional_context],dim=0) # [batch_size, sequence_length, hidden_size] -> [batch_size+num_additional_context, sequence_length, hidden_size]
            else:
                encoder_hidden_states = encoder_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Add added context
        if is_cross:
            if query.size(0)>batch_size:
                chunked_query = query.chunk(2)
                uncond_query = chunked_query[0]
                cond_query = chunked_query[1]
                split_size = [2*batch_size, *[1]*self.attnstore.num_additional_context]
            elif query.size(0)==batch_size:
                uncond_query = query
                cond_query = query
                split_size = [batch_size,*[1]*self.attnstore.num_additional_context]
            else:
                raise ValueError("query size is not correct")
            if self.focus_on_unconditional:
                additional_query=uncond_query[0:1].repeat(self.attnstore.num_additional_context,1,1)
            else:
                additional_query=cond_query[0:1].repeat(self.attnstore.num_additional_context,1,1)
            query = torch.cat([query, additional_query], dim=0) 
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Register handler
        if is_cross and hasattr(self.attnstore,"key_handler") : 
            key = self.attnstore.key_handler(key)
        
        # Get attention probs
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Register handler
        self.attnstore(attention_probs,
                       is_cross=is_cross,
                       place_in_unet=self.place_in_unet,
                       )
        if is_cross and hasattr(self.attnstore,"value_handler") : 
            hidden_states = self.attnstore.value_handler(attention_probs,value,self.focus_on_unconditional)
        else:
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Remove cross-attention
        if is_cross:
            chunked_hidden_states = hidden_states.split(split_size)
        else:
            chunked_hidden_states = [hidden_states]
        hidden_states = chunked_hidden_states[0]

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
    
def prepare_unet(unet,attention_store=None, filter_set: List[str]=['mid'],self_attn: bool=False, cross_attn: bool=True, CrossAttnProcessor=CCCrossAttnProcessor):
    attn_procs = {}
    num_attn = 0
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        if attention_store is None:
            attn_procs[name] = AttnProcessor()
        elif ("attn2" in name and cross_attn) and (place_in_unet not in filter_set):
            attn_procs[name] = CrossAttnProcessor(
                attention_store=attention_store,
                place_in_unet=place_in_unet)
            num_attn += 1
        elif ("attn1" in name and self_attn) and( place_in_unet not in filter_set):
            attn_procs[name] = CrossAttnProcessor(
                attention_store=attention_store,
                place_in_unet=place_in_unet)
            num_attn += 1
        else:
            attn_procs[name] = AttnProcessor() # default
    if attention_store is not None:
        attention_store.num_attn=num_attn
    unet.set_attn_processor(attn_procs)