from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np
import torch 
import torch.nn as nn
from jaxtyping import Float
from transformers import CLIPTextModel, CLIPTokenizer
from functools import reduce
import re
MAX_LEN = 77
LOW_RESOURCE = True
DEBUG=False
def count_adjacent_duplicates(lst):
    if not lst: return []
    counts, current_count = [], 1
    for i in range(1, len(lst)):
        current_count = current_count + 1 if lst[i] == lst[i-1] else (counts.append(current_count) or 1)
    counts.append(current_count)
    return counts

@dataclass(order=True)
class Span:
    start: int = field(compare=True) # start index of the span
    end: int # end `index` of the span (inclusive) dont index as [start:end] but [start:end+1]
    word: str
    token_index: List[int]
    # embedding: Float[torch.Tensor,'s d']
    mask: Float[torch.Tensor,'s'] = field(init=False,default=None)
    section: List[int] = field(init=False,default=None)
    def __post_init__(self):
        self.mask = torch.zeros(MAX_LEN)
        self.mask[self.start:self.end+1] = 1
        self.section = count_adjacent_duplicates(self.mask.int().tolist())
    @staticmethod
    def merge(spans: List[Span]) -> Span:
        assert len(spans)>0
        start = min([span.start for span in spans])
        end = max([span.end for span in spans])
        word = ' '.join([span.word for span in spans])
        token_index = []
        for span in spans:
            token_index.extend(span.token_index)
        # embedding = torch.cat([span.embedding for span in spans],dim=1)
        # mask = reduce(lambda x,y:x.bool()|y.bool(),[span.mask for span in spans]) 
        return Span(start=start,end=end,word=word,token_index=token_index)#,embedding=embedding)
    
@dataclass
class PromptEmbedding:
    prompt: str
    parsed_prompt: str = field(init=False)
    text_model: CLIPTextModel = field(default=CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',subfolder ='text_encoder'))
    tokenizers: CLIPTokenizer = field(default=CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5',subfolder ='tokenizer'))
    embedding: Float[torch.Tensor,'n s d'] = field(init=False,default=None) # n: number of prompt, s: number of token, d: dimension of embedding
    word_wise_index: Dict[str, List[Span]] = field(init=False,default_factory=dict) # word: Span list by Order of appearance in the prompt
    SOS: Span = field(init=False,default=None)
    EOS: Span = field(init=False,default=None)
    padding: Span = field(init=False,default=None)

    def __post_init__(self):
        self.word_wise_index,self.parsed_prompt,self.embedding = self._get_word_wise_index()
        self.SOS = self.word_wise_index['<|startoftext|>'][0]
        self.EOS = self.word_wise_index['<|endoftext|>'][0]
        self.padding = Span.merge(self.word_wise_index['<|endoftext|>'][1:] ) if len(self.word_wise_index['<|endoftext|>'])>1 else self.EOS
        print(f"full prompt: {self.prompt}, parsed prompt: {self.parsed_prompt}") if DEBUG else None

    def _get_text_embeddings(self) -> Float[torch.Tensor,'n s d']:
        prompt = self.parsed_prompt
        tokenized_prompt = self.tokenizers(prompt,padding='max_length',max_length=MAX_LEN,truncation=True,return_tensors='pt').input_ids
        with torch.no_grad():
            embedding = self.text_model(tokenized_prompt.to(self.text_model.device)).last_hidden_state
        return embedding
    
    def _get_word_wise_index(self) -> Tuple[Dict[str, List[Span]],str,Float[torch.Tensor,'n s d']]:
        tmp = re.split(r'(\([^)]+\))',self.prompt)
        parts=[]
        for part in tmp:
            if part.startswith('(') and part.endswith(')'):
                parts.append(part[1:-1])
            else:
                sub_parts = part.split()
                for sub_part in sub_parts:
                    if sub_part and sub_part[-1] in '.,;!?':
                        parts.append(sub_part[:-1])
                        parts.append(sub_part[-1])
                    else:
                        parts.append(sub_part)
        parts = [part for part in parts if part]
        token_ids=[]
        for part in parts:
            token_ids.append((self.tokenizers.encode(part,add_special_tokens=False)))
        full_prompt= self.tokenizers.decode(reduce(lambda x,y:x+y,token_ids)) if len(token_ids)>0 else ''

        assert '</w>' not in full_prompt 
        
        token_span=[]
        start=1
        for token,word in zip(token_ids,parts):
            token_span.append((start,start+len(token)-1,word,token))
            start+=len(token)
        print(token_span) if DEBUG else None
        full_token = self.tokenizers(full_prompt,padding='max_length',max_length=MAX_LEN,truncation=True).input_ids
        self.parsed_prompt = full_prompt
        self.embedding = self._get_text_embeddings()

        Spans=[]
        end=0
        Spans.append(Span(start=0,end=0,word='<|startoftext|>',token_index=full_token[0:1]))#embedding=self.embedding[0:1,0:1]))
        for start,end,word,token in token_span:
            Spans.append(Span(start=start,end=end,word=word,token_index=token))#,embedding=self.embedding[0:1,start:end+1]))
        Spans.append(Span(start=end+1,end=end+1,word='<|endoftext|>',token_index=full_token[end+1:end+2]))#,embedding=self.embedding[0:1,end+1:end+2]))
        pad_start=end+2
        for pad in full_token[end+2:]:
            Spans.append(Span(start=pad_start,end=pad_start,word='<|endoftext|>',token_index=[pad]))#,embedding=self.embedding[0:1,pad_start:pad_start+1]))
            pad_start+=1

        word_wise_index={}
        for span in Spans:
            if span.word not in word_wise_index:
                word_wise_index[span.word]=[]
            word_wise_index[span.word].append(span)
        
        return word_wise_index,full_prompt,self.embedding
    
    def get_word_emb(self,word:str,ord:int=0) -> Float[torch.Tensor,'n s d']:
        if word == 'EOS':
            word='<|endoftext|>'
            ord=0
        elif word == 'SOS':
            word='<|startoftext|>'
        elif word == 'PAD':
            word='<|endoftext|>'
            return torch.split(self.embedding,self.padding.section,dim=1)[-1]
        if word not in self.word_wise_index:
            raise ValueError(f"word '{word}' is not in the prompt")
        if ord>=len(self.word_wise_index[word]):
            raise ValueError(f"word '{word}' is not used {ord+1} times in the prompt")
        
        res = torch.split(self.embedding,self.word_wise_index[word][ord].section,dim=1)
        return res[0] if len(res)==1 else res[1]
    
    def __getitem__(self,word:str) -> List[Span]:
        if word == 'EOS':
            word='<|endoftext|>'
        elif word == 'SOS':
            word='<|startoftext|>'
        if word not in self.word_wise_index:
            raise ValueError(f"word '{word}' is not in the prompt")
        return self.word_wise_index[word]
    

class EmbeddingEdit(nn.Module):
    def __init__(self,seq:int=77,dim:int=768):
        super().__init__()
        self.delta =torch.randn(1,seq,dim)
    def forward(self,text_embedding:PromptEmbedding,target_word:str,alpha:float,order:int=0,inplace:bool = True) -> PromptEmbedding:
        # order -> if same word is used multiple times in the prompt, which one to change
        self.delta =  self.delta.to(text_embedding.embedding.device)

        match target_word:
            case 'EOS':
                target_span =text_embedding.EOS
            case 'EOS+PAD':
                target_span = Span.merge([text_embedding.EOS,text_embedding.padding])
            case "PAD":
                target_span = text_embedding.padding
            case _:
                target_span = text_embedding[target_word][order]
        mask= target_span.mask.unsqueeze(0).unsqueeze(-1).expand_as(text_embedding.embedding)
        # print( target_span.mask.unsqueeze(0).unsqueeze(-1).expand_as(text_embedding.embedding).shape)
        if self.delta.shape[1]==1:
            delta = self.delta.expand_as(text_embedding.embedding)
        elif self.delta.shape[1]==MAX_LEN:
            delta = self.delta
        else:
            raise ValueError('delta shape is not valid')
        if inplace:
            text_embedding.embedding = text_embedding.embedding + (delta).to(text_embedding.embedding.device) * (mask * alpha).to(text_embedding.embedding.device,dtype=delta.dtype)
        else:
            text_embedding = text_embedding.embedding + (delta).to(text_embedding.embedding.device) * (mask * alpha).to(text_embedding.embedding.device,dtype=delta.dtype)
        
        return text_embedding
    def replace(self,text_embedding:PromptEmbedding, new_embedding:Float[torch.Tensor,'n s d'],target_word:str,order:int=0) -> PromptEmbedding:
        match target_word:
            case 'EOS':
                target_span =text_embedding.EOS
            case 'EOS+PAD':
                target_span = Span.merge([text_embedding.EOS,text_embedding.padding])
            case "PAD":
                target_span = text_embedding.padding
            case _:
                target_span = text_embedding[target_word][order]
        text_embedding.embedding=text_embedding.embedding.slice_scatter(new_embedding.to(text_embedding.embedding.device),dim=1,start=target_span.start,end=target_span.end+1) 

        return text_embedding
    
if __name__ == '__main__':
    # global DEBUG 
    DEBUG=True
    prompt = 'A person who is'
    prompt_embedding = PromptEmbedding(prompt)
    print(prompt_embedding)
