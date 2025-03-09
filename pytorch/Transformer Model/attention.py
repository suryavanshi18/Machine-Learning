import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()

        #Creating weight matrix
        #No bias term as per the paper
        self.w_q=nn.Linear(d_model,d_model,bias=False) 
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)

    def forward(self,token_encoding):
        #token_encoding->Word Embedding + Positional Encoding
        q=self.w_q(token_encoding)
        k=self.w_k(token_encoding)
        v=self.w_v(token_encoding)
        #0->rows 1->cols
        sims=torch.matmul(q,k.transpose(dim0=0,dim1=1))
        scaled_sims=sims/torch.tensor(k.size(1)**0.5)
        attention_percent=F.softmax(scaled_sims,dim=1)
        attention=torch.matmul(attention_percent,v)

        return attention

class MaskedSelfAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)
        self.w_q=nn.Linear(d_model,d_model,bias=False)
    
    def forward(self,token_encoding,mask=None):
        q=self.w_q(token_encoding)
        v=self.w_v(token_encoding)
        k=self.w_k(token_encoding)

        sims=torch.matmul(q,k.transpose(0,1))
        scaled_sims=sims/torch.tensor(k.size(1)**0.5)
        if mask is not None:
            #Replaces True masked values with very larg negative value
            #Replaces False masked value with 0
            scaled_sims.masked_fill(mask=mask,value=-1e9)
        attention_percent=F.softmax(scaled_sims,dim=1)
        attention_score=torch.matmul(attention_percent,v)
        return attention_score


encoding_matrix=torch.tensor([[1.16,0.23],[0.57,1.36],[4.41,-2.16]])
torch.manual_seed(42)
self_attention=SelfAttention(d_model=2) #Initializing Object
self_attention_score=self_attention(encoding_matrix) #Forward method called 
print(self_attention_score)
mask=torch.tril(torch.ones(3,3))
mask=mask==0
masked_attention=MaskedSelfAttention(d_model=2)
masked_attention_score=masked_attention(encoding_matrix,mask=mask)
print(masked_attention_score)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_heads):
        super().__init__()
        self.heads=nn.ModuleList([SelfAttention(d_model) for _ in range(num_heads)])

    def forward(self,encoded_matrix):
        return torch.cat([head(encoded_matrix) for head in self.heads],dim=1)

multiheadAttention=MultiHeadAttention(d_model=2,num_heads=2)

mul_score=multiheadAttention(encoding_matrix)

print(mul_score)
    

