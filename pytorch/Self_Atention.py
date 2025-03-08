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

encoding_matrix=torch.tensor([[1.16,0.23],[0.57,1.36],[4.41,-2.16]])
torch.manual_seed(42)
self_attention=SelfAttention(d_model=2) #Initializing Object
attention=self_attention(encoding_matrix) #Forward method called 
print(attention)

    

