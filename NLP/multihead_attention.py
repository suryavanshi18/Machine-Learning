import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return np.dot(X,W_q),np.dot(X,W_k),np.dot(X,W_v)
def calc_softmax(X:np.ndarray):
    mx=np.max(X,axis=-1,keepdims=True)
    exp_x = np.exp(X - mx)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)    
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    _,d=Q.shape
    soft_max=calc_softmax(np.dot(Q,K.T)/np.sqrt(d))
    return np.dot(soft_max,V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len,d_model=Q.shape
    d_k = d_model // n_heads
    heads = []
    for i in range(n_heads):
        s = slice(i * d_k, (i + 1) * d_k)
        heads.append(self_attention(Q[:, s], K[:, s], V[:, s]))
    return np.concatenate(heads, axis=-1)
