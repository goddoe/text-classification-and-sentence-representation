import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embed_dim (int): dimension of embedding.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = embed_dim

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        """Single head naive self-attention.
        Args:
            X (torch.Tensor): inputs, shape of (batch_size, sequence).
        Returns:
            torch.tensor, Sentence representation.
        """
        batch_size, seq_len = X.size()
        X = self.embeddings(X) # batch size x seq_len x embed_dim
        
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X) # batch_size x seq_len
        
        attention_score_raw = q @ k.transpose(-2,-1) / math.sqrt(self.embed_dim) # batch_size x seq_len x seq_len
        
        attention_score = torch.softmax(attention_score_raw, dim=2) # batch_size x seq_len x seq_len
        
        weighted_sum = attention_score @ v # batch_size x seq_len x dim

        context = torch.mean(weighted_sum, dim=1) # self attention된 임베딩들을 average
        
        return context
