import torch
import torch.nn as nn
import math

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, max_seq_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.token_embed.weight.requires_grad = True

        # positional encoding — fixed sine/cosine (no parameters needed)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer — moves with model to device but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, token_ids):
        seq_len = token_ids.shape[0]
        token_vecs = self.token_embed(token_ids)
        return token_vecs + self.pe[:seq_len]


def get_embedding(vocab_size, embed_dim=768, max_seq_len=512):
    return EmbeddingWithPosition(vocab_size, embed_dim, max_seq_len)