import torch.nn as nn

def get_embedding(vocab_size, embed_dim=512):
    embed = nn.Embedding(vocab_size, embed_dim)
    embed.weight.requires_grad = True
    return embed