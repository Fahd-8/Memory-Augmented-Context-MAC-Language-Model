import torch
import torch.nn as nn
from model.lmm import LMM
from model.attention import Attention

class MAC_Layer(nn.Module):
    def __init__(self, dim=512, vocab_size=None):
        super().__init__()
        self.lmm = LMM(dim)
        self.attention = Attention(dim)
        self.output_proj = nn.Linear(dim, vocab_size) if vocab_size else None

    def forward(self, tokens):
        memory_vecs = self.lmm(tokens)
        combined = torch.cat([tokens, memory_vecs], dim=0)
        output, weights = self.attention(combined)
        hidden = output[:len(tokens)]
        if self.output_proj:
            return self.output_proj(hidden), weights
        return hidden, weights


class DeepMAC(nn.Module):
    def __init__(self, num_layers=3, dim=512, vocab_size=50257):
        super().__init__()
        self.mac_layers = nn.ModuleList([
            MAC_Layer(dim, vocab_size=None) for _ in range(num_layers - 1)
        ])
        self.final_mac = MAC_Layer(dim, vocab_size=vocab_size)

    def forward(self, tokens):
        x = tokens
        for mac in self.mac_layers:
            hidden, _ = mac(x)
            x = x + hidden
        return self.final_mac(x)