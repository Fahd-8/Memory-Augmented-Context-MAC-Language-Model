import torch
import torch.nn as nn
from model.lmm import LMM
from model.attention import Attention

class MAC_Layer(nn.Module):
    def __init__(self, dim=768, vocab_size=None):
        super().__init__()
        self.lmm = LMM(dim)
        self.attention = Attention(dim)
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size) if vocab_size else None

        # learnable memory position — gives memory token a unique identity
        self.memory_pos = nn.Parameter(torch.zeros(1, dim))


    def forward(self, tokens, do_ttt=False):
        # memory_summary is always shape [1, dim] regardless of sequence length
        memory_summary = self.lmm(tokens, do_ttt=do_ttt)

        # stamp memory with its own positional identity
        memory_summary = memory_summary + self.memory_pos

        # combined is always seq_len + 1
        combined = torch.cat([tokens, memory_summary], dim=0)

        output, weights = self.attention(combined)

        # slice back only the token positions, discard memory position
        hidden = output[:len(tokens)]

        # layer norm for stability
        hidden = self.norm(hidden)

        if self.output_proj:
            return self.output_proj(hidden), weights
        return hidden, weights

    def reset_memory(self):
        # light reset between sequences — keeps TTT optimizer state
        self.lmm.reset_memory_state()

    def reset_ttt(self):
        # full reset between epochs
        self.lmm.reset_ttt()


class DeepMAC(nn.Module):
    def __init__(self, num_layers=6, dim=768, vocab_size=50257):
        super().__init__()
        self.mac_layers = nn.ModuleList([
            MAC_Layer(dim, vocab_size=None) for _ in range(num_layers - 1)
        ])
        self.final_mac = MAC_Layer(dim, vocab_size=vocab_size)

    def forward(self, tokens, do_ttt=False):
        x = tokens
        for mac in self.mac_layers:
            hidden, _ = mac(x, do_ttt=do_ttt)
            x = x + hidden
        return self.final_mac(x, do_ttt=do_ttt)

    def reset_memory(self):
        # light reset between sequences
        for mac in self.mac_layers:
            mac.reset_memory()
        self.final_mac.reset_memory()

    def reset_ttt(self):
        # full reset between epochs
        for mac in self.mac_layers:
            mac.reset_ttt()
        self.final_mac.reset_ttt()