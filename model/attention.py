import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768 / 12 = 64 per head

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # projects all heads back to original dim after concat
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        seq_len = x.shape[0]

        q = self.query(x)  # [seq_len, dim]
        k = self.key(x)
        v = self.value(x)

        # split into heads — reshape to [num_heads, seq_len, head_dim]
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # attention scores per head
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)

        # weighted sum per head
        out = torch.matmul(attn_weights, v)  # [num_heads, seq_len, head_dim]

        # concat all heads back together
        out = out.transpose(0, 1).contiguous().view(seq_len, self.dim)

        # final projection
        output = self.out_proj(out)

        return output, attn_weights