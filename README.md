# Memory-Augmented Context (MAC) Language Model

A from-scratch PyTorch implementation of the Memory-Augmented Context (MAC) architecture inspired by Google's Titans paper. Combines neural long-term memory with multi-head attention to enable real test-time learning and compressed memory retrieval.

## What makes this different

Most language models forget everything outside their context window. This implementation gives the model a genuine long-term memory module (LMM) that:

- **Learns while running** — Test-Time Training (TTT) updates LMM weights on the fly during both training and inference
- **Detects what matters** — gradient-norm surprise metric identifies novel tokens worth remembering
- **Compresses intelligently** — entire past context compressed into a single memory vector, not mirrored token-by-token
- **Stays fast** — memory is always exactly +1 token to attention, sequence length never explodes

## Architecture

### Flow
```
Input Tokens
    ↓
Embeddings (512-dim, sine/cosine positional encoding)
    ↓
MAC Layer 1
    ├── LMM (TTT) → surprise detection → weight update → mean pool → [1, 512]
    ├── memory_pos (learnable) → stamps memory with unique identity
    ├── cat([tokens, memory]) → [seq_len+1, 512]
    ├── 8-head Attention (causal mask)
    └── Residual connection
    ↓
MAC Layer 2 (same structure)
    ↓
MAC Layer 3 + Output Projection → Logits (50,257 vocab)
    ↓
Next Token Prediction
```

### Long-term Memory Module (LMM)

The LMM is a 4-layer MLP that stores memory in its own weights via TTT:
```
dim=512 → 1024 → 1024 → 1024 → 512
```

**TTT pipeline per token:**
1. Forward pass → compute MSE loss against input token
2. Gradient norm → surprise magnitude
3. Exponential momentum → smoothed surprise over recent context
4. Threshold check → only surprising tokens update memory
5. Weight decay → forgetting gate, discards stale memory
6. Weight update → LMM weights adapt in real time

Memory is compressed to a single summary vector via mean pooling before being handed to attention. LMM weights are owned exclusively by TTT — excluded from the outer Adam optimizer to prevent conflicts.

### Multi-head Attention

- 8 heads, 64 dims per head
- Scaled dot-product attention
- Causal mask for autoregressive generation
- Output projection blends all heads

### MAC Layer

- LMM produces `[1, 512]` memory summary
- Learnable `memory_pos` embedding stamps memory with a distinct positional identity so attention knows it's not a regular token
- Combined sequence is always `seq_len + 1` — memory never bloats the sequence
- After attention, token positions are sliced back — memory position discarded

### DeepMAC

- 3 stacked MAC layers
- Residual connections between layers (`x = x + hidden`)
- Final layer has output projection to vocab size

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Embedding dim | 512 |
| LMM hidden dim | 1024 |
| LMM layers | 4 |
| Attention heads | 8 |
| Head dim | 64 |
| MAC layers | 3 |
| Vocab size | 50,257 (GPT-2) |
| Context length | 50 tokens |

## Training

### Reset strategy

TTT has two levels of reset — this is important for correct behavior:

- `reset_ttt()` — called between epochs. Full reset including TTT optimizer. LMM starts fresh each epoch.
- `reset_memory()` — called between sequences. Only clears momentum buffer. TTT optimizer state persists across sequences within an epoch, allowing the LMM to build surprise trends over time.

### Optimizer separation

| Component | Optimizer |
|-----------|-----------|
| Attention, embeddings, memory_pos | Adam (lr=0.0003) |
| LMM weights | TTT's own SGD (lr=0.001) |

LMM weights are completely excluded from Adam. Two optimizers, two jobs, no conflicts.

### Dataset

- TinyStories (roneneldan/TinyStories)
- GPT-2 tokenizer
- 50-token chunks

## Usage

### Install
```bash
pip install torch transformers datasets
```

### Train
```bash
python experiments/run.py
```

### Generate
```bash
python inference/generate.py
```

Runs both TTT ON and TTT OFF comparisons automatically.

### Generate in code
```python
from inference.generate import load_model, generate_text

mac, embed, tokenizer = load_model('checkpoints/mac_best.pt', device='mps')

output = generate_text(
    prompt="Once upon a time",
    mac=mac,
    embed=embed,
    tokenizer=tokenizer,
    device='mps',
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.85,
    repetition_penalty=1.8,
    do_ttt=True
)
print(output)
```

## File Structure
```
Memory-Augmented-Context-MAC-Language-Model/
├── model/
│   ├── lmm.py          — LMM with TTT, surprise detection, forgetting gate
│   ├── attention.py    — 8-head causal attention
│   ├── mac_layer.py    — MAC_Layer and DeepMAC
│   └── embeddings.py   — token + sine/cosine positional embeddings
├── training/
│   └── train.py        — training loop with TTT/Adam separation
├── inference/
│   └── generate.py     — text generation with top-p sampling
├── data/
│   └── dataset.py      — TinyStories loading and tokenization
├── experiments/
│   └── run.py          — entry point
└── checkpoints/        — saved model weights
```

## Theoretical Background

This implementation is based on:

- **Titans (Google, 2025)** — Neural long-term memory modules with test-time learning and surprise-based memory updates
- **MIRAS Framework** — Unified view of sequence models as associative memory with four design axes: memory architecture, attentional bias, retention gate, memory algorithm
- **Transformer (Vaswani et al., 2017)** — Multi-head attention
- **ResNet (He et al., 2015)** — Residual connections

### MAC vs MAL

The Titans paper proposes two ways to integrate the LMM:

**MAC (Memory as Context)** — this implementation. LMM produces a memory vector that is concatenated to input as an extra token. Attention sees memory alongside real tokens and decides how much to attend to it. Memory is visible and optional.

**MAL (Memory as a Layer)** — alternative approach. LMM sits inline in the pipeline. Tokens flow through it and are transformed before reaching attention. Memory effect is mandatory and invisible.

MAC was chosen because it gives attention the choice to use or ignore memory, which is more appropriate for open-ended generation tasks.

## References

- [Titans + MIRAS: Helping AI have long-term memory — Google Research](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [TinyStories — Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)