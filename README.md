# Memory-Augmented Context (MAC) Language Model

A PyTorch implementation of a Memory-Augmented Context (MAC) architecture for text generation, inspired by Google's Titans research on neural long-term memory modules.

## Overview

This project implements a hybrid neural architecture that combines:
- Long-term Memory Modules (LMM) for compressed memory storage
- Multi-head Attention mechanisms for contextual processing
- Residual connections for stable deep learning
- Token-by-token autoregressive text generation

The model is trained on the TinyStories dataset and demonstrates the core concepts behind modern memory-augmented language models.

## Architecture

### Components

**Long-term Memory Module (LMM)**
- 5-layer MLP (512 → 1024 → 1024 → 1024 → 512)
- Processes and compresses contextual information
- ~3M parameters

**Attention Layer**
- Query, Key, Value projections (512-dim each)
- Scaled dot-product attention
- ~786K parameters

**MAC Layer**
- Combines LMM memory with attention-based processing
- Integrates both compressed memory and precise attention

**DeepMAC**
- Stacks multiple MAC layers with residual connections
- 3 stacked MACs = 18 total layers
- ~80M total parameters

### Key Features

- **Residual Connections**: Enable stable training of deep architectures
- **Unfrozen Embeddings**: Allows adaptation to specific domains
- **GPU Acceleration**: CUDA-optimized training
- **Gradient Clipping**: Prevents exploding gradients
- **Token-by-token Generation**: Autoregressive language modeling

## Model Specifications

- **Embedding Dimension**: 512
- **Hidden Dimension**: 1024
- **Number of Layers**: 18 (3 stacked MACs)
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Parameters**: ~80 million
- **Context Length**: 50 tokens per sequence

## Training

### Dataset
- TinyStories (100 stories subset)
- Tokenized using GPT-2 tokenizer
- Sequences split into 50-token chunks

### Hyperparameters
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Processing: Sequential (single sequence at a time)
- Gradient Clipping: 1.0
- Loss Function: CrossEntropyLoss

### Training Results
- Initial Loss: ~2083
- Final Loss (50 epochs): ~427
- Training Time: ~15 minutes on GPU

## Usage

### Requirements
```bash
pip install torch transformers datasets
```

### Training
```python
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load dataset and tokenizer
dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize model
mac = DeepMAC(num_layers=3, dim=512, vocab_size=tokenizer.vocab_size)
mac = mac.to('cuda')

# Train (see full training code in repository)
```

### Text Generation
```python
def generate_next_word(prompt):
    token_ids = tokenizer.encode(prompt)
    seq_vecs = embed(torch.tensor(token_ids).to(device))
    
    with torch.no_grad():
        logits, _ = mac(seq_vecs)
        next_token_id = torch.argmax(logits[-1]).item()
        next_word = tokenizer.decode([next_token_id])
    
    return next_word

# Example
print(generate_next_word("Once upon a time"))  # Output: "there"
```

## Technical Implementation

### Memory Processing Flow
```
Input Tokens
    ↓
Embeddings (512-dim)
    ↓
MAC Layer 1 (LMM + Attention + Residual)
    ↓
MAC Layer 2 (LMM + Attention + Residual)
    ↓
MAC Layer 3 (LMM + Attention + Output Projection)
    ↓
Logits (50K vocab)
    ↓
Next Token Prediction
```

### Residual Connection Pattern
Each intermediate MAC layer uses skip connections:
```python
hidden, _ = mac(x)
x = x + hidden  # Residual connection
```

This preserves gradient flow and enables stable training of deep networks.

## Theoretical Background

This implementation draws inspiration from:
- **Titans (Google, 2024)**: Neural long-term memory modules with test-time learning
- **MIRAS Framework**: Unified view of sequence modeling as associative memory
- **Transformer Architecture**: Multi-head attention mechanisms
- **ResNet**: Residual connections for deep networks

### Memory-Augmented Context (MAC)
The MAC architecture combines:
1. **Short-term Memory**: Attention over recent tokens (precise but limited)
2. **Long-term Memory**: Neural memory module (compressed but scalable)
3. **Persistent Memory**: Learned task-specific parameters

This hybrid approach aims to balance the efficiency of linear RNNs with the expressiveness of attention mechanisms.

## Limitations

- Small training dataset (100 stories)
- Limited context window (50 tokens)
- Single-token generation (no beam search or sampling strategies)
- No test-time training implemented yet
- Basic training setup (no learning rate scheduling, mixed precision, etc.)

## Future Enhancements

Potential improvements include:
- Test-time training with surprise metrics (Titans approach)
- Larger training dataset (full TinyStories or other corpora)
- Multi-token generation with sampling strategies
- Learning rate scheduling and warmup
- Layer normalization
- Positional encodings
- Evaluation on standard benchmarks

## Performance

### Training Progress
| Phase | Changes | Best Loss |
|-------|---------|-----------|
| Baseline | No causal mask, no positional encoding | 1.48 |
| Phase 1 | Causal masking + positional encodings | 0.58 |

### Sample Generations (Phase 1 Model)

| Prompt | Generated Text |
|--------|---------------|
| "Once upon a time" | "Once upon a time, there was a little boy named Joe. He had 8 small legs exploring in the hill outside..." |
| "The little girl" | "The little girl was very fast that she, and promised to the warrior quickly..." |
| "One day there was" | "One day there was sitting on the grass in a girl named Susie. She wanted to run..." |
| "The boy wanted to" | "The boy wanted to its rightful warrior the kitty was so he ran for a..." |

### Observations
- Model learned character names (Joe, Susie, Lily, Cindy, Jim)
- Model learned story structure (One day... Then... But...)
- Model learned dialogue patterns ("I don't want...", "That's wonderful...")
- Grammar still inconsistent — expected with 100 training stories
- Coherence improves significantly after Phase 1 fixes

### Next: Phase 2 (TTT)
Test-Time Training will allow memory weights to adapt during inference,
enabling the model to maintain coherent narrative throughout generation.

## References

- Titans + MIRAS: Helping AI have long-term memory (Google Research, 2024)
- Attention Is All You Need (Vaswani et al., 2017)
- Deep Residual Learning (He et al., 2015)
- TinyStories Dataset (Eldan & Li, 2023)
