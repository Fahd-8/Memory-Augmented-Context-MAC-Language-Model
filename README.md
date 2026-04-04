# Memory-Augmented Context (MAC) Language Model

PyTorch implementation of Google's Titans MAC architecture, built from scratch. No pretrained models. Real Test-Time Training — the LMM updates its own weights while running.

## Status

**Done**

- Full Titans MAC architecture from scratch
- TTT with gradient-norm surprise detection, momentum, forgetting gate
- Memory compressed to single vector via mean pooling
- Learnable memory_pos — attention knows memory token is not a regular token
- 12-head causal attention with layer norm
- Two-level reset — reset_ttt between epochs, reset_memory between sequences
- LMM and Adam optimizer fully separated — no weight conflicts
- TTT ON vs OFF comparison proven working
- FastAPI backend with stateful memory across requests
- Next.js UI — chat panel + memory proof panel
- prove_ttt.py — mathematical proof TTT is real

**In progress**

- Training larger model (dim=768, 6 layers, ~200M params) on Colab T4

**Planned**

- Full end-to-end demo with trained model
- Public deployment
- Memory recall benchmark
- Investor demo with live proof

---

## The problem

Every LLM today forgets everything outside its context window. Each conversation starts from zero. There is no persistent memory — just a sliding window pretending to be one.

## The solution

A Long-term Memory Module (LMM) that compresses past context into its own weights via Test-Time Training. The model learns while it runs. Surprising tokens update the LMM. Boring tokens get skipped. Memory is always exactly one extra token handed to attention — sequence length never grows.

---

## Architecture

Input Tokens
↓
Embeddings — 768-dim, sine/cosine positional encoding
↓
MAC Layer 1
├── LMM — TTT runs, surprise detected, weights update, mean pool → [1, 768]
├── memory_pos added — stamps memory with unique positional identity
├── cat([tokens, memory]) → [seq_len+1, 768]
├── 12-head Attention — causal mask
├── LayerNorm
└── Residual connection
↓
MAC Layers 2-5 — same structure
↓
MAC Layer 6 — output projection → logits (50,257 vocab)

## LMM — how TTT works

The LMM is a 4-layer MLP. Memory lives in its weights, not in a vector or KV cache.
768 → 2048 → 2048 → 2048 → 768

Per token:

1. Forward pass through LMM → MSE loss against input token
2. Compute gradient norm → surprise score
3. Apply momentum → smoothed surprise over recent context
4. If surprise > threshold → apply weight decay (forgetting gate) → update weights
5. Otherwise → skip, token wasn't worth remembering

After processing all tokens, mean pool the outputs → single [1, 768] memory summary → handed to attention.

LMM weights are owned entirely by TTT's SGD. Adam never touches them.

## Model specs


|                 |            |
| --------------- | ---------- |
| Embedding dim   | 768        |
| LMM hidden dim  | 2048       |
| LMM layers      | 4          |
| Attention heads | 12         |
| Head dim        | 64         |
| MAC layers      | 6          |
| Layer norm      | yes        |
| Vocab size      | 50,257     |
| Context length  | 256 tokens |
| Parameters      | ~200M      |


## Optimizer separation


| Component                         | Optimizer          |
| --------------------------------- | ------------------ |
| Attention, embeddings, memory_pos | Adam — lr=0.0001   |
| LMM weights                       | TTT SGD — lr=0.001 |


Two optimizers. Two jobs. No conflicts.

## Reset strategy


| Call           | When              | What it resets                  |
| -------------- | ----------------- | ------------------------------- |
| reset_ttt()    | Between epochs    | Momentum buffer + TTT optimizer |
| reset_memory() | Between sequences | Momentum buffer only            |


TTT optimizer state persists across sequences within an epoch — the LMM builds surprise trends over time.

---

## Proof it works

```bash
python experiments/prove_ttt.py
```

Three proofs:

**Proof 1 — LMM weights actually change**
TTT ON:  weight change = 0.00847291
TTT OFF: weight change = 0.00000000

**Proof 2 — memory token influences output**
output difference with vs without memory = 0.03421847

**Proof 3 — surprise detection is real**
boring text    weight change = 0.00012453
surprising text weight change = 0.00931847

---

## Usage

### Install

```bash
pip install torch transformers datasets fastapi uvicorn
```

### Train

```bash
python experiments/run.py
```

### Generate

```bash
python inference/generate.py
```

Runs TTT ON and TTT OFF side by side automatically.

### Prove TTT works

```bash
python experiments/prove_ttt.py
```

### API

```bash
python api/main.py
```


| Endpoint  | Method | What it does                    |
| --------- | ------ | ------------------------------- |
| /health   | GET    | Check model is loaded           |
| /generate | POST   | Generate with LMM memory active |
| /compare  | POST   | Same prompt, TTT ON vs OFF      |
| /reset    | POST   | Wipe memory, fresh start        |


### UI

```bash
cd ui && npm install && npm run dev
```

---

## File structure

├── model/
│   ├── lmm.py           — LMM, TTT, surprise detection, forgetting gate
│   ├── attention.py     — 12-head causal attention
│   ├── mac_layer.py     — MAC_Layer, DeepMAC, layer norm
│   └── embeddings.py    — token + positional embeddings
├── training/
│   └── train.py         — training loop, optimizer separation, Drive saving
├── inference/
│   └── generate.py      — top-p sampling, TTT ON/OFF comparison
├── experiments/
│   ├── run.py           — entry point
│   └── prove_ttt.py     — mathematical proof of TTT and MAC
├── api/
│   ├── main.py          — FastAPI app
│   ├── model_loader.py  — stateful model, memory persists across requests
│   └── schemas.py       — request/response types
├── ui/
│   ├── app/             — Next.js pages
│   ├── components/      — ChatPanel, MemoryProof, MessageBubble
│   └── lib/             — API client
└── data/
└── dataset.py       — TinyStories tokenization

---

## MAC vs MAL

The Titans paper proposes two integration approaches:

**MAC — Memory as Context** (this implementation)
LMM produces a memory vector. It gets prepended to the input as an extra token. Attention sees memory alongside real tokens and decides how much to use it. Memory is visible. Attention has a choice.

**MAL — Memory as a Layer**
LMM sits inline between input and attention. Tokens flow through it and get transformed before attention sees them. Memory is invisible. Attention has no choice.

MAC was chosen because giving attention the option to ignore memory produces better results for open-ended generation.

---

## Branches


| Branch      | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| mac-scratch | Pure Titans MAC from scratch — this branch                   |
| qwen-mac    | Experimental — MAC wrapped around frozen Qwen2-1.5B-Instruct |


---

## References

- [Titans + MIRAS — Google Research, 2025](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [TinyStories — Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)

