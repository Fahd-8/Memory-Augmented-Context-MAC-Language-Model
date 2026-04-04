import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Tokenizer
from model.mac_layer import DeepMAC
from model.embeddings import get_embedding

def prove_ttt(checkpoint_path='checkpoints/mac_best.pt'):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Loading model from {checkpoint_path}...\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    mac = DeepMAC(num_layers=6, dim=768, vocab_size=tokenizer.vocab_size)
    embed = get_embedding(tokenizer.vocab_size, embed_dim=768)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    mac.load_state_dict(checkpoint['mac_state_dict'])
    embed.load_state_dict(checkpoint['embed_state_dict'])

    mac = mac.to(device)
    embed = embed.to(device)

    # ── proof 1: TTT weight change ────────────────────────────────────────
    print("=" * 60)
    print("PROOF 1 — LMM weights change during TTT")
    print("=" * 60)

    test_text = "Once upon a time there was a little girl named Lucy who loved to explore the forest near her home."
    token_ids = tokenizer.encode(test_text)
    seq_vecs = embed(torch.tensor(token_ids).to(device))

    # snapshot weights before
    weights_before = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]

    # TTT ON — weights should change
    mac.reset_ttt()
    mac.reset_memory()
    mac(seq_vecs, do_ttt=True)

    weights_after_ttt_on = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]

    change_ttt_on = sum(
        (a - b).abs().mean().item()
        for a, b in zip(weights_after_ttt_on, weights_before)
    )

    # TTT OFF — weights should NOT change
    mac.reset_ttt()
    mac.reset_memory()
    weights_before_off = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]

    mac(seq_vecs, do_ttt=False)

    weights_after_ttt_off = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]

    change_ttt_off = sum(
        (a - b).abs().mean().item()
        for a, b in zip(weights_after_ttt_off, weights_before_off)
    )

    print(f"LMM weight change TTT ON:  {change_ttt_on:.8f}")
    print(f"LMM weight change TTT OFF: {change_ttt_off:.8f}")

    if change_ttt_on > 0 and change_ttt_off == 0.0:
        print("PROOF 1 PASSED — TTT is real, weights update only when TTT is on")
    else:
        print("PROOF 1 FAILED — check TTT implementation")

    # ── proof 2: memory token influences output ───────────────────────────
    print("\n" + "=" * 60)
    print("PROOF 2 — Memory token influences generation")
    print("=" * 60)

    mac.reset_ttt()
    mac.reset_memory()

    # run with memory
    output_with_memory, _ = mac(seq_vecs, do_ttt=True)

    # run without memory — zero out memory_pos so memory token is silent
    mac.reset_ttt()
    mac.reset_memory()
    original_memory_pos = mac.final_mac.memory_pos.data.clone()
    mac.final_mac.memory_pos.data.zero_()
    output_without_memory, _ = mac(seq_vecs, do_ttt=False)

    # restore memory_pos
    mac.final_mac.memory_pos.data.copy_(original_memory_pos)

    diff = (output_with_memory - output_without_memory).abs().mean().item()
    print(f"Output difference with vs without memory: {diff:.8f}")

    if diff > 0:
        print("PROOF 2 PASSED — memory token is influencing generation")
    else:
        print("PROOF 2 FAILED — memory token has no effect")

    # ── proof 3: surprise detection working ──────────────────────────────
    print("\n" + "=" * 60)
    print("PROOF 3 — Surprise detection working")
    print("=" * 60)

    # boring text — low surprise expected
    boring_text = "the the the the the the the the the the the the the the"
    boring_ids = tokenizer.encode(boring_text)
    boring_vecs = embed(torch.tensor(boring_ids).to(device))

    mac.reset_ttt()
    mac.reset_memory()
    weights_before_boring = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]
    mac(boring_vecs, do_ttt=True)
    weights_after_boring = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]
    boring_change = sum(
        (a - b).abs().mean().item()
        for a, b in zip(weights_after_boring, weights_before_boring)
    )

    # surprising text — high surprise expected
    surprising_text = "Suddenly the quantum dragon teleported to Mars and discovered ancient cryptocurrency."
    surprising_ids = tokenizer.encode(surprising_text)
    surprising_vecs = embed(torch.tensor(surprising_ids).to(device))

    mac.reset_ttt()
    mac.reset_memory()
    weights_before_surprising = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]
    mac(surprising_vecs, do_ttt=True)
    weights_after_surprising = [p.clone().detach() for p in mac.final_mac.lmm.net.parameters()]
    surprising_change = sum(
        (a - b).abs().mean().item()
        for a, b in zip(weights_after_surprising, weights_before_surprising)
    )

    print(f"Weight change on boring text:     {boring_change:.8f}")
    print(f"Weight change on surprising text: {surprising_change:.8f}")

    if surprising_change > boring_change:
        print("PROOF 3 PASSED — surprising text causes more weight updates than boring text")
    else:
        print("PROOF 3 FAILED — surprise detection not working as expected")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TTT weight change:        {change_ttt_on:.8f}")
    print(f"Memory influence on output: {diff:.8f}")
    print(f"Boring vs surprising delta: {surprising_change - boring_change:.8f}")
    print("=" * 60)

if __name__ == "__main__":
    prove_ttt()