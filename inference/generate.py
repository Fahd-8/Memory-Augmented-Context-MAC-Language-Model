import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model.mac_layer import DeepMAC
from model.embeddings import get_embedding

def load_model(checkpoint_path='checkpoints/mac_best.pt', device='mps'):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    mac = DeepMAC(num_layers=3, dim=512, vocab_size=tokenizer.vocab_size)
    embed = get_embedding(tokenizer.vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mac.load_state_dict(checkpoint['mac_state_dict'])
    embed.load_state_dict(checkpoint['embed_state_dict'])
    mac = mac.to(device)
    embed = embed.to(device)
    print(f"Model loaded from {checkpoint_path}")
    return mac, embed, tokenizer

def generate_text(prompt, mac, embed, tokenizer, device='mps', max_new_tokens=50, temperature=0.9, top_p=0.9, repetition_penalty=1.3, do_ttt=True):
    mac.eval()
    mac.reset_ttt()    # fresh TTT state for new prompt
    mac.reset_memory()

    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        seq_vecs = embed(torch.tensor(generated).to(device))
        logits, _ = mac(seq_vecs, do_ttt=do_ttt)
        next_logits = logits[-1] / temperature

        # repetition penalty
        for token_id in set(generated):
            next_logits[token_id] /= repetition_penalty

        # top-p sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_logits[indices_to_remove] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)

def run_tests(mac, embed, tokenizer, device='mps', do_ttt=True):
    test_prompts = [
        "Once upon a time",
        "The little girl",
        "One day there was",
        "A cat and a dog",
        "The boy wanted to"
    ]

    print(f"\nTesting MAC Generation (TTT={'on' if do_ttt else 'off'}):\n")

    for prompt in test_prompts:
        generated = generate_text(
            prompt, mac, embed, tokenizer, device,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.85,
            repetition_penalty=1.8,
            do_ttt=do_ttt
        )
        print(f"Prompt:    '{prompt}'")
        print(f"Generated: '{generated}'")
        print()

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    mac, embed, tokenizer = load_model(device=device)

    print("=== TTT ON ===")
    run_tests(mac, embed, tokenizer, device=device, do_ttt=True)

    print("=== TTT OFF ===")
    run_tests(mac, embed, tokenizer, device=device, do_ttt=False)