import torch

def generate_text(prompt, mac, embed, tokenizer, device='mps', max_new_tokens=30, temperature=0.8):
    mac.eval()
    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_vecs = embed(torch.tensor(generated).to(device))
            logits, _ = mac(seq_vecs)

            # Apply temperature
            next_logits = logits[-1] / temperature

            # Sample from distribution instead of always taking max
            probs = torch.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token_id)

            # Stop at end of sentence
            if next_token_id == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(generated)
    return full_text


def run_tests(mac, embed, tokenizer, device='mps'):
    test_prompts = [
        "Once upon a time",
        "The little girl",
        "One day there was",
        "A cat and a dog",
        "The boy wanted to"
    ]

    print("\nTesting MAC Generation:\n")
    for prompt in test_prompts:
        generated = generate_text(prompt, mac, embed, tokenizer, device)
        print(f"Prompt:    '{prompt}'")
        print(f"Generated: '{generated}'")
        print()