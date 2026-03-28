import torch

def generate_next_word(prompt, mac, embed, tokenizer, device='cuda'):
    token_ids = tokenizer.encode(prompt)
    seq_vecs = embed(torch.tensor(token_ids).to(device))

    with torch.no_grad():
        logits, _ = mac(seq_vecs)
        next_token_id = torch.argmax(logits[-1]).item()
        next_word = tokenizer.decode([next_token_id])

    return next_word

def run_tests(mac, embed, tokenizer, device='cuda'):
    test_prompts = [
        "Once upon a time",
        "The little girl",
        "One day there was",
        "A cat and a dog",
        "The boy wanted to"
    ]

    print("\nTesting MAC Generation:\n")
    for prompt in test_prompts:
        next_word = generate_next_word(prompt, mac, embed, tokenizer, device)
        print(f"'{prompt}' → '{next_word}'")