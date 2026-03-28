import torch
import torch.nn as nn
from model.embeddings import get_embedding

def train(mac, sequences, tokenizer, epochs=50, lr=0.0003, device='cuda'):
    embed = get_embedding(tokenizer.vocab_size).to(device)
    mac = mac.to(device)

    optimizer = torch.optim.Adam(
        list(mac.parameters()) + list(embed.parameters()), 
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining MAC on {len(sequences)} sequences for {epochs} epochs...\n")

    for epoch in range(epochs):
        total_loss = 0

        for token_ids in sequences:
            seq_vecs = embed(torch.tensor(token_ids).to(device))

            logits, _ = mac(seq_vecs)

            loss = 0
            for i in range(len(token_ids) - 1):
                pred_logits = logits[i]
                target_id = torch.tensor([token_ids[i + 1]]).to(device)
                loss += criterion(pred_logits.unsqueeze(0), target_id)

            loss = loss / (len(token_ids) - 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mac.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.2f}")

    print("\nTraining complete!")
    return mac, embed