import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from model.embeddings import get_embedding

def train(mac, sequences, tokenizer, epochs=20, lr=0.0001, device='cuda', save_dir='checkpoints'):
    embed = get_embedding(tokenizer.vocab_size, embed_dim=768).to(device)
    mac = mac.to(device)

    non_lmm_params = [
        p for name, p in mac.named_parameters()
        if 'lmm' not in name
    ]

    optimizer = torch.optim.Adam(
        non_lmm_params + list(embed.parameters()),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    print(f"\nTraining MAC on {len(sequences)} sequences for {epochs} epochs...\n")

    best_loss = float('inf')

    for epoch in range(epochs):
        mac.reset_ttt()
        total_loss = 0

        for token_ids in sequences:
            mac.reset_memory()

            seq_vecs = embed(torch.tensor(token_ids).to(device))

            with autocast():
                logits, _ = mac(seq_vecs, do_ttt=True)
                targets = torch.tensor(token_ids[1:]).to(device)
                loss = criterion(logits[:-1], targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                non_lmm_params + list(embed.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(sequences)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'mac_state_dict': mac.state_dict(),
                'embed_state_dict': embed.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'{save_dir}/mac_best.pt')
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

            drive_dir = '/content/drive/MyDrive/MAC_checkpoints'
            if os.path.exists('/content/drive/MyDrive'):
                os.makedirs(drive_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'mac_state_dict': mac.state_dict(),
                    'embed_state_dict': embed.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, f'{drive_dir}/mac_best.pt')
                print(f"  ✓ Also saved to Google Drive")

    torch.save({
        'epoch': epochs,
        'mac_state_dict': mac.state_dict(),
        'embed_state_dict': embed.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'{save_dir}/mac_final.pt')

    if os.path.exists('/content/drive/MyDrive'):
        torch.save({
            'epoch': epochs,
            'mac_state_dict': mac.state_dict(),
            'embed_state_dict': embed.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{drive_dir}/mac_final.pt')

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to {save_dir}/")

    return mac, embed