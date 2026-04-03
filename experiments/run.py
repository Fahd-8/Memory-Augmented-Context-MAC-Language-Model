import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from data.dataset import get_tokenizer, get_dataset, prepare_data
from model.mac_layer import DeepMAC
from training.train import train
from inference.generate import run_tests

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Data
    tokenizer = get_tokenizer()
    dataset = get_dataset()
    sequences = prepare_data(dataset, tokenizer)
    print(f"Prepared {len(sequences)} sequences")

    # Model
    mac = DeepMAC(num_layers=6, dim=768, vocab_size=tokenizer.vocab_size)

    # Train
    mac, embed = train(mac, sequences, tokenizer, epochs=20, device=str(device))

    # Test
    run_tests(mac, embed, tokenizer, device=str(device))

if __name__ == "__main__":
    main()