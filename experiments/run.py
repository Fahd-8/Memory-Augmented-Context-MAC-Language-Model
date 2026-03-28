import torch
from data.dataset import get_tokenizer, get_dataset, prepare_data
from model.mac_layer import DeepMAC
from training.train import train
from inference.generate import run_tests

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    tokenizer = get_tokenizer()
    dataset = get_dataset()
    sequences = prepare_data(dataset, tokenizer)
    print(f"Prepared {len(sequences)} sequences")

    # Model
    mac = DeepMAC(num_layers=3, dim=512, vocab_size=tokenizer.vocab_size)

    # Train
    mac, embed = train(mac, sequences, tokenizer, epochs=50, device=str(device))

    # Test
    run_tests(mac, embed, tokenizer, device=str(device))

if __name__ == "__main__":
    main()