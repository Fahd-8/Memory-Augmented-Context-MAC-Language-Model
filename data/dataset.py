from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

def get_dataset(split="train[:10000]"):
    return load_dataset("roneneldan/TinyStories", split=split)

def prepare_data(dataset, tokenizer, max_len=256, num_stories=5000):
    sequences = []

    for i in range(min(num_stories, len(dataset))):
        text = dataset[i]['text']
        tokens = tokenizer.encode(text, add_special_tokens=True)

        for j in range(0, len(tokens) - max_len, max_len):
            chunk = tokens[j:j + max_len]
            if len(chunk) == max_len:
                sequences.append(chunk)

    return sequences