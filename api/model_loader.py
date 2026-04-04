import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model.mac_layer import DeepMAC
from model.embeddings import get_embedding


class MACModelLoader:
    def __init__(self):
        self.mac = None
        self.embed = None
        self.tokenizer = None
        self.device = None
        self.loaded = False

        # stateful conversation — LMM accumulates memory across requests
        self.conversation_history = []

    def load(self, checkpoint_path='checkpoints/mac_best.pt'):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        print(f"Loading MAC model on {self.device}...")

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.mac = DeepMAC(num_layers=6, dim=768, vocab_size=self.tokenizer.vocab_size)
        self.embed = get_embedding(self.tokenizer.vocab_size, embed_dim=768)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.mac.load_state_dict(checkpoint['mac_state_dict'])
        self.embed.load_state_dict(checkpoint['embed_state_dict'])

        self.mac = self.mac.to(self.device)
        self.embed = self.embed.to(self.device)
        self.mac.eval()

        self.loaded = True
        print(f"Model loaded from {checkpoint_path}")

    def generate(self, prompt, max_new_tokens=100, temperature=0.8,
                 top_p=0.9, repetition_penalty=1.3, do_ttt=True):

        # stateful — do NOT reset between requests
        # LMM memory accumulates across conversation turns
        token_ids = self.tokenizer.encode(prompt)
        generated = list(token_ids)

        for _ in range(max_new_tokens):
            seq_vecs = self.embed(torch.tensor(generated).to(self.device))
            logits, _ = self.mac(seq_vecs, do_ttt=do_ttt)
            next_logits = logits[-1] / temperature

            # repetition penalty
            for token_id in set(generated):
                next_logits[token_id] /= repetition_penalty

            # top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break

        response = self.tokenizer.decode(generated)

        # save to conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "ttt": do_ttt
        })

        return response

    def compare(self, prompt, max_new_tokens=100, temperature=0.8,
                top_p=0.9, repetition_penalty=1.3):
        # TTT ON
        self.reset()
        response_ttt_on = self.generate(
            prompt, max_new_tokens, temperature,
            top_p, repetition_penalty, do_ttt=True
        )

        # TTT OFF — same prompt, fresh state
        self.reset()
        response_ttt_off = self.generate(
            prompt, max_new_tokens, temperature,
            top_p, repetition_penalty, do_ttt=False
        )

        return response_ttt_on, response_ttt_off

    def reset(self):
        # wipe all memory — fresh conversation
        self.mac.reset_ttt()
        self.mac.reset_memory()
        self.conversation_history = []


# singleton — one model instance shared across all requests
mac_model = MACModelLoader()