import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.lmm import LMM


class QwenMAC(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2-1.5B-Instruct', device='mps'):
        super().__init__()
        self.device = device
        self.dim = 1536

        # load Qwen — frozen
        print("Loading Qwen...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32
        ).to(device)

        # freeze all Qwen weights
        for param in self.qwen.parameters():
            param.requires_grad = False

        print("Qwen loaded and frozen.")

        # LMM — owns its own TTT optimizer
        self.lmm = LMM(dim=self.dim).to(device)

        # learnable memory position — stamps memory token with unique identity
        self.memory_pos = nn.Parameter(
            torch.randn(1, self.dim, device=device) * 0.02
        )

        # conversation history for display only — never passed to Qwen
        self.history = []

    def _get_embeddings(self, token_ids):
        token_tensor = torch.tensor(token_ids, device=self.device)
        with torch.no_grad():
            embeddings = self.qwen.model.embed_tokens(token_tensor)
        return embeddings  # [seq_len, 1536]

    def generate(self, text, max_new_tokens=200, temperature=0.7, do_ttt=True):
        # step 1 — tokenize current message only, no history
        messages = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # step 2 — get Qwen embeddings for current message
        embeddings = self._get_embeddings(token_ids)  # [seq_len, 1536]

        # step 3 — LMM processes embeddings
        # TTT updates weights across turns — memory accumulates in weights
        # do NOT reset between turns — LMM carries memory forward
        memory_summary = self.lmm(embeddings, do_ttt=do_ttt)  # [1, 1536]

        # step 4 — stamp memory with positional identity
        memory_summary = memory_summary + self.memory_pos  # [1, 1536]

        # step 5 — prepend memory token to current message embeddings only
        # Qwen sees: [memory_token, current_message_tokens]
        # NO history text — memory comes purely from LMM weights
        combined = torch.cat([memory_summary, embeddings], dim=0)  # [seq_len+1, 1536]

        # step 6 — autoregressive generation
        generated_ids = []
        current_embeds = combined.unsqueeze(0)  # [1, seq_len+1, 1536]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.qwen(
                    inputs_embeds=current_embeds,
                    output_hidden_states=False
                )

            next_logits = outputs.logits[0, -1, :] / temperature
            next_token_id = torch.multinomial(
                torch.softmax(next_logits, dim=-1), num_samples=1
            ).item()

            generated_ids.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break

            # embed next token and append
            next_embed = self._get_embeddings([next_token_id])  # [1, 1536]
            current_embeds = torch.cat(
                [current_embeds, next_embed.unsqueeze(0)], dim=1
            )

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # save to history for display only — never used by Qwen
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        # full reset between separate conversations
        self.history = []
        self.lmm.reset_ttt()
        self.lmm.reset_memory_state()