import torch
from model.qwen_mac import QwenMAC


def run_memory_demo(device='mps'):
    print("\nInitializing QwenMAC...")
    model = QwenMAC(device=device)

    # ── TTT ON demo ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TTT ON — LMM memory active, no history passed to Qwen")
    print("="*60)

    model.reset_conversation()

    conversation = [
        "Hi! My name is Fahad, I am 23 years old and I live in Pakistan.",
        "I work as a COO and lead developer at a company called HolisticTLC.",
        "My favorite sport is MMA. I follow Islam Makhachev and Khabib closely.",
        "What is the capital of France?",
        "Can you tell me what you know about me so far?",
    ]

    for message in conversation:
        print(f"\nUser: {message}")
        response = model.generate(message, do_ttt=True)
        print(f"QwenMAC: {response}")

    # ── TTT OFF demo ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TTT OFF — LMM disabled, no history passed to Qwen")
    print("="*60)

    model.reset_conversation()

    for message in conversation:
        print(f"\nUser: {message}")
        response = model.generate(message, do_ttt=False)
        print(f"QwenMAC: {response}")

    # ── direct recall test ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MEMORY TEST — pure LMM recall, zero history text")
    print("="*60)

    recall_question = "What is my name, where am I from, how old am I, and what do I love?"

    # TTT ON — LMM accumulates memory across 4 turns then recalls
    print("\n--- TTT ON ---")
    model.reset_conversation()

    context_turns = [
        "My name is Fahad.",
        "I live in Pakistan and I am 23 years old.",
        "I love MMA. I work at HolisticTLC as COO.",
        "I follow Islam Makhachev closely.",
    ]

    for turn in context_turns:
        model.generate(turn, do_ttt=True, max_new_tokens=50)

    print(f"User: {recall_question}")
    response_on = model.generate(recall_question, do_ttt=True, max_new_tokens=150)
    print(f"QwenMAC: {response_on}")

    # TTT OFF — same turns, LMM does nothing, no history text
    print("\n--- TTT OFF ---")
    model.reset_conversation()

    for turn in context_turns:
        model.generate(turn, do_ttt=False, max_new_tokens=50)

    print(f"User: {recall_question}")
    response_off = model.generate(recall_question, do_ttt=False, max_new_tokens=150)
    print(f"QwenMAC: {response_off}")

    print("\n" + "="*60)
    print("Demo complete.")
    print("="*60)


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    run_memory_demo(device=device)