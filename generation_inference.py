"""Load and sample the new from-scratch BPE language model."""

from pathlib import Path
from collections import Counter

import torch

from generation_model import SmallLanguageModel
from tokenizer import BPETokenizer


CHECKPOINT_FILE = "generation_model.pth"
TOKENIZER_FILE = "generation_tokenizer.json"


def load_generation_model(checkpoint_path=CHECKPOINT_FILE,
                          tokenizer_path=TOKENIZER_FILE, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"New model checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"New model tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["vocabulary_size"] != len(tokenizer.symbols):
        raise ValueError("The BPE tokenizer does not match this model checkpoint.")

    model = SmallLanguageModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def generate_text(model, tokenizer, prompt, number_of_words=10,
                  context_window=64, temperature=0.55, top_k=8):
    """Generate a requested number of complete word pieces and return only them."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a prompt before generating text.")
    if number_of_words < 1:
        raise ValueError("Generation length must be at least one word.")
    if not 1 <= context_window <= model.context_length:
        raise ValueError(f"Context window must be between 1 and {model.context_length}.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k < 1:
        raise ValueError("top_k must be at least one.")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Please enter a prompt before generating text.")
    all_ids = list(prompt_ids)
    continuation_ids = []
    completed_words = 0
    token_limit = max(50, number_of_words * 12)
    device = next(model.parameters()).device

    model.eval()
    with torch.inference_mode():
        for _ in range(token_limit):
            context = all_ids[-context_window:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1, :] / temperature

            # Block a token that would repeat a recent three-piece sequence.
            if len(all_ids) >= 2:
                previous_pair = tuple(all_ids[-2:])
                for index in range(len(all_ids) - 2):
                    if tuple(all_ids[index:index + 2]) == previous_pair:
                        repeated_next = all_ids[index + 2]
                        logits[repeated_next] = float("-inf")

            # Downweight the first BPE piece of words already used in this
            # continuation. This helps prevent loops such as "episode ... episode".
            recent_completed_words = _completed_words(tokenizer, continuation_ids)
            for word, count in Counter(recent_completed_words).items():
                first_piece = tokenizer._encode_word(word)[0]
                logits[first_piece] -= min(1.5, 0.9 + 0.2 * (count - 1))

            keep = min(top_k, logits.numel())
            top_values, _ = torch.topk(logits, keep)
            logits[logits < top_values[-1]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()
            all_ids.append(next_id)
            continuation_ids.append(next_id)

            piece = tokenizer.symbols[next_id]
            if piece.endswith(tokenizer.END_OF_WORD):
                word = piece[:-len(tokenizer.END_OF_WORD)]
                if any(character.isalnum() for character in word):
                    completed_words += 1
                if completed_words >= number_of_words:
                    break

    return tokenizer.decode(continuation_ids).strip()


def _completed_words(tokenizer, token_ids):
    """Recover completed word strings from BPE pieces for repetition control."""
    words = []
    current = ""
    for token_id in token_ids:
        piece = tokenizer.symbols[token_id]
        if piece == "<unk>":
            current += piece
            continue
        is_end = piece.endswith(tokenizer.END_OF_WORD)
        current += piece[:-len(tokenizer.END_OF_WORD)] if is_end else piece
        if is_end:
            if current:
                words.append(current)
            current = ""
    return words
