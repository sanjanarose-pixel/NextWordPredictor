"""Load and generate text with the model trained on sentence-level data."""

from collections import Counter
from pathlib import Path

import torch

from generation_model import SmallLanguageModel
from tokenizer import BPETokenizer


CHECKPOINT_FILE = "high_quality_model.pth"
TOKENIZER_FILE = "high_quality_tokenizer.json"


def load_high_quality_model(checkpoint_path=CHECKPOINT_FILE,
                            tokenizer_path=TOKENIZER_FILE, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Matching tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.eos_id is None:
        raise ValueError("Tokenizer is missing sentence-boundary token <eos>.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["vocabulary_size"] != len(tokenizer.symbols):
        raise ValueError("Tokenizer and checkpoint vocabularies do not match.")

    model = SmallLanguageModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def _completed_words(tokenizer, token_ids):
    completed = []
    current = ""
    for token_id in token_ids:
        piece = tokenizer.symbols[token_id]
        if piece == tokenizer.END_OF_SENTENCE:
            break
        if piece == "<unk>":
            current += piece
        elif piece.endswith(tokenizer.END_OF_WORD):
            current += piece[:-len(tokenizer.END_OF_WORD)]
            if current and any(character.isalnum() for character in current):
                completed.append(current)
            current = ""
        else:
            current += piece
    return completed


def generate_text(model, tokenizer, prompt, number_of_words=10,
                  context_window=64, temperature=0.7, top_k=24):
    """Generate a continuation, stopping at EOS or the requested word count."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a prompt before generating text.")
    if number_of_words < 1:
        raise ValueError("Generation length must be at least one word.")
    if not 1 <= context_window <= model.context_length:
        raise ValueError(f"Context window must be between 1 and {model.context_length}.")
    if temperature <= 0 or top_k < 1:
        raise ValueError("Sampling settings are invalid.")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Please enter a prompt containing at least one word.")
    all_ids = list(prompt_ids)
    continuation_ids = []
    minimum_words_before_eos = min(4, number_of_words)
    token_limit = max(64, number_of_words * 12)
    device = next(model.parameters()).device

    model.eval()
    with torch.inference_mode():
        for _ in range(token_limit):
            input_ids = torch.tensor(
                [all_ids[-context_window:]], dtype=torch.long, device=device
            )
            logits = model(input_ids)[0, -1, :] / temperature

            # Avoid repeating an already generated BPE trigram.
            if len(all_ids) >= 2:
                previous_pair = tuple(all_ids[-2:])
                for index in range(len(all_ids) - 2):
                    if tuple(all_ids[index:index + 2]) == previous_pair:
                        logits[all_ids[index + 2]] = float("-inf")

            # Downweight complete words repeated in the current continuation.
            for word, count in Counter(_completed_words(tokenizer, continuation_ids)).items():
                first_piece = tokenizer._encode_word(word)[0]
                logits[first_piece] -= min(1.5, 0.8 + 0.2 * (count - 1))

            completed_count = len(_completed_words(tokenizer, continuation_ids))
            if completed_count < minimum_words_before_eos:
                logits[tokenizer.eos_id] = float("-inf")

            keep = min(top_k, logits.numel())
            top_values, _ = torch.topk(logits, keep)
            logits[logits < top_values[-1]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()
            if next_id == tokenizer.eos_id:
                break

            all_ids.append(next_id)
            continuation_ids.append(next_id)
            completed_count = len(_completed_words(tokenizer, continuation_ids))
            if completed_count >= number_of_words:
                break

    return tokenizer.decode(continuation_ids).strip()
