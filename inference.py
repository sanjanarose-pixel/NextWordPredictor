"""Load a trained checkpoint and generate text without running training."""

import re
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer import LegacyWordTokenizer


DEFAULT_CHECKPOINT_CANDIDATES = (
    "language_model_best_stage2.pth",
    "language_model_checkpoint.pth",
)
DEFAULT_VOCABULARY = "vocabulary.json"
EXPECTED_CONFIG_KEYS = {
    "vocabulary_size", "embedding_dimension", "context_length",
    "number_of_heads", "number_of_blocks",
}


def find_checkpoint(directory="."):
    directory = Path(directory)
    for filename in DEFAULT_CHECKPOINT_CANDIDATES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    names = ", ".join(DEFAULT_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(f"No trained checkpoint found in {directory}. Expected one of: {names}")


def load_model(checkpoint_path=None, vocabulary_path=None, device="cpu"):
    """Load weights only after confirming that vocabulary and model sizes agree."""
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else find_checkpoint()
    vocabulary_path = Path(vocabulary_path or DEFAULT_VOCABULARY)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocabulary_path}")

    tokenizer = LegacyWordTokenizer.load(vocabulary_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not EXPECTED_CONFIG_KEYS.issubset(config):
        raise ValueError("Checkpoint is missing the model configuration needed for safe loading.")
    if config["vocabulary_size"] != len(tokenizer.vocabulary):
        raise ValueError(
            "Checkpoint vocabulary size does not match vocabulary.json. "
            "A tokenizer change requires matching newly trained model weights."
        )

    model = LanguageModel(
        vocabulary_size=config["vocabulary_size"],
        embedding_dimension=config["embedding_dimension"],
        context_length=config["context_length"],
        number_of_heads=config["number_of_heads"],
        number_of_blocks=config["number_of_blocks"],
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"Checkpoint weights do not match the configured model: {error}") from error
    model.eval()
    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, number_of_words=30,
                  context_window=32, temperature=0.8, top_k=20):
    """Generate exactly the requested number of tokens after the prompt."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a prompt containing at least one word.")
    if number_of_words < 0:
        raise ValueError("Generation length cannot be negative.")
    if not 1 <= context_window <= model.context_length:
        raise ValueError(f"Context window must be between 1 and {model.context_length}.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k < 1:
        raise ValueError("Top-k must be at least 1.")

    generated_ids = tokenizer.encode(prompt)
    if not generated_ids:
        raise ValueError("Please enter a prompt containing at least one word.")
    vocabulary_size = len(tokenizer.vocabulary)

    with torch.inference_mode():
        for _ in range(number_of_words):
            context = generated_ids[-context_window:]
            input_ids = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
            logits = model(input_ids)[0, -1, :] / temperature
            keep = min(top_k, vocabulary_size)
            top_values, _ = torch.topk(logits, keep)
            logits[logits < top_values[-1]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)
            generated_ids.append(torch.multinomial(probabilities, 1).item())

    return tokenizer.decode(generated_ids)
