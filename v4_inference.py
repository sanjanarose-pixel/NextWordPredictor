"""V4-only checkpoint loading and narrative generation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import torch

from v4_model import V4LanguageModel
from v4_tokenizer import V4BPETokenizer


def load_v4(checkpoint_path: str | Path, tokenizer_path: str | Path, device: str = "cpu"):
    if device != "cpu":
        raise ValueError("V4 is CPU-only")
    tokenizer = V4BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "nextword-v4":
        raise ValueError("Checkpoint is not a V4 checkpoint")
    config = checkpoint["config"]
    if config["vocab_size"] != len(tokenizer.symbols) or config["pad_id"] != tokenizer.pad_id:
        raise ValueError("Checkpoint and tokenizer are incompatible")
    model = V4LanguageModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 48,
             temperature: float = 0.8, top_k: int = 40, repetition_penalty: float = 1.12,
             seed: int | None = None, min_new_words: int = 5, max_new_words: int = 8) -> str:
    if (not prompt.strip() or max_new_tokens < 1 or temperature <= 0 or top_k < 1
            or min_new_words < 0 or max_new_words < max(1, min_new_words)):
        raise ValueError("Invalid prompt or generation settings")
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    prompt_ids = tokenizer.encode(prompt)
    generated = []
    ids = list(prompt_ids)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            current = ids[-model.context_length:]
            x = torch.tensor([current], dtype=torch.long)
            logits = model(x)[0, -1] / temperature
            logits[tokenizer.pad_id] = float("-inf")
            if tokenizer.reserved_ids:
                logits[list(tokenizer.reserved_ids)] = float("-inf")
            completed_text = tokenizer.decode(generated)
            completed_words = len(re.findall(r"\b[\w]+(?:['’][\w]+)?\b", completed_text))
            if completed_words < min_new_words:
                logits[tokenizer.eos_id] = float("-inf")
            for token in set(generated):
                if logits[token] > 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty
            if len(generated) >= 3:
                # Suppress exact repeated trigram continuations.
                prior = tuple(generated[-2:])
                for i in range(max(0, len(generated) - 30)):
                    if tuple(generated[i:i + 2]) == prior and i + 2 < len(generated):
                        logits[generated[i + 2]] = float("-inf")
            k = min(top_k, logits.numel())
            top_values, _ = torch.topk(logits, k)
            logits[logits < top_values[-1]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator).item()
            if token == tokenizer.eos_id:
                break
            ids.append(token)
            generated.append(token)
            piece = tokenizer.symbols[token]
            completed_words = len(re.findall(r"\b[\w]+(?:['’][\w]+)?\b", tokenizer.decode(generated)))
            if completed_words >= max_new_words and piece.endswith(tokenizer.END):
                break
    return prompt + tokenizer.decode(generated)
