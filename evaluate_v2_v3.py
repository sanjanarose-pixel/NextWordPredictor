"""Reproducibly compare Version 2 and Version 3 text generation.

Version 2 here means the sentence-level high-quality model used as V2 by
compare_v2_v3.py. The original word-level Stage 2 checkpoints are a separate
model track and are not compared by this script.

The inference APIs count completed words, so this evaluator loads both models
through their inference pipelines and applies one shared token-limited decoder
to each model. No training, dataset access, or checkpoint writes occur.
"""

from collections import Counter
import hashlib
from pathlib import Path
import platform
import re

import torch

from high_quality_inference import load_high_quality_model
from v3_inference import load_v3_model


OUTPUT_FILE = Path("v2_v3_generation_report.txt")
MAX_NEW_TOKENS = 30
CONTEXT_WINDOW = 64
TEMPERATURE = 0.7
TOP_K = 24
BASE_SEED = 321

PROMPTS = (
    ("Robotics", "A warehouse robot noticed that one package had fallen beside the conveyor, so"),
    ("Robotics", "The robotic arm paused above the delicate glass sample when"),
    ("Science", "The scientist repeated the experiment and discovered that the tiny change in temperature"),
    ("Science", "As the telescope turned toward the distant planet, the research team"),
    ("Everyday life", "On a rainy Tuesday morning, Maya missed her bus and decided to"),
    ("Everyday life", "After dinner, the neighbors gathered in the garden to talk about"),
    ("Technology", "The new phone updated overnight, and by morning its owner noticed"),
    ("Technology", "When the office network went silent, the technician checked the server room and"),
    ("Storytelling", "At midnight, a letter appeared beneath the door of the old lighthouse, addressed to"),
    ("Storytelling", "The map led the two children through the forest to a wooden gate that"),
)


def completed_words(tokenizer, token_ids):
    """Recover completed words from BPE pieces, matching inference behavior."""
    words = []
    current = ""
    for token_id in token_ids:
        piece = tokenizer.symbols[token_id]
        if piece == tokenizer.END_OF_SENTENCE:
            break
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


def generate_fixed_tokens(model, tokenizer, prompt, seed):
    """Generate exactly 30 tokens with shared inference-style sampling rules."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt did not produce tokens: {prompt!r}")

    all_ids = list(prompt_ids)
    continuation_ids = []
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    model.eval()

    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            input_ids = torch.tensor(
                [all_ids[-CONTEXT_WINDOW:]], dtype=torch.long, device=device
            )
            logits = model(input_ids)[0, -1, :] / TEMPERATURE

            # Match the existing inference pipelines' trigram repeat avoidance.
            if len(all_ids) >= 2:
                previous_pair = tuple(all_ids[-2:])
                for index in range(len(all_ids) - 2):
                    if tuple(all_ids[index:index + 2]) == previous_pair:
                        logits[all_ids[index + 2]] = float("-inf")

            # Match their penalty for words repeated within this continuation.
            for word, count in Counter(completed_words(tokenizer, continuation_ids)).items():
                first_piece = tokenizer._encode_word(word)[0]
                logits[first_piece] -= min(1.5, 0.8 + 0.2 * (count - 1))

            # The request is for exactly 30 generated tokens per output. Mask
            # EOS so both models produce the same fixed number of content
            # tokens; outputs are therefore intentionally cut off at token 30.
            logits[tokenizer.eos_id] = float("-inf")

            keep = min(TOP_K, logits.numel())
            top_values, _ = torch.topk(logits, keep)
            logits[logits < top_values[-1]] = float("-inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            all_ids.append(next_id)
            continuation_ids.append(next_id)

    text = tokenizer.decode(continuation_ids).strip()
    return {
        "text": text,
        "token_ids": continuation_ids,
        "hit_token_limit": len(continuation_ids) == MAX_NEW_TOKENS,
    }


def diagnose(tokenizer, result):
    """Return transparent heuristic flags for issues worth human review."""
    text = result["text"]
    token_ids = result["token_ids"]
    flags = []

    if "<unk>" in text or any(tokenizer.symbols[token_id] == "<unk>" for token_id in token_ids):
        flags.append("unknown token")

    ngrams = [tuple(token_ids[i:i + 3]) for i in range(max(0, len(token_ids) - 2))]
    repeated_ngrams = sum(count - 1 for count in Counter(ngrams).values() if count > 1)
    repeated_words = [
        word for word, count in Counter(completed_words(tokenizer, token_ids)).items()
        if count >= 4 and any(character.isalnum() for character in word)
    ]
    if repeated_ngrams:
        flags.append(f"repeated 3-gram ({repeated_ngrams} extra occurrence(s))")
    if repeated_words:
        flags.append("word repeated 4+ times: " + ", ".join(repeated_words[:5]))

    if result["hit_token_limit"]:
        flags.append("fixed 30-token cutoff; may end mid-sentence")
    if token_ids:
        final_piece = tokenizer.symbols[token_ids[-1]]
        if final_piece != "<unk>" and not final_piece.endswith(tokenizer.END_OF_WORD):
            flags.append("incomplete final word")

    formatting_issues = []
    if re.search(r"\s+[,.!?;:%)]", text):
        formatting_issues.append("space before punctuation")
    if re.search(r"[!?.,;:]{2,}", text):
        formatting_issues.append("repeated punctuation")
    if re.match(r"^[,.!?;:%)]", text):
        formatting_issues.append("leading punctuation")
    if text.count('"') % 2:
        formatting_issues.append("unmatched double quote")
    if re.search(r"\w\s+[’']\s+\w", text):
        formatting_issues.append("split apostrophe/contraction spacing")
    if re.search(r"\s+[’']$", text):
        formatting_issues.append("dangling quotation mark")
    if formatting_issues:
        flags.append("formatting: " + ", ".join(formatting_issues))

    return flags or ["no automatic flags"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_model(name, checkpoint_path, tokenizer_path, loaded):
    model, tokenizer, checkpoint = loaded
    return [
        f"{name}",
        f"  checkpoint: {checkpoint_path}",
        f"  checkpoint SHA-256: {sha256(checkpoint_path)}",
        f"  tokenizer: {tokenizer_path}",
        f"  tokenizer SHA-256: {sha256(tokenizer_path)}",
        f"  config: {checkpoint.get('config')}",
        f"  tokenizer symbols: {len(tokenizer.symbols)}",
        f"  checkpoint epoch: {checkpoint.get('epoch', 'unavailable')}",
        f"  validation loss: {checkpoint.get('validation_loss', 'unavailable')}",
        f"  model context length: {model.context_length}",
    ]


def main():
    # Version 2 means the sentence-level model used as V2 by compare_v2_v3.py.
    v2 = load_high_quality_model("high_quality_model.pth", "high_quality_tokenizer.json")
    v3 = load_v3_model()
    models = (
        ("Version 2 (sentence-level)", v2, "high_quality_model.pth", "high_quality_tokenizer.json"),
        ("Version 3 (paragraph-level)", v3, "v3_model.pth", "v3_tokenizer.json"),
    )

    lines = [
        "NextWordPredictor: Version 2 vs Version 3 generation review",
        "=" * 68,
        "",
        "Model naming: Version 2 is the sentence-level high-quality model, matching compare_v2_v3.py.",
        "The original word-level Stage 2 checkpoints are a separate model track.",
        "Both models were loaded through their project inference loaders.",
        "The shared evaluator decoder is used because the public inference functions count words,",
        "while this comparison requires exactly 30 generated model tokens.",
        "",
        "Shared decoding settings:",
        f"  exactly {MAX_NEW_TOKENS} new tokens; context={CONTEXT_WINDOW}; temperature={TEMPERATURE}; top-k={TOP_K}",
        f"  seed per prompt=base {BASE_SEED} + prompt index; EOS is suppressed to keep fixed-length outputs.",
        "  sampling is performed on CPU; every continuation is intentionally cut off at token 30.",
        "  Repetition avoidance and repeated-word penalty follow the existing inference implementations.",
        f"  Python={platform.python_version()}; PyTorch={torch.__version__}",
        "",
        "Checkpoint metadata:",
    ]
    for name, loaded, checkpoint_path, tokenizer_path in models:
        lines.extend(describe_model(name, checkpoint_path, tokenizer_path, loaded))
        lines.append("")
    lines.extend([
        "Validation-loss caution: each loss comes from a different training corpus/tokenizer setup;",
        "these values are context only and do not by themselves establish generation quality.",
        "Automatic flags are heuristics to guide inspection, not quality scores.",
        "",
        "Generated outputs",
        "-" * 68,
        "",
    ])

    for index, (category, prompt) in enumerate(PROMPTS, start=1):
        seed = BASE_SEED + index
        lines.extend([f"{index:02d}. [{category}] {prompt}", f"   seed: {seed}"])
        for name, loaded, _, _ in models:
            model, tokenizer, _ = loaded
            result = generate_fixed_tokens(model, tokenizer, prompt, seed)
            flags = diagnose(tokenizer, result)
            count_note = f"{len(result['token_ids'])} generated token(s)"
            lines.append(f"   {name}: {result['text'] or '[empty continuation]'}")
            lines.append(f"      output length: {count_note}")
            lines.append(f"      review flags: {'; '.join(flags)}")
        lines.append("")

    report = "\n".join(lines)
    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved report to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
