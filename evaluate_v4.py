"""Fixed-prompt V4 evaluation; heuristic scores support, not replace, human review."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from v4_inference import generate, load_v4

PROMPTS = [
    "Once upon a time, there was a little girl who",
    "One morning, a rabbit found a strange key",
    "A little boy went into the forest and",
    "The old man had a secret that",
    "One day, a small robot decided to",
    "The princess opened the mysterious box and",
    "A little bird fell from its nest",
    "Deep inside the forest, there lived",
    "The little dog followed the sound",
    "When the sun went down, the children",
    "A tiny dragon was afraid of the dark, so",
    "Mia and her brother found a map under the bed",
    "The friendly monster knocked on the door and",
    "At the edge of the lake, a turtle",
    "When the magic seed began to grow",
]


def repetition_metrics(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.casefold())
    ngrams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    repeated = len(ngrams) - len(set(ngrams))
    return {"word_count": len(words), "unique_trigram_ratio": (len(set(ngrams)) / len(ngrams)) if ngrams else 1.0,
            "repeated_trigram_count": repeated, "longest_word_run": _longest_run(words)}


def _longest_run(words):
    best = run = 0
    prior = None
    for word in words:
        run = run + 1 if word == prior else 1
        best = max(best, run)
        prior = word
    return best


def surface_checks(text: str, prompt: str) -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    continuation = text[len(prompt):].strip()
    continuation_words = re.findall(r"\b[\w]+(?:['’][\w]+)?\b", continuation)
    prompt_words = set(re.findall(r"\b\w+\b", prompt.casefold()))
    continuation_words = set(re.findall(r"\b\w+\b", text[len(prompt):].casefold()))
    overlap = len(prompt_words & continuation_words) / max(len(prompt_words), 1)
    repeated_punctuation = bool(re.search(r"[!?.,]{3,}", text))
    unbalanced_quotes = (text.count('"') % 2 != 0 or text.count("“") != text.count("”"))
    fragments = sum(len(re.findall(r"\b\w+\b", s)) < 3 for s in sentences)
    return {"sentence_count": len(sentences), "prompt_lexical_overlap": round(overlap, 3),
            "fragment_like_sentences": fragments, "repeated_punctuation": repeated_punctuation,
            "unbalanced_curly_quotes": unbalanced_quotes,
            "continuation_word_count": len(continuation_words),
            "empty_continuation": not continuation}


def evaluate(checkpoint: str, tokenizer_path: str, out: str | None = None):
    model, tokenizer, _ = load_v4(checkpoint, tokenizer_path)
    rows = []
    for i, prompt in enumerate(PROMPTS):
        continuation = generate(model, tokenizer, prompt, max_new_tokens=48,
                                temperature=0.8, top_k=40, seed=800 + i,
                                min_new_words=5, max_new_words=9)
        rows.append({"prompt": prompt, "output": continuation,
                     "repetition": repetition_metrics(continuation),
                     "surface_checks": surface_checks(continuation, prompt)})
    checks = [row["surface_checks"] for row in rows]
    report = {"settings": {"temperature": 0.8, "top_k": 40, "max_new_tokens": 48,
                            "min_new_words": 5, "max_new_words": 9,
                            "seed_base": 800}, "samples": rows,
              "summary": {
                  "samples": len(rows),
                  "empty_continuations": sum(item["empty_continuation"] for item in checks),
                  "continuation_word_counts": [item["continuation_word_count"] for item in checks],
                  "repeated_punctuation_samples": sum(item["repeated_punctuation"] for item in checks),
                  "unbalanced_quote_samples": sum(item["unbalanced_curly_quotes"] for item in checks),
                  "fragment_like_sentences": sum(item["fragment_like_sentences"] for item in checks),
                  "repeated_trigram_total": sum(row["repetition"]["repeated_trigram_count"] for row in rows),
              },
              "human_review": ["relevance", "grammaticality", "causal/context continuity", "dialogue voice"]}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if out:
        Path(out).write_text(serialized, encoding="utf-8")
    print(serialized)
    return report


def compare_v2_v3_v4(v4_model, v4_tokenizer):
    """Compare on shared prompts/seed where decoders permit; report settings honestly."""
    from high_quality_inference import load_high_quality_model, generate_text as generate_legacy
    from v3_inference import load_v3_model
    v2, v3 = load_high_quality_model(), load_v3_model()
    rows = []
    for i, prompt in enumerate(PROMPTS):
        legacy = {}
        for label, loaded in (("V2", v2), ("V3", v3)):
            lm, lt, _ = loaded
            torch.manual_seed(1000 + i)
            legacy[label] = generate_legacy(lm, lt, prompt, number_of_words=70,
                                            context_window=min(64, lm.context_length),
                                            temperature=0.8, top_k=40)
        legacy["V4"] = generate(v4_model, v4_tokenizer, prompt, max_new_tokens=140,
                                temperature=0.8, top_k=40, seed=1000 + i)
        rows.append({"prompt": prompt, "outputs": legacy,
                     "note": "Sampling algorithms/tokenizers differ; equal temperature/top-k and seeds are approximate controls."})
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("tokenizer")
    parser.add_argument("--out")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.tokenizer, args.out)
