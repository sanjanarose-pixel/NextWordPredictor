"""Compare Version 2 and 3 on the same prompts and decoding settings."""

from pathlib import Path

import torch

from high_quality_inference import generate_text as generate_v2
from high_quality_inference import load_high_quality_model
from v3_inference import generate_text as generate_v3
from v3_inference import load_v3_model


PROMPTS = ["The robot moves", "A scientist", "The person", "The weather today"]
CONTEXT_WINDOW = 64
NUMBER_OF_WORDS = 14
TEMPERATURE = 0.7
SAMPLE_TOP_K = 24
SEED = 321


def generate_pair(v2, v3, prompt, top_k, seed):
    v2_model, v2_tokenizer, _ = v2
    v3_model, v3_tokenizer, _ = v3
    torch.manual_seed(seed)
    v2_text = generate_v2(
        v2_model, v2_tokenizer, prompt,
        number_of_words=NUMBER_OF_WORDS,
        context_window=CONTEXT_WINDOW,
        temperature=TEMPERATURE,
        top_k=top_k,
    )
    torch.manual_seed(seed)
    v3_text = generate_v3(
        v3_model, v3_tokenizer, prompt,
        number_of_words=NUMBER_OF_WORDS,
        context_window=CONTEXT_WINDOW,
        temperature=TEMPERATURE,
        top_k=top_k,
    )
    return v2_text, v3_text


def main():
    v2 = load_high_quality_model()
    v3 = load_v3_model()
    lines = [
        "Version 2 vs Version 3",
        f"Settings: context={CONTEXT_WINDOW}, words={NUMBER_OF_WORDS}, "
        f"temperature={TEMPERATURE}; sample top-k={SAMPLE_TOP_K}; greedy top-k=1",
        "",
    ]
    for index, prompt in enumerate(PROMPTS):
        for mode, top_k in (("sample", SAMPLE_TOP_K), ("greedy", 1)):
            old_text, new_text = generate_pair(v2, v3, prompt, top_k, SEED + index)
            lines.extend([
                f"[{mode}] {prompt}",
                f"V2: {old_text}",
                f"V3: {new_text}",
                "",
            ])
    report = "\n".join(lines)
    print(report)
    Path("v3_comparison.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
