"""Small standalone V4 prompt/continuation demo; does not touch the V3 app."""
from __future__ import annotations

import argparse
from pathlib import Path

from v4_inference import generate, load_v4

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "v4"


def latest_pair() -> tuple[Path, Path]:
    checkpoints = sorted(ARTIFACTS.glob("v4_best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    for checkpoint in checkpoints:
        run_id = checkpoint.stem.removeprefix("v4_best_")
        tokenizer = ARTIFACTS / f"tokenizer_v4_{run_id}.json"
        if tokenizer.exists():
            return checkpoint, tokenizer
    raise FileNotFoundError("No matching V4 checkpoint and tokenizer found under artifacts/v4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Generate one continuation; omit to use the interactive prompt loop")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.tokenizer):
        parser.error("--checkpoint and --tokenizer must be supplied together")
    checkpoint, tokenizer = ((args.checkpoint, args.tokenizer) if args.checkpoint
                              else latest_pair())
    model, tok, _ = load_v4(checkpoint, tokenizer)

    def run(prompt: str, seed: int) -> None:
        if not prompt.strip():
            print("Enter a nonempty prompt.")
            return
        print(generate(model, tok, prompt, max_new_tokens=48, temperature=0.8,
                       top_k=40, seed=seed, min_new_words=5, max_new_words=9))

    if args.prompt is not None:
        run(args.prompt, args.seed)
        return
    print("V4 CPU demo. Enter a prompt; type /quit to exit.")
    seed = args.seed
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().casefold() in {"/quit", "/exit"}:
            break
        run(prompt, seed)
        seed += 1


if __name__ == "__main__":
    main()
