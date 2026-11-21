"""Create a separate BPE tokenizer artifact for a future model-training run.

This utility does not modify checkpoints or vocabulary.json. The resulting
tokenizer cannot be used with existing word-level model weights.
"""

import argparse
from pathlib import Path

from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset.txt", help="Training text file")
    parser.add_argument("--output", default="bpe_tokenizer.json", help="Output tokenizer file")
    parser.add_argument("--vocabulary-size", type=int, default=8000)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train(text, vocabulary_size=args.vocabulary_size)
    tokenizer.save(args.output)
    print(f"Saved {len(tokenizer.symbols):,} BPE symbols to {args.output}")
    print("Existing checkpoints were not changed. Retrain a model before using this tokenizer for predictions.")


if __name__ == "__main__":
    main()
