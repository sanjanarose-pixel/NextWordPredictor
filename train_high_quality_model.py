"""Train a new word-boundary-aware model on the High-Quality English Sentences data."""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generation_model import SmallLanguageModel
from high_quality_data import load_test_sentences, load_train_and_validation
from tokenizer import BPETokenizer, split_words


TRAIN_FILE = "high_quality_train.txt.gz"
TEST_FILE = "high_quality_test.txt.gz"
TOKENIZER_FILE = "high_quality_tokenizer.json"
CHECKPOINT_FILE = "high_quality_model.pth"

VOCABULARY_SIZE = 8000
CONTEXT_LENGTH = 96
DIMENSION = 128
NUMBER_OF_HEADS = 4
NUMBER_OF_BLOCKS = 3
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
SEED = 42
VALIDATION_BATCHES = 40
TEST_BATCHES = 200


def encode_sentences(sentences, tokenizer):
    """Encode each sentence and append EOS so the model learns where it ends."""
    word_cache = {}
    token_ids = []
    for sentence in sentences:
        for word in split_words(sentence, lowercase=tokenizer.lowercase):
            if word not in word_cache:
                word_cache[word] = tokenizer._encode_word(word)
            token_ids.extend(word_cache[word])
        token_ids.append(tokenizer.eos_id)
    return torch.tensor(token_ids, dtype=torch.long)


def make_batch(token_ids, batch_size, context_length, generator=None):
    starts = torch.randint(
        0, len(token_ids) - context_length - 1, (batch_size,), generator=generator
    )
    offsets = torch.arange(context_length + 1)
    windows = token_ids[starts[:, None] + offsets[None, :]]
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def evaluate(model, token_ids, batch_size, context_length, batches, seed):
    """Evaluate on the same random context windows every epoch for fair comparison."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = make_batch(
                token_ids, batch_size, context_length, generator=generator
            )
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-train-sentences", type=int, default=100_000,
        help="Uniform reservoir sample from official train split (0 means use all rows).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train_sentences < 0:
        parser.error("epochs must be positive and max-train-sentences cannot be negative")

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")

    maximum_rows = None if args.max_train_sentences == 0 else args.max_train_sentences
    train_sentences, validation_sentences, train_counts = load_train_and_validation(
        TRAIN_FILE, maximum_training_sentences=maximum_rows, seed=SEED
    )
    test_sentences, test_counts = load_test_sentences(TEST_FILE)
    print("Clean train split:", dict(train_counts))
    print("Clean test split:", dict(test_counts))
    print(
        f"Using {len(train_sentences):,} training sentences and "
        f"{len(validation_sentences):,} validation sentences."
    )

    tokenizer_path = Path(TOKENIZER_FILE)
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
        if tokenizer.lowercase:
            raise ValueError("This tokenizer is lowercase; delete it to train the case-preserving tokenizer.")
        print(f"Loaded case-preserving BPE tokenizer with {len(tokenizer.symbols):,} symbols.")
    else:
        print("Learning BPE pieces from the sampled training sentences...")
        tokenizer = BPETokenizer.train(
            "\n".join(train_sentences),
            vocabulary_size=VOCABULARY_SIZE,
            lowercase=False,
        )
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}.")

    print("Encoding sentences with explicit EOS boundary tokens...")
    train_ids = encode_sentences(train_sentences, tokenizer)
    validation_ids = encode_sentences(validation_sentences, tokenizer)
    test_ids = encode_sentences(test_sentences, tokenizer)
    del train_sentences, validation_sentences, test_sentences
    print(
        f"Training tokens: {len(train_ids):,}; validation tokens: "
        f"{len(validation_ids):,}; held-out test tokens: {len(test_ids):,}."
    )

    model = SmallLanguageModel(
        vocabulary_size=len(tokenizer.symbols),
        dimension=DIMENSION,
        context_length=CONTEXT_LENGTH,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_blocks=NUMBER_OF_BLOCKS,
        dropout=DROPOUT,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = math.ceil(len(train_ids) / (BATCH_SIZE * CONTEXT_LENGTH))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(20, total_steps // 20)
    best_validation_loss = float("inf")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Training steps: {steps_per_epoch:,} per epoch x {args.epochs} epochs")
    print("Starting from random weights. The official test split is not used for training or selection.")

    initial_validation_loss = evaluate(
        model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH,
        batches=VALIDATION_BATCHES, seed=SEED + 1,
    )
    print(f"Initial validation loss: {initial_validation_loss:.3f}")

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        start_time = time.perf_counter()
        for _ in range(steps_per_epoch):
            inputs, targets = make_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
            if step < warmup_steps:
                learning_rate = LEARNING_RATE * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                learning_rate = LEARNING_RATE * (
                    0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
                )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1

        train_loss = sum(losses) / len(losses)
        validation_loss = evaluate(
            model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH,
            batches=VALIDATION_BATCHES, seed=SEED + 1,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train {train_loss:.3f} | "
            f"validation {validation_loss:.3f} | "
            f"{time.perf_counter() - start_time:.1f}s",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocabulary_size": len(tokenizer.symbols),
                    "dimension": DIMENSION,
                    "context_length": CONTEXT_LENGTH,
                    "number_of_heads": NUMBER_OF_HEADS,
                    "number_of_blocks": NUMBER_OF_BLOCKS,
                    "dropout": DROPOUT,
                },
                "validation_loss": validation_loss,
                "epoch": epoch,
                "training_tokens": len(train_ids),
            }, CHECKPOINT_FILE)
            print(f"  Saved best checkpoint to {CHECKPOINT_FILE}.", flush=True)

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(
        model, test_ids, BATCH_SIZE, CONTEXT_LENGTH,
        batches=TEST_BATCHES, seed=SEED + 2,
    )
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Initial validation loss: {initial_validation_loss:.3f}")
    print(f"Best validation loss: {checkpoint['validation_loss']:.3f}")
    print(f"Held-out test loss (200 fixed batches): {test_loss:.3f}")


if __name__ == "__main__":
    main()
