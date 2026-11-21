"""Train the Version 3 Transformer on paragraph-level FineWeb-Edu text."""

import math
import random
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from generation_model import SmallLanguageModel
from tokenizer import BPETokenizer, split_words


TRAIN_FILE = "v3_train.txt"
VALIDATION_FILE = "v3_validation.txt"
TEST_FILE = "v3_test.txt"
TOKENIZER_FILE = "v3_tokenizer.json"
CHECKPOINT_FILE = "v3_model.pth"

VOCABULARY_SIZE = 8000
TRAIN_TOKEN_BUDGET = 3_300_000
CONTEXT_LENGTH = 96
DIMENSION = 128
NUMBER_OF_HEADS = 4
NUMBER_OF_BLOCKS = 3
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
SEED = 123
VALIDATION_BATCHES = 40
TEST_BATCHES = 200


def read_paragraphs(path):
    """Read one already-filtered paragraph from each line."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def encode_paragraphs(paragraphs, tokenizer):
    """Encode each paragraph and put EOS only at its paragraph boundary."""
    word_cache = {}
    token_ids = []
    for paragraph in paragraphs:
        for word in split_words(paragraph, lowercase=tokenizer.lowercase):
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
        "--dry-run", action="store_true",
        help="Fit/load the tokenizer and report token counts without training weights.",
    )
    args = parser.parse_args()

    for required_file in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        if not Path(required_file).is_file():
            raise FileNotFoundError(
                f"Missing {required_file}; run prepare_v3_data.py first."
            )

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")

    train_paragraphs = read_paragraphs(TRAIN_FILE)
    validation_paragraphs = read_paragraphs(VALIDATION_FILE)
    test_paragraphs = read_paragraphs(TEST_FILE)
    print(
        f"Paragraphs: train={len(train_paragraphs):,}, "
        f"validation={len(validation_paragraphs):,}, test={len(test_paragraphs):,}"
    )

    tokenizer_path = Path(TOKENIZER_FILE)
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Loaded Version 3 BPE tokenizer ({len(tokenizer.symbols):,} symbols).")
    else:
        print("Learning a fresh BPE tokenizer from Version 3 training paragraphs...")
        tokenizer_sample = train_paragraphs.copy()
        random.shuffle(tokenizer_sample)
        tokenizer_sample = tokenizer_sample[:30_000]
        tokenizer = BPETokenizer.train(
            "\n".join(tokenizer_sample),
            vocabulary_size=VOCABULARY_SIZE,
            lowercase=False,
        )
        tokenizer.save(tokenizer_path)
        print(f"Saved separate tokenizer to {tokenizer_path}.")

    print("Selecting a shuffled training sample close to the Version 2 token budget...")
    random.shuffle(train_paragraphs)
    selected_paragraphs = []
    selected_token_count = 0
    word_cache = {}
    for paragraph in train_paragraphs:
        paragraph_ids = []
        for word in split_words(paragraph, lowercase=tokenizer.lowercase):
            if word not in word_cache:
                word_cache[word] = tokenizer._encode_word(word)
            paragraph_ids.extend(word_cache[word])
        paragraph_ids.append(tokenizer.eos_id)
        if selected_paragraphs and selected_token_count >= TRAIN_TOKEN_BUDGET:
            break
        selected_paragraphs.append(paragraph_ids)
        selected_token_count += len(paragraph_ids)

    train_ids = torch.tensor(
        [token for paragraph_ids in selected_paragraphs for token in paragraph_ids],
        dtype=torch.long,
    )
    validation_ids = encode_paragraphs(validation_paragraphs, tokenizer)
    test_ids = encode_paragraphs(test_paragraphs, tokenizer)
    del train_paragraphs, validation_paragraphs, test_paragraphs
    print(
        f"Tokens with paragraph EOS: train={len(train_ids):,}, "
        f"validation={len(validation_ids):,}, test={len(test_ids):,}"
    )

    if args.dry_run:
        print("Dry run complete; no model weights were initialized or trained.")
        return

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
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(20, total_steps // 20)
    best_validation_loss = float("inf")
    history = []
    print(f"Model parameters: {parameter_count:,}")
    print(f"Training steps: {steps_per_epoch:,} per epoch x {EPOCHS} epochs")
    initial_validation_loss = evaluate(
        model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH,
        batches=VALIDATION_BATCHES, seed=SEED + 1,
    )
    print(f"Initial validation loss: {initial_validation_loss:.3f}")

    step = 0
    for epoch in range(1, EPOCHS + 1):
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
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train {train_loss:.3f} | "
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
                "training_history": history.copy(),
                "dataset": "FineWeb-Edu sample-10BT shard, filtered paragraphs",
            }, CHECKPOINT_FILE)
            print(f"  Saved separate Version 3 checkpoint to {CHECKPOINT_FILE}.", flush=True)

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(
        model, test_ids, BATCH_SIZE, CONTEXT_LENGTH,
        batches=TEST_BATCHES, seed=SEED + 2,
    )
    print(f"Best Version 3 epoch: {checkpoint['epoch']}")
    print(f"Initial validation loss: {initial_validation_loss:.3f}")
    print(f"Best validation loss: {checkpoint['validation_loss']:.3f}")
    print(f"Held-out test loss (200 fixed batches): {test_loss:.3f}")


if __name__ == "__main__":
    main()
