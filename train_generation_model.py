"""Train a small text generator from scratch on the local WikiText-2 corpus."""

import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generation_model import SmallLanguageModel
from tokenizer import BPETokenizer


TRAIN_FILE = "dataset.txt"
VALIDATION_FILE = "validation.txt"
TOKENIZER_FILE = "generation_tokenizer.json"
CHECKPOINT_FILE = "generation_model.pth"

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


def clean_wikitext(text):
    """Remove WikiText article-title markup and normalize its punctuation codes."""
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Raw WikiText titles look like '= Article title =' or '== Section =='.
        if re.fullmatch(r"(?:=+\s*){1,3}\S.*?\S(?:\s*=+){1,3}", line):
            continue
        line = line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def make_batch(token_ids, batch_size, context_length):
    starts = torch.randint(0, len(token_ids) - context_length - 1, (batch_size,))
    offsets = torch.arange(context_length + 1)
    windows = token_ids[starts[:, None] + offsets[None, :]]
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def evaluate(model, token_ids, batch_size, context_length, batches=40):
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = make_batch(token_ids, batch_size, context_length)
            logits = model(inputs)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")

    train_text = clean_wikitext(Path(TRAIN_FILE).read_text(encoding="utf-8"))
    validation_text = clean_wikitext(Path(VALIDATION_FILE).read_text(encoding="utf-8"))

    tokenizer_path = Path(TOKENIZER_FILE)
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Loaded BPE tokenizer: {len(tokenizer.symbols):,} pieces")
    else:
        print("Learning a BPE vocabulary from training text...")
        tokenizer = BPETokenizer.train(train_text, vocabulary_size=VOCABULARY_SIZE)
        tokenizer.save(tokenizer_path)
        print(f"Saved {len(tokenizer.symbols):,} pieces to {tokenizer_path}")

    print("Encoding training and validation text...")
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    print(f"Training pieces: {len(train_ids):,}; validation pieces: {len(validation_ids):,}")

    model = SmallLanguageModel(
        vocabulary_size=len(tokenizer.symbols),
        dimension=DIMENSION,
        context_length=CONTEXT_LENGTH,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_blocks=NUMBER_OF_BLOCKS,
        dropout=DROPOUT,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_ids) / (BATCH_SIZE * CONTEXT_LENGTH))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(20, total_steps // 20)
    best_validation_loss = float("inf")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Training steps: {steps_per_epoch:,} per epoch x {EPOCHS} epochs")
    print("Training from a fresh random initialization; no old checkpoint is loaded.")

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        start_time = time.perf_counter()
        for _ in range(steps_per_epoch):
            inputs, targets = make_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)

            if step < warmup_steps:
                learning_rate = LEARNING_RATE * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                learning_rate = LEARNING_RATE * 0.1 + 0.9 * LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            step += 1

        train_loss = sum(epoch_losses) / len(epoch_losses)
        validation_loss = evaluate(model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH)
        seconds = time.perf_counter() - start_time
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.3f} | "
            f"validation loss {validation_loss:.3f} | {seconds:.1f}s"
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
            }, CHECKPOINT_FILE)
            print(f"  Saved best checkpoint: {CHECKPOINT_FILE}")

    print(f"Best validation loss: {best_validation_loss:.3f}")


if __name__ == "__main__":
    main()
