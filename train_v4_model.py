"""Train a small V4 model from local document JSONL with best-validation saving."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from v4_data import DATA_ROOT, make_windows
from v4_model import MODEL_CONFIG, V4LanguageModel, masked_next_token_loss, parameter_count
from v4_tokenizer import V4BPETokenizer

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "v4"
DEFAULT_DOCS = DATA_ROOT / "prepared" / "tinystories_documents.jsonl"


def read_documents(path: Path, require_splits: bool = True) -> list[dict]:
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate document IDs")
    groups: dict[str, set[str]] = {}
    for doc in docs:
        group = doc.get("author_group", doc["author"])
        groups.setdefault(group, set()).add(doc["split"])
    leakage = {group: splits for group, splits in groups.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Document/author leakage across splits: {leakage}")
    if require_splits and not {"train", "validation", "test"}.issubset({d["split"] for d in docs}):
        raise ValueError("Train, validation, and test documents are all required")
    return docs


def encode_splits(docs: list[dict], tokenizer: V4BPETokenizer, context: int):
    result = {}
    for split in ("train", "validation", "test"):
        split_docs = [d for d in docs if d["split"] == split]
        windows, token_count = [], 0
        for doc in split_docs:
            ids = tokenizer.encode(doc["text"])
            token_count += len(ids) + 1  # one EOS target per independent article
            windows.extend(make_windows(ids, context, tokenizer.eos_id))
        result[split] = {"documents": split_docs, "windows": windows, "tokens": token_count}
    return result


def collate(rows, pad_id):
    batch = pad_sequence([torch.tensor(row, dtype=torch.long) for row in rows],
                         batch_first=True, padding_value=pad_id)
    x, y = batch[:, :-1], batch[:, 1:]
    return x, y, x.ne(pad_id)


def atomic_save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate_loss(model, windows, pad_id, batch_size=16):
    model.eval()
    total_loss = total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            x, y, mask = collate(windows[start:start + batch_size], pad_id)
            logits = model(x, mask)
            count = int(y.ne(pad_id).sum())
            loss = masked_next_token_loss(logits, y, pad_id)
            total_loss += float(loss) * count
            total_tokens += count
    return total_loss / max(total_tokens, 1)


def train(args):
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    docs = read_documents(Path(args.documents))
    train_texts = [d["text"] for d in docs if d["split"] == "train"]
    if not train_texts:
        raise ValueError("Training split is empty")

    tokenizer = (V4BPETokenizer.load(args.tokenizer_path) if args.tokenizer_path else
                 V4BPETokenizer.train(train_texts, args.vocab_size))
    if len(tokenizer.symbols) != args.vocab_size:
        raise ValueError("Supplied tokenizer vocabulary does not match --vocab-size")
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ARTIFACT_ROOT / f"tokenizer_v4_{run_id}.json"
    tokenizer.save(tokenizer_path)
    splits = encode_splits(docs, tokenizer, args.context)
    model = V4LanguageModel(vocab_size=len(tokenizer.symbols), dimension=args.dimension,
                            context_length=args.context, number_of_heads=args.heads,
                            number_of_blocks=args.blocks, dropout=args.dropout,
                            pad_id=tokenizer.pad_id).to("cpu")
    initial = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        if initial.get("format") != "nextword-v4" or initial.get("config") != model.config:
            raise ValueError("Initial checkpoint does not match the selected V4 architecture")
        if not args.tokenizer_path or initial.get("tokenizer_file") != Path(args.tokenizer_path).name:
            raise ValueError("Resume requires the exact tokenizer paired with the initial checkpoint")
        model.load_state_dict(initial["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_windows = splits["train"]["windows"]
    if not train_windows:
        raise ValueError("Training documents yield no windows")

    steps_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    best_val = float(initial.get("best_validation_loss", float("inf"))) if initial else float("inf")
    best_state = ({name: value.detach().cpu().clone()
                   for name, value in model.state_dict().items()} if initial else None)
    best_epoch = int(initial.get("best_epoch", 0)) if initial else 0
    steps = int(initial.get("step", 0)) if initial else 0
    stale_epochs = 0
    history = list(initial.get("training_history", [])) if initial else []
    first_epoch = max((row["epoch"] for row in history), default=best_epoch) + 1
    for epoch in range(first_epoch, first_epoch + args.epochs):
        model.train()
        order_generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_windows), generator=order_generator).tolist()
        total_train_loss = total_train_tokens = 0
        for start in range(0, len(order), args.batch_size):
            rows = [train_windows[i] for i in order[start:start + args.batch_size]]
            x, y, mask = collate(rows, tokenizer.pad_id)
            logits = model(x, mask)
            loss = masked_next_token_loss(logits, y, tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int(y.ne(tokenizer.pad_id).sum())
            total_train_loss += float(loss.detach()) * n
            total_train_tokens += n
            steps += 1
            if steps >= args.max_steps:
                break

        val_loss = evaluate_loss(model, splits["validation"]["windows"], tokenizer.pad_id, args.batch_size)
        train_loss = total_train_loss / max(total_train_tokens, 1)
        history.append({"epoch": epoch, "steps": steps, "train_loss": train_loss,
                        "validation_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_val:
            best_val, best_epoch, stale_epochs = val_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience or steps >= args.max_steps:
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    test_loss = evaluate_loss(model, splits["test"]["windows"], tokenizer.pad_id, args.batch_size)
    checkpoint_path = ARTIFACT_ROOT / f"v4_best_{run_id}.pth"
    payload = {"format": "nextword-v4", "config": model.config,
               "model_state_dict": best_state, "tokenizer_file": tokenizer_path.name,
               "step": steps, "best_epoch": best_epoch, "seed": args.seed,
               "best_validation_loss": best_val, "test_loss": test_loss,
               "split_token_counts": {k: v["tokens"] for k, v in splits.items()},
               "dataset": "fhswf/TinyStoriesV2_cleaned, deterministic story-level subset",
               "training_history": history,
               "resumed_from": Path(args.initial_checkpoint).name if initial else None,
               "optimizer_state_reinitialized": bool(initial)}
    atomic_save(checkpoint_path, payload)
    summary = {"dataset_documents": {k: len(v["documents"]) for k, v in splits.items()},
               "actual_bpe_tokens_including_eos": {k: v["tokens"] for k, v in splits.items()},
               "windows_per_split": {k: len(v["windows"]) for k, v in splits.items()},
               "steps_per_epoch": steps_per_epoch, "completed_steps": steps,
               "best_epoch": best_epoch, "training_history": history,
               "best_validation_loss": best_val, "test_loss": test_loss,
               "model_parameters": parameter_count(model), "model_config": model.config,
               "checkpoint": str(checkpoint_path), "tokenizer": str(tokenizer_path)}
    summary_path = ARTIFACT_ROOT / f"v4_results_{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default=str(DEFAULT_DOCS))
    parser.add_argument("--tokenizer-path", default=None,
                        help="Reuse a training-only V4 tokenizer fitted during token count/benchmark")
    parser.add_argument("--initial-checkpoint", default=None,
                        help="Continue a V4 run from its best checkpoint (optimizer state restarts)")
    parser.add_argument("--vocab-size", type=int, default=MODEL_CONFIG["vocab_size"])
    parser.add_argument("--dimension", type=int, default=MODEL_CONFIG["dimension"])
    parser.add_argument("--context", type=int, default=MODEL_CONFIG["context_length"])
    parser.add_argument("--heads", type=int, default=MODEL_CONFIG["number_of_heads"])
    parser.add_argument("--blocks", type=int, default=MODEL_CONFIG["number_of_blocks"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    return parser


if __name__ == "__main__":
    train(parse_args().parse_args())
