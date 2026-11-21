"""Short CPU throughput check using actual V4 training-document windows."""
from __future__ import annotations

import ctypes
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from train_v4_model import DEFAULT_DOCS, collate, encode_splits, read_documents
from v4_model import MODEL_CONFIG, V4LanguageModel, masked_next_token_loss, parameter_count
from v4_tokenizer import V4BPETokenizer


def current_memory_bytes():
    if not hasattr(ctypes, "windll"):
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    f = ctypes.windll.psapi.GetProcessMemoryInfo
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    f.restype = ctypes.c_int
    p = ctypes.windll.kernel32.GetCurrentProcess
    p.restype = ctypes.c_void_p
    return int(c.WorkingSetSize) if f(p(), ctypes.byref(c), c.cb) else None


def run(documents=DEFAULT_DOCS, tokenizer_path=None, vocab_size=2048, dimension=128,
        context=128, heads=4, blocks=3, repeats=1, warmup=10, measured=60,
        batch_size=16, threads=4):
    torch.set_num_threads(threads)
    docs = read_documents(Path(documents), require_splits=False)
    train_docs = [d for d in docs if d["split"] == "train"]
    tokenizer = (V4BPETokenizer.load(tokenizer_path) if tokenizer_path else
                 V4BPETokenizer.train([d["text"] for d in train_docs], vocab_size))
    config = {"vocab_size": len(tokenizer.symbols), "dimension": dimension,
              "context_length": context, "number_of_heads": heads,
              "number_of_blocks": blocks, "dropout": 0.1}
    splits = encode_splits(docs, tokenizer, context)
    windows = splits["train"]["windows"]
    if len(windows) < batch_size:
        raise ValueError("Not enough independent-document training windows")
    rows = windows.copy()
    random.Random(631).shuffle(rows)
    model_params = parameter_count(V4LanguageModel(**config, pad_id=tokenizer.pad_id))
    reports = []
    for rep in range(repeats):
        torch.manual_seed(900 + rep)
        model = V4LanguageModel(**config, pad_id=tokenizer.pad_id).to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        step_index = 0

        def step():
            nonlocal step_index
            start = (step_index * batch_size) % len(rows)
            batch_rows = rows[start:start + batch_size]
            if len(batch_rows) < batch_size:
                batch_rows += rows[:batch_size - len(batch_rows)]
            x, y, mask = collate(batch_rows, tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_next_token_loss(model(x, mask), y, tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_index += 1
            return int(y.ne(tokenizer.pad_id).sum()), float(loss.detach())

        model.train()
        for _ in range(warmup):
            step()
        start_time = time.perf_counter()
        valid_tokens = 0
        losses = []
        for _ in range(measured):
            n, loss = step()
            valid_tokens += n
            losses.append(loss)
        duration = time.perf_counter() - start_time
        reports.append({"repeat": rep + 1, "warmup_steps": warmup, "measured_steps": measured,
                        "seconds_per_step": duration / measured,
                        "valid_tokens_per_second": valid_tokens / duration,
                        "mean_loss": sum(losses) / len(losses),
                        "working_set_bytes_after_pass": current_memory_bytes()})
        del model, optimizer
    return {"device": "cpu", "threads": threads, "batch_size": batch_size,
            "architecture": {**config, "pad_id": tokenizer.pad_id},
            "parameter_count": model_params,
            "training_documents": len(train_docs), "training_tokens": splits["train"]["tokens"],
            "training_windows": len(windows), "steps_per_epoch": math.ceil(len(windows) / batch_size),
            "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default=str(DEFAULT_DOCS))
    parser.add_argument("--tokenizer", dest="tokenizer_path")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
