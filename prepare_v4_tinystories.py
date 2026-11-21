"""Prepare a deterministic, story-independent TinyStories V4 subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "v4" / "raw" / "tinystories"
OUT = ROOT / "data" / "v4" / "prepared" / "tinystories_documents.jsonl"
MANIFEST = ROOT / "data" / "v4" / "tinystories_manifest.json"
END = "<|endoftext|>"
SOURCE = "fhswf/TinyStoriesV2_cleaned"
TRAIN_COUNT = 27_500
TRAIN_SPLIT_COUNT = 25_000
TEST_COUNT = 2_500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_story_file(path: Path, split_name: str, source_file: str):
    raw = path.read_text(encoding="utf-8", errors="strict")
    if END not in raw:
        raise ValueError(f"No story separators found in {path}")
    if raw.count(END) < 1:
        raise ValueError(f"Unexpectedly few stories in {path}")
    stories = []
    seen: set[str] = set()
    empty = duplicate = corrupt = 0
    for index, part in enumerate(raw.split(END)):
        text = part.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            empty += 1
            continue
        if "\x00" in text or "\ufffd" in text:
            corrupt += 1
            continue
        key = fingerprint(text)
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        stories.append({"source_index": index, "text": text, "fingerprint": key,
                        "selection_key": hashlib.sha256(("tinystories-v4-v1:" + key).encode()).hexdigest(),
                        "split_source": split_name, "source_file": source_file})
    return stories, {"markers": raw.count(END), "nonempty_unique_stories": len(stories),
                     "empty_segments": empty, "within_file_duplicates_removed": duplicate,
                     "corrupt_stories_removed": corrupt}


def prepare() -> dict:
    train_path = RAW / "train_part1.txt"
    test_path = RAW / "test_part1.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected official source shards under {RAW}")

    train_source, train_stats = read_story_file(
        train_path, "train", "data/train/part1.txt")
    test_source, test_stats = read_story_file(
        test_path, "test", "data/test/part1.txt")
    train_keys = {item["fingerprint"] for item in train_source}
    cross_duplicates = [item for item in test_source if item["fingerprint"] in train_keys]
    if cross_duplicates:
        duplicate_keys = {item["fingerprint"] for item in cross_duplicates}
        test_source = [item for item in test_source if item["fingerprint"] not in duplicate_keys]

    train_candidates = sorted(train_source, key=lambda item: item["selection_key"])
    test_candidates = sorted(test_source, key=lambda item: item["selection_key"])
    if len(train_candidates) < TRAIN_COUNT or len(test_candidates) < TEST_COUNT:
        raise ValueError("The selected official shards do not contain the requested unique story counts")
    chosen_train = train_candidates[:TRAIN_COUNT]
    chosen_test = test_candidates[:TEST_COUNT]

    docs = []
    for i, item in enumerate(chosen_train):
        split = "train" if i < TRAIN_SPLIT_COUNT else "validation"
        docs.append({"id": f"tiny_train_{item['fingerprint'][:16]}",
                     "author": "GPT-4-generated synthetic story (dataset attribution in manifest)",
                     "author_group": f"story:{item['fingerprint']}",
                     "title": f"TinyStories train record {item['source_index']}",
                     "split": split, "source": f"{SOURCE}/{item['source_file']}",
                     "text": item["text"]})
    for item in chosen_test:
        docs.append({"id": f"tiny_test_{item['fingerprint'][:16]}",
                     "author": "GPT-4-generated synthetic story (dataset attribution in manifest)",
                     "author_group": f"story:{item['fingerprint']}",
                     "title": f"TinyStories test record {item['source_index']}",
                     "split": "test", "source": f"{SOURCE}/{item['source_file']}",
                     "text": item["text"]})
    doc_ids = [doc["id"] for doc in docs]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Duplicate prepared document IDs")
    if any(END in doc["text"] for doc in docs):
        raise ValueError("A story separator remained inside a document")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs),
                   encoding="utf-8")
    counts = {split: sum(doc["split"] == split for doc in docs)
              for split in ("train", "validation", "test")}
    manifest = {
        "dataset": SOURCE,
        "dataset_url": "https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned",
        "description": "Synthetic GPT-4-generated children's stories, cleaned by dataset publisher.",
        "license": "CDLA-Sharing-1.0",
        "license_url": "https://cdla.dev/sharing-1-0/",
        "attribution": "TinyStoriesV2_cleaned by Fachhochschule Südwestfalen (fhswf); source stories generated by GPT-4. Original TinyStories authors are acknowledged by the dataset publisher.",
        "sharing_notes": "If publishing source or modified data, publish under unmodified CDLA-Sharing-1.0, preserve provider attribution/links, and prominently mark modifications. CDLA section 3.5 places no obligations on Results that contain no more than a de minimis portion of source data.",
        "sources": {
            "train": {"path": "data/train/part1.txt", "bytes": train_path.stat().st_size,
                      "sha256": sha256_file(train_path), **train_stats},
            "test": {"path": "data/test/part1.txt", "bytes": test_path.stat().st_size,
                     "sha256": sha256_file(test_path), **test_stats},
        },
        "selection": {
            "method": "Sort unique story text SHA-256 selection keys ascending; take the first requested records. Train/validation are disjoint partitions of the selected official train shard; test stories come only from the official test shard.",
            "seed_label": "tinystories-v4-v1",
            "train_shard_unique_stories_available": len(train_source),
            "test_shard_unique_stories_available": len(test_source),
            "train_selected": TRAIN_SPLIT_COUNT,
            "validation_selected_from_train_shard": TRAIN_COUNT - TRAIN_SPLIT_COUNT,
            "test_selected_from_official_test": len(chosen_test),
            "cross_split_duplicate_stories_removed": len(cross_duplicates),
            "documents_total": len(docs), "document_counts": counts,
            "document_jsonl": str(OUT),
            "preservation": "Split on publisher <|endoftext|> separator, discard only that terminal marker, normalize line endings to LF, trim only outer whitespace; internal paragraph newlines, dialogue, punctuation, and sentence text are retained.",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"document_counts": counts, "train_source_stats": train_stats,
                      "test_source_stats": test_stats,
                      "cross_split_duplicate_stories_removed": len(cross_duplicates),
                      "documents_jsonl": str(OUT), "manifest": str(MANIFEST)}, indent=2))
    return manifest


if __name__ == "__main__":
    prepare()
