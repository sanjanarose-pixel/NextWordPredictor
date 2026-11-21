"""Convert the already-local WikiText-2 splits into independent V4 articles."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from v4_data import DATA_ROOT

ROOT = Path(__file__).resolve().parent
OUT = DATA_ROOT / "prepared" / "wikitext_documents.jsonl"
MANIFEST = DATA_ROOT / "wikitext_manifest.json"
SPLITS = {"train": ROOT / "dataset.txt", "validation": ROOT / "validation.txt",
          "test": ROOT / "test.txt"}
ARTICLE = re.compile(r"^=\s+([^=].*?)\s+=$")
SECTION = re.compile(r"^[=\s]+(.+?)[=\s]+$")


def detokenize_line(value: str) -> str:
    value = re.sub(r"\s*@-@\s*", "-", value)
    value = re.sub(r"\s*@,@\s*", ", ", value)
    value = re.sub(r"\s*@\.@\s*", ".", value)
    value = re.sub(r"\s+([,.;:!?%])", r"\1", value)
    value = re.sub(r"([([{])\s+", r"\1", value)
    value = re.sub(r"\s+([)\]}])", r"\1", value)
    value = re.sub(r"\s+(['’])\s*(?=(?:s|re|ve|d|ll|m|t|n)\b)", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r'"\s*(.*?)\s*"', r'"\1"', value)
    value = re.sub(r"\s+", " ", value).strip()
    return unicodedata.normalize("NFC", value)


def _finish_doc(title: str, lines: list[str], split: str) -> dict | None:
    text = "\n\n".join(item for item in lines if item).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 500 or len(text) > 500_000:
        return None
    if "\ufffd" in text or "\x00" in text:
        raise ValueError(f"Corruption in {split} article {title!r}")
    return {"id": f"wikitext:{split}:{title}", "title": title, "author": "Wikipedia contributors",
            "author_group": f"wiki_article:{title.casefold()}",
            "split": split, "source": "WikiText-2 raw v1 / English Wikipedia Good and Featured articles",
            "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
            "license": "CC BY-SA 3.0 (verify source article notices for individual credits)", "text": text}


def parse_split(split: str, path: Path) -> tuple[list[dict], Counter]:
    if not path.is_file():
        raise FileNotFoundError(path)
    docs, counts = [], Counter()
    corrupted_titles = []
    title, body = None, []

    def flush():
        nonlocal title, body
        if title is not None:
            try:
                doc = _finish_doc(title, body, split)
            except ValueError:
                corrupted_titles.append(title)
                counts["corrupted_articles_excluded"] += 1
            else:
                if doc is None:
                    counts["short_or_outlier_articles"] += 1
                else:
                    doc["id"] = f"wikitext:{split}:{len(docs)}:{title}"
                    docs.append(doc)
        title, body = None, []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        article = ARTICLE.match(line)
        section = SECTION.match(line)
        if article:
            flush()
            title = detokenize_line(article.group(1))
            counts["article_headers"] += 1
        elif section and title is not None and not article:
            heading = detokenize_line(section.group(1).strip())
            if heading:
                body.append(heading)
                body.append("")
        elif title is not None:
            if not line:
                body.append("")
            else:
                body.append(detokenize_line(line))
    flush()
    counts["documents_kept"] = len(docs)
    counts["corrupted_titles"] = corrupted_titles
    counts["source_bytes"] = path.stat().st_size
    counts["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return docs, counts


def fingerprint(text: str) -> str:
    words = re.findall(r"\w+", text.casefold())
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()


def prepare() -> dict:
    all_docs, stats = [], {}
    for split, path in SPLITS.items():
        docs, counts = parse_split(split, path)
        all_docs.extend(docs)
        stats[split] = dict(counts)

    seen: dict[str, str] = {}
    duplicates = []
    for doc in all_docs:
        key = fingerprint(doc["text"])
        if key in seen:
            duplicates.append((seen[key], doc["id"]))
        else:
            seen[key] = doc["id"]
    if duplicates:
        raise ValueError(f"Duplicate article documents detected: {duplicates[:10]}")
    if not all(any(d["split"] == split for d in all_docs) for split in SPLITS):
        raise ValueError("Every split must contain at least one usable article")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in all_docs), encoding="utf-8")
    tmp.replace(OUT)
    manifest = {"dataset": "WikiText-2 raw v1", "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
                "license": "CC BY-SA 3.0 according to source repository metadata",
                "attribution": "Salesforce AI Research; WikiText-2 from English Wikipedia Good and Featured articles.",
                "cleaning": ["per-file official train/validation/test split retained",
                             "article-level documents parsed from single-equals headers",
                             "section headings retained as prose headers",
                             "Moses @-@/@,@/@.@ codes detokenized",
                             "blank lines retained as paragraph boundaries",
                             "articles shorter than 500 chars or over 500,000 chars excluded",
                             "word-normalized exact duplicates checked across every split"],
                "statistics": stats, "documents_total": len(all_docs), "duplicate_pairs": duplicates,
                "documents_jsonl": str(OUT)}
    temp_manifest = MANIFEST.with_suffix(".tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_manifest.replace(MANIFEST)
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
