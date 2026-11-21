"""Filter FineWeb-Edu web documents into clean multi-sentence paragraphs."""

import argparse
import hashlib
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_SOURCE = "v3_fineweb_edu_sample_013.parquet"
MIN_SCORE = 3.5
MIN_CHARACTERS = 180
MAX_CHARACTERS = 1800
MIN_WORDS = 35
MAX_WORDS = 280

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9_.:-]*)(?:\s+[^<>]*?)?\s*/?>")
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'”’)]*)?(?:\s|$)")
URL_OR_MARKUP_RE = re.compile(r"https?://|www\.|<[^>]+>|\[[0-9]{1,3}\]", re.IGNORECASE)
CITATION_RE = re.compile(
    r"\b(?:doi\s*:|references\s*:|bibliography\s*:|et al\.)"
    r"|\([A-Z][A-Za-z.'-]+(?:\s+et al\.)?,?\s+(?:19|20)\d{2}[a-z]?\)",
    re.IGNORECASE,
)
LIST_START_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
BOILERPLATE_RE = re.compile(
    r"\b(?:share this article|previous abstract|next abstract|click here|"
    r"read more|all rights reserved|cookie policy|privacy policy|"
    r"created\s*[|:]\s*updated|leave a comment|comment on paragraph|"
    r"\d+\s+comments?|\d[,.]?\d*\s+views?|slide\s+\d+\s+of\s+\d+)\b",
    re.IGNORECASE,
)
PARAGRAPH_START_RE = re.compile(r"^(?:[\"'“‘(\[]\s*)?[A-Z]")
PARAGRAPH_END_RE = re.compile(r"[.!?][\"'”’)\]]*$")
MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": "“", "â€\x9d": "”",
    "â€“": "–", "â€”": "—", "â€¦": "…",
}


def clean_paragraph(raw_text):
    """Return a usable multi-sentence paragraph, or None for noisy fragments."""
    text = html.unescape(raw_text)
    for broken, corrected in MOJIBAKE.items():
        text = text.replace(broken, corrected)
    text = HTML_TAG_RE.sub(r"\1", text)
    text = CONTROL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS:
        return None
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return None
    if len(SENTENCE_END_RE.findall(text)) < 2:
        return None
    if not PARAGRAPH_START_RE.search(text) or not PARAGRAPH_END_RE.search(text):
        return None
    if LIST_START_RE.match(text):
        return None
    if URL_OR_MARKUP_RE.search(text) or CITATION_RE.search(text):
        return None
    if BOILERPLATE_RE.search(text):
        return None
    if text.count("|") or text.count(";") > 4:
        return None
    if sum(character.isalpha() for character in text) / len(text) < 0.45:
        return None
    return text


def split_for_document(document_id):
    """Make a stable 90/5/5 split so a page never leaks between datasets."""
    digest = hashlib.blake2b(document_id.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def prepare_dataset(source_path=DEFAULT_SOURCE):
    source_path = Path(source_path)
    parquet = pq.ParquetFile(source_path)
    outputs = {
        "train": Path("v3_train.txt"),
        "validation": Path("v3_validation.txt"),
        "test": Path("v3_test.txt"),
    }
    counts = Counter()
    seen_paragraphs = set()

    with outputs["train"].open("w", encoding="utf-8", newline="\n") as train_file, \
            outputs["validation"].open("w", encoding="utf-8", newline="\n") as validation_file, \
            outputs["test"].open("w", encoding="utf-8", newline="\n") as test_file:
        files = {"train": train_file, "validation": validation_file, "test": test_file}
        for group_index in range(parquet.metadata.num_row_groups):
            rows = parquet.read_row_group(
                group_index, columns=["text", "id", "score"]
            ).to_pylist()
            for row in rows:
                counts["documents_seen"] += 1
                if row["score"] < MIN_SCORE:
                    counts["documents_below_score"] += 1
                    continue
                counts["documents_kept_by_score"] += 1
                split = split_for_document(row["id"])
                for raw_paragraph in row["text"].splitlines():
                    paragraph = clean_paragraph(raw_paragraph)
                    if paragraph is None:
                        counts["paragraphs_rejected"] += 1
                        continue
                    fingerprint = hashlib.blake2b(
                        paragraph.encode("utf-8"), digest_size=16
                    ).digest()
                    if fingerprint in seen_paragraphs:
                        counts["duplicate_paragraphs"] += 1
                        continue
                    seen_paragraphs.add(fingerprint)
                    files[split].write(paragraph + "\n")
                    counts[f"{split}_paragraphs"] += 1

    if not counts["train_paragraphs"] or not counts["validation_paragraphs"] or not counts["test_paragraphs"]:
        raise ValueError("Filtering did not leave paragraphs in all three splits.")
    for split, path in outputs.items():
        counts[f"{split}_file_bytes"] = path.stat().st_size
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    counts = prepare_dataset(args.source)
    for name, value in counts.items():
        print(f"{name}: {value}")
    print("Paragraph splits saved to v3_train.txt, v3_validation.txt, and v3_test.txt.")


if __name__ == "__main__":
    main()
