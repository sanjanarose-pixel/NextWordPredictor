"""Isolated V4 source registry, Gutenberg cleaning, splitting, and windows."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "v4"
RAW = DATA_ROOT / "raw"
PREPARED = DATA_ROOT / "prepared"
MANIFEST = DATA_ROOT / "manifest.json"
EOS = "<eos>"


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    author: str
    gutenberg_id: int
    split: str

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt.utf8"


# IDs point to English text editions, not audio, translation, or illustrated editions.
NOVELS = [
    Source("bleak_house", "Bleak House", "Charles Dickens", 1023, "train"),
    Source("david_copperfield", "David Copperfield", "Charles Dickens", 766, "train"),
    Source("great_expectations", "Great Expectations", "Charles Dickens", 1400, "train"),
    Source("oliver_twist", "Oliver Twist", "Charles Dickens", 730, "train"),
    Source("tale_two_cities", "A Tale of Two Cities", "Charles Dickens", 98, "train"),
    Source("middlemarch", "Middlemarch", "George Eliot", 145, "train"),
    Source("vanity_fair", "Vanity Fair", "William Makepeace Thackeray", 5998, "train"),
    Source("moby_dick", "Moby-Dick", "Herman Melville", 2701, "train"),
    Source("jane_eyre", "Jane Eyre", "Charlotte Brontë", 1260, "train"),
    Source("little_women", "Little Women", "Louisa May Alcott", 514, "train"),
    Source("north_south", "North and South", "Elizabeth Gaskell", 4276, "train"),
    Source("dracula", "Dracula", "Bram Stoker", 345, "train"),
    Source("tess", "Tess of the d’Urbervilles", "Thomas Hardy", 1183, "train"),
    Source("wuthering_heights", "Wuthering Heights", "Emily Brontë", 768, "train"),
    Source("anne_green_gables", "Anne of Green Gables", "L. M. Montgomery", 45, "train"),
    Source("secret_garden", "The Secret Garden", "Frances Hodgson Burnett", 113, "train"),
    Source("frankenstein", "Frankenstein", "Mary Shelley", 84, "train"),
    Source("treasure_island", "Treasure Island", "Robert Louis Stevenson", 120, "train"),
    Source("wizard_oz", "The Wonderful Wizard of Oz", "L. Frank Baum", 55, "train"),
    Source("call_wild", "The Call of the Wild", "Jack London", 215, "train"),
    Source("alice", "Alice’s Adventures in Wonderland", "Lewis Carroll", 11, "train"),
    Source("emma", "Emma", "Jane Austen", 158, "validation"),
    Source("sense_sensibility", "Sense and Sensibility", "Jane Austen", 161, "validation"),
    Source("pride_prejudice", "Pride and Prejudice", "Jane Austen", 1342, "validation"),
    Source("huckleberry_finn", "Adventures of Huckleberry Finn", "Mark Twain", 76, "test"),
    Source("tom_sawyer", "The Adventures of Tom Sawyer", "Mark Twain", 74, "test"),
    Source("sherlock_holmes", "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", 1661, "test"),
    Source("hound_baskervilles", "The Hound of the Baskervilles", "Arthur Conan Doyle", 285, "test"),
    Source("war_worlds", "The War of the Worlds", "H. G. Wells", 36, "test"),
    Source("time_machine", "The Time Machine", "H. G. Wells", 35, "test"),
]

OHENRY_COLLECTIONS = {
    # Selected official Gutenberg texts; IDs must be checked against the source
    # pages before acquisition. None are downloaded from mirrors.
    "The Four Million": 2776,
    "The Voice of the City": 1444,
    "The Trimmed Lamp": 3707,
    "Options": 1583,
    "Whirligigs": 1595,
    "Roads of Destiny": 1646,
    "Heart of the West": 1725,
    "The Gentle Grafter": 1805,
    "Strictly Business": 2141,
    "Waifs and Strays": 2295,
    "Cabbages and Kings": 2775,
    "Sixes and Sevens": 2851,
    "Rolling Stones": 3815,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source: Source, timeout: int = 30) -> Path:
    """Download one official PG plain-text file. Existing files are never overwritten."""
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"pg{source.gutenberg_id}.txt.utf8"
    if target.exists():
        return target
    request = urllib.request.Request(source.url, headers={"User-Agent": "NextWordPredictor-V4/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("xb") as out:
        out.write(response.read())
    return target


def clean_gutenberg(raw: str, minimum_chars: int = 10_000) -> str:
    """Strip only standard PG wrapper; preserve the work's internal layout."""
    text = raw.lstrip("\ufeff")
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Missing or malformed Project Gutenberg start/end markers")
    text = text[start.end():end.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    # Normalize only canonically equivalent Unicode; don't flatten whitespace.
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) < minimum_chars:
        raise ValueError(f"Suspiciously short extracted work ({len(text)} characters)")
    if text.count("\ufffd") or text.count("�"):
        raise ValueError("Replacement-character corruption detected")
    return text


def duplicate_groups(documents: list[dict]) -> dict[str, list[str]]:
    """Return normalized-text duplicate groups without discarding any source record."""
    fingerprints: dict[str, list[str]] = {}
    for doc in documents:
        normalized = re.sub(r"\s+", " ", doc["text"]).strip().casefold()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        fingerprints.setdefault(digest, []).append(doc["id"])
    return {digest: ids for digest, ids in fingerprints.items() if len(ids) > 1}


def validate_document_lengths(documents: list[dict], min_chars: int = 1500,
                              max_chars: int = 5_000_000) -> list[str]:
    """Flag length outliers for review; callers must not silently delete them."""
    return [doc["id"] for doc in documents if not min_chars <= len(doc.get("text", "")) <= max_chars]


def extract_stories(collection_text: str, title: str) -> list[tuple[str, str]]:
    """Conservative heading-based extraction; ambiguous contents are rejected for review."""
    # The Gutenberg collection is an anthology. Heading markup/uppercase contents
    # provides exact story titles in current editions; content is segmented at
    # repeated title headings. If any listed heading is absent, do not guess.
    start = re.search(r"(?im)^\s*CONTENTS\s*$", collection_text)
    if not start:
        raise ValueError(f"No contents section found for {title}")
    body = collection_text[start.end():]
    headings = []
    for match in re.finditer(r"(?m)^\s{0,8}([A-Z][A-Z0-9’'.,:;!?() -]{2,90})\s*$", body):
        heading = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if heading not in {"CONTENTS", "TABLE OF CONTENTS"}:
            headings.append((heading, match.start(), match.end()))
    # This deliberately returns only high-confidence body segments; editorial
    # front matter and unrecognized contents cannot silently become training text.
    stories = []
    for i, (heading, _, body_start) in enumerate(headings):
        next_start = headings[i + 1][1] if i + 1 < len(headings) else len(body)
        story = body[body_start:next_start].strip()
        if len(story) >= 1500 and not re.search(r"(?i)copyright|publisher|contents", story[:500]):
            stories.append((heading.title(), story))
    return stories


def make_windows(tokens: list[int], context: int, eos_id: int, stride: int | None = None) -> list[list[int]]:
    """Independent within-document next-token windows; never pads across docs."""
    if context < 2:
        raise ValueError("context must be at least 2")
    stride = stride or context - 1
    if stride < 1 or stride > context - 1:
        raise ValueError("stride must be in [1, context-1]")
    sequence = list(tokens)
    if not sequence or sequence[-1] != eos_id:
        sequence.append(eos_id)
    return [sequence[start:start + context + 1] for start in range(0, max(1, len(sequence) - 1), stride)
            if len(sequence[start:start + context + 1]) >= 2]


def validate_author_splits(sources: Iterable[Source]) -> None:
    author_splits: dict[str, set[str]] = {}
    for source in sources:
        author_splits.setdefault(source.author, set()).add(source.split)
    leaked = {author: splits for author, splits in author_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Author leakage across splits: {leaked}")


def save_manifest(records: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
