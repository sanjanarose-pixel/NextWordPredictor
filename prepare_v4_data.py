"""Prepare locally acquired official V4 sources; no mirrors or existing corpus reuse."""
from __future__ import annotations

import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from v4_data import (DATA_ROOT, MANIFEST, NOVELS, OHENRY_COLLECTIONS, RAW, PREPARED,
                     clean_gutenberg, sha256, validate_author_splits)


class _ChapterText(HTMLParser):
    """Extract prose/heading text; omit images, figures, tables and attribution boxes."""
    SKIP_TAGS = {"figure", "table", "svg", "math", "script", "style", "nav", "aside"}
    BOUNDARIES = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "section", "div"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.stack = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").casefold()
        skip = tag in self.SKIP_TAGS or any(label in classes for label in ("figure", "caption", "credit", "copyright", "equation", "table"))
        self.stack.append((tag, skip))
        if skip:
            self.skip_depth += 1
        if self.skip_depth == 0 and tag in self.BOUNDARIES:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        while self.stack:
            open_tag, skipped = self.stack.pop()
            if skipped:
                self.skip_depth -= 1
            if open_tag == tag:
                break
        if self.skip_depth == 0 and tag in self.BOUNDARIES:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


ROBOTICS_ATTRIBUTION = (
    "Ben-Ari, Mordechai, and Francesco Mondada. Elements of Robotics. Springer, 2018. "
    "https://doi.org/10.1007/978-3-319-62533-1. CC BY 4.0, "
    "https://creativecommons.org/licenses/by/4.0/. Extracted prose only; figure/table/image content omitted."
)


def extract_robotics_epub(epub_path: Path) -> list[dict]:
    """Extract two selected chapter prose bodies; visual/marked credit content excluded."""
    if not epub_path.is_file():
        return []
    selected = {"A419055_1_En_2_Chapter.html": "Robots and Their Applications",
                "A419055_1_En_3_Chapter.html": "Sensors"}
    docs = []
    with zipfile.ZipFile(epub_path) as book:
        for member, title in selected.items():
            payload = book.read(f"OEBPS/{member}").decode("utf-8")
            parser = _ChapterText()
            parser.feed(payload)
            text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
            text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", text).strip()
            text = "\n\n".join(p for p in text.split("\n\n")
                                 if not re.match(r"(?i)\s*(image|figure|photo|illustration)\s+(credit|source)", p))
            if len(text) < 3000:
                raise ValueError(f"Robotics chapter extraction looks incomplete: {title} ({len(text)} chars)")
            docs.append({"id": f"robotics:{member}", "title": title, "author": "Mordechai Ben-Ari; Francesco Mondada",
                         "split": "train", "source": "Elements of Robotics, selected chapter prose",
                         "source_url": "https://link.springer.com/book/10.1007/978-3-319-62533-1",
                         "license": "CC BY 4.0", "attribution": ROBOTICS_ATTRIBUTION, "text": text})
    return docs


def prepare() -> dict:
    validate_author_splits(NOVELS)
    records, docs, missing = [], [], []
    # Read exact files that were separately downloaded from the official PG domain.
    for source in NOVELS:
        raw_path = RAW / f"pg{source.gutenberg_id}.txt.utf8"
        record = {"key": source.key, "title": source.title, "author": source.author,
                  "split": source.split, "source_url": source.url,
                  "license": "Project Gutenberg terms; catalog records public domain in USA;"
                             " check local-law status before redistribution",
                  "source_file": raw_path.name, "status": "missing"}
        if not raw_path.is_file():
            missing.append(source.key)
            records.append(record)
            continue
        raw = raw_path.read_text(encoding="utf-8-sig")
        cleaned = clean_gutenberg(raw)
        record.update({"sha256": sha256(raw_path), "raw_characters": len(raw),
                       "clean_characters": len(cleaned), "status": "prepared"})
        docs.append({"id": f"novel:{source.key}", "title": source.title, "author": source.author,
                     "split": source.split, "source": f"Project Gutenberg #{source.gutenberg_id}",
                     "source_url": source.url, "license": record["license"], "text": cleaned})
        records.append(record)

    # O. Henry requires anthology-level duplicate audit and human review of the
    # exact individual story manifest before any story joins the corpus.
    for title, ebook_id in OHENRY_COLLECTIONS.items():
        path = RAW / f"pg{ebook_id}.txt.utf8"
        records.append({"key": f"ohenry_{ebook_id}", "title": title, "author": "O. Henry",
                        "split": "train", "source_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
                        "license": "Project Gutenberg terms; local-law verification required",
                        "source_file": path.name, "status": "requires verified download and story audit" if path.exists()
                        else "missing"})

    robotics_path = RAW / "elements_of_robotics.epub"
    robotics_docs = extract_robotics_epub(robotics_path)
    docs.extend(robotics_docs)
    if robotics_path.exists():
        records.append({"key": "elements_of_robotics", "title": "Elements of Robotics",
                        "author": "Mordechai Ben-Ari; Francesco Mondada", "split": "train",
                        "source_url": "https://link.springer.com/book/10.1007/978-3-319-62533-1",
                        "license": "CC BY 4.0", "attribution": ROBOTICS_ATTRIBUTION,
                        "sha256": sha256(robotics_path), "source_file": robotics_path.name,
                        "status": "selected prose extracted; figures/tables omitted"})

    PREPARED.mkdir(parents=True, exist_ok=True)
    doc_path = PREPARED / "documents.jsonl"
    tmp = doc_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    tmp.replace(doc_path)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    mt = MANIFEST.with_suffix(".tmp")
    mt.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    mt.replace(MANIFEST)
    return {"documents_prepared": len(docs), "novels_missing": missing, "robotics_docs": len(robotics_docs),
            "manifest": str(MANIFEST), "documents": str(doc_path),
            "note": "O. Henry remains unprepared pending source availability and story-level boundary review."}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
