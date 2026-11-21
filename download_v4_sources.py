"""Acquire only the explicitly selected official V4 source files (no archives/mirrors)."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError

from v4_data import NOVELS, OHENRY_COLLECTIONS, RAW, Source, download_source, sha256


def acquire() -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    sources = list(NOVELS) + [
        Source(f"ohenry_{i}", title, "O. Henry", i, "train")
        for title, i in OHENRY_COLLECTIONS.items()
    ]
    records = []
    for source in sources:
        row = {"key": source.key, "title": source.title, "author": source.author,
               "split": source.split, "url": source.url}
        try:
            path = download_source(source)
            row.update({"status": "downloaded-or-already-present", "file": path.name,
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            records.append(row)
            # PG maintenance is a site-wide blocker; don't hammer the origin.
            break
        records.append(row)
    report = RAW.parent / "acquisition_report.json"
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    print(json.dumps(acquire(), ensure_ascii=False, indent=2))
