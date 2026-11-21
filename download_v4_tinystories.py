"""Fetch only the selected official TinyStories shards; never downloads the full dataset."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "v4" / "raw" / "tinystories"
FILES = {
    "train_part1.txt": {
        "url": "https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned/resolve/main/data/train/part1.txt?download=true",
        "bytes": 80_000_480,
        "sha256": "2d8de3cd74ff26513d2530dd5837f159d665458de192e6d6222a1879d2655359",
    },
    "test_part1.txt": {
        "url": "https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned/resolve/main/data/test/part1.txt?download=true",
        "bytes": 56_015_628,
        "sha256": "600b5e22686f78d2aef762afee9a51502886cf45291653791cd6acc873e82f1f",
    },
}


def digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def download() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, metadata in FILES.items():
        target = DEST / name
        if target.exists():
            size, digest = digest_file(target)
            if size != metadata["bytes"] or digest != metadata["sha256"]:
                raise ValueError(f"Refusing to overwrite unexpected existing file: {target}")
            print(f"verified existing {name}: {size} bytes sha256={digest}")
            continue
        request = urllib.request.Request(metadata["url"],
                                         headers={"User-Agent": "NextWordPredictor-V4/1.0"})
        partial = target.with_suffix(target.suffix + ".partial")
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
                expected_header = response.headers.get("Content-Length")
                if expected_header and int(expected_header) != metadata["bytes"]:
                    raise ValueError(f"Unexpected remote size for {name}: {expected_header}")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size % (10 * 1024 * 1024) < len(chunk):
                        print(f"{name}: {size}/{metadata['bytes']} bytes", flush=True)
            if size != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch for {name}")
            partial.replace(target)
            print(f"saved {target}: {size} bytes sha256={digest.hexdigest()}")
        except Exception:
            partial.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    download()
