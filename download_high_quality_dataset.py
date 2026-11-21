"""Download the official train and test files for the selected dataset."""

from pathlib import Path
from urllib.request import urlretrieve


FILES = {
    "high_quality_train.txt.gz":
        "https://huggingface.co/datasets/hrkhosravi/high-quality-english-sentences/resolve/main/train.txt.gz",
    "high_quality_test.txt.gz":
        "https://huggingface.co/datasets/hrkhosravi/high-quality-english-sentences/resolve/main/test.txt.gz",
}


def main():
    for filename, url in FILES.items():
        destination = Path(filename)
        if destination.exists():
            print(f"Already downloaded: {destination}")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
