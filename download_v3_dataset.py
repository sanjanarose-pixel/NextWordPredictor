"""Download one pinned FineWeb-Edu sample shard for the Version 3 pipeline."""

from pathlib import Path
from urllib.request import urlopen


URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
    "651a648da38bf545cc5487530dbf59d8168c8de3/"
    "sample/10BT/013_00000.parquet"
)
OUTPUT = Path("v3_fineweb_edu_sample_013.parquet")


def main():
    if OUTPUT.exists():
        print(f"Dataset shard already exists: {OUTPUT}")
        return

    temporary_path = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    print("Downloading the pinned 515 MiB FineWeb-Edu sample shard...")
    try:
        with urlopen(URL, timeout=60) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temporary_path.replace(OUTPUT)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved dataset shard to {OUTPUT}.")


if __name__ == "__main__":
    main()
