
import urllib.request
import zipfile
from io import BytesIO

url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

print("Downloading WikiText-2...")

response = urllib.request.urlopen(url, timeout=120)

print("Download complete!")

archive = zipfile.ZipFile(BytesIO(response.read()))

files = {
    "wiki.train.raw": "dataset.txt",
    "wiki.valid.raw": "validation.txt",
    "wiki.test.raw": "test.txt"
}

for source_name, output_name in files.items():

    matching_file = next(
        name for name in archive.namelist()
        if name.endswith(source_name)
    )

    text = archive.read(matching_file).decode("utf-8")

    with open(output_name, "w", encoding="utf-8") as file:
        file.write(text)

    print(output_name, "created:", len(text), "characters")

print("\nAll datasets downloaded successfully!")
