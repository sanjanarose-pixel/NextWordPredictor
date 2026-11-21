"""Download and clean the sentence-level training data without merging splits."""

import gzip
import hashlib
import html
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
}


def clean_sentence(raw_sentence, minimum_characters=20, maximum_characters=1000):
    """Remove scraped markup while retaining words and ordinary punctuation."""
    sentence = html.unescape(raw_sentence)
    for broken, corrected in MOJIBAKE_REPLACEMENTS.items():
        sentence = sentence.replace(broken, corrected)

    # Gutenberg-style page breaks can split a word as ``cor-<160>porations``.
    sentence = re.sub(r"-\s*<\d{2,4}>\s*", "", sentence)
    sentence = re.sub(r"<\d{2,4}>", "", sentence)
    sentence = re.sub(r"<-([A-Za-z][A-Za-z0-9_-]*)>", r"-\1", sentence)

    # Keep the text inside simple HTML/XML tags, but remove the tag brackets.
    sentence = re.sub(
        r"</?([A-Za-z][A-Za-z0-9_.:-]*)(?:\s+[^<>]*?)?\s*/?>",
        r"\1",
        sentence,
        flags=re.IGNORECASE,
    )

    # Drop control and zero-width formatting characters, but retain Unicode text.
    sentence = "".join(
        character for character in sentence
        if character in "\t\r\n" or unicodedata.category(character) not in {"Cc", "Cf"}
    )
    sentence = unicodedata.normalize("NFC", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\s+([,.;:!?%])", r"\1", sentence)

    if len(sentence) < minimum_characters or len(sentence) > maximum_characters:
        return None
    if "\ufffd" in sentence or sentence.count("|") > 2:
        return None
    return sentence


def load_train_and_validation(path, maximum_training_sentences=100_000,
                              validation_modulus=20, seed=42,
                              maximum_validation_sentences=20_000):
    """Sample raw rows first, then clean them; validation stays inside train split."""
    path = Path(path)
    random_source = random.Random(seed)
    raw_training_sample = []
    raw_validation_sample = []
    eligible_training_rows = 0
    eligible_validation_rows = 0
    counts = Counter()

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as source:
        for raw_sentence in source:
            counts["source_rows"] += 1
            raw_sentence = raw_sentence.strip()
            digest = hashlib.blake2b(raw_sentence.encode("utf-8"), digest_size=8).digest()
            if int.from_bytes(digest, "big") % validation_modulus == 0:
                eligible_validation_rows += 1
                if maximum_training_sentences is None:
                    raw_validation_sample.append(raw_sentence)
                elif len(raw_validation_sample) < maximum_validation_sentences:
                    raw_validation_sample.append(raw_sentence)
                else:
                    replacement_index = random_source.randrange(eligible_validation_rows)
                    if replacement_index < maximum_validation_sentences:
                        raw_validation_sample[replacement_index] = raw_sentence
                continue

            eligible_training_rows += 1
            if maximum_training_sentences is None:
                raw_training_sample.append(raw_sentence)
            elif len(raw_training_sample) < maximum_training_sentences:
                raw_training_sample.append(raw_sentence)
            else:
                # Reservoir sampling gives every training row an equal chance
                # of appearing in the CPU-sized subset.
                replacement_index = random_source.randrange(eligible_training_rows)
                if replacement_index < maximum_training_sentences:
                    raw_training_sample[replacement_index] = raw_sentence

    training_sample = [clean_sentence(sentence) for sentence in raw_training_sample]
    validation = [clean_sentence(sentence) for sentence in raw_validation_sample]
    training_sample = [sentence for sentence in training_sample if sentence is not None]
    validation = [sentence for sentence in validation if sentence is not None]
    counts["removed_sampled_rows"] = (
        len(raw_training_sample) + len(raw_validation_sample)
        - len(training_sample) - len(validation)
    )
    counts["clean_training_rows"] = len(training_sample)
    counts["clean_validation_rows"] = len(validation)

    if not training_sample or not validation:
        raise ValueError("The training split did not contain enough clean sentences.")
    counts["eligible_training_rows"] = eligible_training_rows
    counts["sampled_training_rows"] = len(training_sample)
    counts["validation_rows_available"] = eligible_validation_rows
    counts["validation_rows_sampled"] = len(validation)
    return training_sample, validation, counts


def load_test_sentences(path, maximum_sentences=20_000, seed=43):
    """Uniformly sample raw rows from official test, then clean for final evaluation."""
    raw_sample = []
    random_source = random.Random(seed)
    counts = Counter()
    with gzip.open(Path(path), "rt", encoding="utf-8", errors="replace") as source:
        for raw_sentence in source:
            counts["source_rows"] += 1
            raw_sentence = raw_sentence.strip()
            if maximum_sentences is None:
                raw_sample.append(raw_sentence)
            elif len(raw_sample) < maximum_sentences:
                raw_sample.append(raw_sentence)
            else:
                replacement_index = random_source.randrange(counts["source_rows"])
                if replacement_index < maximum_sentences:
                    raw_sample[replacement_index] = raw_sentence
    sentences = [clean_sentence(sentence) for sentence in raw_sample]
    counts["removed_sampled_rows"] = sum(sentence is None for sentence in sentences)
    sentences = [sentence for sentence in sentences if sentence is not None]
    counts["clean_sampled_rows"] = len(sentences)
    return sentences, counts
