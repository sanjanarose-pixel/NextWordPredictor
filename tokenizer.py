"""Tokenizers for this project.

LegacyWordTokenizer exactly matches the vocabulary used by existing weights.
BPETokenizer is a small from-scratch subword tokenizer for future training.
Changing token IDs changes the meaning of embedding/output rows, so BPE cannot
be substituted into an existing word-token checkpoint.
"""

import json
import re
from collections import Counter
from pathlib import Path


def split_words(text, lowercase=True):
    """Split on words and punctuation, optionally preserving capitalization."""
    if lowercase:
        text = text.lower()
    text = text.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    # Match the original training scripts exactly so token IDs stay stable.
    return re.findall(r"\w+(?:['â€™]\w+)*|[^\w\s]", text, flags=re.UNICODE)


def format_tokens(tokens):
    """Join word tokens and attach common punctuation to the previous token."""
    result = " ".join(tokens)
    result = re.sub(r"\s+([.,!?;:%)\]])", r"\1", result)
    return re.sub(r"([(])\s+", r"\1", result)


class LegacyWordTokenizer:
    """The fixed 20,000-word tokenizer that existing checkpoints require."""

    def __init__(self, vocabulary):
        if not vocabulary or vocabulary[0] != "<unk>":
            raise ValueError("Vocabulary must start with the <unk> token.")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary contains duplicate tokens.")
        self.vocabulary = list(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self.unknown_id = self.token_to_id["<unk>"]

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def encode(self, text):
        return [self.token_to_id.get(word, self.unknown_id) for word in split_words(text)]

    def decode(self, token_ids):
        return format_tokens(
            self.vocabulary[int(token_id)]
            if 0 <= int(token_id) < len(self.vocabulary)
            else "<unk>"
            for token_id in token_ids
        )


class BPETokenizer:
    """A beginner-friendly byte-pair encoding tokenizer trained from text.

    It learns frequent character merges within words. The end marker preserves
    word boundaries, while punctuation remains its own initial symbol.
    """

    END_OF_WORD = "</w>"
    END_OF_SENTENCE = "<eos>"

    def __init__(self, symbols, merges, lowercase=True):
        self.symbols = list(symbols)
        self.merges = [tuple(pair) for pair in merges]
        self.lowercase = lowercase
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.unknown_id = self.symbol_to_id.get("<unk>", 0)
        self.eos_id = self.symbol_to_id.get(self.END_OF_SENTENCE)
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def train(cls, text, vocabulary_size=2000, lowercase=True):
        if vocabulary_size < 10:
            raise ValueError("BPE vocabulary_size must be at least 10.")
        words = split_words(text, lowercase=lowercase)
        if not words:
            raise ValueError("Cannot train a tokenizer from empty text.")

        word_counts = Counter(words)
        word_symbols = {
            word: list(word) + [cls.END_OF_WORD]
            for word in word_counts
        }
        symbols = {"<unk>", cls.END_OF_WORD, cls.END_OF_SENTENCE}
        for pieces in word_symbols.values():
            symbols.update(pieces)

        # Keep pair frequencies incrementally. Recounting every word after
        # every merge makes a useful sized vocabulary unnecessarily slow.
        pair_counts = Counter()
        pair_words = {}
        for word, pieces in word_symbols.items():
            for pair in zip(pieces, pieces[1:]):
                pair_counts[pair] += word_counts[word]
                pair_words.setdefault(pair, set()).add(word)

        merges = []
        while len(symbols) < vocabulary_size and pair_counts:
            best_pair, count = max(pair_counts.items(), key=lambda item: item[1])
            if count < 2:
                break
            merged_symbol = "".join(best_pair)
            merges.append(best_pair)
            symbols.add(merged_symbol)

            # Only words containing this pair can change. Update counts for
            # their old and new neighboring pairs as each merge is applied.
            for word in list(pair_words.get(best_pair, ())):
                pieces = word_symbols[word]
                word_frequency = word_counts[word]
                old_pair_counts = Counter(zip(pieces, pieces[1:]))
                for pair, pair_frequency in old_pair_counts.items():
                    pair_counts[pair] -= word_frequency * pair_frequency
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(word)
                    if not pair_words[pair]:
                        del pair_words[pair]

                merged = []
                index = 0
                while index < len(pieces):
                    if index + 1 < len(pieces) and tuple(pieces[index:index + 2]) == best_pair:
                        merged.append(merged_symbol)
                        index += 2
                    else:
                        merged.append(pieces[index])
                        index += 1
                word_symbols[word] = merged

                new_pair_counts = Counter(zip(merged, merged[1:]))
                for pair, pair_frequency in new_pair_counts.items():
                    pair_counts[pair] += word_frequency * pair_frequency
                    pair_words.setdefault(pair, set()).add(word)

        ordered_symbols = ["<unk>", cls.END_OF_SENTENCE] + sorted(
            symbol for symbol in symbols
            if symbol not in {"<unk>", cls.END_OF_SENTENCE}
        )
        return cls(ordered_symbols, merges, lowercase=lowercase)

    def _encode_word(self, word):
        pieces = list(word) + [self.END_OF_WORD]
        while len(pieces) > 1:
            candidates = [
                (self.merge_rank[pair], index, pair)
                for index, pair in enumerate(zip(pieces, pieces[1:]))
                if pair in self.merge_rank
            ]
            if not candidates:
                break
            _, _, best_pair = min(candidates)
            merged = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and tuple(pieces[index:index + 2]) == best_pair:
                    merged.append("".join(best_pair))
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged
        return [self.symbol_to_id.get(piece, self.unknown_id) for piece in pieces]

    def encode(self, text):
        # Most words recur many times in a corpus, so cache their piece IDs.
        cache = {}
        encoded = []
        for word in split_words(text, lowercase=self.lowercase):
            if word not in cache:
                cache[word] = self._encode_word(word)
            encoded.extend(cache[word])
        return encoded

    def decode(self, token_ids):
        words = []
        current = ""
        for token_id in token_ids:
            symbol = self.symbols[int(token_id)] if 0 <= int(token_id) < len(self.symbols) else "<unk>"
            if symbol == self.END_OF_SENTENCE:
                if current:
                    words.append(current)
                    current = ""
                continue
            if symbol == "<unk>":
                current += "<unk>"
            else:
                current += symbol
                if current.endswith(self.END_OF_WORD):
                    words.append(current[:-len(self.END_OF_WORD)])
                    current = ""
        if current:
            words.append(current)
        return format_tokens(words)

    def save(self, path):
        data = {
            "type": "bpe",
            "symbols": self.symbols,
            "merges": self.merges,
            "lowercase": self.lowercase,
        }
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("type") != "bpe":
            raise ValueError("This file does not contain a BPE tokenizer.")
        return cls(data["symbols"], data["merges"], lowercase=data.get("lowercase", True))
