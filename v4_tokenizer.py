"""Lossless V4 character-BPE tokenizer; vocabulary fitting uses train texts only."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class V4BPETokenizer:
    EOS = "<eos>"
    PAD = "<pad>"
    UNK = "<unk>"
    END = "</w>"
    BYTE_BASE = 0xE000

    def __init__(self, symbols: list[str], merges: list[tuple[str, str]]):
        self.symbols = symbols
        self.merges = [tuple(pair) for pair in merges]
        self.ids = {piece: i for i, piece in enumerate(symbols)}
        self.eos_id = self.ids[self.EOS]
        self.pad_id = self.ids[self.PAD]
        self.unk_id = self.ids[self.UNK]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.reserved_ids = {i for i, piece in enumerate(symbols) if piece.startswith("<unused:")}

    @staticmethod
    def _segments(text: str) -> list[str]:
        # Whitespace is a first-class segment, so decode preserves paragraphs,
        # indentation, line breaks, quotes and punctuation exactly.
        return re.findall(r"\s+|[^\s]+", text, flags=re.UNICODE)

    @classmethod
    def _byte_symbols(cls, text: str) -> str:
        return "".join(chr(cls.BYTE_BASE + byte) for byte in text.encode("utf-8"))

    @classmethod
    def train(cls, training_texts: list[str], vocab_size: int = 8000) -> "V4BPETokenizer":
        if vocab_size < 32:
            raise ValueError("vocab_size must be at least 32")
        frequencies = Counter(cls._byte_symbols(segment) for text in training_texts
                              for segment in cls._segments(text))
        if not frequencies:
            raise ValueError("No training text supplied")
        pieces_by_segment = {segment: list(segment) + [cls.END] for segment in frequencies}
        symbols = {cls.EOS, cls.PAD, cls.UNK, cls.END}
        symbols.update(chr(cls.BYTE_BASE + byte) for byte in range(256))
        for pieces in pieces_by_segment.values():
            symbols.update(pieces)
        if len(symbols) > vocab_size:
            raise ValueError(f"Initial character alphabet ({len(symbols)}) exceeds vocabulary size")
        pair_counts = Counter()
        pair_segments: dict[tuple[str, str], set[str]] = {}
        for segment, pieces in pieces_by_segment.items():
            count = frequencies[segment]
            for pair, n in Counter(zip(pieces, pieces[1:])).items():
                pair_counts[pair] += count * n
                pair_segments.setdefault(pair, set()).add(segment)
        merges: list[tuple[str, str]] = []
        while len(symbols) < vocab_size and pair_counts:
            pair, count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if count < 2:
                break
            merged_symbol = pair[0] + pair[1]
            merges.append(pair)
            symbols.add(merged_symbol)
            for segment in list(pair_segments.get(pair, ())):
                old = pieces_by_segment[segment]
                weight = frequencies[segment]
                old_pairs = Counter(zip(old, old[1:]))
                for p, n in old_pairs.items():
                    pair_counts[p] -= weight * n
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_segments[p].discard(segment)
                    if not pair_segments[p]:
                        del pair_segments[p]
                new = []
                i = 0
                while i < len(old):
                    if i + 1 < len(old) and (old[i], old[i + 1]) == pair:
                        new.append(merged_symbol)
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                pieces_by_segment[segment] = new
                for p, n in Counter(zip(new, new[1:])).items():
                    pair_counts[p] += weight * n
                    pair_segments.setdefault(p, set()).add(segment)
        # IDs are deterministic. Pad/EOS/UNK stay fixed at the front.
        ordered = [cls.PAD, cls.EOS, cls.UNK] + sorted(symbols - {cls.PAD, cls.EOS, cls.UNK})
        reserved = 0
        while len(ordered) < vocab_size:
            ordered.append(f"<unused:{reserved:04d}>")
            reserved += 1
        return cls(ordered, merges)

    def _encode_segment(self, segment: str) -> list[int]:
        pieces = list(self._byte_symbols(segment)) + [self.END]
        while len(pieces) > 1:
            candidates = [(self.ranks[pair], i, pair) for i, pair in enumerate(zip(pieces, pieces[1:])) if pair in self.ranks]
            if not candidates:
                break
            _, _, selected = min(candidates)
            merged = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == selected:
                    merged.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            pieces = merged
        return [self.ids.get(p, self.unk_id) for p in pieces]

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        cache: dict[str, list[int]] = {}
        for segment in self._segments(text):
            if segment not in cache:
                cache[segment] = self._encode_segment(segment)
            encoded.extend(cache[segment])
        return encoded

    def decode(self, ids: list[int]) -> str:
        output = bytearray()
        for token_id in ids:
            if token_id == self.eos_id or token_id == self.pad_id or token_id in self.reserved_ids:
                continue
            piece = self.symbols[token_id] if 0 <= token_id < len(self.symbols) else self.UNK
            if piece.endswith(self.END):
                payload = piece[:-len(self.END)]
                output.extend(ord(char) - self.BYTE_BASE for char in payload)
            elif piece == self.UNK:
                output.extend(self.UNK.encode("utf-8"))
            else:
                output.extend(ord(char) - self.BYTE_BASE for char in piece)
        return output.decode("utf-8", errors="replace")

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps({"format": "v4-char-bpe", "symbols": self.symbols,
                                    "merges": self.merges}, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> "V4BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "v4-char-bpe":
            raise ValueError("Not a V4 tokenizer")
        return cls(data["symbols"], data["merges"])
