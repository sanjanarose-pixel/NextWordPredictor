"""Small offline tests for tokenizer behavior and inference safety."""

import tempfile
import unittest
from pathlib import Path

import torch

from inference import generate_text, load_model
from model import LanguageModel
from tokenizer import BPETokenizer, LegacyWordTokenizer


class TokenizerTests(unittest.TestCase):
    def test_legacy_word_tokenizer_round_trip(self):
        tokenizer = LegacyWordTokenizer(["<unk>", "hello", ",", "world", "!"])
        ids = tokenizer.encode("Hello, world!")
        self.assertEqual(tokenizer.decode(ids), "hello, world!")
        self.assertEqual(tokenizer.encode("missing"), [0])

    def test_bpe_can_encode_decode_and_save(self):
        tokenizer = BPETokenizer.train("hello hello help", vocabulary_size=40)
        original_ids = tokenizer.encode("hello help")
        self.assertEqual(tokenizer.decode(original_ids), "hello help")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bpe.json"
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.decode(loaded.encode("hello help")), "hello help")


class InferenceTests(unittest.TestCase):
    def _write_tiny_checkpoint(self, folder):
        vocabulary = ["<unk>", "hello", "world", "."]
        vocabulary_path = folder / "vocabulary.json"
        import json
        vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
        config = {
            "vocabulary_size": 4,
            "embedding_dimension": 8,
            "context_length": 8,
            "number_of_heads": 2,
            "number_of_blocks": 1,
        }
        model = LanguageModel(**{
            "vocabulary_size": config["vocabulary_size"],
            "embedding_dimension": config["embedding_dimension"],
            "context_length": config["context_length"],
            "number_of_heads": config["number_of_heads"],
            "number_of_blocks": config["number_of_blocks"],
        })
        checkpoint_path = folder / "tiny.pth"
        torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)
        return checkpoint_path, vocabulary_path

    def test_model_load_and_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint, vocabulary = self._write_tiny_checkpoint(Path(temp_dir))
            model, tokenizer, config = load_model(checkpoint, vocabulary)
            result = generate_text(model, tokenizer, "hello", 2, 4, 0.8, 2)
            self.assertTrue(result.startswith("hello"))
            self.assertEqual(config["context_length"], 8)

    def test_loader_rejects_incompatible_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint, vocabulary = self._write_tiny_checkpoint(Path(temp_dir))
            vocabulary.write_text('["<unk>", "different"]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vocabulary size"):
                load_model(checkpoint, vocabulary)

    def test_generation_checks_context_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint, vocabulary = self._write_tiny_checkpoint(Path(temp_dir))
            model, tokenizer, _ = load_model(checkpoint, vocabulary)
            with self.assertRaisesRegex(ValueError, "Context window"):
                generate_text(model, tokenizer, "hello", 1, 9)


if __name__ == "__main__":
    unittest.main()
