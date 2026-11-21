"""Tests for clean sentence data, EOS boundaries, and the new inference path."""

import tempfile
import unittest
from pathlib import Path

import torch

from generation_model import SmallLanguageModel
from high_quality_data import clean_sentence
from high_quality_inference import generate_text, load_high_quality_model
from tokenizer import BPETokenizer
from train_high_quality_model import encode_sentences


class HighQualityDatasetTests(unittest.TestCase):
    def test_cleaning_removes_scrape_markers_and_keeps_punctuation(self):
        source = "The cor-<160>porations &amp; <iframe> worked well; it stayed safe."
        self.assertEqual(
            clean_sentence(source),
            "The corporations & iframe worked well; it stayed safe.",
        )

    def test_cleaning_keeps_comparison_symbols(self):
        self.assertEqual(
            clean_sentence("A negative skew has mean < median < mode and Q2 – Q1 > Q3 – Q2."),
            "A negative skew has mean < median < mode and Q2 – Q1 > Q3 – Q2.",
        )

    def test_each_training_sentence_gets_an_eos_token(self):
        tokenizer = BPETokenizer.train(
            "The first sentence.\nA second sentence!",
            vocabulary_size=40,
            lowercase=False,
        )
        first_sentence_ids = tokenizer.encode("The first sentence.")
        ids = encode_sentences(["The first sentence.", "A second sentence!"], tokenizer)
        self.assertEqual(ids[len(first_sentence_ids)].item(), tokenizer.eos_id)
        self.assertEqual(ids[-1].item(), tokenizer.eos_id)


class HighQualityInferenceTests(unittest.TestCase):
    def test_matched_model_and_tokenizer_generate(self):
        tokenizer = BPETokenizer.train(
            "The person opened the door.\nA small dog ran home.",
            vocabulary_size=40,
            lowercase=False,
        )
        config = {
            "vocabulary_size": len(tokenizer.symbols),
            "dimension": 16,
            "context_length": 12,
            "number_of_heads": 4,
            "number_of_blocks": 1,
            "dropout": 0,
        }
        model = SmallLanguageModel(**config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            checkpoint_path = folder / "model.pth"
            tokenizer_path = folder / "tokenizer.json"
            torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)
            tokenizer.save(tokenizer_path)
            loaded_model, loaded_tokenizer, _ = load_high_quality_model(
                checkpoint_path, tokenizer_path
            )
            text = generate_text(loaded_model, loaded_tokenizer, "The person", 2, 8)
            self.assertTrue(text)

    def test_empty_prompt_is_rejected(self):
        tokenizer = BPETokenizer.train("A sentence with enough words.", 40, lowercase=False)
        model = SmallLanguageModel(len(tokenizer.symbols), 16, 12, 4, 1, 0)
        with self.assertRaisesRegex(ValueError, "prompt"):
            generate_text(model, tokenizer, " ", 3, 8)


if __name__ == "__main__":
    unittest.main()
