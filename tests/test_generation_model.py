"""Offline checks for the replacement model and its matched BPE tokenizer."""

import tempfile
import unittest
from pathlib import Path

import torch

from generation_inference import generate_text, load_generation_model
from generation_model import SmallLanguageModel
from tokenizer import BPETokenizer


class NewGenerationModelTests(unittest.TestCase):
    def test_bpe_round_trip_and_model_shape(self):
        tokenizer = BPETokenizer.train("the robot moves. the robot stops.", 40)
        self.assertEqual(tokenizer.decode(tokenizer.encode("the robot moves.")), "the robot moves.")

        model = SmallLanguageModel(
            vocabulary_size=len(tokenizer.symbols),
            dimension=16,
            context_length=12,
            number_of_heads=4,
            number_of_blocks=1,
            dropout=0,
        )
        logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        self.assertEqual(logits.shape, (1, 3, len(tokenizer.symbols)))

    def test_matched_checkpoint_loads_and_generates(self):
        tokenizer = BPETokenizer.train("the robot moves. a person opens the door.", 40)
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
            checkpoint_path = folder / "small.pth"
            tokenizer_path = folder / "bpe.json"
            torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)
            tokenizer.save(tokenizer_path)

            loaded_model, loaded_tokenizer, _ = load_generation_model(checkpoint_path, tokenizer_path)
            generated = generate_text(
                loaded_model, loaded_tokenizer, "the robot", 2, context_window=8
            )
            self.assertTrue(generated)

    def test_empty_prompt_is_rejected(self):
        tokenizer = BPETokenizer.train("a simple sentence.", 40)
        model = SmallLanguageModel(len(tokenizer.symbols), 16, 12, 4, 1, 0)
        with self.assertRaisesRegex(ValueError, "prompt"):
            generate_text(model, tokenizer, "  ", 2, context_window=8)


if __name__ == "__main__":
    unittest.main()
