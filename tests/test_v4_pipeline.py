import tempfile
import unittest
from pathlib import Path

import torch

from v4_data import Source, clean_gutenberg, duplicate_groups, make_windows, validate_author_splits
from v4_inference import generate, load_v4
from v4_model import MODEL_CONFIG, V4LanguageModel, masked_next_token_loss, parameter_count
from v4_tokenizer import V4BPETokenizer
from train_v4_model import atomic_save as atomic_checkpoint
from prepare_v4_wikitext import detokenize_line, parse_split
from prepare_v4_tinystories import read_story_file
from evaluate_v4 import PROMPTS


class V4PipelineTests(unittest.TestCase):
    def test_default_small_model_matches_v4_config(self):
        model = V4LanguageModel()
        self.assertEqual(model.config["vocab_size"], MODEL_CONFIG["vocab_size"])
        self.assertEqual(model.config["context_length"], 64)
        self.assertEqual(parameter_count(model), 237_312)

    def test_tinystories_separator_keeps_paragraphs_and_deduplicates(self):
        story = 'She said, "Come along!"\n\nThe rabbit ran.'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stories.txt"
            source.write_text(f"{story}\n<|endoftext|>\n{story}<|endoftext|>\n",
                              encoding="utf-8")
            docs, stats = read_story_file(source, "train", "fixture")
        self.assertEqual(len(docs), 1)
        self.assertIn("\n\n", docs[0]["text"])
        self.assertIn('"Come along!"', docs[0]["text"])
        self.assertEqual(stats["within_file_duplicates_removed"], 1)

    def test_fixed_tinystories_evaluation_has_fifteen_prompts(self):
        self.assertEqual(len(PROMPTS), 15)

    def test_wikitext_article_extraction_keeps_paragraphs_and_sections(self):
        self.assertEqual(detokenize_line('role @-@ playing , " words "'), 'role-playing, "words"')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wiki.txt"
            source.write_text("= First Article =\n\n" + "A natural sentence with enough words. " * 20
                              + "\n\n= = Background = =\n\n" + "Another paragraph stays here. " * 20
                              + "\n\n= Second Article =\n\n" + "A separate natural article sentence. " * 20,
                              encoding="utf-8")
            docs, stats = parse_split("train", source)
            self.assertEqual(len(docs), 2)
            self.assertIn("\n\n", docs[0]["text"])
            self.assertIn("Background", docs[0]["text"])
            self.assertNotIn("Second Article", docs[0]["text"])
            self.assertEqual(stats["documents_kept"], 2)

    def test_clean_preserves_internal_structure_and_removes_wrapper(self):
        raw = """*** START OF THE PROJECT GUTENBERG EBOOK TEST ***
CHAPTER I

“Don't go,” she said—quietly.

He answered, "I won't."
*** END OF THE PROJECT GUTENBERG EBOOK TEST ***
"""
        result = clean_gutenberg(raw, minimum_chars=0)
        self.assertIn("CHAPTER I\n\n", result)
        self.assertIn("“Don't go,” she said—quietly.", result)
        self.assertIn("\n\nHe answered", result)
        self.assertNotIn("PROJECT GUTENBERG", result)

    def test_document_windows_terminate_at_each_document(self):
        eos = 2
        first = make_windows([10, 11, 12], 4, eos)
        second = make_windows([20, 21], 4, eos)
        self.assertEqual(first, [[10, 11, 12, eos]])
        self.assertEqual(second, [[20, 21, eos]])
        self.assertNotIn(20, first[0])

    def test_author_split_leakage_is_rejected(self):
        validate_author_splits([Source("a", "A", "Author", 1, "train"),
                                Source("b", "B", "Author", 2, "train")])
        with self.assertRaises(ValueError):
            validate_author_splits([Source("a", "A", "Author", 1, "train"),
                                    Source("b", "B", "Author", 2, "test")])

    def test_duplicate_documents_are_detected(self):
        docs = [{"id": "a", "text": "A story.\n\nNext line."},
                {"id": "b", "text": "a STORY. Next line."}]
        self.assertEqual(len(duplicate_groups(docs)), 1)

    def test_tokenizer_exact_round_trip_and_training_only_fit(self):
        training = ["He said, \"don't—wait!\"\n\nShe replied: ‘yes.’\n",
                    "A short training paragraph.\n"]
        tokenizer = V4BPETokenizer.train(training, vocab_size=512)
        self.assertEqual(len(tokenizer.symbols), 512)
        sample = training[0]
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
        unseen_unicode = "A new passage!\n\n“Résumé—🙂”"
        self.assertEqual(tokenizer.decode(tokenizer.encode(unseen_unicode)), unseen_unicode)
        self.assertNotIn("validation-only-word", tokenizer.symbols)

    def test_causal_attention_does_not_read_future_tokens(self):
        torch.manual_seed(1)
        model = V4LanguageModel(vocab_size=40, dimension=20, context_length=8,
                                number_of_heads=5, number_of_blocks=2, dropout=0.0)
        model.eval()
        a = torch.tensor([[1, 4, 5, 6]])
        b = torch.tensor([[1, 4, 5, 12]])
        with torch.inference_mode():
            out_a, out_b = model(a), model(b)
        self.assertTrue(torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6))

    def test_padding_mask_and_loss_mask(self):
        torch.manual_seed(2)
        model = V4LanguageModel(vocab_size=40, dimension=20, context_length=8,
                                number_of_heads=5, number_of_blocks=1, dropout=0.0, pad_id=0)
        model.eval()
        ids = torch.tensor([[1, 3, 4, 0]])
        mask = ids.ne(0)
        with torch.inference_mode():
            out = model(ids, mask)
        self.assertTrue(torch.isfinite(out).all())
        logits = torch.randn(1, 3, 40)
        targets = torch.tensor([[2, 3, 0]])
        loss = masked_next_token_loss(logits, targets, pad_id=0)
        reference = torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, 40), targets[:, :2].reshape(-1))
        self.assertTrue(torch.allclose(loss, reference))

    def test_parameter_count_and_tied_embeddings(self):
        model = V4LanguageModel(**{**{"vocab_size": 8000}, **{k: v for k, v in {
            "dimension": 160, "context_length": 256, "number_of_heads": 5,
            "number_of_blocks": 6, "dropout": 0.1}.items()}})
        self.assertEqual(parameter_count(model), 3_184_960)
        self.assertIs(model.token_embedding.weight, model.token_embedding.weight)
        self.assertEqual(sum(1 for name, _ in model.named_parameters() if name == "token_embedding.weight"), 1)

    def test_checkpoint_round_trip_and_eos_stop(self):
        tokenizer = V4BPETokenizer.train(["Once upon a time.\n"], vocab_size=512)
        model = V4LanguageModel(vocab_size=len(tokenizer.symbols), dimension=20,
                                context_length=32, number_of_heads=5, number_of_blocks=1,
                                dropout=0.0, pad_id=tokenizer.pad_id)
        model.train()
        ids = torch.tensor([[tokenizer.encode("Once upon a time.")[0], tokenizer.eos_id]])
        loss = masked_next_token_loss(model(ids[:, :-1]), ids[:, 1:], tokenizer.pad_id)
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        model.eval()
        with torch.no_grad():
            model.output_bias.fill_(-10.0)
            model.output_bias[tokenizer.eos_id] = 10.0
            model.output_bias[tokenizer.pad_id] = -float("inf")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v4.pth"
            token_path = Path(directory) / "v4.json"
            tokenizer.save(token_path)
            atomic_checkpoint(checkpoint, {"format": "nextword-v4", "config": model.config,
                                           "model_state_dict": model.state_dict()})
            loaded, loaded_tokenizer, _ = load_v4(checkpoint, token_path)
            self.assertEqual(len(loaded.state_dict()), len(model.state_dict()))
            result = generate(loaded, loaded_tokenizer, "Once", max_new_tokens=20, seed=4,
                              min_new_words=0)
            self.assertEqual(result, "Once")


if __name__ == "__main__":
    unittest.main()
