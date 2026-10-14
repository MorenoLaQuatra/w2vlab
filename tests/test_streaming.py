"""Small checks for full-corpus, bounded-batch training."""

import os
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import select_device, train  # noqa: E402
from w2vlab.data import count_corpus, make_pairs, stream_tokens  # noqa: E402
from w2vlab.vectors import WordVectors, load_saved  # noqa: E402


class StreamingTest(unittest.TestCase):
    def test_device_selection(self):
        self.assertEqual(str(select_device("cpu")), "cpu")
        with patch("train.torch.cuda.is_available", return_value=True):
            self.assertEqual(str(select_device("auto")), "cuda")
        with patch("train.torch.cuda.is_available", return_value=False), \
                patch("train.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(select_device("auto")), "mps")
        with self.assertRaises(ValueError):
            select_device("unknown")

    def test_counts_and_article_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "tiny.txt"
            corpus.write_text("one two three four five\nred blue green yellow\n", encoding="utf-8")
            vocab, tokens, examples = count_corpus(corpus, -1, 1, 20, 1, "cbow")
            self.assertEqual((tokens, examples), (9, 5))
            self.assertEqual(sum(vocab.counts), 9)
            ids = (vocab.to_id.get(word, 0) if word is not None else None
                   for word in stream_tokens(corpus))
            self.assertEqual(len(list(make_pairs(ids, 1, "cbow"))), examples)
            _, tokens, examples = count_corpus(corpus, 7, 1, 20, 1, "skipgram")
            self.assertEqual((tokens, examples), (7, 6))
            ids = (vocab.to_id.get(word, 0) if word is not None else None
                   for word in stream_tokens(corpus, 7))
            self.assertEqual(len(list(make_pairs(ids, 1, "skipgram"))), examples)

    def test_train_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "tiny.txt"
            corpus.write_text(("cat cats dog dogs bird birds\n" * 8), encoding="utf-8")
            runs = (("word2vec", "cbow", "negative"),
                    ("fasttext", "cbow", "negative"),
                    ("word2vec", "skipgram", "softmax"))
            for kind, architecture, loss in runs:
                output = Path(directory) / f"{kind}-{architecture}-{loss}.pt"
                config = SimpleNamespace(
                    data=SimpleNamespace(corpus=corpus, max_tokens=-1, min_count=1,
                                         max_words=20, window=1),
                    model=SimpleNamespace(type=kind, architecture=architecture,
                                          dimensions=8, loss=loss, negatives=2,
                                          min_n=3, max_n=4),
                    training=SimpleNamespace(epochs=1, batch_size=8, shuffle_buffer=16,
                                             lr=0.01, seed=1),
                    output=output,
                )
                with redirect_stdout(StringIO()):
                    train(config)
                saved = torch.load(output, weights_only=True)
                self.assertEqual(saved["vectors"].shape, (7, 8))
                self.assertEqual(sum(saved["counts"]), 48)
                checkpoint = output.with_name(output.stem + ".checkpoint.pt")
                loaded = load_saved(checkpoint)
                self.assertTrue(torch.allclose(loaded["vectors"], saved["vectors"]))
                self.assertTrue(torch.equal(loaded["output_vectors"], saved["output_vectors"]))
                examples = 64 if architecture == "skipgram" else 32
                self.assertEqual(loaded["checkpoint_progress"], (1, 0, examples))
                self.assertEqual(WordVectors(checkpoint).most_similar("cat", 2),
                                 WordVectors(output).most_similar("cat", 2))
                if kind == "fasttext":
                    self.assertTrue(torch.allclose(WordVectors(checkpoint).vector("catss"),
                                                    WordVectors(output).vector("catss")))

    def test_cli_accepts_full_corpus_override(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "tiny.txt"
            output = Path(directory) / "vectors.pt"
            corpus.write_text("cat cats dog dogs bird birds\n" * 4, encoding="utf-8")
            root = Path(__file__).resolve().parents[1]
            command = [
                sys.executable, "train.py", "--config", "configs/train.yaml",
                "--data.corpus", str(corpus), "--data.max_tokens", "-1",
                "--data.min_count", "1", "--data.window", "1",
                "--model.dimensions", "8", "--model.negatives", "2",
                "--training.epochs", "1", "--training.batch_size", "8",
                "--training.shuffle_buffer", "16", "--output", str(output),
            ]
            environment = {**os.environ, "PYTHONPATH": str(root / "src")}
            result = subprocess.run(command, cwd=root, env=environment,
               capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            self.assertIn("Vocabulary:", result.stdout)
            self.assertIn("ETA", result.stdout)
            checkpoint = Path(directory) / "vectors.checkpoint.pt"
            resumed = subprocess.run(command + ["--training.resume", str(checkpoint)],
                                     cwd=root, env=environment, capture_output=True,
                                     text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertIn("Resuming", resumed.stdout)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "tiny.txt"
            corpus.write_text("cat cats dog dogs bird birds\n" * 8, encoding="utf-8")
            config = SimpleNamespace(
                data=SimpleNamespace(corpus=corpus, max_tokens=-1, min_count=1,
                                     max_words=20, window=1),
                model=SimpleNamespace(type="fasttext", architecture="cbow",
                                      dimensions=8, loss="negative", negatives=2,
                                      min_n=3, max_n=4),
                training=SimpleNamespace(epochs=1, batch_size=8, shuffle_buffer=16,
                                         checkpoint_every=8, device="cpu", resume="",
                                         lr=0.01, seed=1),
                output=Path(directory) / "baseline.pt",
            )
            with redirect_stdout(StringIO()):
                train(config)
            baseline = torch.load(config.output, weights_only=True)

            config.output = Path(directory) / "resumed.pt"
            from train import save_checkpoint

            class Interrupted(Exception):
                pass

            def stop_after_save(*args):
                save_checkpoint(*args)
                if args[-2] > 0:  # A mid-epoch save, not the epoch-end save.
                    raise Interrupted

            with patch("train.save_checkpoint", side_effect=stop_after_save), \
                    redirect_stdout(StringIO()):
                with self.assertRaises(Interrupted):
                    train(config)
            checkpoint = Path(directory) / "resumed.checkpoint.pt"
            saved = torch.load(checkpoint, weights_only=True)
            self.assertGreater(saved["seen"], 0)
            config.training.resume = str(checkpoint)
            with redirect_stdout(StringIO()):
                train(config)
            resumed = torch.load(config.output, weights_only=True)
            self.assertTrue(torch.equal(baseline["vectors"], resumed["vectors"]))
            self.assertTrue(torch.equal(baseline["output_vectors"],
                                        resumed["output_vectors"]))


if __name__ == "__main__":
    unittest.main()
