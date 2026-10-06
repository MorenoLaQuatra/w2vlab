"""Train word2vec with an explicit YAML config."""

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from yaml_config_manager import load_config

from w2vlab.data import Vocabulary, make_pairs, tokenize
from w2vlab.model import NegativeSampler, Word2Vec


def train(config):
    data, model_config, fit = config.data, config.model, config.training
    if model_config.loss not in ("negative", "softmax"):
        raise ValueError("model.loss must be negative or softmax")
    sizes = (data.window, data.min_count, data.max_words, model_config.dimensions,
             model_config.negatives, fit.epochs, fit.batch_size)
    if min(sizes) < 1 or fit.lr <= 0 or data.max_tokens == 0 or data.max_tokens < -1:
        raise ValueError("sizes and learning rate must be positive; max_tokens may be -1")

    torch.manual_seed(fit.seed)
    tokens = tokenize(Path(data.corpus).read_text(encoding="utf-8"))
    if data.max_tokens > 0:
        tokens = tokens[:data.max_tokens]
    vocab = Vocabulary(tokens, data.min_count, data.max_words)
    pairs = make_pairs(vocab.encode(tokens), data.window, model_config.architecture)
    if len(vocab.words) < 3 or not pairs:
        raise ValueError("corpus is too small; try more text or lower data.min_count")

    model = Word2Vec(len(vocab.words), model_config.dimensions, model_config.architecture)
    sampler = NegativeSampler(vocab.counts) if model_config.loss == "negative" else None
    loader = DataLoader(pairs, batch_size=fit.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=fit.lr)

    source, target = pairs[0]
    example = ([vocab.words[i] for i in source] if model_config.architecture == "cbow"
               else vocab.words[source])
    print(f"{model_config.architecture} / {model_config.loss}: {len(tokens)} tokens, "
          f"{len(vocab.words)} words, {len(pairs)} examples")
    print(f"Example: {example} -> {vocab.words[target]}")
    print(f"W1 input and W2 output: {tuple(model.input.weight.shape)} each")

    first_batch = True
    for epoch in range(fit.epochs):
        total = seen = 0
        started = time.perf_counter()
        for step, (source, target) in enumerate(loader, 1):
            if model_config.architecture == "cbow":
                source = torch.stack(source, dim=1)  # DataLoader transposes nested lists.
            negatives = sampler.sample(target, model_config.negatives) if sampler else None
            if first_batch:
                scores = (f"positive {(len(target),)}, negative {tuple(negatives.shape)}"
                          if negatives is not None else f"softmax {(len(target), len(vocab.words))}")
                print(f"First batch: source {tuple(source.shape)} -> hidden "
                      f"{(len(target), model_config.dimensions)} -> scores {scores}")
                first_batch = False
            loss = model(source, target, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
            seen += len(target)
            if sys.stdout.isatty() and (step % max(1, len(loader) // 30) == 0 or step == len(loader)):
                filled = round(24 * step / len(loader))
                print(f"\repoch {epoch + 1} [{'#' * filled}{'.' * (24 - filled)}] "
                      f"{step}/{len(loader)} loss {total / seen:.3f}", end="", flush=True)
        if sys.stdout.isatty():
            print()
        print(f"epoch {epoch + 1}: loss {total / len(pairs):.3f} "
              f"({time.perf_counter() - started:.1f}s)")

    torch.save({"words": vocab.words, "vectors": model.input.weight.detach()}, config.output)
    print(f"Saved {config.output}")


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python train.py --config configs/train.yaml")
    train(load_config())
