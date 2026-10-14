"""Train word2vec or fastText with an explicit YAML config."""

import sys
import time
import random
from itertools import islice
from pathlib import Path

import torch
from yaml_config_manager import load_config

from w2vlab.data import Vocabulary, Subwords, count_corpus, make_pairs, stream_tokens
from w2vlab.model import FastText, NegativeSampler, Word2Vec


def batches(path, limit, vocab, window, architecture, batch_size, shuffle_buffer,
            seed, skip=0, on_skip=None):
    ids = (vocab.to_id.get(word, 0) if word is not None else None
           for word in stream_tokens(path, limit))
    pairs = make_pairs(ids, window, architecture)
    randomizer = random.Random(seed)
    original_skip = skip
    while chunk := list(islice(pairs, shuffle_buffer)):
        randomizer.shuffle(chunk)
        if skip >= len(chunk):
            skip -= len(chunk)
            if on_skip:
                on_skip(original_skip - skip)
            continue
        for start in range(0, len(chunk), batch_size):
            batch = chunk[start:start + batch_size]
            if skip:
                if skip < len(batch):
                    raise ValueError("checkpoint is not aligned to a training batch")
                skip -= len(batch)
                if on_skip:
                    on_skip(original_skip - skip)
                continue
            source, target = zip(*batch)
            yield torch.tensor(source), torch.tensor(target)
    if skip:
        raise ValueError("checkpoint progress exceeds the corpus")


def select_device(requested):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError("training.device must be auto, cuda, mps, or cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use training.device: auto or cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use training.device: auto or cpu")
    return torch.device(requested)


def clock(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def checkpoint_settings(data, model_config, fit, kind, shuffle_buffer):
    corpus = Path(data.corpus).resolve()
    stat = corpus.stat()
    return {
        "corpus": str(corpus), "corpus_size": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "max_tokens": data.max_tokens, "min_count": data.min_count,
        "max_words": data.max_words, "window": data.window,
        "type": kind, "architecture": model_config.architecture,
        "dimensions": model_config.dimensions, "loss": model_config.loss,
        "negatives": model_config.negatives, "min_n": model_config.min_n,
        "max_n": model_config.max_n, "batch_size": fit.batch_size,
        "shuffle_buffer": shuffle_buffer, "lr": fit.lr, "seed": fit.seed,
    }


def save_checkpoint(path, settings, model, optimizer, vocab, token_count,
                    example_count, epoch, seen, loss_sum):
    saved = {
        "format": "w2vlab-training-1", "settings": settings,
        "words": vocab.words, "counts": vocab.counts,
        "token_count": token_count, "example_count": example_count,
        "epoch": epoch, "seen": seen, "loss_sum": loss_sum,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    staging = path.with_name(path.name + ".tmp")
    started = time.perf_counter()
    torch.save(saved, staging)
    staging.replace(path)  # Keep the previous checkpoint if writing was interrupted.
    progress = (f"epoch {epoch + 1}, {seen:,}/{example_count:,}"
                if seen else f"after epoch {epoch}")
    print(f"Checkpoint: {path} ({progress}; {time.perf_counter() - started:.1f}s)",
          flush=True)


def train(config):
    data, model_config, fit = config.data, config.model, config.training
    shuffle_buffer = getattr(fit, "shuffle_buffer", 10000)
    checkpoint_every = getattr(fit, "checkpoint_every", 10_000_000)
    progress_seconds = getattr(fit, "progress_seconds", 5)
    resume = getattr(fit, "resume", "")
    kind = getattr(model_config, "type", "word2vec")
    if kind not in ("word2vec", "fasttext"):
        raise ValueError("model.type must be word2vec or fasttext")
    if model_config.loss not in ("negative", "softmax"):
        raise ValueError("model.loss must be negative or softmax")
    if model_config.architecture not in ("skipgram", "cbow"):
        raise ValueError("model.architecture must be skipgram or cbow")
    sizes = (data.window, data.min_count, data.max_words, model_config.dimensions,
             model_config.negatives, fit.epochs, fit.batch_size, shuffle_buffer,
             checkpoint_every)
    if min(sizes) < 1 or fit.lr <= 0 or data.max_tokens == 0 or data.max_tokens < -1:
        raise ValueError("sizes and learning rate must be positive; max_tokens may be -1")
    if kind == "fasttext" and not 1 <= model_config.min_n <= model_config.max_n:
        raise ValueError("fasttext needs 1 <= model.min_n <= model.max_n")
    if shuffle_buffer < fit.batch_size:
        raise ValueError("training.shuffle_buffer must be at least training.batch_size")
    if progress_seconds <= 0:
        raise ValueError("training.progress_seconds must be positive")

    device = select_device(getattr(fit, "device", "auto"))
    name = f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""
    print(f"Device: {device}{name}", flush=True)
    settings = checkpoint_settings(data, model_config, fit, kind, shuffle_buffer)
    output = Path(config.output)
    checkpoint_path = output.with_name(output.stem + ".checkpoint.pt")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "w2vlab-training-1":
            raise ValueError("training.resume needs a training checkpoint, not a final model")
        if checkpoint["settings"] != settings:
            changed = [key for key in settings if checkpoint["settings"].get(key) != settings[key]]
            raise ValueError(f"checkpoint settings differ: {', '.join(changed)}")
        vocab = Vocabulary.from_saved(checkpoint["words"], checkpoint["counts"])
        token_count, example_count = checkpoint["token_count"], checkpoint["example_count"]
        print(f"Resuming {resume}: epoch {checkpoint['epoch'] + 1}, "
              f"{checkpoint['seen']:,}/{example_count:,} examples", flush=True)
    else:
        if checkpoint_path.exists():
            raise FileExistsError(f"{checkpoint_path} exists; use --training.resume "
                                  "or choose a new --output")
        torch.manual_seed(fit.seed)
        print("Counting corpus and building vocabulary...", flush=True)
        count_started = time.perf_counter()

        def show_count(count):
            terminal = sys.stdout.isatty()
            prefix = "\r" if terminal else ""
            elapsed = max(time.perf_counter() - count_started, 0.001)
            print(f"{prefix}Counted {count:,} tokens | {count / elapsed:,.0f}/s "
                  f"| elapsed {clock(elapsed)}",
                  end="" if terminal else "\n", flush=True)

        vocab, token_count, example_count = count_corpus(
            data.corpus, data.max_tokens, data.min_count, data.max_words,
            data.window, model_config.architecture, progress=show_count
        )
        if token_count >= 1_000_000 and sys.stdout.isatty():
            print()
        print(f"Vocabulary counted in {clock(time.perf_counter() - count_started)}")
    if len(vocab.words) < 3 or not example_count:
        raise ValueError("corpus is too small; try more text or lower data.min_count")

    subwords = (Subwords(vocab.words, model_config.min_n, model_config.max_n)
                if kind == "fasttext" else None)
    model = (FastText(len(vocab.words), subwords.ids, len(subwords.grams),
                      model_config.dimensions, model_config.architecture)
             if subwords else Word2Vec(len(vocab.words), model_config.dimensions,
                                       model_config.architecture)).to(device)
    sampler = NegativeSampler(vocab.counts) if model_config.loss == "negative" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=fit.lr)
    if resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])

    source, target = next(make_pairs(
        (vocab.to_id.get(word, 0) if word is not None else None
         for word in stream_tokens(data.corpus, data.max_tokens)),
        data.window, model_config.architecture))
    example = ([vocab.words[i] for i in source] if model_config.architecture == "cbow"
               else vocab.words[source])
    print(f"{kind} / {model_config.architecture} / {model_config.loss}: "
          f"{token_count:,} tokens, {len(vocab.words):,} words, {example_count:,} examples")
    print(f"Vocabulary: {len(vocab.words) - 1:,} words + <unk>; "
          f"{vocab.counts[0] / token_count:.1%} of tokens -> <unk>")
    print("IDs: " + ", ".join(f"{i}={vocab.words[i]} ({vocab.counts[i]:,})"
                              for i in range(min(6, len(vocab.words)))))
    if len(vocab.words) - 1 == data.max_words:
        print("Vocabulary reached data.max_words; other words map to <unk>.")
    if model_config.architecture == "cbow":
        print(f"CBOW: {2 * data.window} context words -> mean vector -> center word")
    else:
        print(f"Skip-gram: center word -> {2 * data.window} context targets")
    if model_config.loss == "negative":
        print(f"Negative sampling: 1 true word + {model_config.negatives} sampled words "
              f"(instead of scoring all {len(vocab.words):,})")
    else:
        print(f"Softmax: score all {len(vocab.words):,} vocabulary entries")
    print(f"Streaming batches of {fit.batch_size}; shuffle buffer {shuffle_buffer:,} "
          "(no full pair list in RAM)")
    print(f"Saving every {checkpoint_every:,} examples to {checkpoint_path}")
    if subwords:
        print(f"G character n-grams: {(len(subwords.grams), model_config.dimensions)} "
              f"(including padding); "
              f"padded pieces per word: {subwords.width}")
    print(f"Example: {example} -> {vocab.words[target]}")
    print(f"W1 input and W2 output: {tuple(model.input.weight.shape)} each")

    first_batch = True
    start_epoch = checkpoint["epoch"] if resume else 0
    for epoch in range(start_epoch, fit.epochs):
        seen = checkpoint["seen"] if resume and epoch == start_epoch else 0
        seen_at_start = seen
        total = torch.tensor(checkpoint["loss_sum"] if resume and epoch == start_epoch
                             else 0.0, device=device)
        started = time.perf_counter()
        active_started = None
        last_progress = started
        terminal = sys.stdout.isatty()
        interval = progress_seconds if terminal else max(progress_seconds, 60)
        print(f"Epoch {epoch + 1}/{fit.epochs}: {example_count:,} examples"
              + (f"; replaying {seen:,} already trained examples" if seen else ""),
              flush=True)
        replay_report = [started]

        def show_replay(skipped):
            now = time.perf_counter()
            if now - replay_report[0] < interval:
                return
            rate = skipped / max(now - started, 0.001)
            line = (f"Replay {skipped:,}/{seen_at_start:,} "
                    f"({skipped / seen_at_start:.1%}) | {rate:,.0f}/s "
                    f"| ETA {clock((seen_at_start - skipped) / max(rate, 0.001))}")
            print(("\r" if terminal else "") + line,
                  end="" if terminal else "\n", flush=True)
            replay_report[0] = now

        loader = batches(data.corpus, data.max_tokens, vocab, data.window,
                         model_config.architecture, fit.batch_size, shuffle_buffer,
                         fit.seed + epoch, skip=seen,
                         on_skip=show_replay if seen else None)
        next_checkpoint = (seen // checkpoint_every + 1) * checkpoint_every
        for source, target in loader:
            if active_started is None:
                active_started = time.perf_counter()
                if seen_at_start:
                    if terminal and replay_report[0] != started:
                        print()
                    print(f"Replay complete in {clock(active_started - started)}; "
                          "continuing updates", flush=True)
            negatives = sampler.sample(target, model_config.negatives) if sampler else None
            if first_batch:
                scores = (f"positive {(len(target),)}, negative {tuple(negatives.shape)}"
                          if negatives is not None else f"softmax {(len(target), len(vocab.words))}")
                print(f"First batch: source {tuple(source.shape)} -> hidden "
                      f"{(len(target), model_config.dimensions)} -> scores {scores}")
                first_batch = False
            source, target = source.to(device), target.to(device)
            if negatives is not None:
                negatives = negatives.to(device)
            loss = model(source, target, negatives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(target)
            seen += len(target)
            now = time.perf_counter()
            if now - last_progress >= interval or seen == example_count:
                rate = (seen - seen_at_start) / max(now - active_started, 0.001)
                filled = round(24 * seen / example_count)
                line = (f"epoch {epoch + 1}/{fit.epochs} "
                        f"[{'#' * filled}{'.' * (24 - filled)}] "
                        f"{seen:,}/{example_count:,} ({seen / example_count:.1%}) "
                        f"loss {total.item() / seen:.3f} | {rate:,.0f}/s "
                        f"| ETA {clock((example_count - seen) / max(rate, 0.001))}")
                print(("\r" if terminal else "") + line,
                      end="" if terminal else "\n", flush=True)
                last_progress = now
            if seen >= next_checkpoint and seen < example_count:
                if terminal:
                    print()
                save_checkpoint(checkpoint_path, settings, model, optimizer, vocab,
                                token_count, example_count, epoch, seen, total.item())
                next_checkpoint = seen + checkpoint_every
        if seen != example_count:
            raise ValueError("example count changed while training; corpus may have changed")
        if terminal:
            print()
        finished = time.perf_counter()
        elapsed = finished - started
        active_elapsed = finished - (active_started or started)
        print(f"epoch {epoch + 1}: loss {total.item() / seen:.3f} | "
              f"elapsed {clock(elapsed)} | "
              f"{(seen - seen_at_start) / max(active_elapsed, 0.001):,.0f} examples/s")
        save_checkpoint(checkpoint_path, settings, model, optimizer, vocab,
                        token_count, example_count, epoch + 1, 0, 0.0)

    with torch.no_grad():
        vectors = torch.cat([model.embed(torch.arange(i, min(i + 512, len(vocab.words)),
                                                     device=device)).cpu()
                             for i in range(0, len(vocab.words), 512)])
    saved = {
        "words": vocab.words,
        "counts": vocab.counts,
        "vectors": vectors,  # W1 for word2vec; W1 + G for fasttext
        "output_vectors": model.output.weight.detach().cpu(),  # W2: used during training
        "model_type": kind,
        "architecture": model_config.architecture,
        "loss": model_config.loss,
        "window": data.window,
        "negatives": model_config.negatives,
    }
    if subwords:
        saved.update({"input_vectors": model.input.weight.detach().cpu(),
                      "subword_vectors": model.subword.weight.detach().cpu(),
                      "subwords": subwords.grams, "min_n": model_config.min_n,
                      "max_n": model_config.max_n, "max_subwords": subwords.width})
    torch.save(saved, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python train.py --config configs/train.yaml")
    train(load_config())
