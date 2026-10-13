"""Look inside a trained word2vec or fastText checkpoint in class."""

import sys

import torch
from yaml_config_manager import load_config
from w2vlab.data import char_ngrams


def summary(saved):
    words, w1 = saved["words"], saved["vectors"]
    kind = saved.get("model_type", "word2vec")
    print(f"{kind} / {saved.get('architecture', 'unknown')} / {saved.get('loss', 'unknown')}")
    print(f"Vocabulary: {len(words)} words; <unk> has ID 0")
    if kind == "fasttext":
        print(f"W1 (whole words): {tuple(saved['input_vectors'].shape)}")
        print(f"G (character n-grams): {tuple(saved['subword_vectors'].shape)}")
        print(f"W1 + G (inference vectors): {tuple(w1.shape)}")
    else:
        print(f"W1 (input):  {tuple(w1.shape)}  <- vectors used for inference")
    w2 = saved.get("output_vectors")
    if w2 is None:
        print("W2 (output): not saved in this older checkpoint; retrain to inspect it")
    else:
        print(f"W2 (output): {tuple(w2.shape)}  <- trained to predict context words")


def shapes(saved):
    if "architecture" not in saved:
        print("Training flow unavailable in this older checkpoint; retrain to inspect it")
        return
    vocab, dims = saved["vectors"].shape
    window = saved["window"]
    print("B = batch size, V = vocabulary size, D = embedding dimensions")
    print(f"Here: V={vocab}, D={dims}, window={window}")
    if saved.get("model_type") == "fasttext":
        pieces = saved["max_subwords"]
        print(f"G has {len(saved['subwords']) - 1} learned n-grams; M={pieces} padded pieces/word")
        if saved["architecture"] == "skipgram":
            print("center IDs [B] -> W1 [B, D] + n-gram IDs [B, M] "
                  "-> G sum [B, D] -> hidden [B, D]")
        else:
            print(f"context IDs [B, {2 * window}] -> W1 [B, {2 * window}, D] "
                  f"+ n-gram IDs [B, {2 * window}, M]")
            print(f"G sums [B, {2 * window}, D] -> add + mean -> hidden [B, D]")
    elif saved["architecture"] == "skipgram":
        print("center IDs [B] -> W1 lookup -> hidden [B, D]")
    else:
        print(f"context IDs [B, {2 * window}] -> W1 lookup "
              f"[B, {2 * window}, D] -> mean -> hidden [B, D]")
    if saved["loss"] == "softmax":
        print("hidden [B, D] @ W2.T [D, V] -> scores [B, V] -> scalar loss []")
    else:
        k = saved["negatives"]
        print(f"target [B] -> W2 lookup [B, D]; negatives [B, {k}] "
              f"-> W2 lookup [B, {k}, D]")
        print(f"dot with hidden -> scores [B] + [B, {k}] -> scalar loss []")


def show_word(saved, index, word_first=False):
    words = saved["words"]
    if not 0 <= index < len(words):
        raise ValueError(f"ID must be between 0 and {len(words) - 1}")
    count = saved.get("counts")
    frequency = f"  count={count[index]}" if count is not None else ""
    mapping = (f"{words[index]} -> ID {index}" if word_first
               else f"ID {index:>5} -> {words[index]}")
    print(f"{mapping}{frequency}")


def show_row(name, row, limit):
    numbers = ", ".join(f"{value:.3f}" for value in row[:limit].tolist())
    suffix = ", ..." if row.numel() > limit else ""
    print(f"{name} [{row.numel()}]: [{numbers}{suffix}]")


def show_weights(saved, index, gram_to_id, limit=8):
    if limit < 1:
        raise ValueError("number of dimensions must be positive")
    if gram_to_id is not None:
        ids = [gram_to_id[gram] for gram in char_ngrams(
            saved["words"][index], saved["min_n"], saved["max_n"]) if gram in gram_to_id]
        if index == 0:
            ids = []  # <unk> is synthetic, so it has no character pieces.
        show_row(f"W1 word[{index}]", saved["input_vectors"][index], limit)
        show_row("G sum", saved["subword_vectors"][ids].sum(dim=0), limit)
        show_row(f"W1 + G[{index}]", saved["vectors"][index], limit)
    else:
        show_row(f"W1 input[{index}]", saved["vectors"][index], limit)
    if "output_vectors" in saved:
        show_row(f"W2 output[{index}]", saved["output_vectors"][index], limit)
    else:
        print("W2 output: unavailable in this older checkpoint")


def run_command(saved, line, to_id, gram_to_id):
    parts = line.lower().split()
    if not parts:
        return
    command, *args = parts
    if command == "summary" and not args:
        summary(saved)
    elif command == "shapes" and not args:
        shapes(saved)
    elif command == "vocab" and len(args) <= 1:
        limit = int(args[0]) if args else 20
        if limit < 1:
            raise ValueError("vocab size must be positive")
        for i in range(min(limit, len(to_id))):
            show_word(saved, i)
    elif command == "id" and len(args) == 1:
        show_word(saved, int(args[0]))
    elif command == "ngrams" and len(args) == 1:
        if gram_to_id is None:
            print("This word2vec model has no character n-grams")
            return
        grams = char_ngrams(args[0], saved["min_n"], saved["max_n"])
        print(f"{args[0]}: {sum(gram in gram_to_id for gram in grams)}/{len(grams)} "
              "character n-grams learned")
        for gram in grams:
            print(f"  {gram:<10} -> {gram_to_id.get(gram, 'unseen')}")
    elif (command == "word" and len(args) == 1) or (command == "weights" and len(args) in (1, 2)):
        limit = int(args[1]) if command == "weights" and len(args) == 2 else 8
        if limit < 1:
            raise ValueError("number of dimensions must be positive")
        if args[0] not in to_id:
            print(f"OOV: {args[0]} (no whole-word ID)")
            if command == "weights" and gram_to_id is not None:
                ids = [gram_to_id[gram] for gram in char_ngrams(
                    args[0], saved["min_n"], saved["max_n"]) if gram in gram_to_id]
                if ids:
                    show_row("G sum (OOV vector)", saved["subword_vectors"][ids].sum(dim=0),
                             limit)
                else:
                    print("No learned character n-grams; cannot form a vector")
            return
        index = to_id[args[0]]
        show_word(saved, index, word_first=True)
        if command == "weights":
            show_weights(saved, index, gram_to_id, limit)
    else:
        print("Use: summary | shapes | vocab [N] | word WORD | id N | "
              "weights WORD [N] | ngrams WORD | quit")


def main():
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python model_info.py --config configs/model_info.yaml")
    config = load_config()
    saved = torch.load(config.model, map_location="cpu", weights_only=True)
    to_id = {word: i for i, word in enumerate(saved["words"])}
    gram_to_id = ({gram: i for i, gram in enumerate(saved["subwords"])}
                  if "subwords" in saved else None)
    try:
        run_command(saved, config.command, to_id, gram_to_id)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    if config.interactive:
        print("Commands: summary, shapes, vocab [N], word WORD, id N, "
              "weights WORD [N], ngrams WORD, quit")
        while True:
            try:
                line = input("> ").strip()
                if line.lower() == "quit":
                    break
                run_command(saved, line, to_id, gram_to_id)
            except ValueError as error:
                print(error)
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
