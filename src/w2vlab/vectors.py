"""Simple questions to ask learned word vectors."""

import torch
from torch.nn import functional as F

from .data import char_ngrams, tokenize


def load_saved(path):
    """Open a finished model or reconstruct vectors from a training checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "w2vlab-training-1":
        return saved

    settings, weights = saved["settings"], saved["model"]
    words = saved["words"]
    vectors = weights["input.weight"].clone()
    model = {
        "words": words, "counts": saved["counts"], "vectors": vectors,
        "output_vectors": weights["output.weight"],
        "model_type": settings["type"], "architecture": settings["architecture"],
        "loss": settings["loss"], "window": settings["window"],
        "negatives": settings["negatives"],
        "checkpoint_progress": (saved["epoch"], saved["seen"], saved["example_count"]),
    }
    if settings["type"] == "fasttext":
        min_n, max_n = settings["min_n"], settings["max_n"]
        grams = [""] + list(dict.fromkeys(
            gram for word in words[1:] for gram in char_ngrams(word, min_n, max_n)))
        pieces = weights["subword.weight"]
        if len(grams) != len(pieces):
            raise ValueError("checkpoint subword table does not match its vocabulary")
        for start in range(0, len(words), 128):
            ids = weights["subword_ids"][start:start + 128]
            vectors[start:start + 128] += F.embedding_bag(ids, pieces, mode="sum",
                                                             padding_idx=0)
        model.update({
            "input_vectors": weights["input.weight"], "subword_vectors": pieces,
            "subwords": grams, "min_n": min_n, "max_n": max_n,
            "max_subwords": weights["subword_ids"].shape[1],
        })
    return model


class WordVectors:
    def __init__(self, path):
        saved = load_saved(path)
        self.checkpoint_progress = saved.get("checkpoint_progress")
        self.words = saved["words"]
        self.to_id = {word: i for i, word in enumerate(self.words)}
        self.vectors = saved["vectors"]
        self.unit = F.normalize(self.vectors, dim=1)
        self.subword_vectors = saved.get("subword_vectors")
        self.subword_to_id = ({gram: i for i, gram in enumerate(saved["subwords"])}
                              if self.subword_vectors is not None else None)
        self.min_n, self.max_n = saved.get("min_n"), saved.get("max_n")

    def _token(self, word):
        tokens = tokenize(word)
        if len(tokens) != 1:
            raise KeyError(f"OOV: {word}")
        return tokens[0]

    def index(self, word):
        token = self._token(word)
        if token not in self.to_id:
            raise KeyError(f"OOV: {word}")
        return self.to_id[token]

    def vector(self, word):
        token = self._token(word)
        if token in self.to_id:
            return self.vectors[self.to_id[token]]
        if self.subword_to_id is None:
            raise KeyError(f"OOV: {word}")
        ids = [self.subword_to_id[gram] for gram in char_ngrams(token, self.min_n, self.max_n)
               if gram in self.subword_to_id]
        if not ids:
            raise KeyError(f"OOV: {word} (no learned character n-grams)")
        return self.subword_vectors[ids].sum(dim=0)

    def closest(self, vector, top=10, exclude=()):
        if top < 1:
            raise ValueError("top must be positive")
        scores = self.unit @ F.normalize(vector, dim=0)
        for i in (0, *exclude):
            scores[i] = -float("inf")
        count = min(top, len(self.words) - len(set((0, *exclude))))
        return [(self.words[i], scores[i].item()) for i in scores.topk(count).indices.tolist()]

    def most_similar(self, word, top=10):
        i = self.to_id.get(self._token(word))
        return self.closest(self.vector(word), top, (i,) if i is not None else ())

    def similarity(self, first, second):
        return F.cosine_similarity(self.vector(first), self.vector(second), dim=0).item()

    def analogy(self, a, b, c, top=10):
        guess = self.vector(b) - self.vector(a) + self.vector(c)
        known = []
        for word in (a, b, c):
            token = self._token(word)
            if token in self.to_id:
                known.append(self.to_id[token])
        return self.closest(guess, top, known)
