"""Simple questions to ask learned word vectors."""

import torch
from torch.nn import functional as F

from .data import tokenize


class WordVectors:
    def __init__(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.words = saved["words"]
        self.to_id = {word: i for i, word in enumerate(self.words)}
        self.vectors = saved["vectors"]
        self.unit = F.normalize(self.vectors, dim=1)

    def index(self, word):
        tokens = tokenize(word)
        if len(tokens) != 1 or tokens[0] not in self.to_id:
            raise KeyError(f"OOV: {word}")
        return self.to_id[tokens[0]]

    def closest(self, vector, top=10, exclude=()):
        if top < 1:
            raise ValueError("top must be positive")
        scores = self.unit @ F.normalize(vector, dim=0)
        for i in (0, *exclude):
            scores[i] = -float("inf")
        count = min(top, len(self.words) - len(set((0, *exclude))))
        return [(self.words[i], scores[i].item()) for i in scores.topk(count).indices.tolist()]

    def most_similar(self, word, top=10):
        i = self.index(word)
        return self.closest(self.vectors[i], top, (i,))

    def similarity(self, first, second):
        return (self.unit[self.index(first)] @ self.unit[self.index(second)]).item()

    def analogy(self, a, b, c, top=10):
        ia, ib, ic = self.index(a), self.index(b), self.index(c)
        guess = self.vectors[ib] - self.vectors[ia] + self.vectors[ic]
        return self.closest(guess, top, (ia, ib, ic))
