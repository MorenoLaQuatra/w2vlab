"""The two embedding tables and the two word2vec objectives."""

from torch import nn
from torch.nn import functional as F
import torch


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, dimensions=50, architecture="skipgram"):
        super().__init__()
        if architecture not in ("skipgram", "cbow"):
            raise ValueError("architecture must be skipgram or cbow")
        self.architecture = architecture
        self.input = nn.Embedding(vocab_size, dimensions)   # W1: saved word vectors
        self.output = nn.Embedding(vocab_size, dimensions)  # W2: training targets
        nn.init.uniform_(self.input.weight, -0.5 / dimensions, 0.5 / dimensions)
        nn.init.zeros_(self.output.weight)

    def forward(self, source, target, negatives=None):
        hidden = self.input(source)
        if self.architecture == "cbow":
            hidden = hidden.mean(dim=1)

        if negatives is None:  # Full softmax scores every word in the vocabulary.
            scores = hidden @ self.output.weight.T
            return F.cross_entropy(scores, target)

        positive = (hidden * self.output(target)).sum(dim=1)
        negative = (hidden[:, None, :] * self.output(negatives)).sum(dim=2)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negative).sum(dim=1)).mean()


class NegativeSampler:
    def __init__(self, counts):
        weights = torch.tensor(counts, dtype=torch.float).pow(0.75)
        if (weights > 0).sum() < 2:
            raise ValueError("negative sampling needs at least two observed words")
        self.probabilities = weights / weights.sum()

    def sample(self, targets, k):
        # A positive target must not also be a negative example.
        negatives = torch.multinomial(self.probabilities, targets.numel() * k,
                                      replacement=True).view(-1, k)
        collision = negatives == targets[:, None]
        while collision.any():
            negatives[collision] = torch.multinomial(
                self.probabilities, int(collision.sum()), replacement=True)
            collision = negatives == targets[:, None]
        return negatives
