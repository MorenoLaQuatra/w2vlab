"""Words, vocabulary, and training examples."""

import re
from collections import Counter


def tokenize(text):
    # Unicode letters: keeps Italian accents, drops numbers and punctuation.
    return re.findall(r"[^\W\d_]+", text.lower())


class Vocabulary:
    def __init__(self, tokens, min_count=2, max_words=10000):
        counts = Counter(tokens)
        kept = [(word, count) for word, count in counts.most_common(max_words)
                if count >= min_count]
        self.words = ["<unk>"] + [word for word, _ in kept]
        self.to_id = {word: i for i, word in enumerate(self.words)}
        self.counts = [sum(counts.values()) - sum(count for _, count in kept)]
        self.counts += [count for _, count in kept]

    def encode(self, tokens):
        return [self.to_id.get(word, 0) for word in tokens]


def char_ngrams(word, min_n=3, max_n=6):
    """Character pieces with boundary markers (trigrams of cat: <ca, cat, at>)."""
    bounded = f"<{word}>"
    return list(dict.fromkeys(bounded[i:i + n]
                              for n in range(min_n, max_n + 1)
                              for i in range(len(bounded) - n + 1)))


class Subwords:
    def __init__(self, words, min_n=3, max_n=6):
        self.grams = [""] + list(dict.fromkeys(
            gram for word in words[1:] for gram in char_ngrams(word, min_n, max_n)))
        if len(self.grams) == 1:
            raise ValueError("no character n-grams found; lower model.min_n")
        to_id = {gram: i for i, gram in enumerate(self.grams)}
        rows = [[]] + [[to_id[gram] for gram in char_ngrams(word, min_n, max_n)]
                       for word in words[1:]]
        self.width = max(map(len, rows))
        self.ids = [row + [0] * (self.width - len(row)) for row in rows]


def make_pairs(ids, window=2, architecture="skipgram"):
    """Skip-gram: (center, neighbor); CBOW: (neighbors, center)."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if architecture not in ("skipgram", "cbow"):
        raise ValueError("architecture must be skipgram or cbow")
    pairs = []
    for center in range(window, len(ids) - window):
        context = ids[center - window:center] + ids[center + 1:center + window + 1]
        if architecture == "skipgram":
            pairs.extend((ids[center], neighbor) for neighbor in context)
        else:
            pairs.append((context, ids[center]))
    return pairs
