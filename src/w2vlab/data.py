"""Words, vocabulary, and training examples."""

import re
import sqlite3
import tempfile
from collections import Counter, deque
from contextlib import closing
from pathlib import Path


WORD = re.compile(r"[^\W\d_]+")


def tokenize(text):
    # Unicode letters: keeps Italian accents, drops numbers and punctuation.
    return WORD.findall(text.lower())


def stream_tokens(path, limit=-1):
    """Read one article at a time; None marks an article boundary."""
    seen = 0
    with Path(path).open(encoding="utf-8") as corpus:
        for line in corpus:
            if 0 < limit <= seen:
                return
            for match in WORD.finditer(line.lower()):
                if 0 < limit <= seen:
                    return
                yield match.group()
                seen += 1
            yield None


def count_corpus(path, limit, min_count, max_words, window, architecture, progress=None):
    """Count words exactly on disk, keeping only the chosen vocabulary in RAM."""
    total = examples = article_words = 0
    with tempfile.TemporaryDirectory() as directory:
        with closing(sqlite3.connect(Path(directory) / "counts.sqlite")) as db:
            db.execute("PRAGMA journal_mode=OFF")  # Scratch database; safe to rebuild.
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE counts (word TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID")
            batch = Counter()

            def flush():
                with db:
                    db.executemany(
                        "INSERT INTO counts VALUES (?, ?) "
                        "ON CONFLICT(word) DO UPDATE SET n = n + excluded.n",
                        batch.items(),
                    )
                batch.clear()

            for word in stream_tokens(path, limit):
                if word is None:
                    examples += max(0, article_words - 2 * window)
                    article_words = 0
                    continue
                batch[word] += 1
                total += 1
                article_words += 1
                if total % 100_000 == 0:
                    flush()
                if progress and total % 1_000_000 == 0:
                    progress(total)
            examples += max(0, article_words - 2 * window)
            if batch:
                flush()
            rows = db.execute(
                "SELECT word, n FROM counts WHERE n >= ? "
                "ORDER BY n DESC, word LIMIT ?", (min_count, max_words)
            ).fetchall()

    vocab = Vocabulary(dict(rows), min_count, max_words, total=total)
    if architecture == "skipgram":
        examples *= 2 * window
    return vocab, total, examples


class Vocabulary:
    def __init__(self, tokens, min_count=2, max_words=10000, total=None):
        counts = Counter(tokens)
        kept = [(word, count) for word, count in counts.most_common(max_words)
                if count >= min_count]
        self.words = ["<unk>"] + [word for word, _ in kept]
        self.to_id = {word: i for i, word in enumerate(self.words)}
        self.counts = [(sum(counts.values()) if total is None else total)
                       - sum(count for _, count in kept)]
        self.counts += [count for _, count in kept]

    def encode(self, tokens):
        return [self.to_id.get(word, 0) for word in tokens]

    @classmethod
    def from_saved(cls, words, counts):
        """Restore the exact word IDs from a training checkpoint."""
        vocab = cls.__new__(cls)
        vocab.words = words
        vocab.counts = counts
        vocab.to_id = {word: i for i, word in enumerate(words)}
        return vocab


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
    """Yield pairs from a rolling window; None resets at an article boundary."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if architecture not in ("skipgram", "cbow"):
        raise ValueError("architecture must be skipgram or cbow")
    nearby = deque(maxlen=2 * window + 1)
    for word_id in ids:
        if word_id is None:
            nearby.clear()
            continue
        nearby.append(word_id)
        if len(nearby) < nearby.maxlen:
            continue
        context = list(nearby)
        center = context.pop(window)
        if architecture == "skipgram":
            for neighbor in context:
                yield center, neighbor
        else:
            yield context, center
