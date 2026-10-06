"""Small word2vec implementation for teaching."""

from .data import Vocabulary, make_pairs, tokenize
from .model import Word2Vec
from .vectors import WordVectors

__all__ = ["Vocabulary", "Word2Vec", "WordVectors", "make_pairs", "tokenize"]
