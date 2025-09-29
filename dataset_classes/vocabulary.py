import re
from collections import Counter
from tqdm import tqdm

def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    return text

def build_vocabulary(texts, min_freq=5, max_vocab=50000):
    """
    Build vocabulary from texts
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
        max_vocab: Maximum vocabulary size
    Returns:
        word2idx: dict mapping word to index
        idx2word: dict mapping index to word  
        word_frequencies: list of word frequencies (for negative sampling)
    """
    
    print("Building vocabulary...")
    word_counts = Counter()
    
    for text in tqdm(texts, desc="Counting words"):
        if text and text.strip():
            words = preprocess_text(text).split()
            word_counts.update(words)
    
    print(f"Found {len(word_counts)} unique words")
    
    vocab_items = [(word, count) for word, count in word_counts.most_common() if count >= min_freq][:max_vocab]
    
    print(f"Vocabulary size after filtering: {len(vocab_items)}")
    
    words, frequencies = zip(*vocab_items) if vocab_items else ([], [])
    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word, list(frequencies)
