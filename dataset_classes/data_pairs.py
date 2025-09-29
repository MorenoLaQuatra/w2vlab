from tqdm import tqdm
from .vocabulary import preprocess_text

def create_skipgram_pairs(texts, word2idx, window_size=2):
    """
    Create (center_word, context_word) pairs for Skip-gram
    Args:
        texts: List of text strings
        word2idx: Dictionary mapping words to indices
        window_size: Size of the context window
    Returns:
        List of (center_word_idx, context_word_idx) tuples
    """
    pairs = []
    raise NotImplementedError("Implement Skip-gram pair creation. Hint: for each word, create (center, context) pairs within the window.")

def create_cbow_pairs(texts, word2idx, window_size=2):
    """
    Create (context_words, center_word) pairs for CBOW
    Args:
        texts: List of text strings
        word2idx: Dictionary mapping words to indices
        window_size: Size of the context window
    Returns:
        List of (context_words_list, center_word_idx) tuples
    """
    pairs = []
    raise NotImplementedError("Implement CBOW pair creation. Hint: for each word, create (context, center) pairs with fixed context size.")
