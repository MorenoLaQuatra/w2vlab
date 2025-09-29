import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """
    Skip-gram Word2Vec Model
    
    Traditional Architecture:
    - W1 (in_embeddings): vocab_size x embed_dim - INPUT WORD EMBEDDINGS
    - W2 (out_weights): embed_dim x vocab_size - OUTPUT PROJECTION MATRIX
    
    Two training modes:
    1. Negative Sampling (efficient): sigmoid loss with negative samples
    2. Full Softmax (slow): standard cross-entropy over full vocabulary
    """
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.init_weights()
        print(f"Skip-gram Model:")
        print(f"  - W1 (in_embeddings):  {vocab_size} x {embed_dim}")
        print(f"  - W2 (out_embeddings): {embed_dim} x {vocab_size}")
        
    def init_weights(self):
        init_range = 0.5 / self.embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-init_range, init_range)
        
    def forward_negative_sampling(self, center_words, target_words, negative_words):
        """
        Forward pass with negative sampling (EFFICIENT)
        Args:
            center_words: (batch_size,) - center word indices
            target_words: (batch_size,) - target context word indices  
            negative_words: (batch_size, num_negative) - negative word indices
        Returns:
            loss: scalar tensor
        """
        batch_size = center_words.size(0)
        
        raise NotImplementedError("Implement negative sampling forward pass for SkipGramModel. Hint: use embeddings, sigmoid, and BCE loss.")
    
    def forward_full_softmax(self, center_words, target_words):
        """
        Forward pass with full softmax (SLOW)
        Args:
            center_words: (batch_size,) - center word indices
            target_words: (batch_size,) - target context word indices
        Returns:
            loss: scalar tensor
        """
        raise NotImplementedError("Implement full softmax forward pass for SkipGramModel. Hint: use embeddings, matmul, and cross-entropy loss.")
    
    def get_word_embeddings(self):
        return self.in_embeddings.weight.detach()
