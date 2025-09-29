import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    """
    CBOW (Continuous Bag of Words) Model
    
    Traditional Architecture:
    - W1 (in_embeddings): vocab_size x embed_dim - INPUT WORD EMBEDDINGS
    - W2 (out_weights): embed_dim x vocab_size - OUTPUT PROJECTION MATRIX
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.init_weights()
        print(f"CBOW Model:")
        print(f"  - W1 (in_embeddings):  {vocab_size} x {embed_dim}")
        print(f"  - W2 (out_embeddings): {vocab_size} x {embed_dim}")
    
    def init_weights(self):
        init_range = 0.5 / self.embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-init_range, init_range)
        
    def forward_negative_sampling(self, context_words, center_words, negative_words):
        """
        Forward pass with negative sampling (EFFICIENT)
        Args:
            context_words: (batch_size, context_size) - context word indices
            center_words: (batch_size,) - center word indices
            negative_words: (batch_size, num_negative) - negative word indices
        Returns:
            loss: scalar tensor
        """
        batch_size = context_words.size(0)
        
        raise NotImplementedError("Implement negative sampling forward pass for CBOWModel. Hint: average context embeddings, use sigmoid and BCE loss.")
    
    def forward_full_softmax(self, context_words, center_words):
        """
        Forward pass with full softmax (SLOW)
        Args:
            context_words: (batch_size, context_size) - context word indices
            center_words: (batch_size,) - center word indices
        Returns:
            loss: scalar tensor
        """
        raise NotImplementedError("Implement full softmax forward pass for CBOWModel. Hint: average context embeddings, matmul, and cross-entropy loss.")
    
    def get_word_embeddings(self):
        return self.in_embeddings.weight.detach()
