import torch
from tqdm import tqdm
import numpy as np

class NegativeSampler:
    """
    Improved negative sampling implementation using unigram distribution^0.75
    More memory efficient than table-based approach.
    """
    def __init__(self, vocab_size, word_frequencies):
        '''
        Creates probability distribution for negative sampling using freq^0.75
        Uses torch.multinomial for efficient sampling instead of large lookup table.
        '''
        
        self.vocab_size = vocab_size
        power = 0.75  # ^3/4
        
        # Convert to torch tensor for efficient computation
        frequencies = torch.FloatTensor(word_frequencies)
        
        # Compute probabilities: freq^0.75 / sum(freq^0.75)
        freq_power = frequencies ** power
        self.probabilities = freq_power / freq_power.sum()
        
        print(f"Negative sampling probabilities created for {vocab_size} words")
    
    def sample(self, batch_size, num_negative):
        """
        Sample negative examples using multinomial distribution
        More memory efficient and faster than table lookup
        """
        total_samples = batch_size * num_negative
        
        # Sample using multinomial distribution
        negative_samples = torch.multinomial(
            self.probabilities, 
            total_samples, 
            replacement=True
        )
        
        return negative_samples.reshape(batch_size, num_negative)

    
'''
The multinomial sampling approach works like this:

Given word frequencies: [f0, f1, f2, ...]
1. Compute powers: [f0^0.75, f1^0.75, f2^0.75, ...]
2. Normalize to probabilities: [p0, p1, p2, ...] where sum(pi) = 1
3. Use torch.multinomial to sample according to these probabilities

Example:
w0 freq=100 -> 100^0.75 = 31.6 -> p0 = 0.45
w1 freq=50  -> 50^0.75  = 18.8 -> p1 = 0.27  
w2 freq=25  -> 25^0.75  = 11.2 -> p2 = 0.16
w3 freq=10  -> 10^0.75  = 5.6  -> p3 = 0.08
w4 freq=5   -> 5^0.75   = 3.3  -> p4 = 0.05

'''
