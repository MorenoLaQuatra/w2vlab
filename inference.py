#!/usr/bin/env python3
"""
Word2Vec Inference Script
Load trained Word2Vec model and perform inference operations.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

# Import our model classes
from model_classes import SkipGramModel, CBOWModel


class Word2VecInference:
    """Class for performing inference with trained Word2Vec model."""
    
    def __init__(self, model_path):
        """
        Load trained Word2Vec model.
        
        Args:
            model_path: Path to the saved model file
        """
        print(f"Loading model from {model_path}...")
        
        # Load model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.vocab_size = model_data['vocab_size']
        self.embed_dim = model_data['embed_dim']
        self.model_type = model_data['model_type']
        self.word2idx = model_data['word2idx']
        self.idx2word = model_data['idx2word']
        
        # Load embeddings directly (more efficient for inference)
        self.embeddings = model_data['embeddings']  # numpy array (vocab_size, embed_dim)
        
        # Also create the model if needed for other operations
        if self.model_type == 'skipgram':
            self.model = SkipGramModel(self.vocab_size, self.embed_dim)
        else:  # cbow
            self.model = CBOWModel(self.vocab_size, self.embed_dim)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"- Model type: {self.model_type}")
        print(f"- Vocabulary size: {self.vocab_size}")
        print(f"- Embedding dimension: {self.embed_dim}")
        print(f"- Embeddings matrix shape: {self.embeddings.shape}")
    
    def get_word_embedding(self, word):
        """
        Get the embedding vector for a word.
        
        Args:
            word: The word to get embedding for
        
        Returns:
            numpy array of shape (embed_dim,) or None if word not in vocabulary
        """
        if word not in self.word2idx:
            return None
        
        word_idx = self.word2idx[word]
        return self.embeddings[word_idx]
    
    def get_word_index(self, word):
        """Get the index of a word in vocabulary."""
        return self.word2idx.get(word, None)
    
    def get_word_from_index(self, idx):
        """Get word from its index."""
        return self.idx2word.get(idx, None)
    
    def words_in_vocabulary(self, words):
        """Check which words are in the vocabulary."""
        return [word for word in words if word in self.word2idx]
    
    def most_similar(self, word, top_k=10):
        """
        Find most similar words to the given word based on cosine similarity.
        
        Args:
            word: Input word
            top_k: Number of similar words to return
        
        Returns:
            List of tuples (word, similarity_score)
        """
        if word not in self.word2idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        word_embedding = self.get_word_embedding(word)
        word_embedding = word_embedding.reshape(1, -1)
        
        # Compute cosine similarities with all words
        similarities = cosine_similarity(word_embedding, self.embeddings)[0]
        
        # Get indices of most similar words (excluding the word itself)
        word_idx = self.word2idx[word]
        similar_indices = np.argsort(-similarities)
        
        results = []
        count = 0
        for idx in similar_indices:
            if idx != word_idx and count < top_k:  # Exclude the input word itself
                similar_word = self.idx2word[idx]
                similarity = similarities[idx]
                results.append((similar_word, similarity))
                count += 1
        
        return results
    
    def word_similarity(self, word1, word2):
        """
        Compute cosine similarity between two words.
        
        Args:
            word1, word2: Words to compare
        
        Returns:
            Similarity score or None if either word not in vocabulary
        """
        if word1 not in self.word2idx or word2 not in self.word2idx:
            return None
        
        emb1 = self.get_word_embedding(word1)
        emb2 = self.get_word_embedding(word2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity
    
    def word_analogy(self, word_a, word_b, word_c, top_k=5):
        """
        Solve word analogy: word_a is to word_b as word_c is to ?
        Using the formula: embedding(word_b) - embedding(word_a) + embedding(word_c)
        
        Args:
            word_a, word_b, word_c: Words for the analogy
            top_k: Number of candidate answers to return
        
        Returns:
            List of tuples (word, similarity_score)
        """
        words = [word_a, word_b, word_c]
        missing_words = [w for w in words if w not in self.word2idx]
        
        if missing_words:
            print(f"Words not in vocabulary: {missing_words}")
            return []
        
        # Get embeddings
        emb_a = self.get_word_embedding(word_a)
        emb_b = self.get_word_embedding(word_b)
        emb_c = self.get_word_embedding(word_c)
        
        # Compute target embedding: b - a + c
        target_embedding = emb_b - emb_a + emb_c
        target_embedding = target_embedding.reshape(1, -1)
        
        # Find most similar words to target embedding
        similarities = cosine_similarity(target_embedding, self.embeddings)[0]
        
        # Exclude the input words from results
        exclude_indices = {self.word2idx[word] for word in words}
        
        # Get top similar words
        similar_indices = np.argsort(-similarities)
        
        results = []
        count = 0
        for idx in similar_indices:
            if idx not in exclude_indices and count < top_k:
                word = self.idx2word[idx]
                similarity = similarities[idx]
                results.append((word, similarity))
                count += 1
        
        return results
    
    def get_vocabulary_stats(self):
        """Get statistics about the vocabulary."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'model_type': self.model_type,
            'sample_words': list(self.word2idx.keys())[:20]
        }
    
    def interactive_mode(self):
        """Interactive mode for exploring word embeddings."""
        print("\n" + "="*50)
        print("Word2Vec Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  similar <word> [k]     - Find k most similar words (default k=10)")
        print("  embedding <word>       - Show embedding vector for word")
        print("  similarity <word1> <word2> - Compute similarity between two words")
        print("  analogy <a> <b> <c>    - Solve analogy: a is to b as c is to ?")
        print("  vocab                  - Show vocabulary statistics")
        print("  search <partial>       - Search for words containing substring")
        print("  quit                   - Exit interactive mode")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n> ").strip().lower()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0]
                
                if command == 'quit':
                    break
                
                elif command == 'similar':
                    if len(parts) < 2:
                        print("Usage: similar <word> [k]")
                        continue
                    
                    word = parts[1]
                    k = int(parts[2]) if len(parts) > 2 else 10
                    
                    similar_words = self.most_similar(word, k)
                    if similar_words:
                        print(f"\nMost similar words to '{word}':")
                        for i, (sim_word, score) in enumerate(similar_words, 1):
                            print(f"  {i:2d}. {sim_word:<15} (similarity: {score:.4f})")
                    else:
                        print(f"Word '{word}' not found in vocabulary")
                
                elif command == 'embedding':
                    if len(parts) < 2:
                        print("Usage: embedding <word>")
                        continue
                    
                    word = parts[1]
                    embedding = self.get_word_embedding(word)
                    if embedding is not None:
                        print(f"\nEmbedding for '{word}':")
                        print(f"Shape: {embedding.shape}")
                        print(f"First 10 dimensions: {embedding[:10]}")
                        print(f"Norm: {np.linalg.norm(embedding):.4f}")
                    else:
                        print(f"Word '{word}' not found in vocabulary")
                
                elif command == 'similarity':
                    if len(parts) < 3:
                        print("Usage: similarity <word1> <word2>")
                        continue
                    
                    word1, word2 = parts[1], parts[2]
                    similarity = self.word_similarity(word1, word2)
                    if similarity is not None:
                        print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
                    else:
                        print("One or both words not found in vocabulary")
                
                elif command == 'analogy':
                    if len(parts) < 4:
                        print("Usage: analogy <word_a> <word_b> <word_c>")
                        print("Solves: word_a is to word_b as word_c is to ?")
                        continue
                    
                    word_a, word_b, word_c = parts[1], parts[2], parts[3]
                    results = self.word_analogy(word_a, word_b, word_c)
                    if results:
                        print(f"\n'{word_a}' is to '{word_b}' as '{word_c}' is to:")
                        for i, (word, score) in enumerate(results, 1):
                            print(f"  {i}. {word:<15} (score: {score:.4f})")
                
                elif command == 'vocab':
                    stats = self.get_vocabulary_stats()
                    print(f"\nVocabulary Statistics:")
                    print(f"  Size: {stats['vocab_size']}")
                    print(f"  Embedding dimension: {stats['embed_dim']}")
                    print(f"  Model type: {stats['model_type']}")
                    print(f"  Sample words: {', '.join(stats['sample_words'][:10])}")
                
                elif command == 'search':
                    if len(parts) < 2:
                        print("Usage: search <partial_word>")
                        continue
                    
                    partial = parts[1]
                    matching_words = [word for word in self.word2idx.keys() if partial in word][:20]
                    if matching_words:
                        print(f"\nWords containing '{partial}':")
                        for word in matching_words:
                            print(f"  {word}")
                        if len(matching_words) == 20:
                            print("  ... (showing first 20 matches)")
                    else:
                        print(f"No words found containing '{partial}'")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'quit' to exit or try one of the available commands.")
            
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Word2Vec Inference')
    parser.add_argument('model_path', help='Path to the trained model file')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Start interactive mode')
    parser.add_argument('--word', '-w', type=str,
                        help='Find similar words to this word')
    parser.add_argument('--top_k', '-k', type=int, default=10,
                        help='Number of similar words to show')
    parser.add_argument('--similarity', '-s', nargs=2, metavar=('WORD1', 'WORD2'),
                        help='Compute similarity between two words')
    parser.add_argument('--analogy', '-a', nargs=3, metavar=('A', 'B', 'C'),
                        help='Solve word analogy: A is to B as C is to ?')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        return
    
    # Load model
    w2v = Word2VecInference(args.model_path)
    
    # Execute requested operations
    if args.word:
        print(f"\nMost similar words to '{args.word}':")
        similar_words = w2v.most_similar(args.word, args.top_k)
        if similar_words:
            for i, (word, score) in enumerate(similar_words, 1):
                print(f"  {i:2d}. {word:<15} (similarity: {score:.4f})")
        else:
            print(f"Word '{args.word}' not found in vocabulary")
    
    if args.similarity:
        word1, word2 = args.similarity
        similarity = w2v.word_similarity(word1, word2)
        if similarity is not None:
            print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
        else:
            print("One or both words not found in vocabulary")
    
    if args.analogy:
        word_a, word_b, word_c = args.analogy
        results = w2v.word_analogy(word_a, word_b, word_c)
        if results:
            print(f"\n'{word_a}' is to '{word_b}' as '{word_c}' is to:")
            for i, (word, score) in enumerate(results, 1):
                print(f"  {i}. {word:<15} (score: {score:.4f})")
    
    # Start interactive mode if requested or no specific operation given
    if args.interactive or not any([args.word, args.similarity, args.analogy]):
        w2v.interactive_mode()


if __name__ == "__main__":
    main()