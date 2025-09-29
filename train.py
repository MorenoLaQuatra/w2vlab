#!/usr/bin/env python3
"""
Word2Vec Training Script (Template)
Most utility code is provided. Core logic must be implemented during the lesson.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os

from dataset_classes import (
    build_vocabulary, 
    create_skipgram_pairs, 
    create_cbow_pairs,
    Word2VecDataset,
    NegativeSampler
)
from model_classes import SkipGramModel, CBOWModel


def train_word2vec(model, dataloader, negative_sampler, epochs=5, lr=0.001, 
                   use_negative_sampling=True, num_negative=5, device='cpu'):
    """
    Main training function.
    """
    raise NotImplementedError("Implement the training loop for Word2Vec. Hint: see the original repo for optimizer, loss, and batching logic.")


def save_model(model, word2idx, idx2word, word_frequencies, filepath):
    """Save trained model and vocabulary"""
    model_data = {
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embed_dim': model.embed_dim,
        'model_type': 'skipgram' if isinstance(model, SkipGramModel) else 'cbow',
        'word2idx': word2idx,
        'idx2word': idx2word,
        'word_frequencies': word_frequencies,
        'embeddings': model.get_word_embeddings().cpu().numpy()
    }
    torch.save(model_data, filepath)
    print(f"Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Educational Word2Vec Training')
    parser.add_argument('--model_type', choices=['skipgram', 'cbow'], default='skipgram', help='Model architecture to use')
    parser.add_argument('--embed_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--window_size', type=int, default=2, help='Context window size')
    parser.add_argument('--dataset_name', type=str, default='wikimedia/wikipedia', help='HuggingFace dataset name (default: wikimedia/wikipedia)')
    parser.add_argument('--dataset_config', type=str, default='20231101.it', help='HuggingFace dataset configuration (default: 20231101.it)')
    parser.add_argument('--dataset_text_field', type=str, default='text', help='Field name in the dataset containing the text (default: text)')
    parser.add_argument('--min_freq', type=int, default=5, help='Minimum word frequency')
    parser.add_argument('--max_vocab', type=int, default=50000, help='Maximum vocabulary size')
    parser.add_argument('--max_texts', type=int, default=None, help='Maximum number of texts to process')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--negative_sampling', action='store_true', default=True, help='Use negative sampling (default: True)')
    parser.add_argument('--no_negative_sampling', action='store_true', help='Disable negative sampling, use full softmax instead')
    parser.add_argument('--num_negative', type=int, default=5, help='Number of negative samples per positive example')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model')
    args = parser.parse_args()
    if args.no_negative_sampling:
        args.negative_sampling = False
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading {args.dataset_name} dataset - {args.dataset_config}...")
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
        print(f"Dataset loaded: {len(dataset)} articles")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    print(f"Extracting texts (max {args.max_texts})...")
    texts = []
    for i, example in enumerate(tqdm(dataset)):
        if args.max_texts is not None and i >= args.max_texts:
            break
        if example[args.dataset_text_field] and len(example[args.dataset_text_field].strip()) > 50:
            texts.append(example[args.dataset_text_field])
    print(f"Extracted {len(texts)} texts")
    word2idx, idx2word, word_frequencies = build_vocabulary(texts, args.min_freq, args.max_vocab)
    vocab_size = len(word2idx)
    if vocab_size == 0:
        print("Error: Empty vocabulary")
        return
    if args.model_type == 'skipgram':
        pairs = create_skipgram_pairs(texts, word2idx, args.window_size)
        model = SkipGramModel(vocab_size, args.embed_dim)
    else:
        pairs = create_cbow_pairs(texts, word2idx, args.window_size)
        model = CBOWModel(vocab_size, args.embed_dim)
    if len(pairs) == 0:
        print("Error: No training pairs created")
        return
    dataset = Word2VecDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    negative_sampler = None
    if args.negative_sampling:
        negative_sampler = NegativeSampler(vocab_size, word_frequencies)
    print("Starting training...")
    train_word2vec(
        model=model,
        dataloader=dataloader, 
        negative_sampler=negative_sampler,
        epochs=args.epochs,
        lr=args.lr,
        use_negative_sampling=args.negative_sampling,
        num_negative=args.num_negative,
        device=device
    )
    suffix = "neg" if args.negative_sampling else "softmax"
    model_path = os.path.join(args.output_dir, f'word2vec_{args.model_type}_{args.embed_dim}d_{suffix}.pth')
    save_model(model, word2idx, idx2word, word_frequencies, model_path)
    print("\nTraining completed!")
    print(f"Final model saved to: {model_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {args.embed_dim}")

if __name__ == "__main__":
    main()
