"""
Utilities for loading and processing cached predictions for neural network training.
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


def load_cache(cache_file):
    """
    Load cached predictions from pickle file.

    Args:
        cache_file: Path to the cache pickle file

    Returns:
        Dictionary with 'metadata' and 'data' keys
    """
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    return cache


class CachedPredictionsDataset(Dataset):
    """
    PyTorch Dataset for cached predictions.

    Converts cached predictions into training examples where:
    - Input: Probabilities from N models for each token in union vocab
    - Target: Index of the selected token in the union vocab (for now)

    Each example corresponds to one generation step (position) in one question.
    """

    def __init__(self, cache_file, max_vocab_size=30):
        """
        Args:
            cache_file: Path to cached predictions pickle file
            max_vocab_size: Maximum union vocabulary size (for padding)
        """
        self.cache = load_cache(cache_file)
        self.max_vocab_size = max_vocab_size
        self.num_models = self.cache['metadata']['num_models']

        # Flatten all generation steps into individual examples
        self.examples = []
        for question_data in self.cache['data']:
            for gen_step in question_data['generations']:
                self.examples.append({
                    'question_id': question_data['question_id'],
                    'question_text': question_data['question_text'],
                    'position': gen_step['position'],
                    'union_vocab': gen_step['union_vocab'],
                    'model_probs': self._extract_model_probs(gen_step),
                    'selected_token': gen_step['selected_token'],
                    'selected_token_idx': self._get_token_idx(gen_step),
                })

    def _extract_model_probs(self, gen_step):
        """Extract model probabilities into a list of dicts."""
        model_probs = []
        for i in range(1, self.num_models + 1):
            model_key = f'model{i}_probs'
            model_probs.append(gen_step[model_key])
        return model_probs

    def _get_token_idx(self, gen_step):
        """Get the index of the selected token in the union vocabulary."""
        selected_token = gen_step['selected_token']
        union_vocab = gen_step['union_vocab']

        try:
            return union_vocab.index(selected_token)
        except ValueError:
            # If selected token not in union vocab, return -1 (will be filtered)
            return -1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Returns:
            dict with keys:
                - model_probs: Tensor of shape [vocab_size, num_models]
                - target_idx: Index of correct token in vocabulary
                - vocab_size: Actual vocabulary size (before padding)
                - mask: Boolean mask for valid tokens (1 for valid, 0 for padding)
        """
        example = self.examples[idx]
        union_vocab = example['union_vocab']
        vocab_size = len(union_vocab)

        # Create probability matrix: [vocab_size, num_models]
        prob_matrix = np.zeros((vocab_size, self.num_models), dtype=np.float32)

        for model_idx, model_prob_dict in enumerate(example['model_probs']):
            for token_idx, token in enumerate(union_vocab):
                if token in model_prob_dict:
                    prob_matrix[token_idx, model_idx] = model_prob_dict[token][0]

        # Pad to max_vocab_size if necessary
        if vocab_size < self.max_vocab_size:
            padded_prob_matrix = np.zeros((self.max_vocab_size, self.num_models), dtype=np.float32)
            padded_prob_matrix[:vocab_size, :] = prob_matrix
            prob_matrix = padded_prob_matrix

        # Create mask (1 for valid tokens, 0 for padding)
        mask = np.zeros(self.max_vocab_size, dtype=np.float32)
        mask[:vocab_size] = 1.0

        return {
            'model_probs': torch.from_numpy(prob_matrix),
            'target_idx': example['selected_token_idx'],
            'vocab_size': vocab_size,
            'mask': torch.from_numpy(mask),
            'question_id': example['question_id'],
            'position': example['position'],
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of examples from CachedPredictionsDataset

    Returns:
        Dictionary of batched tensors
    """
    model_probs = torch.stack([item['model_probs'] for item in batch])  # [batch, vocab, num_models]
    target_idx = torch.tensor([item['target_idx'] for item in batch], dtype=torch.long)
    vocab_sizes = torch.tensor([item['vocab_size'] for item in batch], dtype=torch.long)
    masks = torch.stack([item['mask'] for item in batch])  # [batch, vocab]
    question_ids = [item['question_id'] for item in batch]
    positions = [item['position'] for item in batch]

    return {
        'model_probs': model_probs,
        'target_idx': target_idx,
        'vocab_size': vocab_sizes,
        'mask': masks,
        'question_id': question_ids,
        'position': positions,
    }


def get_dataset_stats(cache_file):
    """
    Get statistics about the cached dataset.

    Args:
        cache_file: Path to cache file

    Returns:
        Dictionary with statistics
    """
    cache = load_cache(cache_file)

    vocab_sizes = []
    num_positions = []

    for question_data in cache['data']:
        num_positions.append(len(question_data['generations']))
        for gen_step in question_data['generations']:
            vocab_sizes.append(len(gen_step['union_vocab']))

    stats = {
        'num_questions': len(cache['data']),
        'num_examples': sum(num_positions),
        'avg_vocab_size': np.mean(vocab_sizes),
        'max_vocab_size': np.max(vocab_sizes),
        'min_vocab_size': np.min(vocab_sizes),
        'avg_generation_length': np.mean(num_positions),
        'dataset': cache['metadata']['dataset'],
        'num_models': cache['metadata']['num_models'],
    }

    return stats


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets.

    Args:
        dataset: CachedPredictionsDataset instance
        train_ratio: Ratio of training data
        seed: Random seed

    Returns:
        train_dataset, val_dataset
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    split_idx = int(len(dataset) * train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset
