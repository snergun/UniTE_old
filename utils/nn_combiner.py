"""
Neural network combiner for ensemble predictions.

Implements a learned non-linear combination function f(p_1, p_2, ..., p_n) -> z
that operates on probabilities from multiple models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCombiner(nn.Module):
    """
    Multi-layer perceptron for combining model probabilities.

    For each token in the union vocabulary, takes probabilities from N models
    and outputs a single score. The universal function is applied to all tokens.

    Architecture:
        Input: [num_models] probabilities
        Hidden layers: [hidden_dim1, hidden_dim2, ...]
        Output: [1] score
    """

    def __init__(self, num_models, hidden_dims=[64, 32], dropout=0.2):
        """
        Args:
            num_models: Number of models being ensembled
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super(MLPCombiner, self).__init__()

        self.num_models = num_models
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        in_dim = num_models

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer (single score per token)
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, model_probs, mask=None):
        """
        Forward pass through the network.

        Args:
            model_probs: Tensor of shape [batch, vocab_size, num_models]
                        Probabilities from each model for each token
            mask: Optional tensor of shape [batch, vocab_size]
                  Binary mask (1 for valid tokens, 0 for padding)

        Returns:
            scores: Tensor of shape [batch, vocab_size]
                   Raw scores for each token (before softmax)
        """
        batch_size, vocab_size, num_models = model_probs.shape
        assert num_models == self.num_models, f"Expected {self.num_models} models, got {num_models}"

        # Reshape to [batch * vocab_size, num_models]
        flat_probs = model_probs.view(-1, num_models)

        # Apply network to each token
        scores = self.network(flat_probs)  # [batch * vocab_size, 1]

        # Reshape back to [batch, vocab_size]
        scores = scores.view(batch_size, vocab_size)

        # Apply mask if provided (set padding scores to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        return scores

    def get_distribution(self, model_probs, mask=None):
        """
        Get probability distribution over tokens.

        Args:
            model_probs: Tensor of shape [batch, vocab_size, num_models]
            mask: Optional tensor of shape [batch, vocab_size]

        Returns:
            probs: Tensor of shape [batch, vocab_size]
                   Probability distribution (sums to 1 across vocab)
        """
        scores = self.forward(model_probs, mask)
        probs = F.softmax(scores, dim=-1)
        return probs


class AttentionCombiner(nn.Module):
    """
    Attention-based combiner that learns to weight model contributions.

    Instead of a fixed MLP, uses attention to dynamically weight each model's
    probability based on the input probabilities themselves.
    """

    def __init__(self, num_models, hidden_dim=64, dropout=0.2):
        """
        Args:
            num_models: Number of models being ensembled
            hidden_dim: Hidden dimension for attention computation
            dropout: Dropout probability
        """
        super(AttentionCombiner, self).__init__()

        self.num_models = num_models
        self.hidden_dim = hidden_dim

        # Attention network
        self.query = nn.Linear(num_models, hidden_dim)
        self.key = nn.Linear(num_models, hidden_dim)
        self.value = nn.Linear(num_models, hidden_dim)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, model_probs, mask=None):
        """
        Forward pass using attention mechanism.

        Args:
            model_probs: Tensor of shape [batch, vocab_size, num_models]
            mask: Optional tensor of shape [batch, vocab_size]

        Returns:
            scores: Tensor of shape [batch, vocab_size]
        """
        batch_size, vocab_size, num_models = model_probs.shape

        # Compute attention scores
        Q = self.query(model_probs)  # [batch, vocab_size, hidden_dim]
        K = self.key(model_probs)    # [batch, vocab_size, hidden_dim]
        V = self.value(model_probs)  # [batch, vocab_size, hidden_dim]

        # Self-attention (simplified)
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores)  # [batch, vocab_size, 1]

        # Weighted value
        weighted_value = attn_weights * V  # [batch, vocab_size, hidden_dim]

        # Output projection
        scores = self.output(weighted_value).squeeze(-1)  # [batch, vocab_size]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        return scores

    def get_distribution(self, model_probs, mask=None):
        """Get probability distribution over tokens."""
        scores = self.forward(model_probs, mask)
        probs = F.softmax(scores, dim=-1)
        return probs


def nn_combine_and_sample(v_list, nn_model, union_vocab, tokenizer, device='cuda'):
    """
    Combine model predictions using trained neural network and sample next token.

    This function mimics the interface of average_and_sample() in unite2.py but
    uses a neural network instead of linear combination.

    Args:
        v_list: List of probability dictionaries from each model (after vocab_softmax)
                Each element is a list of dicts: [{token: [prob, id], ...}, ...]
        nn_model: Trained MLPCombiner or AttentionCombiner model
        union_vocab: List of lists of union vocabulary tokens
        tokenizer: Primary tokenizer for decoding
        device: Device to run NN on

    Returns:
        next_token: List of selected token strings
        v_combined: List of combined probability dictionaries
        next_token_ids: List of lists of token IDs (one per model)
    """
    nn_model.eval()
    batch_size = len(v_list[0])
    num_models = len(v_list)

    # Find max vocab size in batch
    max_vocab_size = max(len(vu) for vu in union_vocab)

    # Prepare input tensor [batch_size, max_vocab_size, num_models]
    model_probs_tensor = torch.zeros(batch_size, max_vocab_size, num_models, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_vocab_size, dtype=torch.float32)

    # Fill in probabilities
    for batch_idx in range(batch_size):
        vocab_size = len(union_vocab[batch_idx])
        mask[batch_idx, :vocab_size] = 1.0

        for token_idx, token in enumerate(union_vocab[batch_idx]):
            for model_idx in range(num_models):
                if token in v_list[model_idx][batch_idx]:
                    prob = v_list[model_idx][batch_idx][token][0]
                    if isinstance(prob, torch.Tensor):
                        prob = prob.item()
                    model_probs_tensor[batch_idx, token_idx, model_idx] = prob

    # Run through NN
    model_probs_tensor = model_probs_tensor.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        combined_probs = nn_model.get_distribution(model_probs_tensor, mask)  # [batch, vocab]

    # Sample (greedy)
    combined_probs = combined_probs.cpu()
    next_token = []
    v_combined = []
    next_token_ids = [[] for _ in range(num_models)]

    for batch_idx in range(batch_size):
        vocab_size = len(union_vocab[batch_idx])
        probs_batch = combined_probs[batch_idx, :vocab_size]

        # Get max prob token
        sample_index = torch.argmax(probs_batch).item()
        selected_token = union_vocab[batch_idx][sample_index]

        next_token.append(selected_token)

        # Build combined probability dict
        v_comb = {}
        for token_idx, token in enumerate(union_vocab[batch_idx]):
            # Use token ID from first model as reference
            token_id = v_list[0][batch_idx][token][1]
            v_comb[token] = [probs_batch[token_idx].item(), token_id]
        v_combined.append(v_comb)

        # Get token IDs from each model
        for model_idx in range(num_models):
            token_id = v_list[model_idx][batch_idx][selected_token][1]
            next_token_ids[model_idx].append(token_id)

    return next_token, v_combined, next_token_ids


def load_nn_combiner(model_path, num_models, device='cuda'):
    """
    Load trained NN combiner from checkpoint.

    Args:
        model_path: Path to saved model checkpoint
        num_models: Number of models being ensembled
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config
    model_type = checkpoint.get('model_type', 'mlp')
    model_config = checkpoint.get('config', {})

    # Create model
    if model_type == 'mlp':
        model = MLPCombiner(
            num_models=num_models,
            hidden_dims=model_config.get('hidden_dims', [64, 32]),
            dropout=model_config.get('dropout', 0.2)
        )
    elif model_type == 'attention':
        model = AttentionCombiner(
            num_models=num_models,
            hidden_dim=model_config.get('hidden_dim', 64),
            dropout=model_config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
