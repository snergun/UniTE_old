"""
Training script for neural network combiner.

Trains a neural network to combine predictions from multiple language models
using cached predictions as training data.

Usage:
    python train_nn_combiner.py \
        --cache_file <path_to_cache.pkl> \
        --output_model <path_to_save_model.pt> \
        --num_models 2 \
        --model_type mlp \
        --hidden_dims 64 32 \
        --epochs 50 \
        --batch_size 256 \
        --lr 0.001
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from utils.cache_utils import CachedPredictionsDataset, collate_fn, split_dataset, get_dataset_stats
from utils.nn_combiner import MLPCombiner, AttentionCombiner


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        model_probs = batch['model_probs'].to(device)  # [batch, vocab, num_models]
        target_idx = batch['target_idx'].to(device)    # [batch]
        mask = batch['mask'].to(device)                 # [batch, vocab]

        # Forward pass
        scores = model(model_probs, mask)  # [batch, vocab]

        # Calculate loss
        loss = criterion(scores, target_idx)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(scores, dim=1)
        correct += (predicted == target_idx).sum().item()
        total += target_idx.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            model_probs = batch['model_probs'].to(device)
            target_idx = batch['target_idx'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            scores = model(model_probs, mask)

            # Calculate loss
            loss = criterion(scores, target_idx)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(scores, dim=1)
            correct += (predicted == target_idx).sum().item()
            total += target_idx.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train neural network combiner")
    parser.add_argument("--cache_file", type=str, required=True, help="Path to cached predictions")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--num_models", type=int, required=True, help="Number of models being ensembled")

    # Model architecture
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "attention"],
                        help="Type of combiner model")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32],
                        help="Hidden layer dimensions for MLP")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for attention model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Other options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vocab_size", type=int, default=30, help="Max vocabulary size for padding")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print dataset statistics
    print("Loading dataset...")
    stats = get_dataset_stats(args.cache_file)
    print("\nDataset Statistics:")
    print(f"  Dataset: {stats['dataset']}")
    print(f"  Number of models: {stats['num_models']}")
    print(f"  Number of questions: {stats['num_questions']}")
    print(f"  Number of examples: {stats['num_examples']}")
    print(f"  Avg vocabulary size: {stats['avg_vocab_size']:.2f}")
    print(f"  Max vocabulary size: {stats['max_vocab_size']}")
    print(f"  Avg generation length: {stats['avg_generation_length']:.2f}")

    # Verify num_models matches
    if stats['num_models'] != args.num_models:
        raise ValueError(f"Cache has {stats['num_models']} models but --num_models={args.num_models}")

    # Load dataset
    dataset = CachedPredictionsDataset(args.cache_file, max_vocab_size=args.max_vocab_size)

    # Split into train/val
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=args.train_ratio, seed=args.seed)
    print(f"\nTrain examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    # Create model
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == "mlp":
        model = MLPCombiner(
            num_models=args.num_models,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout
        )
        model_config = {
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout
        }
    elif args.model_type == "attention":
        model = AttentionCombiner(
            num_models=args.num_models,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        model_config = {
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout
        }

    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                      patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    print("\nStarting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'config': model_config,
                'num_models': args.num_models,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }

            torch.save(checkpoint, args.output_model)
            print(f"  → Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        print()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_model}")


if __name__ == "__main__":
    main()
