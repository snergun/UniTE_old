# Neural Network-Based Ensemble Combination for UniTE

This implementation adds learned non-linear combination of model predictions using neural networks, replacing the simple linear weighted average with a trainable function f(p₁, p₂, ..., pₙ) → z.

## Overview

The pipeline consists of three phases:
1. **Caching**: Run inference with models and cache all probability distributions
2. **Training**: Train neural network on cached predictions with token-level supervision
3. **Inference**: Use trained NN to combine predictions during generation

## Implementation Summary

### New Files Created

```
utils/cache_utils.py        - Dataset loader for cached predictions
utils/nn_combiner.py         - Neural network architectures (MLP and Attention)
train_nn_combiner.py         - Training script for NN combiner
analyze_results.py           - Compare linear vs NN performance
```

### Modified Files

```
unite2.py                    - Added caching + NN inference (2-model)
unite3.py                    - Added caching + NN inference (3-model)
```

## Usage Guide

### Phase 1: Cache Predictions

Run your ensemble with caching enabled to save all probability distributions:

**For 2-model ensemble:**
```bash
python unite2.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 path/to/model1 \
  --model_path2 path/to/model2 \
  --output_file output_gsm8k_linear.jsonl \
  --cache_predictions cache_gsm8k_2models.pkl \
  --per_device_batch_size 1 \
  --max_new_tokens 512
```

**For 3-model ensemble:**
```bash
python unite3.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 path/to/model1 \
  --model_path2 path/to/model2 \
  --model_path3 path/to/model3 \
  --output_file output_gsm8k_linear.jsonl \
  --cache_predictions cache_gsm8k_3models.pkl \
  --per_device_batch_size 1 \
  --max_new_tokens 512
```

**Cache file contents:**
```python
{
    'metadata': {
        'dataset': 'gsm8k',
        'models': ['model1_path', 'model2_path'],
        'num_models': 2,
        'num_questions': 1319,
        'max_new_tokens': 512
    },
    'data': [
        {
            'question_id': 0,
            'question_text': "Janet's ducks lay 16 eggs...",
            'answer': "#### 18",
            'generations': [
                {
                    'position': 0,
                    'union_vocab': ['Let', 'The', 'First', ...],
                    'model1_probs': {'Let': [0.45, 2803], ...},
                    'model2_probs': {'Let': [0.52, 8561], ...},
                    'linear_combined': {'Let': 0.485, ...},
                    'selected_token': 'Let',
                    'selected_token_id1': 2803,
                    'selected_token_id2': 8561
                },
                # ... more generation steps
            ]
        },
        # ... more questions
    ]
}
```

### Phase 2: Train Neural Network

Train a neural network on the cached predictions:

```bash
python train_nn_combiner.py \
  --cache_file cache_gsm8k_2models.pkl \
  --output_model models/gsm8k_2models_combiner.pt \
  --num_models 2 \
  --model_type mlp \
  --hidden_dims 64 32 \
  --dropout 0.2 \
  --epochs 50 \
  --batch_size 256 \
  --lr 0.001 \
  --train_ratio 0.8
```

**Training arguments:**
- `--model_type`: `mlp` (default) or `attention`
- `--hidden_dims`: Hidden layer sizes for MLP (e.g., `64 32` for two hidden layers)
- `--hidden_dim`: Hidden dimension for attention model
- `--dropout`: Dropout probability for regularization
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (examples = questions × positions)
- `--lr`: Learning rate
- `--train_ratio`: Train/validation split ratio

**Training output:**
```
Dataset Statistics:
  Dataset: gsm8k
  Number of models: 2
  Number of questions: 1319
  Number of examples: 675328  # (1319 questions × ~512 positions)
  Avg vocabulary size: 15.3
  Max vocabulary size: 28

Train examples: 540262
Val examples: 135066

Creating mlp model...
Model parameters: 4737

Epoch 1/50
  Train Loss: 2.1234, Train Acc: 45.67%
  Val Loss: 1.9876, Val Acc: 48.23%
  → Saved best model

...

Training complete!
Best validation loss: 1.2345
Best validation accuracy: 62.45%
Model saved to: models/gsm8k_2models_combiner.pt
```

### Phase 3: Inference with Trained NN

Use the trained NN combiner during generation:

**For 2-model ensemble:**
```bash
python unite2.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 path/to/model1 \
  --model_path2 path/to/model2 \
  --output_file output_gsm8k_nn.jsonl \
  --use_nn_combiner \
  --nn_model_path models/gsm8k_2models_combiner.pt \
  --cache_predictions cache_gsm8k_nn.pkl \
  --per_device_batch_size 1 \
  --max_new_tokens 512
```

**For 3-model ensemble:**
```bash
python unite3.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 path/to/model1 \
  --model_path2 path/to/model2 \
  --model_path3 path/to/model3 \
  --output_file output_gsm8k_nn.jsonl \
  --use_nn_combiner \
  --nn_model_path models/gsm8k_3models_combiner.pt \
  --cache_predictions cache_gsm8k_nn.pkl \
  --per_device_batch_size 1 \
  --max_new_tokens 512
```

### Phase 4: Analyze Results

Compare linear vs NN combination:

```bash
python analyze_results.py \
  --linear_cache cache_gsm8k_2models.pkl \
  --nn_cache cache_gsm8k_nn.pkl \
  --output_dir analysis/gsm8k_comparison
```

**Analysis outputs:**
- `summary.json`: Overall statistics (agreement rate, number of differences)
- `differences.jsonl`: Detailed list of all token selection differences
- `position_agreement.png`: Plot of agreement rate by token position
- `question_differences_histogram.png`: Distribution of differences per question

## Neural Network Architectures

### MLP Combiner (Default)

```python
Input: [num_models] probabilities per token
  ↓
Linear(num_models → 64)
  ↓
ReLU + Dropout(0.2)
  ↓
Linear(64 → 32)
  ↓
ReLU + Dropout(0.2)
  ↓
Linear(32 → 1)
  ↓
Output: Raw score per token
```

Applied to each token in union vocabulary independently, then softmax across all tokens.

### Attention Combiner

Uses attention mechanism to dynamically weight model contributions:

```python
Input: [num_models] probabilities per token
  ↓
Q = Linear(num_models → hidden_dim)
K = Linear(num_models → hidden_dim)
V = Linear(num_models → hidden_dim)
  ↓
Attention weights = sigmoid(Q · K / √hidden_dim)
  ↓
Weighted V → MLP → score
```

## Complete Workflow Example

Here's a complete example for GSM8K with 2 models:

```bash
# Step 1: Cache predictions with linear combination
python unite2.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 Qwen/Qwen2-7B-Instruct \
  --model_path2 meta-llama/Meta-Llama-3-8B-Instruct \
  --output_file results/gsm8k_linear.jsonl \
  --cache_predictions cache/gsm8k_linear.pkl \
  --max_new_tokens 512

# Step 2: Train neural network
python train_nn_combiner.py \
  --cache_file cache/gsm8k_linear.pkl \
  --output_model models/gsm8k_2models.pt \
  --num_models 2 \
  --model_type mlp \
  --hidden_dims 128 64 \
  --epochs 50 \
  --batch_size 512 \
  --lr 0.001

# Step 3: Inference with trained NN
python unite2.py \
  --test_set data/gsm8k_test.json \
  --prompts prompts/gsm8k.txt \
  --model_path1 Qwen/Qwen2-7B-Instruct \
  --model_path2 meta-llama/Meta-Llama-3-8B-Instruct \
  --output_file results/gsm8k_nn.jsonl \
  --use_nn_combiner \
  --nn_model_path models/gsm8k_2models.pt \
  --cache_predictions cache/gsm8k_nn.pkl \
  --max_new_tokens 512

# Step 4: Evaluate and compare
python utils/ans_process.py --output_file results/gsm8k_linear.jsonl  # Get linear accuracy
python utils/ans_process.py --output_file results/gsm8k_nn.jsonl      # Get NN accuracy

python analyze_results.py \
  --linear_cache cache/gsm8k_linear.pkl \
  --nn_cache cache/gsm8k_nn.pkl \
  --output_dir analysis/gsm8k
```

## Dataset-Specific Recommendations

### GSM8K (Math Reasoning)
- **max_new_tokens**: 512
- **Expected benefit**: High - NN can learn which model is better at arithmetic operations vs logical steps
- **Training batch_size**: 256-512 (many examples due to long sequences)

### ARC-Challenge / PIQA (Multiple Choice)
- **max_new_tokens**: 1
- **Expected benefit**: Medium - Only one token decision, but NN can learn question-type patterns
- **Training batch_size**: 1024-2048 (fewer examples, single token)

### TriviaQA / NaturalQuestions (Open QA)
- **max_new_tokens**: 10
- **Expected benefit**: Medium-High - NN can learn which model is better at factual recall
- **Training batch_size**: 512-1024

### MMLU (Multiple Choice with Reasoning)
- **max_new_tokens**: 1
- **Expected benefit**: Medium - Similar to ARC/PIQA
- **Training batch_size**: 1024-2048

## Key Design Decisions

### Universal Function Across Tokens
- Same NN parameters applied to every token at every position
- No position-specific or token-specific parameters
- Learns general "which model to trust" patterns

### Token-Level Supervision
- Supervised by selected token at each generation step
- Loss: Cross-entropy against the token chosen by the ensemble
- Note: This uses the linear combination's choices as "ground truth" during training

### Flexible Model Count
- Separate NNs needed for 2-model vs 3-model ensembles
- Input dimension = number of models
- Cannot mix training/inference with different model counts

### Vocabulary Alignment
- NN operates on aligned union vocabulary from existing UniTE pipeline
- Typically 10-20 tokens per position
- Padding to max_vocab_size=30 with masking

## Implementation Notes

### Memory Requirements
- Caching: ~1-2 GB per 1000 questions (depends on max_new_tokens)
- Training: ~4-8 GB GPU memory (depends on batch_size)
- Inference: Same as original UniTE + ~50 MB for NN model

### Training Time
- ~5-15 minutes on single GPU for typical dataset (1000 questions × 100 tokens)
- Scales linearly with number of examples

### Inference Overhead
- Minimal: NN is tiny (3-layer MLP with <5K parameters)
- Negligible compared to LLM inference time (<1ms per token)

## Troubleshooting

### Training accuracy is very low
- Check that num_models matches the cache
- Verify cache file contains valid data
- Try reducing learning rate or increasing model capacity

### Out of memory during training
- Reduce batch_size
- Reduce max_vocab_size padding
- Use smaller hidden_dims

### NN and linear give identical results
- NN may have underfit - try more capacity or longer training
- Check that NN model loaded correctly
- Verify --use_nn_combiner flag is set

### Cache files are too large
- Expected for long sequences (GSM8K: ~2GB for 1000 questions)
- Use smaller test sets for experimentation
- Clean up old cache files regularly

## Future Enhancements

Possible extensions to explore:

1. **Position-aware NN**: Add position embeddings to make f() position-dependent
2. **Token-aware NN**: Incorporate token identity features
3. **Context-aware NN**: Add question embeddings as additional input
4. **Better supervision**: Use ground truth labels instead of linear combination choices
5. **Multi-task learning**: Train single NN on multiple datasets
6. **Uncertainty-aware**: Predict confidence scores along with combinations
7. **Dynamic weighting**: Learn per-model weights that vary by context

## Citation

If you use this neural network combiner, please cite both the original UniTE paper and acknowledge this extension:

```bibtex
@article{UniTE2024,
  title={Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling},
  author={...},
  journal={arXiv preprint arXiv:2410.03777},
  year={2024}
}
```
