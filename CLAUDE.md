# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniTE (UNIon Top-k Ensembling) is a research implementation for ensembling multiple Large Language Models by focusing on the union of top-k tokens from each model. This approach avoids full vocabulary alignment, reducing computational overhead while maintaining effectiveness.

Paper: [Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling](https://arxiv.org/abs/2410.03777)

## Running the Code

### 2-Model Ensembling
```bash
python unite2.py \
  --test_set <path_to_test_data.json> \
  --prompts <path_to_prompts.txt> \
  --model_path1 <path_to_first_model> \
  --model_path2 <path_to_second_model> \
  --output_file <output_path.jsonl> \
  --per_device_batch_size 1 \
  --max_new_tokens <tokens>  # Dataset-specific: ARC/PIQA: 1, NQ/TriviaQA: 10, GSM8K: 512
```

### 3-Model Ensembling
```bash
python unite3.py \
  --test_set <path_to_test_data.json> \
  --prompts <path_to_prompts.txt> \
  --model_path1 <path_to_first_model> \
  --model_path2 <path_to_second_model> \
  --model_path3 <path_to_third_model> \
  --output_file <output_path.jsonl> \
  --per_device_batch_size 1 \
  --max_new_tokens <tokens>
```

### MMLU Evaluation
```bash
python unite_mmlu.py \
  --test_set <mmlu_test_dir> \
  --prompts <mmlu_dev_dir> \
  --model_path1 <path_to_first_model> \
  --model_path2 <path_to_second_model> \
  --output_file <output_path.jsonl> \
  --max_new_tokens 1
```

## Architecture and Key Concepts

### Core Ensemble Algorithm

The ensemble process operates token-by-token through these stages:

1. **Top-k Extraction** (`get_top_k_tokens`): Extract top-10 tokens and their logits from each model
2. **Union Vocabulary** (`get_union_vocab`): Create a unified vocabulary from all models' top-k tokens
3. **Vocabulary Update** (`update_vocab`): Align each model's vocabulary to the union set by:
   - Converting tokens between different tokenizer formats (e.g., '▁' ↔ 'Ġ')
   - Finding logits for missing tokens through tokenizer conversion
   - Handling special tokens per model family
4. **Softmax Normalization** (`vocab_softmax`): Re-normalize probabilities after vocabulary expansion
5. **Weighted Averaging** (`average_and_sample`): Combine probabilities and select next token

### Multi-GPU and Distributed Training

Uses Hugging Face Accelerate for:
- Multi-GPU model placement (device1, device2, device3)
- Distributed data loading with `accelerator.prepare_data_loader`
- Result gathering across processes with `gather_object`
- Flash Attention 2 for memory efficiency

### Model-Specific Token Handling

The codebase supports different LLM families with specific special token IDs for whitespace/padding:

| Model Family | Special Token ID | Token Format |
|--------------|-----------------|--------------|
| Llama 2      | 29871          | '▁' prefix   |
| Llama 3/3.1  | 220            | 'Ġ' prefix   |
| Mistral      | 29473          | '▁' prefix   |
| DeepSeek     | 207            | '▁' prefix   |
| OpenChat     | 28705          | '▁' prefix   |
| Qwen2        | 220            | 'Ġ' prefix   |
| GLM          | 128            | -            |

When updating vocabularies in `update_vocab`, the code handles:
- Token format conversion between models with different tokenizer styles
- OOV (out-of-vocabulary) token ID mapping (0 for Mistral/Llama2)
- Subtoken resolution when direct token conversion fails
- Duplicate token ID prevention within the same position

### Dataset Support

Each dataset has specific collation functions and answer extraction:

**GSM8K** (Math reasoning):
- Collate: `gsm_collate_fn` - Adds "Let's think step by step"
- Extract: `gsm_extract_math_answer` - Parses numeric answers from text
- Parse: `gsm_parse_pred_ans` - Validates exact numeric match

**ARC-Challenge / PIQA** (Multiple choice):
- Collate: `arc_collate_fn` / `piqa_collate_fn` - Formats ABCD/AB choices
- Parse: `arc_parse_pred_ans` - String matching for letter answers

**TriviaQA / NaturalQuestions** (Open QA):
- Collate: `qa_collate_fn` - Simple question format
- Parse: `qa_parse_pred_ans` - Substring matching against multiple acceptable answers

**MMLU** (Multiple choice):
- Collate: `collate_fn` in unite_mmlu.py - Uses few-shot prompts from dev set
- Extract: `extract_math_answer` - Extracts A/B/C/D from generated text
- Parse: `parse_pred_ans` - Computes average accuracy across subjects

### Generation Configuration

Models use these fixed generation settings:
- `num_beams=1`: No beam search (greedy decoding)
- `do_sample=False`: Deterministic generation
- `max_new_tokens=1`: One token per iteration (controlled by outer loop)
- `use_cache=True`: KV-cache enabled with `past_key_values` reuse
- `output_logits=True`: Required for ensemble voting

### Input/Output Format

Input JSON format (test_set):
```json
{"question": "...", "answer": "..."}
```

Output JSONL format:
```json
{"question": "...", "original_sln": "...", "pred_solution": "...", "pred": "...", "label": "..."}
```

## Important Implementation Details

### Token Synchronization Across Models

After selecting the next token from the ensemble:
1. Token is added to model1's input_ids
2. The updated sequence is decoded back to text
3. Text is re-tokenized for model2/model3 to maintain alignment

This ensures different tokenizers stay synchronized despite vocabulary differences.

### Past Key Values Handling

- First iteration (i=0): Generate without past_key_values
- Subsequent iterations: Reuse KV cache from previous step
- Only model1's past_key_values is consistently tracked and updated
- Models 2/3 regenerate from synchronized input_ids each iteration

### Accelerate Integration

The main process handles progress bars with `accelerator.is_main_process`, while all processes participate in:
- Model inference
- Result list building
- `gather_object` synchronization before file writing

## Code Organization

```
unite2.py / unite3.py         # Main ensemble scripts (2 or 3 models)
unite_mmlu.py                 # MMLU-specific evaluation with few-shot prompts
utils/
  ├── collate_fun.py          # Dataset-specific batch collation
  ├── ans_process.py          # Result parsing and accuracy computation
  └── extract_response.py     # Answer extraction from generated text
```

## Hardware Requirements

- Multi-GPU setup required (CUDA devices 0, 1, and optionally 2)
- Flash Attention 2 support
- FP16 precision for memory efficiency
- KV-cache enabled to reduce recomputation
