"""
Analysis script to compare linear combination vs neural network combination.

Loads cached predictions from both methods and compares:
- Per-position token-level accuracy
- Final answer sequence-level accuracy
- Token selection differences

Usage:
    python analyze_results.py \
        --linear_cache <path_to_linear_cache.pkl> \
        --nn_cache <path_to_nn_cache.pkl> \
        --output_dir <path_to_output_analysis>
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os


def load_cache(cache_file):
    """Load cached predictions."""
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    return cache


def compare_token_selections(linear_cache, nn_cache):
    """
    Compare token selections between linear and NN methods.

    Returns:
        dict with comparison statistics
    """
    linear_data = linear_cache['data']
    nn_data = nn_cache['data']

    assert len(linear_data) == len(nn_data), "Caches have different number of questions"

    stats = {
        'total_tokens': 0,
        'same_selections': 0,
        'different_selections': 0,
        'position_agreements': defaultdict(lambda: {'same': 0, 'different': 0}),
        'differences': []  # List of (question_id, position, linear_token, nn_token)
    }

    for linear_q, nn_q in zip(linear_data, nn_data):
        assert linear_q['question_id'] == nn_q['question_id'], "Question IDs don't match"

        for linear_gen, nn_gen in zip(linear_q['generations'], nn_q['generations']):
            position = linear_gen['position']
            linear_token = linear_gen['selected_token']
            nn_token = nn_gen['selected_token']

            stats['total_tokens'] += 1

            if linear_token == nn_token:
                stats['same_selections'] += 1
                stats['position_agreements'][position]['same'] += 1
            else:
                stats['different_selections'] += 1
                stats['position_agreements'][position]['different'] += 1
                stats['differences'].append({
                    'question_id': linear_q['question_id'],
                    'position': position,
                    'linear_token': linear_token,
                    'nn_token': nn_token,
                    'union_vocab': linear_gen['union_vocab']
                })

    stats['agreement_rate'] = stats['same_selections'] / stats['total_tokens'] if stats['total_tokens'] > 0 else 0

    return stats


def analyze_probability_differences(linear_cache, nn_cache):
    """
    Analyze how much the NN changes probabilities compared to linear combination.

    Returns:
        dict with probability statistics
    """
    linear_data = linear_cache['data']
    nn_data = nn_cache['data']

    prob_diffs = []  # List of absolute probability differences for selected tokens

    for linear_q, nn_q in zip(linear_data, nn_data):
        for linear_gen, nn_gen in zip(linear_q['generations'], nn_q['generations']):
            linear_selected = linear_gen['selected_token']
            nn_selected = nn_gen['selected_token']

            # Get probabilities from linear combination
            if linear_selected in linear_gen['linear_combined']:
                linear_prob = linear_gen['linear_combined'][linear_selected]
            else:
                continue

            # For NN, we don't have direct probabilities in cache, skip for now
            # This would require recomputing from model scores

    stats = {
        'mean_prob_diff': np.mean(prob_diffs) if prob_diffs else 0,
        'std_prob_diff': np.std(prob_diffs) if prob_diffs else 0,
    }

    return stats


def plot_position_agreement(stats, output_dir):
    """Plot agreement rate by position."""
    positions = sorted(stats['position_agreements'].keys())
    agreement_rates = []

    for pos in positions:
        same = stats['position_agreements'][pos]['same']
        different = stats['position_agreements'][pos]['different']
        total = same + different
        agreement_rates.append(same / total if total > 0 else 0)

    plt.figure(figsize=(12, 6))
    plt.plot(positions, agreement_rates, marker='o')
    plt.xlabel('Token Position')
    plt.ylabel('Agreement Rate')
    plt.title('Linear vs NN Agreement Rate by Position')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_agreement.png'), dpi=150)
    plt.close()


def save_differences_report(stats, output_dir):
    """Save detailed report of differences."""
    with open(os.path.join(output_dir, 'differences.jsonl'), 'w') as f:
        for diff in stats['differences']:
            f.write(json.dumps(diff) + '\n')

    # Summary statistics
    summary = {
        'total_tokens': stats['total_tokens'],
        'same_selections': stats['same_selections'],
        'different_selections': stats['different_selections'],
        'agreement_rate': stats['agreement_rate'],
        'num_differences': len(stats['differences'])
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def analyze_per_question_differences(linear_cache, nn_cache):
    """
    Analyze differences on a per-question basis.

    Returns:
        dict mapping question_id to number of different tokens
    """
    linear_data = linear_cache['data']
    nn_data = nn_cache['data']

    question_diffs = {}

    for linear_q, nn_q in zip(linear_data, nn_data):
        qid = linear_q['question_id']
        num_diffs = 0

        for linear_gen, nn_gen in zip(linear_q['generations'], nn_q['generations']):
            if linear_gen['selected_token'] != nn_gen['selected_token']:
                num_diffs += 1

        question_diffs[qid] = num_diffs

    return question_diffs


def plot_question_differences_histogram(question_diffs, output_dir):
    """Plot histogram of number of different tokens per question."""
    diff_counts = list(question_diffs.values())

    plt.figure(figsize=(10, 6))
    plt.hist(diff_counts, bins=range(max(diff_counts) + 2), alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Different Token Selections')
    plt.ylabel('Number of Questions')
    plt.title('Distribution of Differences Between Linear and NN Methods')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_differences_histogram.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze linear vs NN combination results")
    parser.add_argument("--linear_cache", type=str, required=True,
                        help="Path to cache from linear combination")
    parser.add_argument("--nn_cache", type=str, required=True,
                        help="Path to cache from NN combination")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Directory to save analysis results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading caches...")
    linear_cache = load_cache(args.linear_cache)
    nn_cache = load_cache(args.nn_cache)

    print("\nLinear Cache Metadata:")
    print(f"  Dataset: {linear_cache['metadata']['dataset']}")
    print(f"  Models: {linear_cache['metadata']['models']}")
    print(f"  Questions: {linear_cache['metadata']['num_questions']}")

    print("\nNN Cache Metadata:")
    print(f"  Dataset: {nn_cache['metadata']['dataset']}")
    print(f"  Models: {nn_cache['metadata']['models']}")
    print(f"  Questions: {nn_cache['metadata']['num_questions']}")

    print("\n" + "="*60)
    print("COMPARING TOKEN SELECTIONS")
    print("="*60)

    # Compare token selections
    print("\nAnalyzing token selection differences...")
    comparison_stats = compare_token_selections(linear_cache, nn_cache)

    print(f"\nToken Selection Statistics:")
    print(f"  Total tokens: {comparison_stats['total_tokens']}")
    print(f"  Same selections: {comparison_stats['same_selections']}")
    print(f"  Different selections: {comparison_stats['different_selections']}")
    print(f"  Agreement rate: {comparison_stats['agreement_rate']*100:.2f}%")

    # Per-question analysis
    print("\nAnalyzing per-question differences...")
    question_diffs = analyze_per_question_differences(linear_cache, nn_cache)

    questions_with_diffs = sum(1 for count in question_diffs.values() if count > 0)
    print(f"  Questions with at least one difference: {questions_with_diffs}/{len(question_diffs)}")
    print(f"  Average differences per question: {np.mean(list(question_diffs.values())):.2f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_position_agreement(comparison_stats, args.output_dir)
    plot_question_differences_histogram(question_diffs, args.output_dir)

    # Save detailed reports
    print("\nSaving detailed reports...")
    save_differences_report(comparison_stats, args.output_dir)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - summary.json: Overall statistics")
    print(f"  - differences.jsonl: Detailed list of all differences")
    print(f"  - position_agreement.png: Agreement rate by token position")
    print(f"  - question_differences_histogram.png: Distribution of differences per question")


if __name__ == "__main__":
    main()
