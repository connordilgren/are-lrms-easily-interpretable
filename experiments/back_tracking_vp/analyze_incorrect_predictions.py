#!/usr/bin/env python3
"""
Analyze Incorrect Predictions: GT Answer in Top-K Vocabulary Projections

For incorrect model predictions, checks how often the ground truth answer
appears in the top-k vocabulary projections at the answer-predicting position
(### for coconut, : for CODI).

Reads results.json from analyze_gt_representation.py.
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from experiments.back_tracking_vp.backtrace_computation_tree import (
    create_visualization_html,
    create_index_html,
)


def clean_token(token: str) -> str:
    """Strip BPE space prefix (Ġ / \\u0120) and surrounding whitespace."""
    return token.replace('\u0120', ' ').strip()


def is_nonzero_int_token(token_str: str) -> bool:
    """Check if a cleaned token string represents a non-zero integer."""
    cleaned = token_str.strip()
    if not cleaned:
        return False
    try:
        val = int(cleaned)
        return val != 0
    except ValueError:
        return False


def find_answer_position_idx(
    vocab_projection_top_k: List[List[str]],
    num_reasoning_positions: int,
) -> Optional[int]:
    """Find the first position where top-1 is a non-zero integer.

    Starts scanning from num_reasoning_positions forward.
    Returns the index, or None if no such position found.
    """
    for idx in range(num_reasoning_positions, len(vocab_projection_top_k)):
        top_k_tokens = vocab_projection_top_k[idx]
        if top_k_tokens:
            top1 = clean_token(top_k_tokens[0])
            if is_nonzero_int_token(top1):
                return idx
    return None


def get_answer_position_idx(model_type: str, num_reasoning_positions: int) -> int:
    """Get the index into vocab_projection_top_k for the answer-predicting position.

    DEPRECATED: Use find_answer_position_idx() for dynamic detection.

    - Coconut: the ### position, which is right after reasoning positions.
    - CODI: the ':' position, which is the 4th delimiter token after reasoning
      (The=+0, answer=+1, is=+2, :=+3).
    """
    if model_type == "coconut":
        return num_reasoning_positions
    elif model_type == "codi":
        return num_reasoning_positions + 3
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def gt_answer_first_token(gt_answer: str, tokenizer) -> str:
    """Tokenize gt_answer, return the first non-zero integer token text.

    For single-token answers (e.g. "36"), this returns "36".
    For multi-token answers (e.g. "70000" -> ["70","000"]), this returns "70".
    For answers like "0.5" -> ["0", ".", "5"], this returns "5".
    """
    token_ids = tokenizer.encode(gt_answer, add_special_tokens=False)
    for tid in token_ids:
        text = clean_token(tokenizer.decode([tid]))
        if is_nonzero_int_token(text):
            return text
    # Fallback to first token if no non-zero integer found
    return clean_token(tokenizer.decode([token_ids[0]]))


def find_gt_answer_rank(
    gt_first_token_text: str,
    top_k_tokens: List[str],
) -> Optional[int]:
    """Check if the GT answer's first token appears in top-k at the answer position.

    Args:
        gt_first_token_text: Cleaned text of the first BPE token of the GT answer.
        top_k_tokens: Raw top-k token strings from vocab_projection_top_k.

    Returns the 0-indexed rank if found, or None.
    """
    for rank, token in enumerate(top_k_tokens):
        if clean_token(token) == gt_first_token_text:
            return rank
    return None


def analyze_results(
    results_path: Path,
    tokenizer,
    num_visualizations: int = 10,
) -> Dict:
    """Run the incorrect-prediction analysis on a results.json file."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    per_sample = data['per_sample']
    model_type = metadata['model_type']
    top_k = metadata['top_k']
    num_latent = metadata['num_latent']

    logging.info(f"Model: {model_type}, top_k: {top_k}, num_latent: {num_latent}")
    logging.info(f"Total samples: {len(per_sample)}")

    # Collect results
    total_samples = len(per_sample)
    total_correct = 0
    total_incorrect = 0
    gt_in_topk = []          # samples where GT answer IS in top-k
    gt_not_in_topk = []      # samples where GT answer is NOT in top-k
    rank_distribution = defaultdict(int)  # rank -> count

    for sample in per_sample:
        if sample['answer_correct']:
            total_correct += 1
            continue

        total_incorrect += 1

        num_reasoning_positions = sample['num_reasoning_positions']
        vp = sample.get('vocab_projection_top_k', [])
        answer_pos_idx = find_answer_position_idx(vp, num_reasoning_positions)

        if answer_pos_idx is None:
            logging.warning(
                f"Sample {sample['sample_idx']}: no answer position found "
                f"(no non-zero integer in top-1 from position {num_reasoning_positions})"
            )
            gt_not_in_topk.append(sample['sample_idx'])
            continue

        top_k_tokens = vp[answer_pos_idx]
        gt_answer = sample['gt_answer']
        gt_first_tok = gt_answer_first_token(gt_answer, tokenizer)
        rank = find_gt_answer_rank(gt_first_tok, top_k_tokens)

        if rank is not None:
            gt_in_topk.append({
                'sample_idx': sample['sample_idx'],
                'gt_answer': gt_answer,
                'gt_first_token': gt_first_tok,
                'model_answer': sample['model_answer'],
                'gt_rank': rank,
                'top_k_tokens_at_answer_pos': [clean_token(t) for t in top_k_tokens],
            })
            rank_distribution[rank] += 1
        else:
            gt_not_in_topk.append(sample['sample_idx'])

    # Summary
    num_gt_in_topk = len(gt_in_topk)
    pct = (num_gt_in_topk / total_incorrect * 100) if total_incorrect > 0 else 0

    summary = {
        'model_type': model_type,
        'top_k': top_k,
        'num_latent': num_latent,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'total_incorrect': total_incorrect,
        'gt_in_topk_count': num_gt_in_topk,
        'gt_in_topk_pct': round(pct, 2),
        'rank_distribution': {str(k): v for k, v in sorted(rank_distribution.items())},
    }

    # Print summary
    logging.info(f"\n{'=' * 60}")
    logging.info(f"INCORRECT PREDICTION ANALYSIS: {model_type.upper()}")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total samples:     {total_samples}")
    logging.info(f"Correct:           {total_correct}")
    logging.info(f"Incorrect:         {total_incorrect}")
    logging.info(f"GT in top-{top_k}:     {num_gt_in_topk} / {total_incorrect} ({pct:.2f}%)")
    logging.info(f"\nRank distribution (0-indexed):")
    for rank in sorted(rank_distribution.keys()):
        logging.info(f"  Rank {rank}: {rank_distribution[rank]}")
    logging.info(f"{'=' * 60}")

    # Output directory
    original_dir_name = results_path.parent.name
    output_dir_name = original_dir_name + "_incorrect_predictions"
    output_dir = results_path.parent.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSONs
    indices_path = output_dir / 'incorrect_with_gt_in_topk.json'
    with open(indices_path, 'w') as f:
        json.dump(gt_in_topk, f, indent=2)
    logging.info(f"Saved {num_gt_in_topk} entries to {indices_path}")

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary to {summary_path}")

    # Generate visualizations for a subset
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    viz_indices = [entry['sample_idx'] for entry in gt_in_topk[:num_visualizations]]
    success_count = 0

    for sample_idx in viz_indices:
        sample = next((s for s in per_sample if s['sample_idx'] == sample_idx), None)
        if sample is None:
            continue

        output_path = viz_dir / f"sample_{sample_idx:03d}_incorrect_gt_in_topk.html"
        if create_visualization_html(
            sample, output_path, top_k=top_k,
            model_type=model_type, num_latent=num_latent,
            force_answer_rank0=False
        ):
            success_count += 1

    if viz_indices:
        # Create index listing only the visualized samples
        found_samples = viz_indices  # These are "found" in the sense of GT-in-top-k
        create_index_html(per_sample, viz_dir, found_samples, [])
        logging.info(f"Created {success_count} visualizations in {viz_dir}")

    logging.info(f"\nOutput directory: {output_dir.absolute()}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze incorrect predictions: GT answer in top-k vocab projections'
    )
    parser.add_argument(
        '--results_json',
        type=str,
        required=True,
        help='Path to results.json from analyze_gt_representation.py'
    )
    parser.add_argument(
        '--base_llm',
        type=str,
        default='gpt2',
        help='Base LLM tokenizer to use (default: gpt2)'
    )
    parser.add_argument(
        '--num_visualizations',
        type=int,
        default=10,
        help='Number of examples to visualize (default: 10)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results_path = Path(args.results_json)
    if not results_path.exists():
        logging.error(f"Results file not found: {results_path}")
        return

    base_llm_to_model_id = {
        'gpt2': 'openai-community/gpt2',
        'llama32-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    }
    model_id = base_llm_to_model_id.get(args.base_llm, args.base_llm)
    logging.info(f"Loading tokenizer: {args.base_llm} ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    analyze_results(results_path, tokenizer, num_visualizations=args.num_visualizations)


if __name__ == "__main__":
    main()
