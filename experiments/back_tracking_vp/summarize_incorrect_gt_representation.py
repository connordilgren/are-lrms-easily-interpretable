#!/usr/bin/env python3
"""
Summarize GT representation results for incorrect predictions.

Reads results.json from analyze_gt_representation.py (which already contains
tree search results for ALL samples) and produces summary CSVs with stats
for incorrect predictions only.

Because analyze_gt_representation.py gates baseline tree search behind
`if answer_correct`, baseline_1_found/baseline_5_found are always False for
incorrect samples. This script recomputes baselines from scratch using the
stored vocab_projection_top_k token strings (no model inference needed).

Produces two CSVs:
  - summary_all_incorrect.csv: denominator = all incorrect predictions
  - summary_gt_in_topk.csv: denominator = incorrect predictions where the
    GT answer's first BPE token is in top-k at the answer position
"""

import argparse
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from experiments.back_tracking_vp.analyze_incorrect_predictions import (
    clean_token,
    find_answer_position_idx,
    gt_answer_first_token,
    find_gt_answer_rank,
)
from experiments.back_tracking_vp.analyze_gt_representation import (
    build_random_baseline_pool,
    select_random_baselines,
    parse_solution,
    extract_all_result_values,
    solution_is_vp_impossible,
    _first_nonzero_int_token,
    convert_steps_to_solution_str,
)


def _value_to_target_string(value: str, tokenizer) -> str:
    """Convert a numeric value to its target token string for matching.

    Encodes the value (with and without space prefix), picks the representative
    token via _first_nonzero_int_token, decodes it, and cleans it.
    Returns the cleaned string to match against stored top-k tokens.
    """
    tokens_no_space = tokenizer.encode(str(value), add_special_tokens=False)
    tokens_with_space = tokenizer.encode(' ' + str(value), add_special_tokens=False)

    tid_no_space = _first_nonzero_int_token(tokens_no_space, tokenizer)
    tid_with_space = _first_nonzero_int_token(tokens_with_space, tokenizer)

    # Return both cleaned strings; we'll check both when matching
    str_no_space = clean_token(tokenizer.decode([tid_no_space]))
    str_with_space = clean_token(tokenizer.decode([tid_with_space]))

    # They're usually the same after cleaning; return the no-space version
    # but _get_rank_in_topk_tokens will handle both variants
    return str_no_space, str_with_space


def _get_rank_in_topk_tokens(
    target_strs: Tuple[str, str],
    topk_tokens: List[str],
) -> Optional[int]:
    """Find the rank of a target string in stored top-k token list.

    Args:
        target_strs: Tuple of (no_space, with_space) target strings from
            _value_to_target_string.
        topk_tokens: List of raw token strings from vocab_projection_top_k.

    Returns:
        0-indexed rank if found, or None.
    """
    for rank, token in enumerate(topk_tokens):
        cleaned = clean_token(token)
        if cleaned == target_strs[0] or cleaned == target_strs[1]:
            return rank
    return None


def _baseline_tree_found(
    solution_steps: List[Dict],
    vocab_projection_top_k: List[List[str]],
    tokenizer,
    num_reasoning_positions: int,
    question_numbers: Optional[set] = None,
) -> bool:
    """Simplified tree search on stored top-k token strings.

    Mirrors the logic of find_gt_tree_with_ranks but operates on stored token
    strings instead of logits. Returns True if a valid tree is found.

    Uses require_answer_position=False (matching baseline config in the
    original analysis), so the final result can appear at any position after
    all operand positions.
    """
    if not solution_steps:
        return False

    num_positions = len(vocab_projection_top_k)

    # Build intermediate results map: result_value -> step that produced it
    intermediate_results = {}
    for step in solution_steps[:-1]:
        result = step['result']
        if isinstance(result, float) and result == int(result):
            result = int(result)
        intermediate_results[result] = step

    # Cache: (value_str, position) -> rank or None
    rank_cache = {}

    def get_rank(value, position):
        key = (value, position)
        if key not in rank_cache:
            if position < num_positions:
                target_strs = _value_to_target_string(str(value), tokenizer)
                rank_cache[key] = _get_rank_in_topk_tokens(
                    target_strs, vocab_projection_top_k[position]
                )
            else:
                rank_cache[key] = None
        return rank_cache[key]

    # Memoization: (operand, max_position) -> earliest valid position or None
    memo = {}

    def find_earliest_position(operand, max_position):
        """Find the earliest position where operand (and its subtree) is satisfied.

        For leaf operands: earliest position in [0, max_position) where it
        appears in top-k.
        For intermediates: earliest position where operand appears AND all
        child operands have valid subtrees at earlier positions.

        Returns position index or None.
        """
        key = (operand, max_position)
        if key in memo:
            return memo[key]

        if operand not in intermediate_results:
            # Leaf operand - find earliest position in top-k
            for pos in range(max_position):
                if get_rank(operand, pos) is not None:
                    memo[key] = pos
                    return pos
            # Also allow question as source for leaf operands
            if question_numbers and operand in question_numbers:
                memo[key] = -1
                return -1
            memo[key] = None
            return None

        # Intermediate result
        prev_step = intermediate_results[operand]

        # Identity step (e.g., <<40=40>>): treat as leaf
        if len(prev_step['operands']) == 1 and prev_step['operands'][0] == operand:
            for pos in range(max_position):
                if get_rank(operand, pos) is not None:
                    memo[key] = pos
                    return pos
            if question_numbers and operand in question_numbers:
                memo[key] = -1
                return -1
            memo[key] = None
            return None

        # Try positions in ascending order for this intermediate
        for pos in range(max_position):
            if get_rank(operand, pos) is None:
                continue
            # All children must be found at positions strictly before pos
            all_children_ok = True
            for child_operand in prev_step['operands']:
                child_pos = find_earliest_position(child_operand, pos)
                if child_pos is None:
                    all_children_ok = False
                    break
            if all_children_ok:
                memo[key] = pos
                return pos

        memo[key] = None
        return None

    # Get final step
    final_step = solution_steps[-1]
    final_result = final_step['result']
    if isinstance(final_result, float) and final_result == int(final_result):
        final_result = int(final_result)

    # All operands of the final step must be found within reasoning positions
    max_operand_pos = -1
    for operand in final_step['operands']:
        pos = find_earliest_position(operand, num_reasoning_positions)
        if pos is None:
            return False
        if pos > max_operand_pos:
            max_operand_pos = pos

    # Final result can appear at any position after all operand positions
    # (require_answer_position=False)
    for pos in range(max_operand_pos + 1, num_positions):
        if get_rank(final_result, pos) is not None:
            return True

    return False


def _answer_in_topk(
    solution_steps: List[Dict],
    vocab_projection_top_k: List[List[str]],
    tokenizer,
) -> bool:
    """Check if a solution's final answer token appears in top-k at any position."""
    if not solution_steps:
        return False

    final_result = solution_steps[-1]['result']
    if isinstance(final_result, float) and final_result == int(final_result):
        final_result = int(final_result)

    target_strs = _value_to_target_string(str(final_result), tokenizer)
    for topk_tokens in vocab_projection_top_k:
        if _get_rank_in_topk_tokens(target_strs, topk_tokens) is not None:
            return True
    return False


def _compute_baseline_for_sample(
    sample: Dict,
    dataset: List[Dict],
    baseline_pool: Dict[int, List],
    rng: random.Random,
    tokenizer,
) -> Tuple[bool, bool, bool, bool]:
    """Compute baseline metrics for one incorrect sample.

    Args:
        sample: Per-sample result dict from results.json
        dataset: Full dataset list (for GT value sets)
        baseline_pool: Pool from build_random_baseline_pool
        rng: Seeded random number generator
        tokenizer: Tokenizer for value encoding

    Returns:
        (baseline_1_found, baseline_5_found,
         baseline_1_in_topk, baseline_5_in_topk)
    """
    sample_idx = sample['sample_idx']
    step_count = sample['step_count']
    num_rp = sample['num_reasoning_positions']
    vp = sample.get('vocab_projection_top_k', [])
    question_numbers = set(sample['question_numbers']) if sample.get('question_numbers') else None

    if not vp:
        return False, False, False, False

    # Build primary solution string from dataset
    ds_sample = dataset[sample_idx]
    steps = ds_sample.get('steps', [])
    if not steps:
        return False, False, False, False

    primary_solution_str = convert_steps_to_solution_str(steps)
    if solution_is_vp_impossible(primary_solution_str):
        return False, False, False, False

    # Collect GT value sets from all solutions
    gt_value_sets = [extract_all_result_values(primary_solution_str)]
    for gen_sol in (ds_sample.get('gen_solutions') or []):
        gen_solution_str = convert_steps_to_solution_str(gen_sol)
        gt_value_sets.append(extract_all_result_values(gen_solution_str))

    # Select up to 5 random baselines
    baselines = select_random_baselines(
        sample_idx, step_count, gt_value_sets, baseline_pool, rng, n=5
    )
    if not baselines:
        return False, False, False, False

    baseline_1_found = False
    baseline_5_found = False
    baseline_1_in_topk = False
    baseline_5_in_topk = False
    for i, (baseline_sol, baseline_idx) in enumerate(baselines):
        baseline_steps = parse_solution(baseline_sol)
        if not baseline_steps:
            continue

        in_topk = _answer_in_topk(baseline_steps, vp, tokenizer)
        if in_topk:
            if i == 0:
                baseline_1_in_topk = True
            baseline_5_in_topk = True

        found = _baseline_tree_found(
            baseline_steps, vp, tokenizer, num_rp,
            question_numbers=question_numbers,
        )
        if found:
            if i == 0:
                baseline_1_found = True
            baseline_5_found = True

    return baseline_1_found, baseline_5_found, baseline_1_in_topk, baseline_5_in_topk


def summarize(
    results_path: Path,
    tokenizer,
    dataset_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    with open(results_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    per_sample = data['per_sample']
    model_type = metadata['model_type']
    top_k = metadata['top_k']

    logging.info(f"Model: {model_type}, top_k: {top_k}, samples: {len(per_sample)}")

    # Load dataset for baseline computation
    if dataset_path is None:
        dataset_path = metadata.get('dataset_path')
    if dataset_path is None:
        logging.warning("No dataset_path available; baselines will be zeros")
        dataset = None
        baseline_pool = None
        rng = None
    else:
        logging.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        baseline_pool = build_random_baseline_pool(dataset)
        rng = random.Random(seed)
        logging.info(f"Built baseline pool with {sum(len(v) for v in baseline_pool.values())} entries")

    # Two tracking dicts: all-incorrect and gt-in-topk
    by_steps_all = defaultdict(lambda: {
        'total': 0, 'incorrect': 0,
        'primary_found': 0, 'any_gt_found': 0,
        'base_1_found': 0, 'base_5_found': 0,
        'base_1_in_topk': 0, 'base_5_in_topk': 0,
    })
    by_steps_topk = defaultdict(lambda: {
        'total': 0, 'incorrect': 0,
        'primary_found': 0, 'any_gt_found': 0,
        'base_1_found': 0, 'base_5_found': 0,
        'base_1_in_topk': 0, 'base_5_in_topk': 0,
    })

    num_incorrect = 0
    for sample in per_sample:
        if sample['answer_correct']:
            continue

        num_incorrect += 1
        step_count = sample['step_count']
        primary_found = sample.get('primary_found', False)
        any_gt_found = sample.get('any_gt_found', False)

        # Compute baselines from scratch
        if dataset is not None and baseline_pool is not None and rng is not None:
            base_1_found, base_5_found, base_1_in_topk, base_5_in_topk = (
                _compute_baseline_for_sample(
                    sample, dataset, baseline_pool, rng, tokenizer
                )
            )
        else:
            base_1_found = False
            base_5_found = False
            base_1_in_topk = False
            base_5_in_topk = False

        if num_incorrect % 50 == 0:
            logging.info(f"  Processed {num_incorrect} incorrect samples...")

        # All-incorrect stats
        s = by_steps_all[step_count]
        s['total'] += 1
        s['incorrect'] += 1
        if primary_found:
            s['primary_found'] += 1
        if any_gt_found:
            s['any_gt_found'] += 1
        if base_1_found:
            s['base_1_found'] += 1
        if base_5_found:
            s['base_5_found'] += 1
        if base_1_in_topk:
            s['base_1_in_topk'] += 1
        if base_5_in_topk:
            s['base_5_in_topk'] += 1

        # Check if GT answer's first token is in top-k
        num_rp = sample['num_reasoning_positions']
        vp = sample.get('vocab_projection_top_k', [])
        answer_pos_idx = find_answer_position_idx(vp, num_rp)
        if answer_pos_idx is not None:
            gt_first_tok = gt_answer_first_token(sample['gt_answer'], tokenizer)
            rank = find_gt_answer_rank(gt_first_tok, vp[answer_pos_idx])
            if rank is not None:
                t = by_steps_topk[step_count]
                t['total'] += 1
                t['incorrect'] += 1
                if primary_found:
                    t['primary_found'] += 1
                if any_gt_found:
                    t['any_gt_found'] += 1
                if base_1_found:
                    t['base_1_found'] += 1
                if base_5_found:
                    t['base_5_found'] += 1
                if base_1_in_topk:
                    t['base_1_in_topk'] += 1
                if base_5_in_topk:
                    t['base_5_in_topk'] += 1

    logging.info(f"Total incorrect samples processed: {num_incorrect}")

    # Output directory
    original_dir_name = results_path.parent.name
    output_dir = results_path.parent.parent / (original_dir_name + "_incorrect_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, by_steps, csv_name in [
        ("All Incorrect", by_steps_all, "summary_all_incorrect.csv"),
        ("GT in Top-K", by_steps_topk, "summary_gt_in_topk.csv"),
    ]:
        csv_path = output_dir / csv_name
        _write_csv(by_steps, csv_path, label)

    logging.info(f"Output: {output_dir}")


def _write_csv(by_steps, csv_path, label):
    total_incorrect = sum(s['incorrect'] for s in by_steps.values())
    total_primary = sum(s['primary_found'] for s in by_steps.values())
    total_any = sum(s['any_gt_found'] for s in by_steps.values())
    total_base1 = sum(s['base_1_found'] for s in by_steps.values())
    total_base5 = sum(s['base_5_found'] for s in by_steps.values())
    total_base1_topk = sum(s['base_1_in_topk'] for s in by_steps.values())
    total_base5_topk = sum(s['base_5_in_topk'] for s in by_steps.values())

    with open(csv_path, 'w') as f:
        f.write("Steps,Samples,Incorrect,Primary Found %,Any GT Found %,"
                "Base-1 %,Base-5 %,Base-1 In Top-k %,Base-5 In Top-k %\n")
        for step_count in sorted(by_steps.keys()):
            s = by_steps[step_count]
            inc = s['incorrect']
            if inc > 0:
                primary_pct = s['primary_found'] / inc * 100
                any_pct = s['any_gt_found'] / inc * 100
                base1_pct = s['base_1_found'] / inc * 100
                base5_pct = s['base_5_found'] / inc * 100
                base1_topk_pct = s['base_1_in_topk'] / inc * 100
                base5_topk_pct = s['base_5_in_topk'] / inc * 100
            else:
                primary_pct = any_pct = base1_pct = base5_pct = 0
                base1_topk_pct = base5_topk_pct = 0
            f.write(f"{step_count},{s['total']},{inc},{primary_pct:.2f},{any_pct:.2f},"
                    f"{base1_pct:.2f},{base5_pct:.2f},{base1_topk_pct:.2f},{base5_topk_pct:.2f}\n")

        # Overall row
        if total_incorrect > 0:
            overall_primary = total_primary / total_incorrect * 100
            overall_any = total_any / total_incorrect * 100
            overall_base1 = total_base1 / total_incorrect * 100
            overall_base5 = total_base5 / total_incorrect * 100
            overall_base1_topk = total_base1_topk / total_incorrect * 100
            overall_base5_topk = total_base5_topk / total_incorrect * 100
        else:
            overall_primary = overall_any = overall_base1 = overall_base5 = 0
            overall_base1_topk = overall_base5_topk = 0
        f.write(f"All,{total_incorrect},{total_incorrect},{overall_primary:.2f},{overall_any:.2f},"
                f"{overall_base1:.2f},{overall_base5:.2f},{overall_base1_topk:.2f},{overall_base5_topk:.2f}\n")

    logging.info(f"  [{label}] {csv_path.name}: {total_incorrect} samples, "
                 f"Primary={overall_primary:.1f}%, Any={overall_any:.1f}%, "
                 f"Base-1={overall_base1:.1f}%, Base-5={overall_base5:.1f}%, "
                 f"Base-1 TopK={overall_base1_topk:.1f}%, Base-5 TopK={overall_base5_topk:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize GT representation for incorrect predictions'
    )
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--base_llm', type=str, default='gpt2')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset JSON (default: read from results metadata)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for baseline selection (default: 42)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    base_llm_to_model_id = {
        'gpt2': 'openai-community/gpt2',
        'llama32-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    }
    model_id = base_llm_to_model_id.get(args.base_llm, args.base_llm)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    summarize(
        Path(args.results_json),
        tokenizer,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
