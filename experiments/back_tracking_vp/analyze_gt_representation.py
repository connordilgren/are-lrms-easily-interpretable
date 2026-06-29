#!/usr/bin/env python3
"""
Analyze Ground Truth Reasoning Trace Representation in Vocabulary Projections

This script analyzes how often ground truth reasoning traces are represented
in the model's vocabulary projections for the GSM-test dataset.

Computes vocab projections on-the-fly using ModelFactory and UnifiedAnalyzer,
supporting both coconut and codi models.
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from models.factory import ModelFactory
from analyzers.base import UnifiedAnalyzer
from experiments.back_tracking_vp.solution_utils import to_number, extract_all_numbers


# Max tree combinations before switching to greedy selection (score is additive,
# so greedy per-operand selection finds the globally optimal tree).
MAX_TREE_COMBOS = 10000

# ============================================================================
# Token Rank Utilities (from collect_inorder_ranks.py pattern)
# ============================================================================

def get_rank_for_token(logits: torch.Tensor, token_id: int) -> int:
    """Get 0-indexed rank for a token in logits (0 = highest probability)."""
    sorted_indices = logits.argsort(descending=True)
    rank_0indexed = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
    return rank_0indexed


def _first_nonzero_int_token(token_ids, tokenizer):
    """Find the token ID of the first token that decodes to a non-zero integer.

    For multi-token numbers, the first non-zero integer token is the most
    informative. E.g. for 0.5 -> ["0", ".", "5"], we use "5".
    For 16.00 -> ["16", ".", "00"], we use "16".

    Falls back to the first token if no non-zero integer token is found.
    """
    for tid in token_ids:
        text = tokenizer.decode([tid]).strip()
        try:
            val = int(text)
            if val != 0:
                return tid
        except ValueError:
            continue
    return token_ids[0]


def _is_space_token(logits: torch.Tensor, tokenizer) -> bool:
    """Check if the top-1 token is a space character."""
    top_token_id = logits.argmax().item()
    top_token_text = tokenizer.decode([top_token_id])
    return top_token_text.strip() == ''


def _is_eos_token(logits: torch.Tensor, tokenizer) -> bool:
    """Check if the top-1 token is an end-of-sequence token."""
    top_token_id = logits.argmax().item()
    return top_token_id == tokenizer.eos_token_id


def get_rank_for_value(logits: torch.Tensor, value: str, tokenizer, top_k: int) -> Optional[int]:
    """Get minimum rank between ' VALUE' and 'VALUE' tokenizations.

    Returns 0-indexed rank if within top_k, else None.
    Uses the first non-zero integer token for multi-token values.
    """
    tokens_no_space = tokenizer.encode(str(value), add_special_tokens=False)
    tokens_with_space = tokenizer.encode(' ' + str(value), add_special_tokens=False)

    target_no_space = _first_nonzero_int_token(tokens_no_space, tokenizer)
    target_with_space = _first_nonzero_int_token(tokens_with_space, tokenizer)

    rank_no_space = get_rank_for_token(logits, target_no_space)
    rank_with_space = get_rank_for_token(logits, target_with_space)

    min_rank = min(rank_no_space, rank_with_space)
    return min_rank if min_rank < top_k else None


def find_key_positions(
    output_ids: torch.Tensor,
    model,
    num_latent: int
) -> Optional[Dict[str, int]]:
    """
    Find key positions in the output sequence.

    Handles different model types:
    - Coconut: <|start-latent|>, <|latent|>, <|end-latent|>, ###, answer
    - CODI: <|bot|>, <|bot|>..., <|eot|>, "The answer is:", answer

    Returns dict with positions for:
    - start_latent: Position of start marker
    - latent_0 to latent_{n-1}: Positions of latent tokens
    - end_latent: Position of end marker
    - For Coconut: hash (position of ###)
    - For CODI: delimiter_0 to delimiter_3 (positions of "The", "answer", "is", ":")
    - first_answer: Position of first answer token
    """
    output_flat = output_ids[0] if output_ids.dim() > 1 else output_ids
    special_tokens = model.special_tokens
    model_type = model.model_type

    # Get start/end token IDs based on model type
    if model_type == "coconut":
        start_id = special_tokens.get('start_latent')
        end_id = special_tokens.get('end_latent')
    elif model_type == "codi":
        # CODI uses bot_id for both start marker and latent placeholder tokens
        start_id = special_tokens.get('bot')
        end_id = special_tokens.get('eot')
    else:
        # Try coconut-style first, then codi-style
        start_id = special_tokens.get('start_latent') or special_tokens.get('bot')
        end_id = special_tokens.get('end_latent') or special_tokens.get('eot')

    # Check if we have valid token IDs
    if start_id is None or end_id is None:
        logging.warning(f"Could not find start/end token IDs. Special tokens: {special_tokens}")
        return None

    # Find start position (first occurrence of start_id)
    start_matches = (output_flat == start_id).nonzero(as_tuple=True)[0]
    if len(start_matches) == 0:
        return None
    start_idx = start_matches[0].item()

    # Find end position
    end_matches = (output_flat == end_id).nonzero(as_tuple=True)[0]
    if len(end_matches) == 0:
        return None
    end_idx = end_matches[0].item()

    # Build positions dict
    positions = {
        'start_latent': start_idx,
        'end_latent': end_idx,
    }

    # Add latent token positions
    for i in range(num_latent):
        positions[f'latent_{i}'] = start_idx + 1 + i

    if model_type == "codi":
        # CODI uses "The answer is:" as delimiter (4 tokens typically)
        positions['delimiter_0'] = end_idx + 1  # "The"
        positions['delimiter_1'] = end_idx + 2  # "answer"
        positions['delimiter_2'] = end_idx + 3  # "is"
        positions['delimiter_3'] = end_idx + 4  # ":"
        positions['first_answer'] = end_idx + 5
    else:
        # Coconut: look for ### token
        hash_idx = None
        for pos in range(end_idx + 1, len(output_flat)):
            try:
                token_text = model.tokenizer.decode([output_flat[pos].item()])
                if '###' in token_text or token_text.strip() == '#':
                    hash_idx = pos
                    break
            except:
                continue

        if hash_idx is None:
            return None

        positions['hash'] = hash_idx
        positions['first_answer'] = hash_idx + 1

    return positions


# ============================================================================
# Solution Parsing (KEPT from original)
# ============================================================================

def solution_is_vp_impossible(solution_str):
    """
    Check if a solution contains values that can't be found in vocab projections.

    VP impossible if:
    - Fractions (e.g., 3/2) appear in the result (fractions in expression are OK - they represent division)
    - Non-numeric results (e.g., "125%")

    Decimals are NOT VP impossible - get_rank_for_value uses the first non-zero
    integer BPE token for multi-token numbers.

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        bool: True if solution is VP impossible
    """
    # Find all <<...=...>> patterns - capture result as any non->> sequence
    pattern = r'<<(.+?)=([^>]+)>>'
    matches = re.findall(pattern, solution_str)

    for expr, result_str in matches:
        result_str = result_str.strip()

        # Check if result is a fraction (e.g., "3/2") - fractions in result are VP impossible
        if '/' in result_str:
            return True

        # Check if result is a valid number (not e.g., "125%")
        try:
            float(result_str)
        except ValueError:
            return True

    return False


def count_solution_steps(solution_str):
    """
    Count number of computation steps in a solution.

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        int: Number of steps
    """
    if not solution_str:
        return 0
    return len(re.findall(r'<<.+?=.+?>>', solution_str))


def extract_all_result_values(solution_str):
    """Extract all step result values (intermediates + final answer)."""
    pattern = r'<<.+?=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)
    return set(to_number(m) for m in matches)


def extract_intermediate_values(solution_str):
    """
    Extract all intermediate result values from a solution.
    Intermediates are all step results except the final one.

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        set: Set of intermediate values (int or float)
    """
    pattern = r'<<.+?=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)
    if len(matches) <= 1:
        return set()
    # All results except the last one are intermediates
    return set(to_number(m) for m in matches[:-1])


def extract_all_values(solution_str):
    """
    Extract all operands and results from a solution.

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        set: Set of all values (int or float) - operands and results
    """
    # Find all <<expr=result>> patterns
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)

    all_values = set()
    for expr, result_str in matches:
        # Add the result
        all_values.add(to_number(result_str))
        # Add all operands from the expression
        all_values.update(extract_all_numbers(expr))

    return all_values


WORD_TO_NUMBER = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000,
    'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3,
}


def extract_question_numbers(question: str) -> set:
    """
    Extract all numbers from a question string.

    Handles:
    - Plain integers and decimals: 16, 5.75
    - Dollar amounts: $2, $80,000
    - Numbers with commas: 80,000
    - Percentages: 150% (extracts 150)
    - Spelled-out numbers: one, two, ..., twenty, thirty, ..., ninety,
      hundred, thousand, half, twice, double, triple

    Args:
        question: The question text

    Returns:
        Set of numeric values (int or float)
    """
    numbers = set()

    # Strip $ signs and collapse commas within digit groups
    cleaned = re.sub(r'\$', '', question)
    while re.search(r'(\d),(\d)', cleaned):
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', cleaned)

    # Extract digit-based numbers (integers and decimals)
    for match in re.finditer(r'\d+(?:\.\d+)?', cleaned):
        val = to_number(match.group())
        if val is not None and val != 0:
            numbers.add(val)

    # Extract spelled-out numbers
    q_lower = question.lower()
    for word, value in WORD_TO_NUMBER.items():
        if value and re.search(r'\b' + word + r'\b', q_lower):
            numbers.add(value)

    return numbers


def parse_solution(solution_str):
    """
    Parse a solution string into list of steps with operands and results.

    Example: "<<3+4=7>> <<(16-7)*2=18>>"
    Returns: [
        {'operands': [3, 4], 'result': 7, 'expression': '<<3+4=7>>'},
        {'operands': [16, 7, 2], 'result': 18, 'expression': '<<(16-7)*2=18>>'}
    ]

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        list: List of step dictionaries with 'operands', 'result', 'expression'
    """
    steps = []

    # Find all <<...=...>> patterns
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)

    for expr, result_str in matches:
        # Handle float results (convert to int if whole number)
        result_float = float(result_str)
        if result_float == int(result_float):
            result = int(result_float)
        else:
            result = result_float

        # Extract all integers from the expression (operands)
        operands = extract_all_numbers(expr)

        steps.append({
            'operands': operands,
            'result': result,
            'expression': f'<<{expr}={result_str}>>'
        })

    return steps


def convert_steps_to_solution_str(steps: List[str]) -> str:
    """
    Convert a list of step strings to a single solution string.

    Args:
        steps: List like ["<<a=b>>", "<<c=d>>"]

    Returns:
        Single string like "<<a=b>> <<c=d>>"
    """
    return ' '.join(steps)


# ============================================================================
# Random Baseline Functions
# ============================================================================

def build_random_baseline_pool(dataset: List[Dict]) -> Dict[int, List]:
    """
    Build pool of solutions grouped by step count.

    Args:
        dataset: List of dataset samples with 'steps' and 'answer' fields

    Returns:
        dict: {step_count: [(sample_idx, solution_str, all_result_values), ...]}
    """
    pool = defaultdict(list)

    for i, sample in enumerate(dataset):
        steps = sample.get('steps', [])
        if not steps:
            continue

        solution = convert_steps_to_solution_str(steps)
        step_count = len(steps)
        all_values = extract_all_result_values(solution)

        pool[step_count].append((i, solution, all_values))

    return dict(pool)


def select_random_baselines(
    sample_idx: int,
    step_count: int,
    gt_value_sets: List[set],
    baseline_pool: Dict[int, List],
    rng: random.Random,
    n: int = 5
) -> List[Tuple[str, int]]:
    """
    Select n random solutions from different samples with:
    - Same step count
    - Different sample
    - At least one result value differs from every GT solution

    Args:
        sample_idx: Current sample's index
        step_count: Number of steps to match
        gt_value_sets: List of result value sets from all GT solutions
        baseline_pool: Pool from build_random_baseline_pool
        rng: Random number generator
        n: Number of baselines to select

    Returns:
        list: List of (baseline_solution_str, baseline_idx) tuples, up to n items
    """
    if step_count not in baseline_pool:
        return []

    candidates = []
    for idx, sol, bl_values in baseline_pool[step_count]:
        if idx == sample_idx:
            continue
        # Baseline must have at least one value not in each GT solution
        if any(bl_values.issubset(gt_vals) for gt_vals in gt_value_sets):
            continue
        candidates.append((idx, sol))

    if not candidates:
        return []

    k = min(n, len(candidates))
    selected = rng.sample(candidates, k)
    return [(sol, idx) for idx, sol in selected]


# ============================================================================
# GT Tree Search with Rank Tracking (MODIFIED)
# ============================================================================

def find_gt_tree_with_ranks(
    solution_steps: List[Dict],
    logits: torch.Tensor,
    tokenizer,
    top_k: int,
    num_reasoning_positions: int,
    answer_position: Optional[int] = None,
    require_answer_position: bool = True,
    question_numbers: Optional[set] = None
) -> Tuple[int, Optional[float], Optional[Dict]]:
    """
    Count how many ways the GT solution tree can be found in vocab projections,
    and compute representation_score as the average rank of all found operands/results.

    Key constraint: When searching for an operand that was an intermediate result,
    search only in positions BEFORE where that intermediate was found.

    Args:
        solution_steps: List of parsed steps with operands and results
        logits: Logits tensor [num_positions, vocab_size] - includes all positions
        tokenizer: Tokenizer for encoding values
        top_k: Number of top tokens to consider
        num_reasoning_positions: Number of reasoning positions (0 to num_latent+1)
        answer_position: Position index where answer appears (for final result edges)
        require_answer_position: If True, final answer must appear at answer_position.
            If False, the tree is valid as long as all operand subtrees are found.
        question_numbers: Optional set of numbers extracted from the question text.
            When provided, leaf operands found in this set can use position -1
            (representing the question) as an additional source. Trees using VP
            table operands are preferred over question-sourced ones.

    Returns:
        Tuple of (tree_count, representation_score, best_tree)
        representation_score = average rank (0-indexed) of all found operands/results.
        Lower is better. None if not found.
        best_tree = dict with 'nodes' and 'edges' for the best tree, or None if not found.
    """
    if not solution_steps:
        return 0, None, None

    # Build intermediate results map: result_value -> (step_idx, step)
    intermediate_results = {}
    for step_idx, step in enumerate(solution_steps[:-1]):  # All steps except the last one
        result = step['result']
        if isinstance(result, float) and result == int(result):
            result = int(result)
        intermediate_results[result] = (step_idx, step)

    # Cache for operand ranks at each position
    rank_cache = {}  # (operand, position) -> rank or None

    def get_operand_rank(operand, position):
        """Get rank of operand at position, or None if not in top_k."""
        key = (operand, position)
        if key not in rank_cache:
            if position < logits.shape[0]:
                rank_cache[key] = get_rank_for_value(logits[position], str(operand), tokenizer, top_k)
            else:
                rank_cache[key] = None
        return rank_cache[key]

    def find_operand_positions_with_ranks(operand, max_position):
        """Find all (position, rank) pairs where operand appears in top_k."""
        positions = []
        for pos in range(max_position):
            rank = get_operand_rank(operand, pos)
            if rank is not None:
                positions.append((pos, rank))
        return positions

    def subtree_score(tree):
        """Score a subtree: (num_question_nodes, vp_position_rank_sum). Lower is better."""
        num_question = sum(1 for n in tree['nodes'] if n.get('type') == 'question')
        vp_score = sum(n['position'] * 10 + n['rank']
                       for n in tree['nodes'] if n.get('type') != 'question')
        return (num_question, vp_score)

    # Memoization for tree counting - now stores full tree info
    memo = {}

    def count_trees_for_operand(operand, max_position, step_context):
        """
        Count ways to find this operand (and its subtree if intermediate).
        max_position: Only search positions 0 to max_position-1
        step_context: Index of the step that uses this operand. Only treat a value
            as an intermediate result if its producing step comes before step_context.
        Returns: (count, list of tree_info dicts)
        Each tree_info = {'positions': [...], 'ranks': [...], 'nodes': [...], 'edges': [...]}
        """
        key = (operand, max_position, step_context)
        if key in memo:
            return memo[key]

        found_positions = find_operand_positions_with_ranks(operand, max_position)

        # For leaf operands, also allow the question as a source
        is_prior_intermediate = (operand in intermediate_results and intermediate_results[operand][0] < step_context)
        if not is_prior_intermediate and question_numbers and operand in question_numbers:
            found_positions.append((-1, 0))

        if not found_positions:
            memo[key] = (0, [])
            return 0, []

        if not is_prior_intermediate:
            # Leaf operand - each position is a valid tree
            trees = []
            for pos, rank in found_positions:
                node_type = 'question' if pos == -1 else 'leaf'
                trees.append({
                    'positions': [pos],
                    'ranks': [rank],
                    'nodes': [{'value': operand, 'position': pos, 'rank': rank, 'type': node_type}],
                    'edges': []
                })
            memo[key] = (len(trees), trees)
            return len(trees), trees

        # Intermediate result - need to find its operands in earlier positions
        prev_step_idx, prev_step = intermediate_results[operand]

        # Identity step (e.g., <<40=40>>): operand equals result, treat as leaf
        if len(prev_step['operands']) == 1 and prev_step['operands'][0] == operand:
            trees = []
            for pos, rank in found_positions:
                trees.append({
                    'positions': [pos],
                    'ranks': [rank],
                    'nodes': [{'value': operand, 'position': pos, 'rank': rank, 'type': 'leaf'}],
                    'edges': []
                })
            memo[key] = (len(trees), trees)
            return len(trees), trees

        all_trees = []

        for pos, rank in found_positions:
            # Get all child subtrees for each operand of prev_step
            child_subtrees_list = []
            valid = True

            for child_operand in prev_step['operands']:
                child_count, child_trees = count_trees_for_operand(child_operand, pos, prev_step_idx)
                if child_count == 0:
                    valid = False
                    break
                child_subtrees_list.append((child_operand, child_trees))

            if not valid:
                continue

            # Enumerate combinations of child subtrees
            from itertools import product
            child_trees_only = [ct for _, ct in child_subtrees_list]

            # Check if full enumeration is feasible
            total_combos = 1
            for ct in child_trees_only:
                total_combos *= len(ct)
                if total_combos > MAX_TREE_COMBOS:
                    break

            if total_combos <= MAX_TREE_COMBOS:
                combos = product(*child_trees_only)
            else:
                # Greedy: pick best subtree per operand (optimal since score is additive)
                combos = [tuple(min(ct, key=subtree_score) for ct in child_trees_only)]

            for combo in combos:
                # Merge all info
                combined_positions = [pos]
                combined_ranks = [rank]
                combined_nodes = [{'value': operand, 'position': pos, 'rank': rank, 'type': 'intermediate'}]
                combined_edges = []

                for i, child_tree in enumerate(combo):
                    combined_positions.extend(child_tree['positions'])
                    combined_ranks.extend(child_tree['ranks'])
                    combined_nodes.extend(child_tree['nodes'])
                    combined_edges.extend(child_tree['edges'])

                    # Add edge from each direct child to this intermediate
                    child_operand = child_subtrees_list[i][0]
                    # Find the root node of the child subtree (first node with this operand value)
                    child_root_pos = child_tree['positions'][0]
                    child_root_rank = child_tree['ranks'][0]
                    combined_edges.append({
                        'from_value': child_operand,
                        'from_position': child_root_pos,
                        'from_rank': child_root_rank,
                        'to_value': operand,
                        'to_position': pos,
                        'to_rank': rank
                    })

                all_trees.append({
                    'positions': combined_positions,
                    'ranks': combined_ranks,
                    'nodes': combined_nodes,
                    'edges': combined_edges
                })

        memo[key] = (len(all_trees), all_trees)
        return len(all_trees), all_trees

    # Get final step
    final_step = solution_steps[-1]
    final_result = final_step['result']
    if isinstance(final_result, float) and final_result == int(final_result):
        final_result = int(final_result)

    # Enumerate all trees by combining operand subtrees for final step
    final_step_idx = len(solution_steps) - 1
    operand_subtrees_list = []
    for operand in final_step['operands']:
        count, trees = count_trees_for_operand(operand, num_reasoning_positions, final_step_idx)
        if count == 0:
            return 0, None, None
        operand_subtrees_list.append((operand, trees))

    # Determine where the final result can appear
    if require_answer_position:
        # Final answer MUST be in the answer column
        final_result_rank = None
        effective_answer_position = answer_position

        if answer_position is not None:
            final_result_rank = get_operand_rank(final_result, answer_position)

            # If not found, skip past leading space tokens to find the actual answer
            # This handles cases like LLaMA outputting " 3" instead of "3"
            if final_result_rank is None:
                candidate_pos = answer_position
                # Loop while: current position has space as top-1, and next position exists and isn't EOS
                while (candidate_pos + 1 < logits.shape[0] and
                       _is_space_token(logits[candidate_pos], tokenizer) and
                       not _is_eos_token(logits[candidate_pos + 1], tokenizer)):
                    candidate_pos += 1

                # If we moved past spaces, try the new position
                if candidate_pos != answer_position:
                    final_result_rank = get_operand_rank(final_result, candidate_pos)
                    effective_answer_position = candidate_pos

        if final_result_rank is None:
            return 0, None, None
        final_result_placements = [(effective_answer_position, final_result_rank)]
    else:
        # Final answer can be at any position (must be after operands, checked per-combo)
        final_result_placements = []
        for pos in range(logits.shape[0]):
            rank = get_operand_rank(final_result, pos)
            if rank is not None:
                final_result_placements.append((pos, rank))
        if not final_result_placements:
            return 0, None, None

    # Combine all operand subtrees
    from itertools import product
    all_complete_trees = []
    operand_trees_only = [ot for _, ot in operand_subtrees_list]

    # Check if full enumeration is feasible
    total_combos = 1
    for ot in operand_trees_only:
        total_combos *= len(ot)
        if total_combos > MAX_TREE_COMBOS:
            break

    if total_combos <= MAX_TREE_COMBOS:
        combos = product(*operand_trees_only)
    else:
        # Greedy: pick best subtree per operand (optimal since score is additive)
        combos = [tuple(min(ot, key=subtree_score) for ot in operand_trees_only)]

    for combo in combos:
        # Collect operand subtree info
        base_positions = []
        base_ranks = []
        base_nodes = []
        base_edges = []

        for i, child_tree in enumerate(combo):
            base_positions.extend(child_tree['positions'])
            base_ranks.extend(child_tree['ranks'])
            base_nodes.extend(child_tree['nodes'])
            base_edges.extend(child_tree['edges'])

        # Filter final result placements: must be strictly after all operand positions
        max_operand_pos = max(base_positions) if base_positions else -1
        valid_placements = [(pos, rank) for pos, rank in final_result_placements
                           if pos > max_operand_pos]

        for result_pos, result_rank in valid_placements:
            tree_positions = list(base_positions)
            tree_ranks = list(base_ranks)
            tree_nodes = list(base_nodes)
            tree_edges = list(base_edges)

            # Add edge from each final step operand to the final result
            for i, child_tree in enumerate(combo):
                child_operand = operand_subtrees_list[i][0]
                child_root_pos = child_tree['positions'][0]
                child_root_rank = child_tree['ranks'][0]
                tree_edges.append({
                    'from_value': child_operand,
                    'from_position': child_root_pos,
                    'from_rank': child_root_rank,
                    'to_value': final_result,
                    'to_position': result_pos,
                    'to_rank': result_rank
                })

            # Add final result node
            tree_positions.append(result_pos)
            tree_ranks.append(result_rank)
            tree_nodes.append({
                'value': final_result,
                'position': result_pos,
                'rank': result_rank,
                'type': 'final'
            })

            all_complete_trees.append({
                'positions': tree_positions,
                'ranks': tree_ranks,
                'nodes': tree_nodes,
                'edges': tree_edges,
                'final_result': final_result
            })

    tree_count = len(all_complete_trees)

    if tree_count == 0:
        return 0, None, None

    # Compute representation_score for each tree:
    # Primary: fewer question-sourced nodes is always better
    # Secondary: sum of (position * 10 + rank) for VP nodes (lower is better)
    def tree_score(tree):
        """Compute score: (num_question_nodes, vp_position_rank_sum)."""
        num_question = sum(1 for n in tree['nodes'] if n.get('type') == 'question')
        vp_score = sum(n['position'] * 10 + n['rank']
                       for n in tree['nodes'] if n.get('type') != 'question')
        return (num_question, vp_score)

    # Select tree with minimum score
    best_tree = min(all_complete_trees, key=tree_score)
    best_score = tree_score(best_tree)

    return tree_count, best_score, best_tree


# ============================================================================
# Analysis Functions (NEW)
# ============================================================================

def analyze_solution(
    solution_str: str,
    is_primary: bool,
    logits: torch.Tensor,
    tokenizer,
    top_k: int,
    num_reasoning_positions: int,
    answer_position: Optional[int] = None,
    question_numbers: Optional[set] = None
) -> Dict:
    """
    Analyze a single solution against the vocab projections.

    Args:
        solution_str: Solution string with <<expr=result>> format
        is_primary: Whether this is the primary solution
        logits: Logits tensor [num_positions, vocab_size]
        tokenizer: Tokenizer for encoding values
        top_k: Number of top tokens to consider
        num_reasoning_positions: Number of reasoning positions
        answer_position: Position index where answer appears (for final result edges)
        question_numbers: Optional set of numbers from the question text

    Returns:
        Dict with solution analysis results
    """
    is_vp_impossible = solution_is_vp_impossible(solution_str)

    if is_vp_impossible:
        return {
            'solution': solution_str,
            'is_primary': is_primary,
            'is_vp_impossible': True,
            'times_found': 0,
            'representation_score': None,
            'is_best_represented': False,
            'best_tree': None
        }

    steps = parse_solution(solution_str)
    if not steps:
        return {
            'solution': solution_str,
            'is_primary': is_primary,
            'is_vp_impossible': False,
            'times_found': 0,
            'representation_score': None,
            'is_best_represented': False,
            'best_tree': None
        }

    tree_count, representation_score, best_tree = find_gt_tree_with_ranks(
        steps, logits, tokenizer, top_k, num_reasoning_positions, answer_position,
        question_numbers=question_numbers
    )

    return {
        'solution': solution_str,
        'is_primary': is_primary,
        'is_vp_impossible': False,
        'times_found': tree_count,
        'representation_score': representation_score,
        'is_best_represented': False,  # Will be set later
        'best_tree': best_tree
    }


def select_best_represented(solutions: List[Dict]) -> Optional[int]:
    """
    Select the best represented solution by representation_score.

    Ties: prefer primary, then first in gen_solutions order.

    Args:
        solutions: List of solution result dicts

    Returns:
        Index of best represented solution, or None if none found
    """
    found_solutions = [(i, s) for i, s in enumerate(solutions) if s['times_found'] > 0]

    if not found_solutions:
        return None

    # Sort by: representation_score (ascending), is_primary (descending), index (ascending)
    def sort_key(item):
        idx, sol = item
        return (sol['representation_score'], not sol['is_primary'], idx)

    found_solutions.sort(key=sort_key)
    return found_solutions[0][0]


def analyze_sample_on_the_fly(
    sample: Dict,
    sample_idx: int,
    model,
    analyzer: UnifiedAnalyzer,
    num_latent: int,
    top_k: int,
    device: str,
    baseline_pool: Optional[Dict[int, List]] = None,
    rng: Optional[random.Random] = None,
    baseline_require_answer: bool = True,
    include_question_tokens: bool = False,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Analyze a single sample by running the model and computing vocab projections.

    Args:
        sample: Dataset sample with question, steps, answer, gen_solutions
        sample_idx: Index of the sample
        model: Loaded model
        analyzer: UnifiedAnalyzer instance
        num_latent: Number of latent tokens
        top_k: Top-k tokens for search
        device: Device string
        baseline_pool: Optional pool for baseline testing
        rng: Optional random number generator for baseline selection
        baseline_require_answer: If True, baseline final answer must appear in
            the answer column. If False, only operand subtrees are required.
        include_question_tokens: If True, numbers from the question text can
            serve as leaf operands in the tree search.
        verbose: Enable debug logging

    Returns:
        Dict with analysis results, or None if analysis failed
    """
    question = sample['question']
    gt_answer = str(sample['answer']).replace(',', '').strip()

    # Extract question numbers if enabled
    question_nums = extract_question_numbers(question) if include_question_tokens else None

    # Prepare input
    inputs = model.prepare_input(question, num_latents=num_latent)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run inference with activation capture
    max_new_tokens = 64
    result = analyzer.analyze_with_capture(
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        layer_indices=None  # Capture all layers
    )

    output = result["output"]
    output_ids = output.output_ids

    # Find key positions
    positions = find_key_positions(output_ids, model, num_latent)
    if positions is None:
        if verbose:
            logging.warning(f"Sample {sample_idx}: Could not find key positions, skipping")
        return None

    # Get position indices for reasoning (positions 0 to num_latent+1)
    # These are: start_latent, latent_0, ..., latent_{n-1}, end_latent
    # Then add delimiter/hash and answer positions
    if model.model_type == "codi":
        # CODI: <reasoning> <eot> "The answer is:" <answer> <eos>
        reasoning_positions = [
            positions['start_latent'],
            *[positions[f'latent_{i}'] for i in range(num_latent)],
            positions['end_latent'],
        ]
        # Add delimiter positions (The answer is:) and first answer
        delimiter_positions = [
            positions.get('delimiter_0'),
            positions.get('delimiter_1'),
            positions.get('delimiter_2'),
            positions.get('delimiter_3'),
        ]
        delimiter_positions = [p for p in delimiter_positions if p is not None]
        answer_position = positions.get('first_answer')

        # Use list() to create a copy, not an alias
        position_indices = list(reasoning_positions) + delimiter_positions
        answer_position_idx = None  # Index in position_indices where answer is PREDICTED
        if answer_position is not None and delimiter_positions:
            # The last delimiter (':') predicts the answer token via next-token prediction.
            # (Unlike Coconut where ### explicitly predicts the answer, CODI's answer
            # is predicted by the autoregressive logits at the ':' position.)
            answer_position_idx = len(reasoning_positions) + len(delimiter_positions) - 1
            position_indices.append(answer_position)
            # Try to get second answer token (only if in bounds)
            if answer_position + 1 < output_ids.shape[1]:
                position_indices.append(answer_position + 1)
    else:
        # Coconut: <reasoning> ### <answer>
        reasoning_positions = [
            positions['start_latent'],
            *[positions[f'latent_{i}'] for i in range(num_latent)],
            positions['end_latent'],
        ]
        hash_position = positions.get('hash')
        answer_position = positions.get('first_answer')

        # Use list() to create a copy, not an alias
        position_indices = list(reasoning_positions)
        answer_position_idx = None  # Index in position_indices where answer is PREDICTED
        if hash_position is not None:
            # The hash position (###) predicts the answer token
            answer_position_idx = len(position_indices)
            position_indices.append(hash_position)
        if answer_position is not None:
            position_indices.append(answer_position)
            # Try to get eos token (only if in bounds)
            if answer_position + 1 < output_ids.shape[1]:
                position_indices.append(answer_position + 1)

    num_reasoning_positions = len(reasoning_positions)

    # Get hidden states from final layer
    activations = result["activations"]
    if not activations:
        if verbose:
            logging.warning(f"Sample {sample_idx}: No activations captured, skipping")
        return None

    final_layer_idx = max(activations.keys())
    hidden_states = activations[final_layer_idx]

    # Check bounds and filter out-of-bounds positions
    seq_len = hidden_states.shape[1]
    valid_position_indices = [p for p in position_indices if p < seq_len]
    if len(valid_position_indices) < num_reasoning_positions:
        if verbose:
            logging.warning(f"Sample {sample_idx}: Not enough valid positions, skipping")
        return None

    # Update answer_position_idx if some positions were filtered out
    if answer_position_idx is not None and answer_position_idx >= len(valid_position_indices):
        answer_position_idx = None

    # Extract hidden states at all positions (reasoning + answer) and project to vocab
    hidden_at_positions = hidden_states[0, valid_position_indices, :]
    proj_result = analyzer.project_activations_to_vocab(
        hidden_at_positions.unsqueeze(0),
        top_k=top_k,
        return_probs=False
    )
    all_logits = proj_result["logits"][0]  # [num_all_positions, vocab_size]

    # Extract top-k tokens at each position for visualization (all positions)
    all_top_k_indices = all_logits.argsort(dim=-1, descending=True)[:, :top_k]
    vocab_projection_top_k = []
    column_labels = []  # Build from actual top tokens
    for pos_idx in range(len(valid_position_indices)):
        position_tokens = []
        for rank in range(top_k):
            token_id = all_top_k_indices[pos_idx, rank].item()
            token_str = model.tokenizer.decode([token_id])
            position_tokens.append(token_str)
        vocab_projection_top_k.append(position_tokens)
        # Use first token (rank 0) as column label, cleaned up
        top_token = position_tokens[0].replace('Ġ', ' ').strip()
        if not top_token:  # Handle empty or whitespace-only tokens
            top_token = repr(position_tokens[0])
        column_labels.append(top_token)

    # Extract model answer
    decoded_output = result["decoded"]
    output_metadata = result["output"].metadata or {}
    delimiter = output_metadata.get("delimiter", "###")

    if delimiter in decoded_output:
        model_answer = decoded_output.split(delimiter)[-1].replace(',', '').strip()
    else:
        model_answer = ""

    # Check correctness
    try:
        answer_correct = float(model_answer.replace(',', '')) == float(gt_answer.replace(',', ''))
    except (ValueError, TypeError):
        answer_correct = model_answer == gt_answer

    # Collect all solutions
    all_solutions = []

    # Primary solution
    primary_solution_str = convert_steps_to_solution_str(sample['steps'])
    step_count = len(sample['steps'])

    # Analyze primary solution
    primary_result = analyze_solution(
        primary_solution_str,
        is_primary=True,
        logits=all_logits,
        tokenizer=model.tokenizer,
        top_k=top_k,
        num_reasoning_positions=num_reasoning_positions,
        answer_position=answer_position_idx,
        question_numbers=question_nums
    )
    all_solutions.append(primary_result)

    # Analyze gen_solutions
    gen_solutions = sample.get('gen_solutions') or []
    if gen_solutions:
        for gen_sol in gen_solutions:
            gen_solution_str = convert_steps_to_solution_str(gen_sol)
            gen_result = analyze_solution(
                gen_solution_str,
                is_primary=False,
                logits=all_logits,
                tokenizer=model.tokenizer,
                top_k=top_k,
                num_reasoning_positions=num_reasoning_positions,
                answer_position=answer_position_idx,
                question_numbers=question_nums
            )
            all_solutions.append(gen_result)

    # Determine best represented
    best_idx = select_best_represented(all_solutions)
    if best_idx is not None:
        all_solutions[best_idx]['is_best_represented'] = True

    # Aggregate flags
    primary_found = all_solutions[0]['times_found'] > 0
    any_gt_found = any(s['times_found'] > 0 for s in all_solutions)

    # Baseline testing (only for correct answers with VP-possible primary solution)
    baseline_result = {
        'baselines': [],
        'baseline_1_found': False,
        'baseline_5_found': False,
        'num_baselines_found': 0,
    }

    if answer_correct and baseline_pool is not None and rng is not None:
        if not solution_is_vp_impossible(primary_solution_str):
            # Collect result value sets from ALL GT solutions
            gt_value_sets = [extract_all_result_values(primary_solution_str)]
            for gen_sol in (sample.get('gen_solutions') or []):
                gen_solution_str = convert_steps_to_solution_str(gen_sol)
                gt_value_sets.append(extract_all_result_values(gen_solution_str))

            # Select up to 5 random baselines
            baselines = select_random_baselines(
                sample_idx, step_count, gt_value_sets, baseline_pool, rng, n=5
            )

            if baselines:
                num_found = 0
                for baseline_sol, baseline_idx in baselines:
                    baseline_steps = parse_solution(baseline_sol)
                    if baseline_steps:
                        baseline_tree_count, baseline_score, baseline_best_tree = find_gt_tree_with_ranks(
                            baseline_steps, all_logits, model.tokenizer, top_k, num_reasoning_positions, answer_position_idx,
                            require_answer_position=baseline_require_answer,
                            question_numbers=question_nums
                        )
                        baseline_result['baselines'].append({
                            'solution': baseline_sol,
                            'source_idx': baseline_idx,
                            'times_found': baseline_tree_count,
                            'representation_score': baseline_score,
                            'best_tree': baseline_best_tree,
                        })
                        if baseline_tree_count > 0:
                            num_found += 1

                baseline_result['num_baselines_found'] = num_found

                # Check if first baseline was found (Baseline-1)
                if baseline_result['baselines'] and baseline_result['baselines'][0]['times_found'] > 0:
                    baseline_result['baseline_1_found'] = True

                # Check if any of 5 baselines was found (Baseline-5)
                if num_found > 0:
                    baseline_result['baseline_5_found'] = True

    return {
        'sample_idx': sample_idx,
        'question': question,
        'gt_answer': gt_answer,
        'model_answer': model_answer,
        'answer_correct': answer_correct,
        'step_count': step_count,
        'solutions': all_solutions,
        'primary_found': primary_found,
        'any_gt_found': any_gt_found,
        'best_represented_idx': best_idx,
        'baseline': baseline_result,
        'vocab_projection_top_k': vocab_projection_top_k,
        'column_labels': column_labels,
        'num_reasoning_positions': num_reasoning_positions,
        'question_numbers': sorted(question_nums) if question_nums else None
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze GT reasoning trace representation in vocab projections (on-the-fly)'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['coconut', 'codi'],
        help='Model type (coconut or codi)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/gsm_valid-gold-reasoning-trace_test.json',
        help='Path to dataset JSON'
    )

    parser.add_argument(
        '--num_latent',
        type=int,
        default=6,
        help='Number of latent tokens'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top tokens to consider for search'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/back_tracking_vp',
        help='Base output directory (run-specific subdirectory will be created)'
    )

    parser.add_argument(
        '--base_llm',
        type=str,
        default='gpt2',
        help='Base LLM name (e.g., gpt2, llama)'
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default=None,
        help='Base model ID for model loading (e.g., meta-llama/Llama-3.2-1B-Instruct). Defaults to base_llm if not specified.'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda, mps, cpu)'
    )

    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )

    parser.add_argument(
        '--sample_indices',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample indices to analyze (overrides --max_samples)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for baseline selection'
    )

    parser.add_argument(
        '--no_baseline_require_answer',
        dest='baseline_require_answer',
        action='store_false',
        help='Do not require baseline final answer to appear in the answer column'
    )

    parser.add_argument(
        '--include_question_tokens',
        action='store_true',
        help='Allow numbers from the question text to serve as leaf operands in the tree search'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.info("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        logging.info("MPS not available, falling back to CPU")
        device = "cpu"
    logging.info(f"Using device: {device}")

    # Load model
    logging.info(f"Loading {args.model_type} model from {args.model_path}...")
    model_id = args.model_id if args.model_id else args.base_llm
    model = ModelFactory.create(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        num_latent=args.num_latent,
        model_id=model_id
    )
    analyzer = UnifiedAnalyzer(model)
    logging.info("Model loaded")

    # Load dataset
    logging.info(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)

    # Determine which samples to process
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
        max_needed = max(sample_indices) + 1
        if max_needed > len(dataset):
            logging.error(f"Sample index {max(sample_indices)} out of range (dataset has {len(dataset)} samples)")
            return
        logging.info(f"Processing {len(sample_indices)} specific samples: {sample_indices}")
    elif args.max_samples is not None:
        dataset = dataset[:args.max_samples]
        sample_indices = list(range(len(dataset)))
        logging.info(f"Processing {len(dataset)} samples")
    else:
        sample_indices = list(range(len(dataset)))
        logging.info(f"Processing {len(dataset)} samples")

    # Build baseline pool (before limiting samples, to have more candidates)
    logging.info("Building random baseline pool...")
    baseline_pool = build_random_baseline_pool(dataset)
    for step_count, solutions in sorted(baseline_pool.items()):
        logging.info(f"  {step_count} steps: {len(solutions)} solutions")

    # Initialize random number generator for reproducibility
    rng = random.Random(args.seed)
    logging.info(f"Random seed: {args.seed}")

    # Initialize tracking
    by_steps = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'primary_found': 0,
        'any_gt_found': 0,
        'baseline_1_found': 0,
        'baseline_5_found': 0,
        'baseline_tested': 0,
    })

    per_sample_results = []
    skipped = 0

    # Process samples
    logging.info("Running analysis...")
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Analyzing samples"):
            sample = dataset[idx]
            result = analyze_sample_on_the_fly(
                sample=sample,
                sample_idx=idx,
                model=model,
                analyzer=analyzer,
                num_latent=args.num_latent,
                top_k=args.top_k,
                device=device,
                baseline_pool=baseline_pool,
                rng=rng,
                baseline_require_answer=args.baseline_require_answer,
                include_question_tokens=args.include_question_tokens,
                verbose=args.verbose
            )

            if result is None:
                skipped += 1
                continue

            per_sample_results.append(result)

            # Update step-wise stats
            step_count = result['step_count']
            by_steps[step_count]['total'] += 1

            if result['answer_correct']:
                by_steps[step_count]['correct'] += 1

                if result['primary_found']:
                    by_steps[step_count]['primary_found'] += 1

                if result['any_gt_found']:
                    by_steps[step_count]['any_gt_found'] += 1

                # Baseline stats
                if result['baseline']['baselines']:
                    by_steps[step_count]['baseline_tested'] += 1

                    if result['baseline']['baseline_1_found']:
                        by_steps[step_count]['baseline_1_found'] += 1

                    if result['baseline']['baseline_5_found']:
                        by_steps[step_count]['baseline_5_found'] += 1

    # Compute summary statistics
    total_processed = len(per_sample_results)
    total_correct = sum(1 for r in per_sample_results if r['answer_correct'])
    total_primary_found = sum(1 for r in per_sample_results if r['answer_correct'] and r['primary_found'])
    total_any_found = sum(1 for r in per_sample_results if r['answer_correct'] and r['any_gt_found'])
    total_baseline_tested = sum(1 for r in per_sample_results if r['answer_correct'] and r['baseline']['baselines'])
    total_baseline_1_found = sum(1 for r in per_sample_results if r['answer_correct'] and r['baseline']['baseline_1_found'])
    total_baseline_5_found = sum(1 for r in per_sample_results if r['answer_correct'] and r['baseline']['baseline_5_found'])

    # Print summary table
    logging.info("\n" + "=" * 95)
    logging.info("RESULTS BY NUMBER OF STEPS")
    logging.info("=" * 95)

    header = f"{'Steps':<8}{'Samples':<10}{'Correct':<10}{'Primary %':<12}{'Any GT %':<12}{'Base-1 %':<12}{'Base-5 %':<12}"
    logging.info(header)
    logging.info("-" * 95)

    for step_count in sorted(by_steps.keys()):
        stats = by_steps[step_count]
        correct = stats['correct']
        if correct > 0:
            primary_pct = stats['primary_found'] / correct * 100
            any_pct = stats['any_gt_found'] / correct * 100
        else:
            primary_pct = 0
            any_pct = 0

        baseline_tested = stats['baseline_tested']
        if baseline_tested > 0:
            baseline_1_pct = stats['baseline_1_found'] / baseline_tested * 100
            baseline_5_pct = stats['baseline_5_found'] / baseline_tested * 100
        else:
            baseline_1_pct = 0
            baseline_5_pct = 0

        row = f"{step_count:<8}{stats['total']:<10}{correct:<10}{primary_pct:<12.2f}{any_pct:<12.2f}{baseline_1_pct:<12.2f}{baseline_5_pct:<12.2f}"
        logging.info(row)

    # Overall row
    logging.info("-" * 95)
    if total_correct > 0:
        overall_primary_pct = total_primary_found / total_correct * 100
        overall_any_pct = total_any_found / total_correct * 100
    else:
        overall_primary_pct = 0
        overall_any_pct = 0

    if total_baseline_tested > 0:
        overall_baseline_1_pct = total_baseline_1_found / total_baseline_tested * 100
        overall_baseline_5_pct = total_baseline_5_found / total_baseline_tested * 100
    else:
        overall_baseline_1_pct = 0
        overall_baseline_5_pct = 0

    overall_row = f"{'All':<8}{total_processed:<10}{total_correct:<10}{overall_primary_pct:<12.2f}{overall_any_pct:<12.2f}{overall_baseline_1_pct:<12.2f}{overall_baseline_5_pct:<12.2f}"
    logging.info(overall_row)
    logging.info("=" * 95)

    # Additional baseline info
    logging.info(f"\nBaseline tested: {total_baseline_tested} samples")
    logging.info(f"Baseline-1 found: {total_baseline_1_found} ({overall_baseline_1_pct:.2f}%)")
    logging.info(f"Baseline-5 found: {total_baseline_5_found} ({overall_baseline_5_pct:.2f}%)")

    if skipped > 0:
        logging.info(f"\nSkipped {skipped} samples due to position finding errors")

    # Prepare output directory with descriptive name
    # Format: {model_type}_{base_llm}_{dataset_name}_k{top_k}
    dataset_name = Path(args.dataset_path).stem  # e.g., "gsm_test_clean"
    answer_suffix = "no-baseline-require-answer" if not args.baseline_require_answer else "yes-baseline-require-answer"
    question_suffix = "yes-question-tokens" if args.include_question_tokens else "no-question-tokens"
    run_name = f"{args.model_type}_{args.base_llm}_{dataset_name}_k{args.top_k}_{answer_suffix}_{question_suffix}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output data
    output_data = {
        'metadata': {
            'model_type': args.model_type,
            'model_path': args.model_path,
            'num_latent': args.num_latent,
            'top_k': args.top_k,
            'dataset_path': args.dataset_path,
            'total_samples': total_processed,
            'seed': args.seed
        },
        'per_sample': per_sample_results
    }

    # Save JSON
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Saved JSON to {json_path}")

    # Save CSV summary
    csv_path = output_dir / 'summary.csv'
    with open(csv_path, 'w') as f:
        f.write("Steps,Samples,Correct,Primary Found %,Any GT Found %,Base-1 %,Base-5 %\n")
        for step_count in sorted(by_steps.keys()):
            stats = by_steps[step_count]
            correct = stats['correct']
            if correct > 0:
                primary_pct = stats['primary_found'] / correct * 100
                any_pct = stats['any_gt_found'] / correct * 100
            else:
                primary_pct = 0
                any_pct = 0

            baseline_tested = stats['baseline_tested']
            if baseline_tested > 0:
                baseline_1_pct = stats['baseline_1_found'] / baseline_tested * 100
                baseline_5_pct = stats['baseline_5_found'] / baseline_tested * 100
            else:
                baseline_1_pct = 0
                baseline_5_pct = 0

            f.write(f"{step_count},{stats['total']},{correct},{primary_pct:.2f},{any_pct:.2f},{baseline_1_pct:.2f},{baseline_5_pct:.2f}\n")
        # Overall row
        f.write(f"All,{total_processed},{total_correct},{overall_primary_pct:.2f},{overall_any_pct:.2f},{overall_baseline_1_pct:.2f},{overall_baseline_5_pct:.2f}\n")
    logging.info(f"Saved CSV to {csv_path}")

    logging.info(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
