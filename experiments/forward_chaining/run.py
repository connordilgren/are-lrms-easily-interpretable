#!/usr/bin/env python3
"""
Forward Chaining Experiment

Discovers computation trees from vocabulary projection tables by:
1. Finding all valid arithmetic steps at each position pair (i, i+1)
2. Chaining these steps to form trees ending at the model's predicted answer
3. Selecting the best tree (most steps, lowest average rank)

For coconut: operands at position i, result (top-1 integer) at position i+1
"""

import argparse
import json
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

import torch
from tqdm import tqdm

from models.factory import ModelFactory
from analyzers.base import UnifiedAnalyzer


def make_json_serializable(obj, seen=None):
    """Recursively convert an object to JSON-serializable format.

    Handles sets, circular references, and non-serializable types.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return "<circular reference>"

    if isinstance(obj, dict):
        seen.add(obj_id)
        result = {}
        for k, v in obj.items():
            try:
                result[str(k)] = make_json_serializable(v, seen)
            except (TypeError, ValueError):
                result[str(k)] = str(v)
        seen.discard(obj_id)
        return result
    elif isinstance(obj, (list, tuple)):
        seen.add(obj_id)
        result = [make_json_serializable(item, seen) for item in obj]
        seen.discard(obj_id)
        return result
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For other types (tensors, etc.), convert to string
        try:
            return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"


def find_key_positions(
    output_ids: torch.Tensor,
    model,
    num_latent: int
) -> Optional[Dict[str, int]]:
    """Find key positions in the output sequence."""
    output_flat = output_ids[0] if output_ids.dim() > 1 else output_ids
    special_tokens = model.special_tokens
    model_type = model.model_type

    if model_type == "coconut":
        start_id = special_tokens.get('start_latent')
        end_id = special_tokens.get('end_latent')
    elif model_type == "codi":
        start_id = special_tokens.get('bot')
        end_id = special_tokens.get('eot')
    else:
        start_id = special_tokens.get('start_latent') or special_tokens.get('bot')
        end_id = special_tokens.get('end_latent') or special_tokens.get('eot')

    if start_id is None or end_id is None:
        return None

    start_matches = (output_flat == start_id).nonzero(as_tuple=True)[0]
    if len(start_matches) == 0:
        return None
    start_idx = start_matches[0].item()

    end_matches = (output_flat == end_id).nonzero(as_tuple=True)[0]
    if len(end_matches) == 0:
        return None
    end_idx = end_matches[0].item()

    positions = {
        'start_latent': start_idx,
        'end_latent': end_idx,
    }

    for i in range(num_latent):
        positions[f'latent_{i}'] = start_idx + 1 + i

    if model_type == "codi":
        positions['delimiter_0'] = end_idx + 1
        positions['delimiter_1'] = end_idx + 2
        positions['delimiter_2'] = end_idx + 3
        positions['delimiter_3'] = end_idx + 4
        positions['first_answer'] = end_idx + 5
    else:
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


def extract_integers_from_topk(top_k_tokens: List[str]) -> List[Tuple[int, int]]:
    """Extract (value, rank) pairs for integer tokens in top-k."""
    integers = []
    for rank, token in enumerate(top_k_tokens):
        cleaned = token.strip()
        try:
            value = int(cleaned)
            integers.append((value, rank))
        except ValueError:
            continue
    return integers


def get_top1_integer(top_k_tokens: List[str]) -> Optional[Tuple[int, int]]:
    """Get the first (top-ranked) integer from top-k tokens.

    Returns (value, rank) where rank is the position of the first integer.
    """
    for rank, token in enumerate(top_k_tokens):
        cleaned = token.strip()
        try:
            value = int(cleaned)
            return (value, rank)
        except ValueError:
            continue
    return None


def _is_space_token_str(token: str) -> bool:
    """Check if a token string is purely whitespace."""
    return token.strip() == ''


def extract_multitoken_integer(
    vocab_projection_top_k: List[List[str]],
    start_position_in_list: int
) -> Optional[Tuple[int, int, int]]:
    """
    Extract a multi-token integer by combining top-1 tokens from consecutive positions.

    Handles cases where the model outputs numbers across multiple tokens,
    e.g., " 6" + "75" = 675.
    Also handles Llama-style tokenization where a leading space token precedes the digits.

    Args:
        vocab_projection_top_k: List of top-k tokens at each position
        start_position_in_list: Index to start extraction from

    Returns:
        (integer_value, rank, spaces_skipped) if found, None if not.
        rank is 0 since we use top-1 at each position.
        spaces_skipped is the number of leading space tokens that were skipped.
    """
    combined_digits = ""

    # Skip leading space tokens (handles Llama's " " + "18" tokenization)
    actual_start = start_position_in_list
    while actual_start < len(vocab_projection_top_k):
        top1_token = vocab_projection_top_k[actual_start][0]
        if not _is_space_token_str(top1_token):
            break
        actual_start += 1

    spaces_skipped = actual_start - start_position_in_list

    for pos_idx in range(actual_start, len(vocab_projection_top_k)):
        top1_token = vocab_projection_top_k[pos_idx][0]  # Get top-1 token
        cleaned = top1_token.strip()

        # Check if this token contributes to the number
        if cleaned.isdigit():
            combined_digits += cleaned
        elif cleaned == "-" and not combined_digits:
            # Allow leading negative sign
            combined_digits += cleaned
        else:
            # Non-digit token (e.g., <|endoftext|>), stop combining
            break

    if combined_digits and combined_digits != "-":
        try:
            return int(combined_digits), 0, spaces_skipped
        except ValueError:
            pass

    return None


def extract_numbers_from_question(question: str) -> Set[int]:
    """Extract all numbers mentioned in the question text.

    This includes:
    - Digit-based numbers (e.g., "16", "25")
    - Comma-formatted numbers (e.g., "80,000", "1,000,000")
    - Decimal numbers (e.g., "1.2", "0.5") - represented by first non-zero integer token
    - Word-based numbers (e.g., "three", "four")
    - Implied 100 when percentages are present (e.g., "20%" implies 20/100)

    For multi-token numbers like decimals, we use the first non-zero integer token
    to match how the model tokenizes them. E.g., "1.2" -> 1, "0.5" -> 5.

    For percentages, we add 100 to the set since percentage calculations require
    dividing by 100 (e.g., "20% of 25" = 20/100 * 25).

    These numbers should not be treated as intermediate results.
    """
    import re

    numbers = set()

    # Track positions of numbers we've already processed so we don't double-count
    processed_positions = set()

    # Extract comma-formatted numbers first (e.g., "80,000", "1,000,000")
    # Use the first non-zero integer token (matching tokenization behavior)
    comma_number_pattern = r'\b(\d{1,3}(?:,\d{3})+)\b'
    for match in re.finditer(comma_number_pattern, question):
        # Mark this position range as processed
        processed_positions.add((match.start(), match.end()))
        # Split by comma and use first non-zero part (like tokenization)
        parts = match.group(1).split(',')
        for part in parts:
            if int(part) != 0:
                numbers.add(int(part))
                break

    # Extract decimal numbers (e.g., "1.2", "2.5%", "0.5")
    # Remove decimal, remove leading zeros, take first integer token
    # E.g., "1.2" -> 12, "2.5%" -> 25, "0.5" -> 5
    decimal_pattern = r'\b(\d+)\.(\d+)'
    for match in re.finditer(decimal_pattern, question):
        integer_part = match.group(1)
        decimal_part = match.group(2)
        # Mark this position range as processed
        processed_positions.add((match.start(), match.end()))
        # Concatenate parts, remove leading zeros, convert to int
        combined = integer_part + decimal_part
        numbers.add(int(combined.lstrip('0') or '0'))

    # Extract whole numbers (but not parts of numbers we already processed)
    # Use \b(\d+) without trailing \b to match numbers followed by units (e.g., "30mph")
    whole_number_pattern = r'\b(\d+)'
    for match in re.finditer(whole_number_pattern, question):
        # Skip if this number is part of a number we already processed
        pos = match.start()
        is_already_processed = any(start <= pos < end for start, end in processed_positions)
        if not is_already_processed:
            try:
                numbers.add(int(match.group(1)))
            except ValueError:
                pass

    # Word to number mapping for common small numbers and multipliers
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        # Multiplier words
        'twice': 2, 'double': 2, 'triple': 3, 'thrice': 3, 'half': 2, 'quarter': 4,
        # Quantity words
        'dozen': 12, 'dozens': 12
    }

    # Ordinal words for fractions (e.g., "two-thirds" -> 2 and 3)
    ordinal_to_num = {
        'half': 2, 'halves': 2,
        'third': 3, 'thirds': 3,
        'fourth': 4, 'fourths': 4, 'quarter': 4, 'quarters': 4,
        'fifth': 5, 'fifths': 5,
        'sixth': 6, 'sixths': 6,
        'seventh': 7, 'sevenths': 7,
        'eighth': 8, 'eighths': 8,
        'ninth': 9, 'ninths': 9,
        'tenth': 10, 'tenths': 10
    }

    question_lower = question.lower()

    # Extract hyphenated fractions (e.g., "two-thirds", "three-fifths")
    for cardinal, cardinal_num in word_to_num.items():
        if cardinal_num == 0 or cardinal_num > 20:
            continue  # Skip zero and large numbers
        for ordinal, ordinal_num in ordinal_to_num.items():
            pattern = f'{cardinal}-{ordinal}'
            if pattern in question_lower:
                numbers.add(cardinal_num)
                numbers.add(ordinal_num)

    # Extract word-based numbers (case insensitive)
    for word, num in word_to_num.items():
        if word in question_lower:
            numbers.add(num)

    # Extract standalone ordinal words (e.g., "a third", "half")
    # These represent divisors in fraction expressions
    for word, num in ordinal_to_num.items():
        if word in question_lower:
            numbers.add(num)

    # Add 100 if percentage is present (% sign or "percent" word)
    # The 100 is implied by percentage notation (e.g., 20% means 20/100)
    if '%' in question or 'percent' in question_lower:
        numbers.add(100)

    return numbers


def parse_gt_step(step_str: str) -> Optional[Dict]:
    """Parse a GT step string like '<<16-3-4=9>>' into operands/result dict.

    Returns dict with:
    - operands: list of operand values (always positive, as operations are infix)
    - operations: set of operations used (+, -, *, /)
    - result: result value (can be negative)
    - expression: original string
    """
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    match = re.match(pattern, step_str)
    if not match:
        return None

    expr, result_str = match.groups()
    result = int(float(result_str)) if float(result_str) == int(float(result_str)) else float(result_str)

    # Extract operands from expression - use only positive numbers since operators are infix
    # E.g., "16-3-4" should give [16, 3, 4], not [16, -3, -4]
    # Note: Convert to float first since strings like '3.00' can't be directly int()'d
    operands = [int(float(x)) if float(x) == int(float(x)) else float(x)
                for x in re.findall(r'\d+(?:\.\d+)?', expr)]

    # Extract operations from expression
    operations = set(re.findall(r'[+\-*/]', expr))

    return {
        'operands': operands,
        'operations': operations,
        'result': result,
        'expression': step_str
    }


def compare_gt_to_found(
    gt_steps: List[str],
    found_steps: List[Dict],
    used_steps: List[Dict],
    vocab_projection_top_k: List[List[str]]
) -> Dict:
    """Compare a GT solution to found computation steps.

    Returns comparison metrics dict.
    """
    # Parse GT steps
    parsed_gt = [parse_gt_step(s) for s in gt_steps]
    parsed_gt = [s for s in parsed_gt if s is not None]

    if not parsed_gt:
        return {'error': 'Failed to parse GT steps'}

    # Get GT intermediate results (all step results, including final answer)
    gt_intermediates = set(s['result'] for s in parsed_gt)

    # 1. Count GT intermediates in vocab projection top-1
    top1_integers = set()
    for pos_tokens in vocab_projection_top_k:
        if pos_tokens:
            tok = pos_tokens[0].strip()
            try:
                val = int(tok)
                top1_integers.add(val)
            except ValueError:
                pass
    gt_intermediates_in_vp_top1 = len(gt_intermediates & top1_integers)

    # 2. Count GT intermediates in used step results vs any found step results
    used_results = set(s['result'] for s in used_steps)
    all_found_results = set(s['result'] for s in found_steps)
    gt_intermediates_in_used_steps = len(gt_intermediates & used_results)
    gt_intermediates_in_any_found = len(gt_intermediates & all_found_results)

    # 3/4. Count matching steps
    def steps_match(gt_step, found_step):
        """Check if a GT step matches a found step (same operands, operations, result)."""
        # Compare operands as sorted lists to preserve counts
        gt_operands = sorted(gt_step['operands'])
        found_operands = sorted([found_step['operand1'], found_step['operand2']])
        if found_step.get('operand3') is not None:
            found_operands = sorted(found_operands + [found_step['operand3']])

        if gt_operands != found_operands:
            return False

        if gt_step['result'] != found_step['result']:
            return False

        # Compare operations
        gt_operations = gt_step.get('operations', set())
        # Extract operations from found step's operation string
        # e.g., "+-" -> {'+', '-'}, "*" -> {'*'}, "*/" -> {'*', '/'}
        found_op_str = found_step.get('operation', '')
        found_operations = set(c for c in found_op_str if c in '+-*/')

        return gt_operations == found_operations

    matching_used_steps = sum(
        1 for gt in parsed_gt
        if any(steps_match(gt, fs) for fs in used_steps)
    )

    matching_any_steps = sum(
        1 for gt in parsed_gt
        if any(steps_match(gt, fs) for fs in found_steps)
    )

    # 5. Check if trees are identical
    is_same_tree = (matching_used_steps == len(parsed_gt) == len(used_steps))

    return {
        'gt_steps': gt_steps,  # Include original step strings
        'gt_step_count': len(parsed_gt),
        'gt_intermediate_count': len(gt_intermediates),
        'is_same_tree': is_same_tree,
        'gt_intermediates_in_vp_top1': gt_intermediates_in_vp_top1,
        'gt_intermediates_in_used_steps': gt_intermediates_in_used_steps,
        'gt_intermediates_in_any_found': gt_intermediates_in_any_found,
        'matching_used_steps': matching_used_steps,
        'matching_any_steps': matching_any_steps,
    }


def compare_all_gt_solutions(
    primary_steps: List[str],
    gen_solutions: List[List[str]],
    found_steps: List[Dict],
    used_steps: List[Dict],
    vocab_projection_top_k: List[List[str]]
) -> Dict:
    """Compare all GT solutions and select best fit.

    Returns dict with:
    - primary: comparison for primary solution
    - best_fit: comparison for best fitting solution (if different from primary)
    - best_fit_index: index in gen_solutions (or -1 for primary)
    """
    # Compare primary
    primary_comparison = compare_gt_to_found(
        primary_steps, found_steps, used_steps, vocab_projection_top_k
    )
    primary_comparison['is_primary'] = True
    primary_comparison['solution_index'] = -1

    all_comparisons = [primary_comparison]

    # Compare gen_solutions
    for i, gen_sol in enumerate(gen_solutions or []):
        comp = compare_gt_to_found(
            gen_sol, found_steps, used_steps, vocab_projection_top_k
        )
        comp['is_primary'] = False
        comp['solution_index'] = i
        all_comparisons.append(comp)

    # Find best fit: most matching_any_steps, then most gt_intermediates_in_any_found
    best_idx = 0
    for i, comp in enumerate(all_comparisons):
        if comp.get('error'):
            continue
        current_best = all_comparisons[best_idx]
        if (comp['matching_any_steps'] > current_best.get('matching_any_steps', 0) or
            (comp['matching_any_steps'] == current_best.get('matching_any_steps', 0) and
             comp['gt_intermediates_in_any_found'] > current_best.get('gt_intermediates_in_any_found', 0))):
            best_idx = i

    result = {
        'primary': primary_comparison,
    }

    # Only include best_fit if different from primary
    if best_idx != 0:
        best_fit = all_comparisons[best_idx]
        best_fit['is_best_fit'] = True
        result['best_fit'] = best_fit
    else:
        primary_comparison['is_best_fit'] = True

    return result


def find_all_steps_at_position(
    operand_tokens: List[str],
    result_tokens: List[str],
    pos_operands: int,
    pos_result: int,
    top_k: int,
    intermediate_results: List[Tuple[int, int, int]] = None,
    question_numbers: Set[int] = None,
    previous_top1_integers: List[Tuple[int, int, int]] = None,
    include_question_tokens: bool = False,
    k_offset: int = 1
) -> List[Dict]:
    """Find all valid arithmetic steps at a position pair.

    Args:
        operand_tokens: Top-k tokens at position i (operands)
        result_tokens: Top-k tokens at position i+1 (result)
        pos_operands: Position index for operands
        pos_result: Position index for result
        top_k: Number of top tokens to consider for operands
        intermediate_results: List of (value, source_position, source_rank) from previous steps
        question_numbers: Set of numbers mentioned in the question (should not be intermediate results)
        previous_top1_integers: List of (value, position, rank) for top-1 integers from all previous positions
        k_offset: Offset from operand position to result position (used to identify early-position results)

    Returns:
        List of step dicts
    """
    # Get top-1 integer as result
    result_info = get_top1_integer(result_tokens)
    if result_info is None:
        return []

    result_val, result_rank = result_info

    # Note: We no longer skip positions where the result is a question number.
    # Even if an intermediate result or final answer matches a question number,
    # we still want to find computation steps that produce it.

    # Get all integers from operand position
    operands_from_position = extract_integers_from_topk(operand_tokens[:top_k])

    # Build a set of intermediate results at the current position for quick lookup
    # These come from question-computed steps and should be marked as intermediate
    ir_at_current_pos = {}
    if intermediate_results:
        for ir_val, ir_pos, ir_rank in intermediate_results:
            if ir_pos == pos_operands:
                ir_at_current_pos[ir_val] = ir_rank

    # Build combined operand list: (value, rank, is_intermediate, source_pos, ir_rank, is_question_number)
    # Intermediate results get rank -1 (preferred) to prioritize them
    # ir_rank is the original rank at the source position (for visualization)
    # Priority 2: top-k integer that is also a question number -> source_pos = pos_operands, NOT intermediate
    # Priority 3: question number NOT in top-k -> source_pos = "question" (added below)
    # Priority 4: top-k integer that is NOT a question number -> source_pos = pos_operands
    all_operands = []
    for val, rank in operands_from_position:
        is_qnum = question_numbers and val in question_numbers
        if is_qnum:
            # Question numbers in top-k should NOT be marked as intermediate
            # This prevents the tree builder from tracing them back to unverified steps
            all_operands.append((val, rank, False, pos_operands, rank, True))
        elif val in ir_at_current_pos:
            # Mark as intermediate (from question-computed step) - but only for non-question numbers
            all_operands.append((val, -1, True, pos_operands, ir_at_current_pos[val], False))
        else:
            # Source is the operand position, not a question number
            all_operands.append((val, rank, False, pos_operands, rank, False))

    # Add top-1 integers from all previous positions as potential operands
    # These are marked as "intermediate" for visualization purposes (arrows from source position)
    if previous_top1_integers:
        for prev_val, prev_pos, prev_rank in previous_top1_integers:
            if prev_pos < pos_operands:
                # Check if this value is already in all_operands from the same position
                already_exists = any(
                    op[0] == prev_val and op[3] == prev_pos
                    for op in all_operands
                )
                if not already_exists:
                    is_qnum = question_numbers and prev_val in question_numbers
                    # Use rank 0 (top-1) for scoring, mark as "intermediate" for arrow drawing
                    all_operands.append((prev_val, 0, True, prev_pos, prev_rank, is_qnum))

    # Add intermediate results as potential operands (from computed steps)
    if intermediate_results:
        for ir_val, ir_pos, ir_rank in intermediate_results:
            # Use intermediate results from positions before this one,
            # OR from early positions (< k_offset) which come from question-computed steps
            # that were processed before the main loop
            if ir_pos < pos_operands or ir_pos < k_offset:
                # Check if this value is already in all_operands from the same position
                already_exists = any(
                    op[0] == ir_val and op[3] == ir_pos
                    for op in all_operands
                )
                if not already_exists:
                    is_qnum = question_numbers and ir_val in question_numbers
                    # Use rank -1 to indicate this is from intermediate result (preferred)
                    # Store original rank as 5th element for visualization
                    all_operands.append((ir_val, -1, True, ir_pos, ir_rank, is_qnum))

    # Add question numbers as potential operands (priority 3 - NOT in top-k)
    # These get rank 1000 + index to make them lower priority than top-k integers
    # Marked with source_pos="question" for visualization (no arrow, just label)
    if include_question_tokens and question_numbers:
        for qnum_idx, qnum_val in enumerate(sorted(question_numbers)):
            # Check if this value is already in all_operands
            already_exists = any(op[0] == qnum_val for op in all_operands)
            if not already_exists:
                # Use high rank (1000 + index) to make them lower priority
                # Mark with special source position "question"
                # is_intermediate=True triggers arrow drawing from source
                all_operands.append((qnum_val, 1000 + qnum_idx, True, "question", 0, True))

    if len(all_operands) < 2:
        return []

    steps = []

    # Helper to create step dict
    # Tuple format: (value, rank, is_intermediate, source_pos, ir_rank, is_question_number)
    def make_step(o1, o1_rank, o1_is_ir, o1_src, o1_ir_rank, o1_is_qnum,
                  o2, o2_rank, o2_is_ir, o2_src, o2_ir_rank, o2_is_qnum, op):
        # Steps using intermediate results are preferred (lower score)
        # Count how many operands are from intermediate results (but not question-only sources)
        o1_from_question_only = o1_src == "question"
        o2_from_question_only = o2_src == "question"
        # For IR count, exclude question-only operands (they're lower priority)
        uses_ir = (o1_is_ir and not o1_from_question_only) or (o2_is_ir and not o2_from_question_only)
        ir_count = int(o1_is_ir and not o1_from_question_only) + int(o2_is_ir and not o2_from_question_only)
        # For ranking: prefer more intermediate results, then lower avg rank
        # Use negative ir_count so more IR usage = lower score
        # Question-only numbers have very high rank (1000+) which pushes them to lower priority
        effective_rank = (o1_rank + o2_rank) / 2 if not uses_ir else -ir_count
        return {
            "operand1": o1,
            "operand2": o2,
            "operation": op,
            "result": result_val,
            "position_operands": pos_operands,
            "position_result": pos_result,
            "operand1_rank": o1_rank,
            "operand2_rank": o2_rank,
            "result_rank": result_rank,
            "avg_operand_rank": effective_rank,
            "uses_intermediate": uses_ir,
            "operand1_is_intermediate": o1_is_ir,
            "operand2_is_intermediate": o2_is_ir,
            "operand1_source_pos": o1_src,
            "operand2_source_pos": o2_src,
            "operand1_ir_rank": o1_ir_rank,  # original rank at source position
            "operand2_ir_rank": o2_ir_rank,  # original rank at source position
            "operand1_is_question_number": o1_is_qnum,
            "operand2_is_question_number": o2_is_qnum,
        }

    # Check self-pairs first (x*x, x/x, x+x) - but NOT x-x=0
    for (op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum) in all_operands:
        # Skip 0 and 1 for self-operations (trivial: 0*0=0, 1*1=1, 0/0=undef, 1/1=1, 0+0=0, 1+1=2)
        # Actually, let's allow them - the user might want 1+1=2 etc.
        # But skip 0 for division (0/0 is undefined)

        # x + x = 2x
        if op + op == result_val:
            steps.append(make_step(op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum,
                                   op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum, "+"))

        # x * x = x^2
        if op * op == result_val:
            steps.append(make_step(op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum,
                                   op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum, "*"))

        # x / x = 1 (only if x != 0)
        if op != 0 and result_val == 1:
            steps.append(make_step(op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum,
                                   op, op_rank, op_is_ir, op_src, op_ir_rank, op_is_qnum, "/"))

    # Check all pairs of distinct operands
    for (op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum), (op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum) in combinations(all_operands, 2):
        # Skip if both operands are the same intermediate result
        if op1_is_ir and op2_is_ir and op1_src == op2_src and op1 == op2:
            continue
        # Skip trivial steps that don't represent real computation:
        # - n*1=n, n/1=n (multiply/divide by 1)
        # - n+0=n, n-0=n (add/subtract 0)
        if op1 == 1 or op2 == 1:
            # Skip if result equals the other operand (trivial identity for * or /)
            if (op1 == 1 and result_val == op2) or (op2 == 1 and result_val == op1):
                continue
        if op1 == 0 or op2 == 0:
            # Skip if result equals the other operand (trivial identity for + or -)
            if (op1 == 0 and result_val == op2) or (op2 == 0 and result_val == op1):
                continue

        # Addition: op1 + op2 = result
        if op1 + op2 == result_val:
            steps.append(make_step(op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                                   op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum, "+"))

        # Subtraction: op1 - op2 = result
        if op1 - op2 == result_val:
            steps.append(make_step(op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                                   op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum, "-"))

        # Subtraction: op2 - op1 = result
        if op2 - op1 == result_val:
            steps.append(make_step(op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum,
                                   op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum, "-"))

        # Multiplication: op1 * op2 = result
        if op1 * op2 == result_val:
            steps.append(make_step(op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                                   op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum, "*"))

        # Division: op1 / op2 = result (integer division)
        if op2 != 0 and op1 % op2 == 0 and op1 // op2 == result_val:
            steps.append(make_step(op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                                   op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum, "/"))

        # Division: op2 / op1 = result (integer division)
        if op1 != 0 and op2 % op1 == 0 and op2 // op1 == result_val:
            steps.append(make_step(op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum,
                                   op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum, "/"))

    # Check triples of operands for 3-operand operations
    if len(all_operands) >= 3:
        def make_step_3op(o1, o1_rank, o1_is_ir, o1_src, o1_ir_rank, o1_is_qnum,
                          o2, o2_rank, o2_is_ir, o2_src, o2_ir_rank, o2_is_qnum,
                          o3, o3_rank, o3_is_ir, o3_src, o3_ir_rank, o3_is_qnum,
                          operation):
            """Create a 3-operand step with the given operation."""
            o1_from_question_only = o1_src == "question"
            o2_from_question_only = o2_src == "question"
            o3_from_question_only = o3_src == "question"
            # For IR count, exclude question-only operands
            uses_ir = ((o1_is_ir and not o1_from_question_only) or
                       (o2_is_ir and not o2_from_question_only) or
                       (o3_is_ir and not o3_from_question_only))
            ir_count = (int(o1_is_ir and not o1_from_question_only) +
                        int(o2_is_ir and not o2_from_question_only) +
                        int(o3_is_ir and not o3_from_question_only))
            effective_rank = (o1_rank + o2_rank + o3_rank) / 3 if not uses_ir else -ir_count
            return {
                "operand1": o1,
                "operand2": o2,
                "operand3": o3,
                "operation": operation,
                "result": result_val,
                "position_operands": pos_operands,
                "position_result": pos_result,
                "operand1_rank": o1_rank,
                "operand2_rank": o2_rank,
                "operand3_rank": o3_rank,
                "result_rank": result_rank,
                "avg_operand_rank": effective_rank,
                "uses_intermediate": uses_ir,
                "operand1_is_intermediate": o1_is_ir,
                "operand2_is_intermediate": o2_is_ir,
                "operand3_is_intermediate": o3_is_ir,
                "operand1_source_pos": o1_src,
                "operand2_source_pos": o2_src,
                "operand3_source_pos": o3_src,
                "operand1_ir_rank": o1_ir_rank,
                "operand2_ir_rank": o2_ir_rank,
                "operand3_ir_rank": o3_ir_rank,
                "operand1_is_question_number": o1_is_qnum,
                "operand2_is_question_number": o2_is_qnum,
                "operand3_is_question_number": o3_is_qnum,
                "is_3op": True,
            }

        for (op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum), \
            (op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum), \
            (op3, op3_rank, op3_is_ir, op3_src, op3_ir_rank, op3_is_qnum) in combinations(all_operands, 3):
            # Skip if any two operands are the same intermediate result from the same position
            if op1_is_ir and op2_is_ir and op1_src == op2_src and op1 == op2:
                continue
            if op1_is_ir and op3_is_ir and op1_src == op3_src and op1 == op3:
                continue
            if op2_is_ir and op3_is_ir and op2_src == op3_src and op2 == op3:
                continue

            # Skip 3-operand identity patterns where all operands equal the result.
            # These are mathematically valid but semantically meaningless (no real computation).
            # Examples: 25 + 25 - 25 = 25, 4 * 4 / 4 = 4
            if op1 == op2 == op3 == result_val:
                continue

            # 3-operand addition: op1 + op2 + op3 = result
            if op1 + op2 + op3 == result_val:
                steps.append(make_step_3op(
                    op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                    op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum,
                    op3, op3_rank, op3_is_ir, op3_src, op3_ir_rank, op3_is_qnum,
                    "+"
                ))

            # 3-operand subtraction variants: op1 + op2 - op3, op1 - op2 + op3, op1 - op2 - op3
            # We use permutations to cover all orderings
            operand_list = [
                (op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum),
                (op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum),
                (op3, op3_rank, op3_is_ir, op3_src, op3_ir_rank, op3_is_qnum),
            ]
            from itertools import permutations as perms
            for (a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum), \
                (b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum), \
                (c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum) in perms(operand_list):
                # a + b - c = result
                if a + b - c == result_val:
                    # Skip trivial cancellation: A + B - A = B or A + B - B = A
                    if a == c or b == c:
                        continue
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "+-"  # a + b - c
                    ))
                # a - b - c = result
                if a - b - c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "--"  # a - b - c
                    ))

            # 3-operand multiplication: op1 * op2 * op3 = result
            if op1 * op2 * op3 == result_val:
                steps.append(make_step_3op(
                    op1, op1_rank, op1_is_ir, op1_src, op1_ir_rank, op1_is_qnum,
                    op2, op2_rank, op2_is_ir, op2_src, op2_ir_rank, op2_is_qnum,
                    op3, op3_rank, op3_is_ir, op3_src, op3_ir_rank, op3_is_qnum,
                    "*"
                ))

            # 3-operand mixed multiplication/division variants
            for (a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum), \
                (b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum), \
                (c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum) in perms(operand_list):
                # a * b / c = result (integer division)
                if c != 0 and (a * b) % c == 0 and (a * b) // c == result_val:
                    # Skip trivial cancellation: A * B / A = B or A * B / B = A
                    if a == c or b == c:
                        continue
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "*/"  # a * b / c
                    ))
                # a / b / c = result (integer division)
                if b != 0 and c != 0 and a % b == 0 and (a // b) % c == 0 and (a // b) // c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "//"  # a / b / c
                    ))

            # 3-operand mixed addition/multiplication variants (standard precedence)
            for (a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum), \
                (b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum), \
                (c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum) in perms(operand_list):
                # a + b * c = a + (b*c)
                if a + b * c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "+*"  # a + (b * c)
                    ))
                # a - b * c = a - (b*c)
                if a - b * c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "-*"  # a - (b * c)
                    ))
                # a * b + c = (a*b) + c
                if a * b + c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "*+"  # (a * b) + c
                    ))
                # a * b - c = (a*b) - c
                if a * b - c == result_val:
                    # Skip trivial identity: X * 2 - X = X (always true for any X)
                    if (result_val == a and b == 2 and c == a) or (result_val == b and a == 2 and c == b):
                        continue
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "*-"  # (a * b) - c
                    ))

            # 3-operand mixed addition/division variants (standard precedence, integer division)
            for (a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum), \
                (b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum), \
                (c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum) in perms(operand_list):
                # a + b / c = a + (b/c)
                if c != 0 and b % c == 0 and a + b // c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "+/"  # a + (b / c)
                    ))
                # a - b / c = a - (b/c)
                if c != 0 and b % c == 0 and a - b // c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "-/"  # a - (b / c)
                    ))
                # a / b + c = (a/b) + c
                if b != 0 and a % b == 0 and a // b + c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "/+"  # (a / b) + c
                    ))
                # a / b - c = (a/b) - c
                if b != 0 and a % b == 0 and a // b - c == result_val:
                    steps.append(make_step_3op(
                        a, a_rank, a_is_ir, a_src, a_ir_rank, a_is_qnum,
                        b, b_rank, b_is_ir, b_src, b_ir_rank, b_is_qnum,
                        c, c_rank, c_is_ir, c_src, c_ir_rank, c_is_qnum,
                        "/-"  # (a / b) - c
                    ))

    return steps


def is_step_explainable(
    step: Dict,
    question_numbers: Set[int],
    prior_explainability: Dict[int, bool]
) -> bool:
    """Check if a step is explainable given prior computed explainability.

    A step is explainable if:
    1. All leaf operands (not intermediate results) are in question_numbers
    2. All intermediate operands come from explainable earlier steps

    Args:
        step: Step dict with operand info
        question_numbers: Numbers from the question
        prior_explainability: Map of position -> explainability for earlier steps

    Returns:
        True if the step is explainable, False otherwise
    """
    # Check each operand
    for op_key in ["operand1", "operand2", "operand3"]:
        if step.get(op_key) is None:
            continue

        is_question_number = step.get(f"{op_key}_is_question_number", False)
        is_intermediate = step.get(f"{op_key}_is_intermediate", False)
        source_pos = step.get(f"{op_key}_source_pos")

        if is_question_number:
            # Question number operands are always explainable
            continue
        elif is_intermediate:
            # Check if this is from a question (always explainable)
            if source_pos == "question":
                continue
            # Check if the source step is explainable
            if source_pos not in prior_explainability or not prior_explainability[source_pos]:
                return False
        else:
            # Leaf operand (vocab projection): check if in question_numbers
            op_val = step[op_key]
            if op_val not in question_numbers:
                return False

    return True


def select_best_step(
    steps: List[Dict],
    question_numbers: Set[int] = None,
    prior_explainability: Dict[int, bool] = None
) -> Optional[Dict]:
    """Select the best step from a list.

    Prefers (in order of dominance):
    1. Explainable steps (if question_numbers provided)
    2. Fewer operands (2-operand steps over 3-operand steps)
    3. Lower average operand rank (as tiebreaker)
    """
    if not steps:
        return None

    # Compute explainability for each candidate
    if question_numbers is not None and prior_explainability is not None:
        for step in steps:
            step["_explainable"] = is_step_explainable(step, question_numbers, prior_explainability)
    else:
        for step in steps:
            step["_explainable"] = True  # Default if no question context

    # Sort by (NOT explainable, num_operands, avg_rank)
    # False < True, so "not explainable" puts explainable first
    return min(steps, key=lambda s: (
        not s.get("_explainable", True),
        1 if s.get("is_3op") else 0,
        s["avg_operand_rank"]
    ))


def find_all_steps(
    vocab_projection_top_k: List[List[str]],
    num_step_positions: int,
    top_k: int,
    k_offset: int = 1,
    question_numbers: Set[int] = None,
    include_question_tokens: bool = False,
    answer_position_idx: int = None,
    validate: bool = False,
    template: Dict = None,
    model = None,
    analyzer = None,
    num_latent: int = None,
    device: str = None,
    validation_n: int = 3,
    verbose_validation: bool = False,
    enable_copy: bool = False,
    validation_required_passes: int = 2,
    validation_max_rank: int = 1,
    check_all_candidates: bool = False
) -> Tuple[Dict[int, List[Dict]], Dict[int, Optional[Dict]], List[Dict], Optional[Dict]]:
    """Find all steps at each position pair.

    Operands can come from:
    1. Top-k integers at the operand position
    2. Top-1 integers from all previous positions
    3. Results of previous steps (intermediate results)

    Steps using intermediate results are preferred.
    Steps whose result is a number from the question are filtered out.

    When validate=True, each step is validated as it's discovered, using validation
    status to guide step selection for subsequent positions.

    Args:
        vocab_projection_top_k: List of top-k tokens at each position
        num_step_positions: Number of position pairs to check (0 to num_step_positions-1)
        top_k: Number of top tokens to consider for operands
        k_offset: Offset from operand position to result position (1 for coconut, 2 for codi)
        question_numbers: Set of numbers mentioned in the question (should not be intermediate results)
        include_question_tokens: Whether to allow question numbers as operands
        answer_position_idx: Position index of the final answer (for copy step detection)
        validate: Whether to validate steps as they are discovered
        template: Template dict with variables (required if validate=True)
        model: Model for inference (required if validate=True)
        analyzer: UnifiedAnalyzer instance (required if validate=True)
        num_latent: Number of latent tokens (required if validate=True)
        device: Device string (required if validate=True)
        validation_n: Number of validation checks per step
        validation_required_passes: Minimum number of validation checks that must pass (default: 2)
        validation_max_rank: Max rank for expected result in integer tokens (default: 1)

    Returns:
        Tuple of (all_steps_by_position, best_step_by_position, copy_from_question_steps, validation_info)
        validation_info is None if validate=False, otherwise a dict with step validation details
    """
    all_steps = {}
    best_steps = {}
    # Track intermediate results: list of (value, result_position, result_rank)
    intermediate_results = []
    # Track explainability of steps by position for incremental computation
    # Maps position_operands -> is_explainable
    step_explainability = {}

    # For integrated validation
    verified_positions = set()  # Set of position_result values that have been verified
    validation_info = {"steps": [], "tree_verified": True} if validate else None
    tree_steps_so_far = []  # Steps selected so far (for operand tracing)

    # Build list of top-1 integers from each position
    # This allows operands to come from any previous position's top-1 integer
    all_top1_integers = []
    for pos_idx, tokens in enumerate(vocab_projection_top_k):
        top1_info = get_top1_integer(tokens)
        if top1_info is not None:
            val, rank = top1_info
            all_top1_integers.append((val, pos_idx, rank))

    # Find steps computed from question tokens at early positions (0 to k_offset-1)
    # These positions can't have results from the normal step-finding loop
    # (which produces results at positions k_offset and beyond)
    # Must be done BEFORE the main loop so results are available as intermediate operands
    if include_question_tokens and question_numbers:
        question_nums_list = sorted(question_numbers)

        for pos_idx in range(k_offset):
            if pos_idx >= len(vocab_projection_top_k):
                break

            tokens = vocab_projection_top_k[pos_idx]
            top1_info = get_top1_integer(tokens)
            if top1_info is None:
                continue

            result_val, result_rank = top1_info

            # Skip if this is just a copy of a question number (handled by copy_from_question)
            if result_val in question_numbers:
                continue

            # Collect ALL candidate steps from question numbers (like main loop does)
            question_step_candidates = []

            def make_question_step(op1, op2, operation, op1_idx, op2_idx):
                return {
                    "operand1": op1,
                    "operand2": op2,
                    "operation": operation,
                    "result": result_val,
                    "position_operands": "question",
                    "position_result": pos_idx,
                    "operand1_rank": 1000 + op1_idx,
                    "operand2_rank": 1000 + op2_idx,
                    "result_rank": result_rank,
                    "avg_operand_rank": (1000 + op1_idx + 1000 + op2_idx) / 2,
                    "uses_intermediate": False,
                    "operand1_is_intermediate": True,
                    "operand2_is_intermediate": True,
                    "operand1_source_pos": "question",
                    "operand2_source_pos": "question",
                    "operand1_ir_rank": 0,
                    "operand2_ir_rank": 0,
                    "operand1_is_question_number": True,
                    "operand2_is_question_number": True,
                    "is_computed_from_question": True,
                }

            def make_question_step_3op(op1, op2, op3, operation, idx1, idx2, idx3):
                return {
                    "operand1": op1,
                    "operand2": op2,
                    "operand3": op3,
                    "operation": operation,
                    "result": result_val,
                    "position_operands": "question",
                    "position_result": pos_idx,
                    "operand1_rank": 1000 + idx1,
                    "operand2_rank": 1000 + idx2,
                    "operand3_rank": 1000 + idx3,
                    "result_rank": result_rank,
                    "avg_operand_rank": (1000 + idx1 + 1000 + idx2 + 1000 + idx3) / 3,
                    "uses_intermediate": False,
                    "operand1_is_intermediate": True,
                    "operand2_is_intermediate": True,
                    "operand3_is_intermediate": True,
                    "operand1_source_pos": "question",
                    "operand2_source_pos": "question",
                    "operand3_source_pos": "question",
                    "operand1_ir_rank": 0,
                    "operand2_ir_rank": 0,
                    "operand3_ir_rank": 0,
                    "operand1_is_question_number": True,
                    "operand2_is_question_number": True,
                    "operand3_is_question_number": True,
                    "is_computed_from_question": True,
                    "is_3op": True,
                }

            # Try all pairs of question numbers (2-operand)
            for i, qnum1 in enumerate(question_nums_list):
                for j, qnum2 in enumerate(question_nums_list):
                    if i > j:
                        continue

                    # Self-operations (same operand twice)
                    if i == j:
                        if qnum1 + qnum1 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum1, "+", i, i))
                        if qnum1 * qnum1 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum1, "*", i, i))
                    else:
                        # Different operands
                        if qnum1 + qnum2 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum2, "+", i, j))
                        if qnum1 - qnum2 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum2, "-", i, j))
                        if qnum2 - qnum1 == result_val:
                            question_step_candidates.append(make_question_step(qnum2, qnum1, "-", j, i))
                        if qnum1 * qnum2 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum2, "*", i, j))
                        if qnum2 != 0 and qnum1 % qnum2 == 0 and qnum1 // qnum2 == result_val:
                            question_step_candidates.append(make_question_step(qnum1, qnum2, "/", i, j))
                        if qnum1 != 0 and qnum2 % qnum1 == 0 and qnum2 // qnum1 == result_val:
                            question_step_candidates.append(make_question_step(qnum2, qnum1, "/", j, i))

            # Try 3-operand combinations from question numbers
            from itertools import permutations as perms
            for i, qnum1 in enumerate(question_nums_list):
                for j, qnum2 in enumerate(question_nums_list):
                    for k, qnum3 in enumerate(question_nums_list):
                        if i >= j or j >= k:
                            continue  # Unique triplets only

                        operand_list = [(qnum1, i), (qnum2, j), (qnum3, k)]

                        # Try all permutations for non-commutative operations
                        for (a, ai), (b, bi), (c, ci) in perms(operand_list):
                            # a * b / c (most common for percentages: 20 * 25 / 100 = 5)
                            if c != 0 and (a * b) % c == 0 and (a * b) // c == result_val:
                                if a != c and b != c:  # Skip trivial cancellation
                                    question_step_candidates.append(make_question_step_3op(a, b, c, "*/", ai, bi, ci))

                            # a / b / c
                            if b != 0 and c != 0 and a % b == 0 and (a // b) % c == 0 and (a // b) // c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "//", ai, bi, ci))

                            # a + b - c
                            if a + b - c == result_val and a != c and b != c:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "+-", ai, bi, ci))

                            # a - b - c
                            if a - b - c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "--", ai, bi, ci))

                            # a + b * c
                            if a + b * c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "+*", ai, bi, ci))

                            # a - b * c
                            if a - b * c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "-*", ai, bi, ci))

                            # a * b + c
                            if a * b + c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "*+", ai, bi, ci))

                            # a * b - c
                            if a * b - c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "*-", ai, bi, ci))

                            # a + b / c
                            if c != 0 and b % c == 0 and a + b // c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "+/", ai, bi, ci))

                            # a - b / c
                            if c != 0 and b % c == 0 and a - b // c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "-/", ai, bi, ci))

                            # a / b + c
                            if b != 0 and a % b == 0 and a // b + c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "/+", ai, bi, ci))

                            # a / b - c
                            if b != 0 and a % b == 0 and a // b - c == result_val:
                                question_step_candidates.append(make_question_step_3op(a, b, c, "/-", ai, bi, ci))

                        # Also try 3-operand same-operation: a + b + c, a * b * c
                        if qnum1 + qnum2 + qnum3 == result_val:
                            question_step_candidates.append(make_question_step_3op(qnum1, qnum2, qnum3, "+", i, j, k))

                        if qnum1 * qnum2 * qnum3 == result_val:
                            question_step_candidates.append(make_question_step_3op(qnum1, qnum2, qnum3, "*", i, j, k))

            # Mark all candidates as explainable (they use only question numbers)
            for step in question_step_candidates:
                step["explainable"] = True

            if question_step_candidates:
                # Select best step using validation (like main loop does)
                if validate and template is not None and model is not None:
                    from experiments.forward_chaining.validation import select_step_with_validation

                    selected_step, is_verified, step_validation_details = select_step_with_validation(
                        question_step_candidates,
                        tree_steps_so_far,
                        template,
                        verified_positions,
                        question_numbers,
                        model,
                        analyzer,
                        num_latent,
                        device,
                        validation_n=validation_n,
                        top_k=top_k,
                        explainable_positions=set(),  # No explainable positions yet
                        top_k_at_operand_pos=set(),  # No top-k at operand position (it's "question")
                        verbose=verbose_validation,
                        required_passes=validation_required_passes,
                        max_rank=validation_max_rank,
                        check_all_candidates=check_all_candidates
                    )

                    # Track validation info
                    validation_info["steps"].append({
                        "position": "question",
                        "result_position": selected_step["position_result"],
                        "step": selected_step,
                        "verified": is_verified,
                        "candidates_tried": len(question_step_candidates),
                        "validations": step_validation_details
                    })

                    # Add verification status to the step itself
                    selected_step["verified"] = is_verified

                    if is_verified:
                        verified_positions.add(selected_step["position_result"])
                    else:
                        validation_info["tree_verified"] = False

                    tree_steps_so_far.append(selected_step)
                else:
                    # No validation - pick the step with the lowest avg_operand_rank
                    selected_step = min(question_step_candidates, key=lambda s: s["avg_operand_rank"])
                    is_verified = False

                # Use special key for question-computed steps
                best_steps[f"question_result_{pos_idx}"] = selected_step
                # Add to intermediate results so later steps can use this value
                intermediate_results.append((
                    selected_step["result"],
                    selected_step["position_result"],
                    selected_step["result_rank"]
                ))
                step_explainability[pos_idx] = True

    # Track explainable positions (position_result values with explainable steps)
    explainable_positions = set()

    for pos in range(num_step_positions):
        pos_result = pos + k_offset
        if pos_result >= len(vocab_projection_top_k):
            break

        operand_tokens = vocab_projection_top_k[pos]
        result_tokens = vocab_projection_top_k[pos_result]

        # Skip step finding if result's top-1 integer is a question number.
        # This position will be handled by copy_from_question logic later.
        if question_numbers:
            result_top1 = get_top1_integer(result_tokens)
            if result_top1 is not None:
                result_val, _ = result_top1
                if result_val in question_numbers:
                    continue

        # Filter to only include top-1 integers from positions before pos
        # Exclude position 0 because it can only have intermediate results if
        # include_question_tokens is set (handled separately)
        previous_top1 = [(v, p, r) for v, p, r in all_top1_integers if 0 < p < pos]

        steps = find_all_steps_at_position(
            operand_tokens, result_tokens, pos, pos_result, top_k,
            intermediate_results=intermediate_results,
            question_numbers=question_numbers,
            previous_top1_integers=previous_top1,
            include_question_tokens=include_question_tokens,
            k_offset=k_offset
        )

        all_steps[pos] = steps

        # Track explainability for ALL steps at this position (not just selected)
        # This is used for priority ranking - we want to know if ANY step could explain
        # an intermediate, not just the selected step
        for step in steps:
            step_is_explainable = is_step_explainable(
                step, question_numbers or set(), step_explainability
            )
            step["explainable"] = step_is_explainable
            if step_is_explainable:
                # Add the result position to explainable_positions
                # This means "there exists an explainable step that produces a result here"
                explainable_positions.add(step["position_result"])

        # Select best step, optionally with validation
        if validate and template is not None and model is not None:
            from experiments.forward_chaining.validation import select_step_with_validation

            # Build set of integer values in top-k at the operand position
            top_k_at_operand_pos = set()
            for val, rank in extract_integers_from_topk(operand_tokens[:top_k]):
                top_k_at_operand_pos.add(val)

            best_step, is_verified, step_validation_details = select_step_with_validation(
                candidates=steps,
                tree_steps=tree_steps_so_far,
                template=template,
                verified_positions=verified_positions,
                question_numbers=question_numbers or set(),
                model=model,
                analyzer=analyzer,
                num_latent=num_latent,
                device=device,
                validation_n=validation_n,
                top_k=top_k,
                explainable_positions=explainable_positions,
                top_k_at_operand_pos=top_k_at_operand_pos,
                verbose=verbose_validation,
                required_passes=validation_required_passes,
                max_rank=validation_max_rank,
                check_all_candidates=check_all_candidates
            )
            best_steps[pos] = best_step

            if best_step is not None:
                # Track validation info
                validation_info["steps"].append({
                    "position": pos,
                    "result_position": best_step["position_result"],
                    "step": best_step,
                    "verified": is_verified,
                    "candidates_tried": len(steps),
                    "validations": step_validation_details
                })

                # Add verification status to the step itself
                best_step["verified"] = is_verified

                if is_verified:
                    verified_positions.add(best_step["position_result"])
                else:
                    validation_info["tree_verified"] = False

                tree_steps_so_far.append(best_step)

                # Track explainability for the selected step (already computed above)
                step_is_explainable = best_step.get("explainable", False)
                step_explainability[best_step["position_result"]] = step_is_explainable
        else:
            best_step = select_best_step(steps, question_numbers, step_explainability)
            best_steps[pos] = best_step

        # Add the best step's result to intermediate results for future positions
        if best_step is not None:
            intermediate_results.append((
                best_step["result"],
                best_step["position_result"],
                best_step["result_rank"]
            ))
            # Track explainability for this step's result position (non-validation path)
            # Use the result position as key since that's where intermediates are sourced from
            if not validate:
                step_is_explainable = best_step.get("explainable", False)
                step_explainability[best_step["position_result"]] = step_is_explainable

    # Create copy_from_question steps for top-1 integers matching question numbers
    # Only when include_question_tokens is True and enable_copy is True
    copy_from_question_steps = []
    if enable_copy and include_question_tokens and question_numbers:
        # Get results produced by computation steps
        step_results_at_pos = {}
        for pos, step in best_steps.items():
            if step is not None:
                step_results_at_pos[step["position_result"]] = step["result"]

        # Check each position's top-1 integer
        for pos_idx, tokens in enumerate(vocab_projection_top_k):
            # Skip answer positions (final answer columns don't need copy steps)
            # This includes the answer position and any positions after it
            if answer_position_idx is not None and pos_idx >= answer_position_idx:
                continue

            top1_info = get_top1_integer(tokens)
            if top1_info is None:
                continue

            top1_val, top1_rank = top1_info

            # Check if this top-1 integer is a question number
            if top1_val not in question_numbers:
                continue

            # Check if any computation step produces this value at this position
            if pos_idx in step_results_at_pos and step_results_at_pos[pos_idx] == top1_val:
                continue

            # This is a copy from question - create synthetic step
            # Find the index of this question number for rank assignment
            qnum_idx = sorted(question_numbers).index(top1_val)
            copy_step = {
                "operation": "copy_from_question",
                "operand1": top1_val,
                "operand2": None,
                "result": top1_val,
                "position_operands": "question",  # Special marker
                "position_result": pos_idx,
                "operand1_rank": 1000 + qnum_idx,
                "operand2_rank": None,
                "result_rank": top1_rank,
                "avg_operand_rank": 1000 + qnum_idx,
                "operand1_is_intermediate": True,
                "operand1_source_pos": "question",
                "operand1_is_question_number": True,
                "is_copy_from_question": True,
            }
            copy_from_question_steps.append(copy_step)

    return all_steps, best_steps, copy_from_question_steps, validation_info


def find_combination_for_answer(
    intermediate_results: List[Tuple[int, Dict]],
    final_answer: int
) -> Optional[Dict]:
    """Find intermediate results that combine to produce the final answer.

    Tries both 2-operand and 3-operand combinations.

    Args:
        intermediate_results: List of (value, step) tuples
        final_answer: The target answer

    Returns:
        Dict with 'values', 'operation', 'steps' keys, or None if no combination found
    """
    from itertools import permutations as perms

    # Try 2-operand combinations first
    for i, (v1, step1) in enumerate(intermediate_results):
        for j, (v2, step2) in enumerate(intermediate_results):
            if i >= j:
                continue
            # Skip if same step
            if step1["position_operands"] == step2["position_operands"]:
                continue

            # Try all operations
            if v1 + v2 == final_answer:
                return {"values": [v1, v2], "operation": "+", "steps": [step1, step2]}
            if v1 - v2 == final_answer:
                return {"values": [v1, v2], "operation": "-", "steps": [step1, step2]}
            if v2 - v1 == final_answer:
                return {"values": [v2, v1], "operation": "-", "steps": [step2, step1]}
            if v1 * v2 == final_answer:
                return {"values": [v1, v2], "operation": "*", "steps": [step1, step2]}
            if v2 != 0 and v1 % v2 == 0 and v1 // v2 == final_answer:
                return {"values": [v1, v2], "operation": "/", "steps": [step1, step2]}
            if v1 != 0 and v2 % v1 == 0 and v2 // v1 == final_answer:
                return {"values": [v2, v1], "operation": "/", "steps": [step2, step1]}

    # Try 3-operand combinations
    if len(intermediate_results) >= 3:
        from itertools import combinations
        for (v1, step1), (v2, step2), (v3, step3) in combinations(intermediate_results, 3):
            # Skip if any two steps are the same
            positions = {step1["position_operands"], step2["position_operands"], step3["position_operands"]}
            if len(positions) < 3:
                continue

            # 3-operand addition: v1 + v2 + v3 = result
            if v1 + v2 + v3 == final_answer:
                return {"values": [v1, v2, v3], "operation": "+", "steps": [step1, step2, step3]}

            # 3-operand multiplication: v1 * v2 * v3 = result
            if v1 * v2 * v3 == final_answer:
                return {"values": [v1, v2, v3], "operation": "*", "steps": [step1, step2, step3]}

            # Try permutations for non-commutative operations
            operand_list = [(v1, step1), (v2, step2), (v3, step3)]
            for (a, sa), (b, sb), (c, sc) in perms(operand_list):
                # a + b - c = result
                if a + b - c == final_answer:
                    return {"values": [a, b, c], "operation": "+-", "steps": [sa, sb, sc]}
                # a - b - c = result
                if a - b - c == final_answer:
                    return {"values": [a, b, c], "operation": "--", "steps": [sa, sb, sc]}
                # a * b / c = result (integer division)
                if c != 0 and (a * b) % c == 0 and (a * b) // c == final_answer:
                    return {"values": [a, b, c], "operation": "*/", "steps": [sa, sb, sc]}
                # a / b / c = result (integer division)
                if b != 0 and c != 0 and a % b == 0 and (a // b) % c == 0 and (a // b) // c == final_answer:
                    return {"values": [a, b, c], "operation": "//", "steps": [sa, sb, sc]}
                # Mixed addition/multiplication (standard precedence)
                # a + b * c = a + (b*c)
                if a + b * c == final_answer:
                    return {"values": [a, b, c], "operation": "+*", "steps": [sa, sb, sc]}
                # a - b * c = a - (b*c)
                if a - b * c == final_answer:
                    return {"values": [a, b, c], "operation": "-*", "steps": [sa, sb, sc]}
                # a * b + c = (a*b) + c
                if a * b + c == final_answer:
                    return {"values": [a, b, c], "operation": "*+", "steps": [sa, sb, sc]}
                # a * b - c = (a*b) - c
                if a * b - c == final_answer:
                    return {"values": [a, b, c], "operation": "*-", "steps": [sa, sb, sc]}
                # Mixed addition/division (standard precedence, integer division)
                # a + b / c = a + (b/c)
                if c != 0 and b % c == 0 and a + b // c == final_answer:
                    return {"values": [a, b, c], "operation": "+/", "steps": [sa, sb, sc]}
                # a - b / c = a - (b/c)
                if c != 0 and b % c == 0 and a - b // c == final_answer:
                    return {"values": [a, b, c], "operation": "-/", "steps": [sa, sb, sc]}
                # a / b + c = (a/b) + c
                if b != 0 and a % b == 0 and a // b + c == final_answer:
                    return {"values": [a, b, c], "operation": "/+", "steps": [sa, sb, sc]}
                # a / b - c = (a/b) - c
                if b != 0 and a % b == 0 and a // b - c == final_answer:
                    return {"values": [a, b, c], "operation": "/-", "steps": [sa, sb, sc]}

    return None


def filter_unverified_backing_steps(
    tree_steps: List[Dict],
    all_steps_by_pos: Dict[int, Dict],
    vocab_projection_top_k: List[List[str]]
) -> List[Dict]:
    """
    Filter out unverified steps that back verified steps.

    When a verified step uses an operand from an unverified step,
    exclude the unverified step and re-source the operand from top-k.

    Args:
        tree_steps: List of steps in the tree (will be modified in place)
        all_steps_by_pos: Dict mapping position_operands -> step dict
        vocab_projection_top_k: Top-k tokens at each position (for rank lookup)

    Returns:
        Filtered list of steps (unverified backing steps removed)
    """
    # Find unverified steps whose results are used by verified steps
    steps_to_exclude = set()

    for step in tree_steps:
        if not step.get("verified", False):
            continue  # Only check verified steps for their sources

        # Check each operand source
        for op_key in ["operand1", "operand2", "operand3"]:
            if step.get(op_key) is None:
                continue
            if not step.get(f"{op_key}_is_intermediate", False):
                continue

            source_pos = step.get(f"{op_key}_source_pos")
            if source_pos is None or source_pos == "question":
                continue

            # Find the source step
            source_step = next(
                (s for s in tree_steps
                 if s["position_result"] == source_pos and s["result"] == step[op_key]),
                None
            )

            if source_step and not source_step.get("verified", False):
                # Mark this unverified source step for exclusion
                steps_to_exclude.add(source_step["position_operands"])

                # Re-source this operand from top-k (not intermediate)
                # Look up the actual rank of this value at the operand position
                operand_value = step[op_key]
                pos_operands = step["position_operands"]
                new_rank = step[f"{op_key}_rank"]  # Default to existing rank

                if pos_operands != "question" and pos_operands < len(vocab_projection_top_k):
                    top_k_at_pos = vocab_projection_top_k[pos_operands]
                    integers_at_pos = extract_integers_from_topk(top_k_at_pos)
                    for val, rank in integers_at_pos:
                        if val == operand_value:
                            new_rank = rank
                            break

                step[f"{op_key}_is_intermediate"] = False
                step[f"{op_key}_source_pos"] = pos_operands
                step[f"{op_key}_rank"] = new_rank

    # Filter out excluded steps
    filtered_steps = [
        s for s in tree_steps
        if s["position_operands"] not in steps_to_exclude
    ]

    return filtered_steps


def build_computation_tree(
    best_steps: Dict[int, Optional[Dict]],
    final_answer: int,
    vocab_projection_top_k: List[List[str]],
    answer_position_idx: int,
    k_offset: int = 1,
    question_numbers: Set[int] = None,
    include_question_tokens: bool = False,
    copy_from_question_steps: List[Dict] = None,
    enable_copy: bool = False
) -> Optional[Dict]:
    """Build a computation tree that produces the final answer.

    The tree is built by:
    1. Strategy 1 (Preferred): Use a direct final answer step at the earliest position,
       adding a copy step if the result is not at the answer position
    2. Strategy 2 (Fallback): If no direct step exists, try combining non-final
       intermediate results to produce the final answer

    Args:
        best_steps: Dict mapping position to best step (or None)
        final_answer: The model's predicted answer
        vocab_projection_top_k: Top-k tokens at each position
        answer_position_idx: Index in vocab projection where answer appears
        k_offset: Offset from operand position to result position
        question_numbers: Set of numbers from the question (for explainability)
        include_question_tokens: Whether question numbers can be operands
        copy_from_question_steps: List of copy_from_question steps to include

    Returns:
        Tree dict or None if no valid tree found
    """
    # Collect all valid steps and their results
    all_valid_steps = [(step["result"], step) for pos, step in best_steps.items() if step is not None]

    if not all_valid_steps:
        return None

    # Separate steps that directly produce the final answer from those that don't
    direct_final_steps = [(v, s) for v, s in all_valid_steps if v == final_answer]
    non_final_steps = [(v, s) for v, s in all_valid_steps if v != final_answer]

    tree_steps = []
    final_combination = None  # (v1, v2, op) if final answer is from combining intermediates
    uses_direct_final_step = False
    copy_step = None

    # Strategy 1 (Preferred): Use a direct final answer step
    # When copy is enabled: prefer earliest position, add copy step if not at answer position
    # When copy is disabled: prefer step at answer position, fallback to earliest
    if direct_final_steps:
        if enable_copy:
            # Sort by position_result ascending (earliest first)
            # Treat "question" as -1 for position_operands comparison
            direct_final_steps_sorted = sorted(
                direct_final_steps,
                key=lambda x: (x[1]["position_result"], -1 if x[1]["position_operands"] == "question" else x[1]["position_operands"])
            )
            _, direct_step = direct_final_steps_sorted[0]

            # If the earliest step is not at the answer position, create a copy step
            if direct_step["position_result"] != answer_position_idx:
                copy_step = {
                    "operation": "copy",
                    "operand1": direct_step["result"],
                    "operand2": None,
                    "result": direct_step["result"],
                    "position_operands": direct_step["position_result"],  # source position
                    "position_result": answer_position_idx,  # destination (answer position)
                    "operand1_rank": direct_step["result_rank"],
                    "operand2_rank": None,
                    "result_rank": 0,  # Should be top-1 at answer position
                    "avg_operand_rank": direct_step["result_rank"],
                    "is_copy": True,
                }
        else:
            # Copy disabled: prefer step at answer position, fallback to earliest
            steps_at_answer_pos = [s for v, s in direct_final_steps if s["position_result"] == answer_position_idx]
            if steps_at_answer_pos:
                # Use the best step at answer position (lowest avg_operand_rank)
                direct_step = min(steps_at_answer_pos, key=lambda s: s["avg_operand_rank"])
            else:
                # Fallback to earliest position (no copy will be created)
                direct_final_steps_sorted = sorted(
                    direct_final_steps,
                    key=lambda x: (x[1]["position_result"], -1 if x[1]["position_operands"] == "question" else x[1]["position_operands"])
                )
                _, direct_step = direct_final_steps_sorted[0]

        tree_steps = [direct_step]
        uses_direct_final_step = True

        # Trace backwards to find operand sources
        used_positions = {direct_step["position_operands"]}

        def trace_backwards(step: Dict):
            """Recursively trace operand sources.

            Only include a previous step if its result is actually used as an operand
            in the current step, checking the specific source position.
            """
            op1, op2 = step["operand1"], step["operand2"]
            op3 = step.get("operand3")  # For 3-operand steps

            # Get the source positions for intermediate operands
            op1_source_pos = step.get("operand1_source_pos") if step.get("operand1_is_intermediate") else None
            op2_source_pos = step.get("operand2_source_pos") if step.get("operand2_is_intermediate") else None
            op3_source_pos = step.get("operand3_source_pos") if step.get("operand3_is_intermediate") else None

            for prev_step_result, prev_step in all_valid_steps:
                if prev_step["position_operands"] in used_positions:
                    continue
                prev_result_pos = prev_step["position_result"]

                # Check if this previous step's result is actually used by the current step
                # by matching both value AND source position
                is_op1_source = (prev_step_result == op1 and op1_source_pos == prev_result_pos)
                is_op2_source = (prev_step_result == op2 and op2_source_pos == prev_result_pos)
                is_op3_source = (op3 is not None and prev_step_result == op3 and op3_source_pos == prev_result_pos)

                if is_op1_source or is_op2_source or is_op3_source:
                    used_positions.add(prev_step["position_operands"])
                    tree_steps.append(prev_step)
                    trace_backwards(prev_step)

        trace_backwards(direct_step)

        # Filter out unverified steps that back verified steps
        if tree_steps:
            tree_steps = filter_unverified_backing_steps(tree_steps, best_steps, vocab_projection_top_k)

    # Strategy 2 (Fallback): Try to combine non-final intermediate results
    # Only used when no direct final step exists
    if not tree_steps and len(non_final_steps) >= 2:
        combo = find_combination_for_answer(non_final_steps, final_answer)
        if combo:
            tree_steps = combo["steps"]
            final_combination = {
                "values": combo["values"],
                "operation": combo["operation"]
            }
            uses_direct_final_step = False

    if not tree_steps:
        return None

    # Sort steps by position (treat "question" as -1 to sort before numeric positions)
    tree_steps.sort(key=lambda s: -1 if s["position_operands"] == "question" else s["position_operands"])

    # Build nodes and edges
    nodes = []
    edges = []
    added_nodes = set()

    for step in tree_steps:
        pos_op = step["position_operands"]
        pos_res = step["position_result"]

        # Add result node
        result_key = (step["result"], pos_res)
        if result_key not in added_nodes:
            # A step's result is "final" only if it directly produces the answer
            # at the answer position (no copy step needed)
            is_final = (step["result"] == final_answer and uses_direct_final_step and copy_step is None)
            nodes.append({
                "value": step["result"],
                "position": pos_res,
                "rank": step["result_rank"],
                "type": "final" if is_final else "intermediate"
            })
            added_nodes.add(result_key)

        # Add operand nodes (only if they're leaves - not intermediate results from tree)
        # Build operand info list with actual positions and ranks
        operand_infos = [
            (step["operand1"], step["operand1_rank"], step.get("operand1_is_intermediate", False),
             step.get("operand1_source_pos", pos_op), step.get("operand1_ir_rank", step["operand1_rank"])),
            (step["operand2"], step["operand2_rank"], step.get("operand2_is_intermediate", False),
             step.get("operand2_source_pos", pos_op), step.get("operand2_ir_rank", step["operand2_rank"]))
        ]
        # Add operand3 for 3-operand steps
        if step.get("operand3") is not None:
            operand_infos.append(
                (step["operand3"], step["operand3_rank"], step.get("operand3_is_intermediate", False),
                 step.get("operand3_source_pos", pos_op), step.get("operand3_ir_rank", step["operand3_rank"]))
            )

        for op, op_rank, op_is_ir, op_src_pos, op_src_rank in operand_infos:
            # Use actual source position and rank for the operand
            actual_pos = op_src_pos if op_is_ir else pos_op
            actual_rank = op_src_rank if op_is_ir else op_rank
            op_key = (op, actual_pos)

            # Check if this operand is an intermediate result from a previous step in tree
            is_tree_intermediate = any(
                s["result"] == op and s["position_result"] == actual_pos
                for s in tree_steps if s["position_operands"] != step["position_operands"]
            )

            if not is_tree_intermediate and op_key not in added_nodes:
                nodes.append({
                    "value": op,
                    "position": actual_pos,
                    "rank": actual_rank,
                    "type": "leaf"
                })
                added_nodes.add(op_key)

        # Add edges from operands to result
        # For intermediate result operands, use the stored source position/rank
        op1_pos = pos_op
        op1_rank = step["operand1_rank"]
        if step.get("operand1_is_intermediate"):
            # Use the stored source position and rank from when the step was found
            op1_pos = step.get("operand1_source_pos", pos_op)
            op1_rank = step.get("operand1_ir_rank", step["operand1_rank"])

        op2_pos = pos_op
        op2_rank = step["operand2_rank"]
        if step.get("operand2_is_intermediate"):
            # Use the stored source position and rank from when the step was found
            op2_pos = step.get("operand2_source_pos", pos_op)
            op2_rank = step.get("operand2_ir_rank", step["operand2_rank"])

        edges.append({
            "from_value": step["operand1"],
            "from_position": op1_pos,
            "from_rank": op1_rank,
            "to_value": step["result"],
            "to_position": pos_res,
            "to_rank": step["result_rank"]
        })
        edges.append({
            "from_value": step["operand2"],
            "from_position": op2_pos,
            "from_rank": op2_rank,
            "to_value": step["result"],
            "to_position": pos_res,
            "to_rank": step["result_rank"]
        })

        # Add edge for operand3 (3-operand steps)
        if step.get("operand3") is not None:
            op3_pos = pos_op
            op3_rank = step["operand3_rank"]
            if step.get("operand3_is_intermediate"):
                op3_pos = step.get("operand3_source_pos", pos_op)
                op3_rank = step.get("operand3_ir_rank", step["operand3_rank"])

            edges.append({
                "from_value": step["operand3"],
                "from_position": op3_pos,
                "from_rank": op3_rank,
                "to_value": step["result"],
                "to_position": pos_res,
                "to_rank": step["result_rank"]
            })

    # If using a copy step, add the final answer node at the answer position and the copy edge
    if copy_step:
        # Add final answer node at the answer position
        final_key = (copy_step["result"], copy_step["position_result"])
        if final_key not in added_nodes:
            nodes.append({
                "value": copy_step["result"],
                "position": copy_step["position_result"],
                "rank": copy_step["result_rank"],
                "type": "final"
            })
            added_nodes.add(final_key)

        # Add edge from source position to answer position (the copy)
        edges.append({
            "from_value": copy_step["operand1"],
            "from_position": copy_step["position_operands"],  # source position
            "from_rank": copy_step["operand1_rank"],
            "to_value": copy_step["result"],
            "to_position": copy_step["position_result"],  # answer position
            "to_rank": copy_step["result_rank"],
            "is_copy": True  # Mark this edge as a copy edge
        })

    # If using combined intermediates, add the final answer node and edges
    if final_combination:
        values = final_combination["values"]
        # Add final answer node (synthetic - not from a step)
        nodes.append({
            "value": final_answer,
            "position": "combined",  # Special position marker
            "rank": 0,  # N/A for combined
            "type": "final"
        })
        # Add edges from intermediate results to final
        for val in values:
            step = next(s for s in tree_steps if s["result"] == val)
            edges.append({
                "from_value": val,
                "from_position": step["position_result"],
                "from_rank": step["result_rank"],
                "to_value": final_answer,
                "to_position": "combined",
                "to_rank": 0
            })

    # Calculate tree metrics
    total_rank = sum(s["avg_operand_rank"] for s in tree_steps)
    avg_rank = total_rank / len(tree_steps) if tree_steps else 0

    # Compute explainability for each step
    step_explainability = {}
    tree_explainable = True
    if question_numbers is not None:
        step_explainability = compute_step_explainability(tree_steps, question_numbers)
        # Add explainability flag to each step
        for step in tree_steps:
            step["explainable"] = step_explainability.get(step["position_operands"], False)
        # Tree is explainable if all steps are explainable
        tree_explainable = all(step_explainability.values()) if step_explainability else True
    else:
        # No question_numbers provided - mark all as explainable by default
        for step in tree_steps:
            step["explainable"] = True

    # Add copy_from_question_steps to the tree if provided
    actual_copy_from_question = copy_from_question_steps or []

    return {
        "steps": tree_steps,
        "nodes": nodes,
        "edges": edges,
        "num_steps": len(tree_steps),
        "avg_rank": avg_rank,
        "final_answer": final_answer,
        "final_combination": final_combination,  # (v1, v2, op) or None
        "uses_direct_final_step": uses_direct_final_step,
        "copy_step": copy_step,  # Copy step info if present, or None
        "step_explainability": step_explainability,  # Dict[position_operands, bool]
        "tree_explainable": tree_explainable,  # True if all steps are explainable
        "include_question_tokens": include_question_tokens,
        "copy_from_question_steps": actual_copy_from_question,
    }


def compute_step_explainability(
    tree_steps: List[Dict],
    question_numbers: Set[int]
) -> Dict[int, bool]:
    """Compute explainability for each step in the tree.

    A step is explainable if:
    1. All leaf operands (not intermediate results) are in question_numbers
    2. All intermediate operands come from explainable steps

    Args:
        tree_steps: List of step dicts from the computation tree
        question_numbers: Set of numbers mentioned in the question

    Returns:
        Dict mapping position_operands -> is_explainable
    """
    if not tree_steps:
        return {}

    # Map from position_operands -> is_explainable
    explainability = {}

    # Process steps in position order (dependencies come first)
    # Treat "question" as -1 for sorting
    for step in sorted(tree_steps, key=lambda s: -1 if s["position_operands"] == "question" else s["position_operands"]):
        pos = step["position_operands"]
        is_explainable = True

        # Check each operand
        for op_key in ["operand1", "operand2", "operand3"]:
            if step.get(op_key) is None:
                continue

            op_val = step[op_key]
            is_ir = step.get(f"{op_key}_is_intermediate", False)
            src_pos = step.get(f"{op_key}_source_pos")
            is_qnum = step.get(f"{op_key}_is_question_number", False)

            # Question number operands are always explainable
            if is_qnum:
                continue

            if is_ir:
                # Check if this is from a question number (always explainable)
                if src_pos == "question":
                    # Question number operands are always explainable
                    continue

                # Intermediate: check if source step is explainable
                # Find the step that produced this intermediate result
                source_step = None
                for s in tree_steps:
                    if s["position_result"] == src_pos and s["result"] == op_val:
                        source_step = s
                        break

                if source_step is None:
                    # Source step not in tree - not explainable
                    is_explainable = False
                    break
                elif not explainability.get(source_step["position_operands"], False):
                    # Source step is not explainable
                    is_explainable = False
                    break
            else:
                # Leaf operand: must be in question_numbers
                if op_val not in question_numbers:
                    is_explainable = False
                    break

        explainability[pos] = is_explainable

    return explainability


def count_valid_trees(
    all_steps: Dict[int, List[Dict]],
    final_answer: int
) -> int:
    """Count how many valid computation trees exist.

    A valid tree is one where we can chain steps to produce the final answer.
    This counts all possible combinations, not just the best one.
    """
    # Find all steps that produce the final answer
    answer_steps = []
    for pos, steps in all_steps.items():
        for step in steps:
            if step["result"] == final_answer:
                answer_steps.append((pos, step))

    if not answer_steps:
        return 0

    # For each answer-producing step, count valid sub-trees
    # This is a simplified count - just count answer-producing steps for now
    # A more complete implementation would enumerate all combinations
    return len(answer_steps)


def analyze_sample(
    sample: Dict,
    sample_idx: int,
    model,
    analyzer: UnifiedAnalyzer,
    num_latent: int,
    top_k: int,
    device: str,
    include_question_tokens: bool = False,
    validate: bool = False,
    validation_n: int = 3,
    template: Optional[Dict] = None,
    verbose_validation: bool = False,
    enable_copy: bool = False,
    validation_required_passes: int = 2,
    validation_max_rank: int = 1,
    check_all_candidates: bool = False
) -> Optional[Dict]:
    """Analyze a single sample for forward chaining computation trees."""
    question = sample['question']
    gt_answer = str(sample['answer']).replace(',', '').strip()
    gt_solution = ' '.join(sample.get('steps', []))

    # Prepare input
    inputs = model.prepare_input(question, num_latents=num_latent)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run inference with activation capture
    result = analyzer.analyze_with_capture(
        input_ids,
        attention_mask,
        max_new_tokens=64,
        layer_indices=None
    )

    output = result["output"]
    output_ids = output.output_ids

    # Find key positions
    positions = find_key_positions(output_ids, model, num_latent)
    if positions is None:
        return None

    # Build position indices for vocab projection
    # For step finding: positions 0 through num_latent (inclusive)
    # This covers position pairs (0,1), (1,2), ..., (num_latent-1, num_latent)
    if model.model_type == "codi":
        # Include additional answer positions for multi-token answers
        first_answer_pos = positions['first_answer']
        position_indices = [
            positions['start_latent'],
            *[positions[f'latent_{i}'] for i in range(num_latent)],
            positions['end_latent'],
            positions['delimiter_0'],
            positions['delimiter_1'],
            positions['delimiter_2'],
            positions['delimiter_3'],
            first_answer_pos,
            first_answer_pos + 1,  # ans_1 or <|endoftext|>
            first_answer_pos + 2,  # ans_2 or <|endoftext|>
            first_answer_pos + 3,  # <|endoftext|> for longer answers
        ]
        # For CODI, the answer is PREDICTED at delimiter_3 (":") position
        # delimiter_3 is at index num_latent + 5 (after start + latents + end + 4 delims - 1)
        answer_position_in_list = num_latent + 5
    else:
        # Include additional answer positions for multi-token answers
        # Add up to 3 extra positions after first_answer (to show <|endoftext|>)
        first_answer_pos = positions['first_answer']
        position_indices = [
            positions['start_latent'],
            *[positions[f'latent_{i}'] for i in range(num_latent)],
            positions['end_latent'],
            positions['hash'],
            first_answer_pos,
            first_answer_pos + 1,  # ans_1 or <|endoftext|>
            first_answer_pos + 2,  # ans_2 or <|endoftext|>
            first_answer_pos + 3,  # <|endoftext|> for longer answers
        ]
        # For coconut, the answer is PREDICTED at the ### position
        # hash is at index num_latent + 2 (after start + latents + end)
        answer_position_in_list = num_latent + 2

    # Get hidden states from final layer
    activations = result["activations"]
    if not activations:
        return None

    final_layer_idx = max(activations.keys())
    hidden_states = activations[final_layer_idx]

    # Check bounds
    seq_len = hidden_states.shape[1]
    valid_position_indices = [p for p in position_indices if p < seq_len]

    # Need at least enough positions for step finding
    num_step_positions = num_latent + 1  # Positions 0 through num_latent
    if len(valid_position_indices) < num_step_positions + 2:  # +2 for end and answer
        return None

    # Project to vocab
    hidden_at_positions = hidden_states[0, valid_position_indices, :]
    proj_result = analyzer.project_activations_to_vocab(
        hidden_at_positions.unsqueeze(0),
        top_k=top_k,
        return_probs=False
    )
    all_logits = proj_result["logits"][0]

    # Extract top-k tokens at each position
    all_top_k_indices = all_logits.argsort(dim=-1, descending=True)[:, :top_k]
    vocab_projection_top_k = []

    for pos_idx in range(len(valid_position_indices)):
        position_tokens = []
        for rank in range(top_k):
            token_id = all_top_k_indices[pos_idx, rank].item()
            token_str = model.tokenizer.decode([token_id])
            position_tokens.append(token_str)
        vocab_projection_top_k.append(position_tokens)

    # Get the model's predicted answer (top-1 integer at answer position)
    # Use multi-token extraction for answer position (handles " 6" + "75" = 675)
    if answer_position_in_list < len(vocab_projection_top_k):
        answer_info = extract_multitoken_integer(vocab_projection_top_k, answer_position_in_list)
    else:
        answer_info = None

    if answer_info is None:
        model_answer = None
        model_answer_rank = None
        answer_token_offset = 0
    else:
        model_answer, model_answer_rank, answer_token_offset = answer_info

    # Also extract model answer from decoded output for comparison
    decoded_output = result["decoded"]
    output_metadata = result["output"].metadata or {}
    delimiter = output_metadata.get("delimiter", "###")

    if delimiter in decoded_output:
        decoded_answer = decoded_output.split(delimiter)[-1].replace(',', '').strip()
    else:
        decoded_answer = ""

    # Check correctness
    try:
        gt_float = float(gt_answer.replace(',', ''))
        if model_answer is not None:
            answer_correct = float(model_answer) == gt_float
        else:
            answer_correct = False
    except (ValueError, TypeError):
        answer_correct = False

    # Determine k_offset based on model type
    # coconut: k_offset=1 (operands at i, result at i+1)
    # codi: k_offset=2 (operands at i, result at i+2)
    k_offset = 2 if model.model_type == "codi" else 1

    # Extract numbers from the question (these should not be intermediate results)
    question_numbers = extract_numbers_from_question(question)

    # Find all steps within latent positions and up to the answer position
    # Position pairs: (0,k), (1,1+k), ..., based on k_offset
    # For coconut: num_latent+2 positions (0 to num_latent+1)
    # For codi: num_latent+5 positions to include delimiter positions ("The", "answer", "is")
    #   which can have operands producing results at the answer position (":")
    num_step_positions = num_latent + 5 if model.model_type == "codi" else num_latent + 2
    all_steps, best_steps, copy_from_question_steps, integrated_validation_info = find_all_steps(
        vocab_projection_top_k,
        num_step_positions=num_step_positions,
        top_k=top_k,
        k_offset=k_offset,
        question_numbers=question_numbers,
        include_question_tokens=include_question_tokens,
        answer_position_idx=answer_position_in_list,
        validate=validate,
        template=template,
        model=model,
        analyzer=analyzer,
        num_latent=num_latent,
        device=device,
        validation_n=validation_n,
        verbose_validation=verbose_validation,
        enable_copy=enable_copy,
        validation_required_passes=validation_required_passes,
        validation_max_rank=validation_max_rank,
        check_all_candidates=check_all_candidates
    )

    # Count total steps found
    total_steps_found = sum(len(steps) for steps in all_steps.values())
    best_steps_found = sum(1 for s in best_steps.values() if s is not None)

    # Build computation tree if we have a valid answer
    tree = None
    num_valid_trees = 0

    if model_answer is not None:
        tree = build_computation_tree(
            best_steps,
            model_answer,
            vocab_projection_top_k,
            answer_position_in_list,
            k_offset=k_offset,
            question_numbers=question_numbers,
            include_question_tokens=include_question_tokens,
            copy_from_question_steps=copy_from_question_steps,
            enable_copy=enable_copy
        )
        num_valid_trees = count_valid_trees(all_steps, model_answer)

    # Determine which steps are used vs unused
    used_positions = set()
    if tree is not None:
        used_positions = {s["position_operands"] for s in tree["steps"]}

    steps_summary = []
    for pos, step in best_steps.items():
        if step is not None:
            step_info = {**step, "used_in_tree": pos in used_positions}
            steps_summary.append(step_info)

    # Compute explainability for ALL steps (used and unused)
    all_steps_explainability = compute_step_explainability(steps_summary, question_numbers)

    # Add explainability to each step
    for step in steps_summary:
        step["explainable"] = all_steps_explainability.get(step["position_operands"], False)

    num_used_steps = len(used_positions)
    num_unused_steps = best_steps_found - num_used_steps

    # Generate position names for visualization
    position_names = generate_position_names(num_latent, model.model_type)
    actual_position_names = position_names[:len(valid_position_indices)]

    # Determine tree explainability
    tree_explainable = None
    if tree is not None:
        tree_explainable = tree.get("tree_explainable", True)

    # Determine tree verification status (all tree steps verified)
    tree_verified = None
    if tree is not None and validate:
        tree_steps = tree.get("steps", [])
        tree_verified = all(step.get("verified", False) for step in tree_steps) if tree_steps else True

    # Compare with ground truth solutions
    gt_comparison = None
    gt_steps = sample.get('steps')
    if gt_steps:
        gt_comparison = compare_all_gt_solutions(
            primary_steps=gt_steps,
            gen_solutions=sample.get('gen_solutions', []),
            found_steps=steps_summary,
            used_steps=tree['steps'] if tree else [],
            vocab_projection_top_k=vocab_projection_top_k
        )

    # Validation result - use integrated validation if available
    validation_result = None
    if validate and integrated_validation_info is not None:
        # Use integrated validation results (validation during tree building)
        validation_result = integrated_validation_info
    elif validate and tree is not None and template is not None:
        # Fallback to post-hoc validation if integrated validation wasn't run
        from experiments.forward_chaining.validation import validate_tree
        validation_result = validate_tree(
            tree=tree,
            template=template,
            model=model,
            analyzer=analyzer,
            num_latent=num_latent,
            device=device,
            n=validation_n,
            top_k=top_k
        )

    return {
        "sample_idx": sample_idx,
        "question": question,
        "question_numbers": sorted(question_numbers),  # Numbers extracted from the question
        "gt_answer": gt_answer,
        "gt_solution": gt_solution,
        "model_answer": model_answer,
        "model_answer_rank": model_answer_rank,
        "decoded_answer": decoded_answer,
        "answer_correct": answer_correct,
        "total_steps_found": total_steps_found,
        "best_steps_found": best_steps_found,
        "num_used_steps": num_used_steps,
        "num_unused_steps": num_unused_steps,
        "num_valid_trees": num_valid_trees,
        "tree_found": tree is not None,
        "tree_explainable": tree_explainable,  # True if all tree steps are explainable
        "tree_verified": tree_verified,  # True if all tree steps are verified (None if not validated)
        "gt_comparison": gt_comparison,
        "best_tree": tree,
        "all_best_steps": steps_summary,
        "vocab_projection_top_k": vocab_projection_top_k,
        "position_names": actual_position_names,
        "answer_position_idx": answer_position_in_list,
        "answer_token_offset": answer_token_offset,  # Offset from answer_position_idx to actual integer token
        "include_question_tokens": include_question_tokens,
        "copy_from_question_steps": copy_from_question_steps,
        "validation": validation_result,
    }


def generate_position_names(num_latent: int, model_type: str) -> List[str]:
    """Generate position names for visualization."""
    if model_type == "codi":
        names = ["<|bot|>"]
        names += [f"<|latent|>_{i}" for i in range(num_latent)]
        names += ["<|eot|>", "The", "answer", "is", ":", "ans_0", "ans_1", "ans_2", "ans_3"]
    else:
        names = ["<|start|>"]
        names += [f"<|latent|>_{i}" for i in range(num_latent)]
        names += ["<|end|>", "###", "ans_0", "ans_1", "ans_2", "ans_3"]
    return names


def save_verbose_validation_html(result: Dict, output_dir: str, sample_idx: int,
                                 model_type: str = "coconut", top_k: int = 10):
    """Save comprehensive validation HTML with per-candidate files.

    Creates:
    - An index file listing all positions and candidates
    - Individual HTML files for each candidate tested at each position
    """
    validation = result.get("validation", {})
    steps = validation.get("steps", [])
    question = result.get("question", "N/A")
    gt_answer = result.get("gt_answer", "N/A")
    model_answer = result.get("model_answer", "N/A")

    # Create sample subdirectory
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    # Build index HTML
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Details - Sample {sample_idx}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #333; }}
        .question {{ background: #e3f2fd; padding: 12px; border-radius: 4px; margin: 10px 0; }}
        .position-card {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .candidate {{ padding: 8px; margin: 4px 0; border-left: 4px solid #ccc; background: #fafafa; }}
        .candidate.selected {{ border-left-color: #4CAF50; background: #e8f5e9; }}
        .candidate.validated {{ border-left-color: #2196F3; }}
        .candidate.untested {{ border-left-color: #9e9e9e; background: #f0f0f0; opacity: 0.7; }}
        .verified {{ color: #4CAF50; }}
        .unverified {{ color: #f44336; }}
        .untested {{ color: #9e9e9e; font-style: italic; }}
        a {{ color: #1976D2; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .priority {{ font-family: monospace; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Details - Sample {sample_idx}</h1>
        <div class="question"><strong>Question:</strong> {question}</div>
        <div><strong>GT Answer:</strong> {gt_answer} | <strong>Model Answer:</strong> {model_answer}</div>
"""

    for vstep in steps:
        pos = vstep.get("position", "?")
        result_pos = vstep.get("result_position", "?")
        selected_step = vstep.get("step", {})
        selected_verified = vstep.get("verified", False)
        validations = vstep.get("validations", [])

        # Format selected step
        op3 = selected_step.get("operand3")
        operation = selected_step.get('operation', '')
        if op3 and len(operation) == 2:
            # 3-operand: operation like "+*" means "operand1 + operand2 * operand3"
            selected_expr = f"{selected_step.get('operand1')} {operation[0]} {selected_step.get('operand2')} {operation[1]} {op3} = {selected_step.get('result')}"
        else:
            selected_expr = f"{selected_step.get('operand1')} {operation} {selected_step.get('operand2')} = {selected_step.get('result')}"

        status_class = "verified" if selected_verified else "unverified"
        status_text = "VERIFIED" if selected_verified else "UNVERIFIED"

        index_html += f"""
        <div class="position-card">
            <h2>Position [{pos}→{result_pos}]</h2>
            <div><strong>Selected:</strong> {selected_expr} <span class="{status_class}">({status_text})</span></div>
            <h3>All Candidates Tested:</h3>
"""

        # Find all_candidates_tried in validations
        all_candidates = []
        for v in validations:
            if "all_candidates_tried" in v:
                all_candidates = v["all_candidates_tried"]
                break

        if all_candidates:
            for cand in all_candidates:
                cand_expr = cand.get("expression", "?")
                cand_priority = cand.get("worst_priority", "?")
                cand_validated = cand.get("validated")
                cand_tested = cand.get("tested", True)  # Default to True for backwards compat
                cand_order = cand.get("order", "?")

                # Check if this is the selected candidate
                is_selected = cand_expr == selected_expr

                cand_class = "candidate"
                if is_selected:
                    cand_class += " selected"
                if not cand_tested:
                    cand_class += " untested"
                elif cand_validated:
                    cand_class += " validated"

                # Status mark
                if not cand_tested:
                    v_mark = "—"
                    status_class = "untested"
                elif cand_validated:
                    v_mark = "✓"
                    status_class = "verified"
                else:
                    v_mark = "✗"
                    status_class = "unverified"

                cand_filename = f"pos{pos}_candidate{cand_order:02d}.html"

                # For untested candidates, don't link to individual HTML
                if cand_tested:
                    index_html += f"""
            <div class="{cand_class}">
                <a href="{cand_filename}">{cand_order}. {cand_expr}</a>
                <span class="priority">priority={cand_priority}</span>
                <span class="{status_class}">{v_mark}</span>
            </div>
"""
                    # Create individual candidate HTML only for tested candidates
                    save_candidate_html(
                        sample_dir, cand_filename, sample_idx, pos, result_pos,
                        cand, question, gt_answer, model_answer, is_selected,
                        model_type=model_type, top_k=top_k
                    )
                else:
                    index_html += f"""
            <div class="{cand_class}">
                <span>{cand_order}. {cand_expr}</span>
                <span class="priority">priority={cand_priority}</span>
                <span class="{status_class}">{v_mark} (not tested)</span>
            </div>
"""
        else:
            index_html += "<div>No candidate details available</div>"

        index_html += "</div>"

    index_html += """
    </div>
</body>
</html>
"""

    # Save index
    with open(os.path.join(sample_dir, "index.html"), 'w') as f:
        f.write(index_html)


def get_position_tokens_for_table(num_latent: int, model_type: str, num_positions: int) -> List[str]:
    """Generate output token names for vocab projection table columns."""
    if model_type == "codi":
        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|eot|>", "The", "answer", "is", ":"]
        num_extra = max(0, num_positions - len(output_tokens))
        output_tokens += [f"ans_{i}" for i in range(num_extra)]
    else:  # coconut
        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|end|>", "###"]
        num_extra = max(0, num_positions - len(output_tokens))
        output_tokens += [f"ans_{i}" for i in range(num_extra)]

    return output_tokens[:num_positions]


def render_vocab_projection_table_html(
    vocab_projection: List[List[str]],
    num_latent: int,
    model_type: str,
    highlight_position: int = None,
    expected_value: int = None,
    observed_value: int = None,
    top_k: int = 10
) -> str:
    """Render a vocab projection table as HTML for validation checks.

    Args:
        vocab_projection: List of lists of top-k tokens at each position
        num_latent: Number of latent tokens
        model_type: Model type ("coconut" or "codi")
        highlight_position: Position index to highlight (validation position)
        expected_value: Expected integer value at the highlight position
        observed_value: Observed integer value at the highlight position
        top_k: Number of top tokens to show per position

    Returns:
        HTML string for the vocab projection table
    """
    if not vocab_projection:
        return "<div>No vocab projection data available</div>"

    num_positions = len(vocab_projection)
    output_tokens = get_position_tokens_for_table(num_latent, model_type, num_positions)

    def extract_number_from_token(token: str):
        """Extract integer value from a token string, or None if not a number."""
        cleaned = token.replace('\u0120', ' ').strip()
        try:
            return int(cleaned)
        except ValueError:
            return None

    html = """
            <div class="vocab-table-container">
                <table class="vocab-projection-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
"""

    # Add column headers
    for pos_idx, out_token in enumerate(output_tokens):
        out_token_escaped = out_token.replace('<', '&lt;').replace('>', '&gt;')
        highlight_class = ' class="highlight-col"' if pos_idx == highlight_position else ''
        html += f'                            <th{highlight_class}>{out_token_escaped}</th>\n'

    html += """                        </tr>
                    </thead>
                    <tbody>
"""

    # Add rows for each rank
    for rank in range(min(top_k, max(len(tokens) for tokens in vocab_projection) if vocab_projection else 0)):
        html += f"                        <tr>\n"
        html += f'                            <td class="rank-col">{rank + 1}</td>\n'

        for pos_idx, tokens in enumerate(vocab_projection):
            if rank < len(tokens):
                token = tokens[rank]
                token_display = token.replace('<', '&lt;').replace('>', '&gt;')
                int_val = extract_number_from_token(token)

                # Determine cell styling
                cell_classes = []
                if pos_idx == highlight_position:
                    cell_classes.append("highlight-col")
                    # Check if this is expected or observed value
                    if int_val is not None:
                        if int_val == expected_value and rank == 0:
                            cell_classes.append("expected-match")
                        elif int_val == observed_value and rank == 0:
                            if observed_value != expected_value:
                                cell_classes.append("observed-mismatch")
                        elif int_val == expected_value:
                            cell_classes.append("expected-in-list")

                if int_val is not None:
                    cell_classes.append("integer-cell")

                class_str = f' class="{" ".join(cell_classes)}"' if cell_classes else ''
                html += f'                            <td{class_str}>{token_display}</td>\n'
            else:
                html += f'                            <td>-</td>\n'

        html += f"                        </tr>\n"

    html += """                    </tbody>
                </table>
            </div>
"""

    return html


def save_candidate_html(
    output_dir: str, filename: str, sample_idx: int, pos: int, result_pos: int,
    candidate: Dict, question: str, gt_answer: str, model_answer: str, is_selected: bool,
    model_type: str = "coconut", top_k: int = 10
):
    """Save HTML for a single candidate validation."""
    expr = candidate.get("expression", "?")
    priority = candidate.get("worst_priority", "?")
    op_priorities = candidate.get("operand_priorities", {})
    validated = candidate.get("validated", False)
    validation_details = candidate.get("validation_details", [])
    step = candidate.get("step", {})

    status_class = "verified" if validated else "unverified"
    status_text = "VERIFIED" if validated else "UNVERIFIED"
    selected_text = " (SELECTED)" if is_selected else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Candidate {candidate.get('order', '?')} - Position [{pos}→{result_pos}]</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .question {{ background: #e3f2fd; padding: 12px; border-radius: 4px; margin: 10px 0; font-size: 13px; }}
        .step-card {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .verified {{ color: #4CAF50; }}
        .unverified {{ color: #f44336; }}
        .selected {{ background: #e8f5e9; padding: 4px 8px; border-radius: 4px; }}
        .modification {{ background: #fff3e0; padding: 12px; border-radius: 4px; margin: 8px 0; font-family: monospace; font-size: 12px; }}
        .mod-header {{ font-weight: bold; margin-bottom: 8px; }}
        .expected {{ color: #2196F3; }}
        .observed {{ color: #9C27B0; }}
        .match {{ color: #4CAF50; font-weight: bold; }}
        .no-match {{ color: #f44336; font-weight: bold; }}
        .operand-info {{ background: #f5f5f5; padding: 12px; border-radius: 4px; margin: 8px 0; }}
        .priority {{ font-family: monospace; background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }}
        a {{ color: #1976D2; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        /* Vocab projection table styles */
        .vocab-table-container {{ overflow-x: auto; margin: 12px 0; }}
        .vocab-projection-table {{ border-collapse: separate; border-spacing: 2px; width: auto; }}
        .vocab-projection-table th {{ background-color: #555; color: white; padding: 6px 10px; text-align: center; font-size: 10px; white-space: nowrap; }}
        .vocab-projection-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: center; font-family: monospace; font-size: 11px; background: #fff; }}
        .vocab-projection-table .rank-col {{ background-color: #f5f5f5; font-weight: bold; width: 40px; }}
        .vocab-projection-table .highlight-col {{ background-color: #fff3e0 !important; }}
        .vocab-projection-table th.highlight-col {{ background-color: #ff9800 !important; }}
        .vocab-projection-table .expected-match {{ background-color: #c8e6c9 !important; font-weight: bold; }}
        .vocab-projection-table .observed-mismatch {{ background-color: #ffcdd2 !important; font-weight: bold; }}
        .vocab-projection-table .expected-in-list {{ background-color: #e8f5e9 !important; }}
        .vocab-projection-table .integer-cell {{ background-color: #fff; }}
    </style>
</head>
<body>
    <div class="container">
        <p><a href="index.html">← Back to all candidates</a></p>
        <h1>[{pos}→{result_pos}] {expr} <span class="{status_class}">({status_text})</span>{f'<span class="selected">{selected_text}</span>' if is_selected else ''}</h1>

        <div class="question"><strong>Question:</strong> {question}</div>

        <div class="step-card">
            <h2>Priority Analysis</h2>
            <div><strong>Worst Priority:</strong> <span class="priority">{priority}</span></div>
            <table>
                <tr><th>Operand</th><th>Value</th><th>Source</th><th>Priority</th></tr>
"""

    # Add operand details
    for op_key in ["operand1", "operand2", "operand3"]:
        if step.get(op_key) is not None:
            op_val = step.get(op_key)
            op_src_raw = candidate.get(f"{op_key}_source", step.get(f"{op_key}_source_pos", "?"))
            # Format source as "latent position X" for numeric positions
            if isinstance(op_src_raw, int):
                op_src = f"latent position {op_src_raw}"
            elif op_src_raw == "question":
                op_src = "question"
            else:
                op_src = str(op_src_raw)
            op_pri = op_priorities.get(op_key, "?")
            html += f"<tr><td>{op_key}</td><td>{op_val}</td><td>{op_src}</td><td><span class='priority'>{op_pri}</span></td></tr>\n"

    html += """
            </table>
        </div>

        <div class="step-card">
            <h2>Validation Checks</h2>
"""

    # Show validation details
    if validation_details:
        for v in validation_details:
            if "status" in v and v["status"] == "trivially_verified":
                reason = v.get("reason", "trivially verified")
                html += f'<div class="modification"><span class="verified">Trivially verified: {reason}</span></div>\n'
            elif "error" in v:
                error_type = v["error"]
                if error_type == "maxed_out_tries":
                    reason = v.get("reason", "unable to be verified -- maxed out tries")
                    found = v.get("found", 0)
                    needed = v.get("needed", 3)
                    modifiable_vars = v.get("modifiable_vars", [])
                    rejections_by_var = v.get("rejections_by_var", {})

                    html += f'<div class="modification"><span class="unverified">{reason}</span>'
                    html += f'<br>Found {found}/{needed} valid modifications'
                    if modifiable_vars:
                        html += f'<br>Modifiable vars: {", ".join(modifiable_vars)}'

                    # Show rejection details for each variable
                    if rejections_by_var:
                        html += '<br><br><strong>Candidates tried:</strong>'
                        for var, rejections in rejections_by_var.items():
                            if rejections:
                                html += f'<br><em>{var}:</em><ul style="margin:2px 0;">'
                                for rej in rejections[:10]:  # Limit to first 10 per var
                                    val = rej.get("value", "?")
                                    rej_reason = rej.get("reason", "?")
                                    expected = rej.get("expected_result")
                                    strategy = rej.get("strategy", "?")
                                    reason_display = {
                                        "result_multi_token": f"result {expected} is multi-token",
                                        "result_unchanged": f"result unchanged ({expected})",
                                        "operand_multi_token": "operand is multi-token",
                                        "non_positive": "value <= 0",
                                        "cannot_compute_result": "cannot compute result",
                                    }.get(rej_reason, rej_reason)
                                    html += f'<li>{val} ({strategy}): {reason_display}</li>'
                                if len(rejections) > 10:
                                    html += f'<li>... and {len(rejections) - 10} more</li>'
                                html += '</ul>'
                    html += '</div>\n'
                elif error_type == "no_traceable_operands":
                    html += f'<div class="modification"><span class="unverified">No traceable operands - cannot validate</span></div>\n'
                else:
                    html += f'<div class="modification"><span class="unverified">Error: {error_type}</span></div>\n'
                if "operand_sources" in v:
                    html += '<div class="operand-info"><strong>Operand Sources:</strong><br>'
                    for op_key, src in v.get("operand_sources", {}).items():
                        src_type = src.get("type", "unknown")
                        src_val = src.get("value", "?")
                        src_var = src.get("var", "")
                        if src_type == "question_var":
                            html += f"  {op_key}: {src_val} ({src_var}) - question variable<br>"
                        elif src_type == "intermediate":
                            html += f"  {op_key}: {src_val} - intermediate result<br>"
                        else:
                            html += f"  {op_key}: {src_val} - <span class='unverified'>{src_type.upper()}</span><br>"
                    html += "</div>\n"
            elif "var_modified" in v:
                var_name = v.get("var_modified", "?")
                original = v.get("original_value", "?")
                modified = v.get("new_value", "?")
                expected = v.get("expected_result", "?")
                observed = v.get("observed_result", "?")
                matched = v.get("passed", False)

                match_class = "match" if matched else "no-match"
                match_text = "MATCHED" if matched else "DID NOT MATCH"

                html += f"""
            <div class="modification">
                <div class="mod-header">Modification: {var_name} = {original} → {modified}</div>
                <div><span class="expected">Expected result:</span> {expected}</div>
                <div><span class="observed">Observed result:</span> {observed}</div>
                <div class="{match_class}">{match_text}</div>
"""
                # Add vocab projection table if available
                vocab_projection = v.get("vocab_projection_top_k")
                if vocab_projection:
                    step_result_pos = v.get("step_result_position")
                    v_num_latent = v.get("num_latent", 6)
                    # Convert expected/observed to int if possible for comparison
                    exp_int = int(expected) if isinstance(expected, (int, float)) or (isinstance(expected, str) and expected.isdigit()) else None
                    obs_int = int(observed) if isinstance(observed, (int, float)) or (isinstance(observed, str) and str(observed).lstrip('-').isdigit()) else None
                    html += render_vocab_projection_table_html(
                        vocab_projection,
                        num_latent=v_num_latent,
                        model_type=model_type,
                        highlight_position=step_result_pos,
                        expected_value=exp_int,
                        observed_value=obs_int,
                        top_k=top_k
                    )
                html += """            </div>
"""
    else:
        html += "<div>No validation details available</div>"

    html += """
        </div>
    </div>
</body>
</html>
"""

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(html)


def save_validation_details_html(result: Dict, output_dir: str, sample_idx: int, verbose: bool = False,
                                 model_type: str = "coconut", top_k: int = 10):
    """Save detailed validation info as HTML for a single sample.

    When verbose=True, creates individual HTML files for each candidate tested.
    """
    validation = result.get("validation", {})
    steps = validation.get("steps", [])

    if not steps:
        return

    question = result.get("question", "N/A")
    gt_answer = result.get("gt_answer", "N/A")
    model_answer = result.get("model_answer", "N/A")

    # If verbose, create per-candidate HTML files
    if verbose:
        save_verbose_validation_html(result, output_dir, sample_idx,
                                     model_type=model_type, top_k=top_k)
        return

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Details - Sample {sample_idx}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .question {{ background: #e3f2fd; padding: 12px; border-radius: 4px; margin: 10px 0; }}
        .step-card {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step-header {{ font-size: 18px; font-weight: bold; margin-bottom: 12px; }}
        .verified {{ color: #4CAF50; }}
        .unverified {{ color: #f44336; }}
        .unexplainable {{ color: #9e9e9e; }}
        .modification {{ background: #fff3e0; padding: 12px; border-radius: 4px; margin: 8px 0; font-family: monospace; font-size: 12px; }}
        .mod-header {{ font-weight: bold; margin-bottom: 8px; }}
        .expected {{ color: #2196F3; }}
        .observed {{ color: #9C27B0; }}
        .match {{ color: #4CAF50; font-weight: bold; }}
        .no-match {{ color: #f44336; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .operand-sources {{ background: #f5f5f5; padding: 8px; border-radius: 4px; margin: 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Details - Sample {sample_idx}</h1>

        <div class="question">
            <strong>Question:</strong> {question}
        </div>

        <div>
            <strong>GT Answer:</strong> {gt_answer} |
            <strong>Model Answer:</strong> {model_answer}
        </div>
"""

    for vstep in steps:
        pos = vstep.get("position", "?")
        result_pos = vstep.get("result_position", "?")
        step = vstep.get("step", {})
        verified = vstep.get("verified", False)
        candidates = vstep.get("candidates_tried", 0)
        validations = vstep.get("validations", [])

        # Format step expression
        op1, op2 = step.get("operand1", "?"), step.get("operand2", "?")
        op3 = step.get("operand3")
        operation = step.get("operation", "?")
        step_result = step.get("result", "?")

        if op3 and len(operation) == 2:
            # 3-operand: operation like "+*" means "operand1 + operand2 * operand3"
            expr = f"{op1} {operation[0]} {op2} {operation[1]} {op3} = {step_result}"
        else:
            expr = f"{op1} {operation} {op2} = {step_result}"

        status_class = "verified" if verified else "unverified"
        status_text = "VERIFIED" if verified else "UNVERIFIED"

        html += f"""
        <div class="step-card">
            <div class="step-header">
                [{pos}→{result_pos}] {expr}
                <span class="{status_class}">({status_text})</span>
            </div>
            <div>Candidates tested: {candidates}</div>
"""

        # Show operand sources
        if validations:
            for v in validations:
                if "operand_sources" in v:
                    html += '<div class="operand-sources"><strong>Operand Sources:</strong><br>'
                    for op_key, src in v.get("operand_sources", {}).items():
                        src_type = src.get("type", "unknown")
                        src_val = src.get("value", "?")
                        src_var = src.get("var", "")
                        if src_type == "question_var":
                            html += f"  {op_key}: {src_val} ({src_var}) - question variable<br>"
                        elif src_type == "domain_constant":
                            html += f"  {op_key}: {src_val} - domain constant<br>"
                        elif src_type == "intermediate":
                            html += f"  {op_key}: {src_val} - intermediate result<br>"
                        else:
                            html += f"  {op_key}: {src_val} - <span class='unverified'>UNKNOWN</span><br>"
                    html += "</div>"

                if "error" in v:
                    error_type = v["error"]
                    if error_type == "maxed_out_tries":
                        reason = v.get("reason", "unable to be verified -- maxed out tries")
                        found = v.get("found", 0)
                        needed = v.get("needed", 3)
                        modifiable_vars = v.get("modifiable_vars", [])
                        rejections_by_var = v.get("rejections_by_var", {})

                        html += f'<div class="modification"><span class="unverified">{reason}</span>'
                        html += f'<br>Found {found}/{needed} valid modifications'
                        if modifiable_vars:
                            html += f'<br>Modifiable vars: {", ".join(modifiable_vars)}'

                        # Show rejection details for each variable
                        if rejections_by_var:
                            html += '<br><br><strong>Candidates tried:</strong>'
                            for var, rejections in rejections_by_var.items():
                                if rejections:
                                    html += f'<br><em>{var}:</em><ul style="margin:2px 0;">'
                                    for rej in rejections[:10]:  # Limit to first 10 per var
                                        val = rej.get("value", "?")
                                        rej_reason = rej.get("reason", "?")
                                        expected = rej.get("expected_result")
                                        strategy = rej.get("strategy", "?")
                                        reason_display = {
                                            "result_multi_token": f"result {expected} is multi-token",
                                            "result_unchanged": f"result unchanged ({expected})",
                                            "operand_multi_token": "operand is multi-token",
                                            "non_positive": "value <= 0",
                                            "cannot_compute_result": "cannot compute result",
                                        }.get(rej_reason, rej_reason)
                                        html += f'<li>{val} ({strategy}): {reason_display}</li>'
                                    if len(rejections) > 10:
                                        html += f'<li>... and {len(rejections) - 10} more</li>'
                                    html += '</ul>'
                        html += '</div>'
                    else:
                        html += f'<div class="modification"><span class="unverified">Error: {error_type}</span></div>'

        # Show prompt modifications
        for v in validations:
            # Handle both formats: nested "modifications" or direct modification fields
            if "modifications" in v:
                mods = v.get("modifications", [])
            elif "var_modified" in v:
                # Direct format from run_validation_checks
                mods = [v]
            else:
                mods = []

            for mod in mods:
                var_name = mod.get("var_modified", mod.get("variable", "?"))
                original = mod.get("original_value", "?")
                modified = mod.get("new_value", mod.get("modified_value", "?"))
                expected = mod.get("expected_result", "?")
                observed = mod.get("observed_result", "?")
                matched = mod.get("passed", mod.get("result_matched", False))

                match_class = "match" if matched else "no-match"
                match_text = "MATCHED" if matched else "DID NOT MATCH"

                html += f"""
            <div class="modification">
                <div class="mod-header">Modification: {var_name} = {original} → {modified}</div>
                <div><span class="expected">Expected result:</span> {expected}</div>
                <div><span class="observed">Observed result:</span> {observed}</div>
                <div class="{match_class}">{match_text}</div>
            </div>
"""

        html += "</div>"

    html += """
    </div>
</body>
</html>
"""

    output_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_validation.html")
    with open(output_path, 'w') as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Forward Chaining Experiment")
    parser.add_argument("--model_type", type=str, required=True, choices=["coconut", "codi"],
                        help="Model type")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset JSON")
    parser.add_argument("--output_dir", type=str, default="results/forward_chaining",
                        help="Output directory")
    parser.add_argument("--num_latent", type=int, default=6,
                        help="Number of latent tokens")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k tokens to consider for operands")
    parser.add_argument("--base_llm", type=str, default="gpt2",
                        help="Base LLM name for output filenames")
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2",
                        help="Base model ID for loading the model (e.g., 'openai-community/gpt2' or 'meta-llama/Llama-3.2-1B-Instruct')")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=None,
                        help="Specific sample indices to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--include_question_tokens", action="store_true",
                        help="Allow question numbers as operands (lowest priority)")
    parser.add_argument("--validate", action="store_true",
                        help="Enable step validation")
    parser.add_argument("--validation_n", type=int, default=3,
                        help="Number of prompt modifications per step for validation")
    parser.add_argument("--validation_required_passes", type=int, default=2,
                        help="Minimum number of validation checks that must pass (default: 2)")
    parser.add_argument("--validation_max_rank", type=int, default=1,
                        help="Max rank for expected result in integer tokens. 1=top-1 only, 2=top-1 or top-2, etc. (default: 1)")
    parser.add_argument("--templates_path", type=str, default="data/gsm_templates.json",
                        help="Path to templates JSON (required for validation)")
    parser.add_argument("--save_validation_details", action="store_true",
                        help="Save detailed validation info (prompt modifications, expected/observed results)")
    parser.add_argument("--verbose_validation", action="store_true",
                        help="Store all candidates tried at each position (for debugging)")
    parser.add_argument("--check_all_candidates", action="store_true",
                        help="Test all candidates even after finding a verified one (for debugging)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate tree visualization HTML files after analysis")
    parser.add_argument("--enable_copy", action="store_true",
                        help="Enable copy steps (copy from question, copy to answer position). Default: disabled")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading {args.model_type} model from {args.model_path}...")
    model_kwargs = {"model_id": args.model_id}
    if args.model_type == "codi":
        model_kwargs["num_latent"] = args.num_latent
    model = ModelFactory.create(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        **model_kwargs
    )
    analyzer = UnifiedAnalyzer(model)
    print("Model loaded")

    # Set the tokenizer for validation (ensures single-token checks use correct tokenizer)
    from experiments.forward_chaining.validation import set_tokenizer
    set_tokenizer(model.tokenizer)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)

    # Load templates if validation is enabled
    templates = None
    template_by_question = {}  # Map question prefix -> template for matching
    if args.validate:
        print(f"\nLoading templates from {args.templates_path}...")
        with open(args.templates_path, 'r') as f:
            templates = json.load(f)
        print(f"Loaded {len(templates)} templates")
        # Build question -> template mapping for subset datasets
        for t in templates:
            orig_q = t.get('original_question', '')
            # Use first 100 chars as key (should be unique enough)
            if orig_q:
                template_by_question[orig_q[:100]] = t

    # Determine which samples to process
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
        print(f"Processing specific samples: {sample_indices}")
    elif args.max_samples is not None:
        sample_indices = list(range(min(args.max_samples, len(dataset))))
        print(f"Processing first {len(sample_indices)} samples")
    else:
        sample_indices = list(range(len(dataset)))
        print(f"Processing all {len(dataset)} samples")

    # Process samples
    results = []
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Analyzing samples"):
            if idx >= len(dataset):
                print(f"Warning: sample index {idx} out of range")
                continue

            sample = dataset[idx]
            # Get template for this sample if validation is enabled
            # Match by question text (handles subset datasets with different indices)
            template = None
            if templates is not None:
                question = sample.get('question', '')
                question_key = question[:100]
                if question_key in template_by_question:
                    template = template_by_question[question_key]
                else:
                    raise ValueError(
                        f"No template found for sample {idx}. "
                        f"Question (first 100 chars): '{question_key}'. "
                        f"This can happen when the dataset and templates file are mismatched. "
                        f"Ensure the templates file contains entries with 'original_question' "
                        f"matching the questions in your dataset."
                    )

            result = analyze_sample(
                sample, idx, model, analyzer,
                args.num_latent, args.top_k, device,
                include_question_tokens=args.include_question_tokens,
                validate=args.validate,
                validation_n=args.validation_n,
                template=template,
                verbose_validation=args.verbose_validation,
                enable_copy=args.enable_copy,
                validation_required_passes=args.validation_required_passes,
                validation_max_rank=args.validation_max_rank,
                check_all_candidates=args.check_all_candidates
            )
            if result is not None:
                results.append(result)

    print(f"\nSuccessfully analyzed {len(results)}/{len(sample_indices)} samples")

    # Create output directory
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    question_token_suffix = "_yes-question-tokens" if args.include_question_tokens else "_no-question-tokens"
    # Include validation config in output subdir name if validation is enabled
    validation_suffix = ""
    if args.validate:
        validation_suffix = f"_rp{args.validation_required_passes}_mr{args.validation_max_rank}"
    output_subdir = f"{args.model_type}_{args.base_llm}_{dataset_name}{question_token_suffix}{validation_suffix}"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    # Save results JSON
    output_data = {
        "metadata": {
            "model_type": args.model_type,
            "model_path": args.model_path,
            "base_llm": args.base_llm,
            "dataset_path": args.dataset_path,
            "num_latent": args.num_latent,
            "top_k": args.top_k,
            "total_samples": len(results),
            "seed": args.seed,
            "include_question_tokens": args.include_question_tokens,
            "validate": args.validate,
            "validation_n": args.validation_n if args.validate else None,
            "validation_required_passes": args.validation_required_passes if args.validate else None,
            "validation_max_rank": args.validation_max_rank if args.validate else None,
            "templates_path": args.templates_path if args.validate else None
        },
        "per_sample": results
    }

    results_path = os.path.join(output_path, "results.json")
    with open(results_path, 'w') as f:
        json.dump(make_json_serializable(output_data), f, indent=2)
    print(f"Results saved to: {results_path}")

    # Save validation details if requested
    if args.save_validation_details and args.validate:
        validation_dir = os.path.join(output_path, "validation_details")
        os.makedirs(validation_dir, exist_ok=True)
        for result in results:
            if result.get("validation") and result["validation"].get("steps"):
                sample_idx = result["sample_idx"]
                save_validation_details_html(result, validation_dir, sample_idx, verbose=args.verbose_validation,
                                             model_type=args.model_type, top_k=args.top_k)
        print(f"Validation details saved to: {validation_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r["answer_correct"])
    trees_found = sum(1 for r in results if r["tree_found"])
    trees_correct = sum(1 for r in results if r["tree_found"] and r["answer_correct"])

    print(f"Total samples: {total}")
    print(f"Correct answers: {correct} ({100*correct/total:.1f}%)" if total > 0 else "No samples")
    print(f"Trees found: {trees_found} ({100*trees_found/total:.1f}%)" if total > 0 else "")
    print(f"Trees found (correct only): {trees_correct}")

    # Stats on number of steps
    if results:
        avg_steps = sum(r["best_steps_found"] for r in results) / len(results)
        avg_tree_steps = sum(r["best_tree"]["num_steps"] for r in results if r["tree_found"]) / max(1, trees_found)
        print(f"Average best steps per sample: {avg_steps:.2f}")
        print(f"Average steps in found trees: {avg_tree_steps:.2f}")

    # Validation statistics
    if args.validate and results:
        validated_results = [r for r in results if r.get("validation") is not None]
        if validated_results:
            print("\n" + "-" * 40)
            print("VALIDATION SUMMARY")
            print("-" * 40)

            # Use tree_verified at result level (correctly tracks only tree steps)
            # Falls back to validation dict for backward compatibility
            def is_verified(r):
                # First check result-level tree_verified (correct for tree steps)
                if r.get("tree_verified") is not None:
                    return r["tree_verified"]
                # Fallback to validation dict
                v = r.get("validation", {})
                if "tree_verified" in v:
                    return v["tree_verified"]
                return v.get("tree_status") == "verified"

            def is_unverified(r):
                if r.get("tree_verified") is not None:
                    return not r["tree_verified"]
                v = r.get("validation", {})
                if "tree_verified" in v:
                    return not v["tree_verified"]
                return v.get("tree_status") == "unverified"

            def is_unexplainable(r):
                v = r.get("validation", {})
                if "tree_verified" in v:
                    return False  # Integrated validation doesn't track unexplainable separately
                return v.get("tree_status") == "unexplainable"

            trees_verified = sum(1 for r in validated_results if is_verified(r))
            trees_unverified = sum(1 for r in validated_results if is_unverified(r))
            trees_unexplainable = sum(1 for r in validated_results if is_unexplainable(r))

            total_validated = len(validated_results)
            print(f"Trees validated: {total_validated}")
            print(f"  Verified: {trees_verified} ({100*trees_verified/total_validated:.1f}%)" if total_validated > 0 else "")
            print(f"  Unverified: {trees_unverified} ({100*trees_unverified/total_validated:.1f}%)" if total_validated > 0 else "")
            print(f"  Unexplainable: {trees_unexplainable} ({100*trees_unexplainable/total_validated:.1f}%)" if total_validated > 0 else "")

    # Generate tree visualizations if requested
    if args.visualize:
        import subprocess
        viz_script = os.path.join(os.path.dirname(__file__), "visualize.py")
        viz_cmd = [
            "python", viz_script,
            "--results_json", results_path,
            "--output_dir", output_path
        ]
        if args.sample_indices:
            viz_cmd.extend(["--sample_indices"] + [str(i) for i in args.sample_indices])
        print(f"\nGenerating visualizations...")
        subprocess.run(viz_cmd)


if __name__ == "__main__":
    main()
