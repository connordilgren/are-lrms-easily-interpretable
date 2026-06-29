#!/usr/bin/env python3
"""
Step Validation for Forward Chaining Experiment

Validates discovered computation steps by modifying question operands and
checking if model intermediate results change as expected.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import torch
from transformers import GPT2Tokenizer

# Import from preprocessing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing.gsm_constants import (
    safe_eval, DOMAIN_CONSTANTS,
    DIVISOR_WORDS, MULTIPLIER_WORDS, COMPOUND_FRACTIONS,
    NUM_TO_DIVISOR_WORD, NUM_TO_MULTIPLIER_WORD, FRACTION_TO_WORD
)

# Import from run.py
from experiments.forward_chaining.run import (
    find_key_positions, get_top1_integer, extract_integers_from_topk,
    extract_multitoken_integer
)


# Global tokenizer instance (lazy loaded)
_tokenizer = None
_tokenizer_id = None  # Track which tokenizer is cached


def set_tokenizer(tokenizer):
    """Set the tokenizer to use for single-token checks.

    This should be called before validation to ensure the correct tokenizer
    is used for the model being validated (e.g., Llama vs GPT-2).
    """
    global _tokenizer, _tokenizer_id, _SINGLE_TOKEN_INTEGERS, _SINGLE_TOKEN_INTEGERS_ID
    _tokenizer = tokenizer
    _tokenizer_id = id(tokenizer)
    # Invalidate cache if tokenizer changed
    if _SINGLE_TOKEN_INTEGERS_ID != _tokenizer_id:
        _SINGLE_TOKEN_INTEGERS = None


def get_tokenizer():
    """Get the current tokenizer (defaults to GPT-2 for backward compatibility)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return _tokenizer


def is_single_token(value: float) -> bool:
    """
    Check if a numeric value is a single token when converted to string.
    Handles both integers and floats.
    """
    tok = get_tokenizer()

    # Format the number as it would appear in steps
    if isinstance(value, float) and value == int(value):
        num_str = str(int(value))
    else:
        num_str = str(value)

    tokens = tok.encode(num_str, add_special_tokens=False)
    return len(tokens) == 1


# Pre-compute single-token integers for common range
_SINGLE_TOKEN_INTEGERS = None
_SINGLE_TOKEN_INTEGERS_ID = None  # Track which tokenizer the cache is for


def get_cached_single_token_integers() -> List[int]:
    """Get cached list of single-token integers 1-1000.

    The cache is invalidated when the tokenizer changes (via set_tokenizer).
    """
    global _SINGLE_TOKEN_INTEGERS, _SINGLE_TOKEN_INTEGERS_ID, _tokenizer_id
    # Check if cache needs to be invalidated due to tokenizer change
    if _SINGLE_TOKEN_INTEGERS is not None and _SINGLE_TOKEN_INTEGERS_ID != _tokenizer_id:
        _SINGLE_TOKEN_INTEGERS = None
    if _SINGLE_TOKEN_INTEGERS is None:
        tok = get_tokenizer()
        _SINGLE_TOKEN_INTEGERS = []
        for n in range(1, 1001):
            if len(tok.encode(str(n), add_special_tokens=False)) == 1:
                _SINGLE_TOKEN_INTEGERS.append(n)
        _SINGLE_TOKEN_INTEGERS_ID = _tokenizer_id
    return _SINGLE_TOKEN_INTEGERS


def trace_operand_sources(
    step: Dict,
    tree_steps: List[Dict],
    template: Dict
) -> Dict[str, Dict]:
    """
    Recursively trace operands to their sources.

    Args:
        step: The step to trace operands for
        tree_steps: All steps in the computation tree
        template: The template with variable mappings

    Returns:
        Dict mapping operand key (operand1, operand2, operand3) to source info:
        - type: "question_var", "domain_constant", "intermediate", or "unknown"
        - var: Variable name (e.g., "VAR_1") if question_var
        - value: The operand value
        - source_step: Source step dict if intermediate
        - leaf_sources: List of leaf sources for intermediates (recursive)
    """
    operand_sources = {}
    variables = template.get('variables', {})

    # Build reverse mapping: value -> list of variable names
    value_to_vars = {}
    for var_name, var_val in variables.items():
        val_key = int(var_val) if isinstance(var_val, float) and var_val == int(var_val) else var_val
        if val_key not in value_to_vars:
            value_to_vars[val_key] = []
        value_to_vars[val_key].append(var_name)

    # Process each operand
    for op_key in ["operand1", "operand2", "operand3"]:
        if step.get(op_key) is None:
            continue

        op_val = step[op_key]
        is_ir = step.get(f"{op_key}_is_intermediate", False)
        src_pos = step.get(f"{op_key}_source_pos")

        # Check if it's from the question
        if src_pos == "question" or step.get(f"{op_key}_is_question_number", False):
            # Find which variable this maps to
            val_key = int(op_val) if isinstance(op_val, float) and op_val == int(op_val) else op_val
            if val_key in value_to_vars:
                operand_sources[op_key] = {
                    "type": "question_var",
                    "var": value_to_vars[val_key][0],  # Use first matching var
                    "value": op_val
                }
            elif op_val in DOMAIN_CONSTANTS:
                operand_sources[op_key] = {
                    "type": "domain_constant",
                    "value": op_val
                }
            else:
                operand_sources[op_key] = {
                    "type": "unknown",
                    "value": op_val
                }
        elif is_ir and src_pos is not None and src_pos != "question":
            # Find the source step that produced this intermediate
            source_step = None
            for s in tree_steps:
                if s["position_result"] == src_pos and s["result"] == op_val:
                    source_step = s
                    break

            if source_step:
                # Recursively trace the source step's operands
                leaf_sources = trace_operand_sources(source_step, tree_steps, template)
                operand_sources[op_key] = {
                    "type": "intermediate",
                    "value": op_val,
                    "source_step": source_step,
                    "leaf_sources": leaf_sources
                }
            else:
                operand_sources[op_key] = {
                    "type": "unknown",
                    "value": op_val
                }
        else:
            # Leaf operand from vocab projection - check if it's a question number
            val_key = int(op_val) if isinstance(op_val, float) and op_val == int(op_val) else op_val
            if val_key in value_to_vars:
                operand_sources[op_key] = {
                    "type": "question_var",
                    "var": value_to_vars[val_key][0],
                    "value": op_val
                }
            elif op_val in DOMAIN_CONSTANTS:
                operand_sources[op_key] = {
                    "type": "domain_constant",
                    "value": op_val
                }
            else:
                operand_sources[op_key] = {
                    "type": "unknown",
                    "value": op_val
                }

    return operand_sources


def get_all_leaf_vars(operand_sources: Dict[str, Dict]) -> Set[str]:
    """
    Get all leaf question variables from operand sources (recursive).

    Args:
        operand_sources: Dict from trace_operand_sources

    Returns:
        Set of variable names (e.g., {"VAR_1", "VAR_2"})
    """
    leaf_vars = set()

    for op_key, source_info in operand_sources.items():
        if source_info["type"] == "question_var":
            leaf_vars.add(source_info["var"])
        elif source_info["type"] == "intermediate":
            # Recursively get leaf vars from the source step
            nested_sources = source_info.get("leaf_sources", {})
            leaf_vars.update(get_all_leaf_vars(nested_sources))

    return leaf_vars


def is_step_fully_explainable(operand_sources: Dict[str, Dict]) -> bool:
    """
    Check if a step is fully explainable (no unknown operand sources).

    Args:
        operand_sources: Dict from trace_operand_sources

    Returns:
        True if all operands trace to question vars or domain constants
    """
    for source_info in operand_sources.values():
        if source_info["type"] == "unknown":
            return False
        elif source_info["type"] == "intermediate":
            # Recursively check the intermediate's sources
            if not is_step_fully_explainable(source_info.get("leaf_sources", {})):
                return False

    return True


def has_any_traceable_operands(operand_sources: Dict[str, Dict]) -> bool:
    """Check if at least one operand can be traced to question variables."""
    for source_info in operand_sources.values():
        if source_info["type"] == "question_var":
            return True
        elif source_info["type"] == "intermediate":
            if has_any_traceable_operands(source_info.get("leaf_sources", {})):
                return True
    return False


def get_word_replacement_candidates(var_metadata: Dict) -> Optional[List[str]]:
    """
    Get valid replacement words based on word type.

    Args:
        var_metadata: Dict with 'word_type' and 'original_text' keys

    Returns:
        List of replacement word strings, or None for regular numbers
    """
    word_type = var_metadata.get("word_type")
    original_word = var_metadata.get("original_text")

    if word_type == "divisor":
        return [w for w in DIVISOR_WORDS if w != original_word]
    elif word_type == "multiplier":
        orig_value = MULTIPLIER_WORDS[original_word]
        # Only swap to multipliers with different value (twice<->triple, not twice<->double)
        return [w for w, v in MULTIPLIER_WORDS.items() if v != orig_value]
    elif word_type == "compound_fraction":
        return [w for w in COMPOUND_FRACTIONS if w != original_word]

    return None  # Regular number, use default behavior


def get_number_scale(value: int) -> int:
    """
    Detect the natural 'scale' of a number for generating smart candidates.

    Returns the largest power of 10 that divides the value evenly,
    capped at 100 to avoid too-large jumps.

    Examples:
        400 -> 100 (try 300, 500, 600)
        60 -> 10 (try 50, 70, 80)
        25 -> 1 (try 24, 26, 27)
        1000 -> 100 (capped, try 900, 1100)
    """
    if value == 0:
        return 1

    abs_val = abs(value)
    scale = 1

    for power in [100, 10]:
        if abs_val >= power and abs_val % power == 0:
            scale = power
            break

    return scale


def generate_smart_candidates(
    original_value: int,
    step_operation: str,
    operand_sources: Dict,
    template: Dict,
    var_to_modify: str,
    exclude_values: Set[int],
    original_step_result: int,
    max_candidates: int = 30,
    collect_rejections: bool = False
) -> Tuple[List[int], List[Dict]]:
    """
    Generate smart candidate values for modification, prioritizing
    candidates likely to produce single-token results.

    Strategies (in order):
    1. Scale-aware neighbors (e.g., 400 -> 300, 500, 600)
    2. Round number neighbors
    3. Increment fallback (+1 through +20)

    All candidates are filtered by:
    - Must be single-token
    - Must produce different result
    - Must preserve single-token result property

    Returns:
        (valid_candidates, rejected_candidates) where rejected_candidates
        contains dicts with {value, reason, expected_result} for debugging.
    """
    candidates = []
    rejections = []
    seen = set(exclude_values)
    original_result_single_token = is_single_token(original_step_result)

    def check_candidate(val: int) -> Tuple[bool, Optional[str], Optional[int]]:
        """Returns (is_valid, rejection_reason, expected_result)"""
        if val in seen:
            return False, "already_used", None
        if val == original_value:
            return False, "same_as_original", None
        if val <= 0:
            return False, "non_positive", None
        if not is_single_token(val):
            return False, "operand_multi_token", None

        expected = compute_expected_step_result(
            step_operation, operand_sources, {var_to_modify: val}, template
        )
        if expected is None:
            return False, "cannot_compute_result", None
        if expected == original_step_result:
            return False, "result_unchanged", expected
        if original_result_single_token and not is_single_token(expected):
            return False, "result_multi_token", expected

        return True, None, expected

    def add_candidate(val: int, strategy: str):
        is_valid, reason, expected = check_candidate(val)
        if is_valid:
            candidates.append(val)
            seen.add(val)
        elif collect_rejections and reason not in ("already_used", "same_as_original"):
            rejections.append({
                "value": val,
                "reason": reason,
                "expected_result": expected,
                "strategy": strategy
            })

    # Strategy 1: Scale-aware neighbors
    scale = get_number_scale(original_value)
    for multiplier in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
        add_candidate(original_value + multiplier * scale, "scale_aware")
        if len(candidates) >= max_candidates:
            return candidates, rejections

    # Strategy 2: Round number neighbors (multiples of 10, 50, 100)
    for step in [10, 50, 100]:
        base = (original_value // step) * step
        for offset in [step, -step, 2*step, -2*step]:
            add_candidate(base + offset, "round_neighbors")
            if len(candidates) >= max_candidates:
                return candidates, rejections

    # Strategy 3: Increment fallback
    for delta in range(1, 21):
        add_candidate(original_value + delta, "increment")
        add_candidate(original_value - delta, "increment")
        if len(candidates) >= max_candidates:
            return candidates, rejections

    return candidates, rejections


def find_next_modification_value(
    template: Dict,
    var_to_modify: str,
    original_step_result: int,
    step_operation: str,
    operand_sources: Dict[str, Dict],
    exclude_values: Set[int] = None,
    collect_rejections: bool = False
) -> Union[Optional[int], Tuple[Optional[int], List[Dict]]]:
    """
    Find a valid single-token modification using smart candidate generation.
    Excludes values already used (for round-robin repeats).
    Returns None if no valid modification found within reasonable range.

    Uses multiple strategies in order:
    1. Scale-aware neighbors (e.g., 400 -> 300, 500, 600)
    2. Round number neighbors
    3. Increment fallback (+1 through +20)

    For word-based numbers (divisors, multipliers, compound fractions), returns
    the numeric value of a different word in the same category.

    Args:
        template: Template dict with variables
        var_to_modify: Variable name to modify (e.g., "VAR_1")
        original_step_result: Original result value of this step
        step_operation: Operation string from the step (e.g., "*", "+", "-", "/")
        operand_sources: Operand sources for this step
        exclude_values: Set of values to skip (already used for this variable)
        collect_rejections: If True, return (value, rejections) tuple with rejection details

    Returns:
        If collect_rejections is False: New value for the variable, or None if no valid modification found
        If collect_rejections is True: Tuple of (value_or_none, list_of_rejection_dicts)
    """
    original_value = template['variables'].get(var_to_modify)
    if original_value is None:
        if collect_rejections:
            return None, []
        return None

    original_value = int(original_value) if isinstance(original_value, float) and original_value == int(original_value) else original_value
    original_result_single_token = is_single_token(original_step_result)
    exclude_values = exclude_values or set()

    # Check if this is a word-based number
    var_metadata = template.get('var_metadata', {}).get(var_to_modify)
    if var_metadata:
        word_type = var_metadata.get('word_type')
        if word_type == 'divisor':
            # Try other divisor words
            candidates = get_word_replacement_candidates(var_metadata)
            if candidates:
                for word in candidates:
                    new_value = DIVISOR_WORDS[word]
                    if new_value in exclude_values:
                        continue
                    expected_result = compute_expected_step_result(
                        step_operation, operand_sources, {var_to_modify: new_value}, template
                    )
                    if expected_result is None or expected_result == original_step_result:
                        continue
                    if original_result_single_token and not is_single_token(expected_result):
                        continue
                    if collect_rejections:
                        return new_value, []
                    return new_value
            if collect_rejections:
                return None, []
            return None

        elif word_type == 'multiplier':
            # Try other multiplier words
            candidates = get_word_replacement_candidates(var_metadata)
            if candidates:
                for word in candidates:
                    new_value = MULTIPLIER_WORDS[word]
                    if new_value in exclude_values:
                        continue
                    expected_result = compute_expected_step_result(
                        step_operation, operand_sources, {var_to_modify: new_value}, template
                    )
                    if expected_result is None or expected_result == original_step_result:
                        continue
                    if original_result_single_token and not is_single_token(expected_result):
                        continue
                    if collect_rejections:
                        return new_value, []
                    return new_value
            if collect_rejections:
                return None, []
            return None

        elif word_type == 'compound_fraction':
            # For compound fractions, we need to modify both numerator and denominator together
            # This function only handles single variable modifications, so compound fractions
            # require special handling at the caller level
            candidates = get_word_replacement_candidates(var_metadata)
            if candidates:
                paired_var = var_metadata.get('paired_var')
                for word in candidates:
                    num, denom = COMPOUND_FRACTIONS[word]
                    if var_metadata.get('is_numerator'):
                        new_value = num
                        paired_value = denom
                    else:
                        new_value = denom
                        paired_value = num
                    if new_value in exclude_values:
                        continue
                    # Need to modify both vars for compound fraction
                    var_modifications = {var_to_modify: new_value, paired_var: paired_value}
                    expected_result = compute_expected_step_result(
                        step_operation, operand_sources, var_modifications, template
                    )
                    if expected_result is None or expected_result == original_step_result:
                        continue
                    if original_result_single_token and not is_single_token(expected_result):
                        continue
                    if collect_rejections:
                        return new_value, []
                    return new_value
            if collect_rejections:
                return None, []
            return None

    # Regular number: use smart candidate generation with multiple strategies
    candidates, rejections = generate_smart_candidates(
        original_value,
        step_operation,
        operand_sources,
        template,
        var_to_modify,
        exclude_values,
        original_step_result,
        collect_rejections=collect_rejections
    )

    result = candidates[0] if candidates else None

    if collect_rejections:
        return result, rejections
    return result


def find_validation_modifications(
    template: Dict,
    var_to_modify: str,
    original_step_result: int,
    step_operation: str,
    operand_sources: Dict[str, Dict],
    n: int = 3
) -> List[int]:
    """
    Find n valid single-token modifications for a variable that change this step's result.

    Args:
        template: Template dict with variables
        var_to_modify: Variable name to modify (e.g., "VAR_1")
        original_step_result: Original result value of this step
        step_operation: Operation string from the step (e.g., "*", "+", "-", "/")
        operand_sources: Operand sources for this step
        n: Number of modifications to find

    Returns:
        List of new values for the variable
    """
    original_value = template['variables'].get(var_to_modify)
    if original_value is None:
        return []

    original_value = int(original_value) if isinstance(original_value, float) and original_value == int(original_value) else original_value

    # Get single-token integers
    single_token_ints = get_cached_single_token_integers()

    # Check if original result was single-token
    original_result_single_token = is_single_token(original_step_result)

    modifications = []

    for candidate in single_token_ints:
        if candidate == original_value:
            continue
        if len(modifications) >= n:
            break

        # Compute expected new result with this modification
        expected_result = compute_expected_step_result(
            step_operation,
            operand_sources,
            {var_to_modify: candidate},
            template
        )

        if expected_result is None:
            continue

        # Check that result actually changes
        if expected_result == original_step_result:
            continue

        # Check single-token preservation
        if original_result_single_token and not is_single_token(expected_result):
            continue

        modifications.append(candidate)

    return modifications


def compute_expected_step_result(
    operation: str,
    operand_sources: Dict[str, Dict],
    var_modifications: Dict[str, int],
    template: Dict
) -> Optional[int]:
    """
    Compute expected step result after variable modifications.

    Args:
        operation: Operation string (e.g., "*", "+", "-/", etc.)
        operand_sources: Operand sources with type info
        var_modifications: Dict mapping VAR_n -> new value
        template: Template dict

    Returns:
        Expected integer result, or None if computation fails
    """
    # Get operand values, applying modifications
    operand_values = []

    for op_key in ["operand1", "operand2", "operand3"]:
        if op_key not in operand_sources:
            continue

        source_info = operand_sources[op_key]

        if source_info["type"] == "question_var":
            var_name = source_info["var"]
            if var_name in var_modifications:
                operand_values.append(var_modifications[var_name])
            else:
                operand_values.append(int(template['variables'][var_name]))
        elif source_info["type"] == "domain_constant":
            operand_values.append(source_info["value"])
        elif source_info["type"] == "intermediate":
            # Recursively compute the intermediate result
            source_step = source_info["source_step"]
            intermediate_result = compute_expected_step_result(
                source_step["operation"],
                source_info["leaf_sources"],
                var_modifications,
                template
            )
            if intermediate_result is None:
                return None
            operand_values.append(intermediate_result)
        elif source_info["type"] == "unknown":
            # For unknown sources, use the original value (it won't change)
            # This allows partial validation when some operands are traceable
            operand_values.append(source_info["value"])
        else:
            return None  # Unrecognized source type

    if len(operand_values) < 2:
        return None

    # Compute result based on operation
    return evaluate_operation(operation, operand_values)


def format_operation(operation: str, op1, op2, op3=None) -> str:
    """
    Format operation as readable math expression.

    Converts 2-char operation codes to proper math notation.
    E.g., "-*" with operands 120, 3, 15 becomes "120 - 3 * 15"

    Args:
        operation: Operation string (e.g., "*", "+", "-*", "*/", etc.)
        op1: First operand value
        op2: Second operand value
        op3: Third operand value (for 3-operand operations)

    Returns:
        Formatted math expression string
    """
    if operation == "-*":
        return f"{op1} - {op2} * {op3}"
    elif operation == "+*":
        return f"{op1} + {op2} * {op3}"
    elif operation == "*-":
        return f"{op1} * {op2} - {op3}"
    elif operation == "*+":
        return f"{op1} * {op2} + {op3}"
    elif operation == "*/":
        return f"{op1} * {op2} / {op3}"
    elif operation == "//":
        return f"{op1} / {op2} / {op3}"
    elif operation == "+/":
        return f"{op1} + {op2} / {op3}"
    elif operation == "-/":
        return f"{op1} - {op2} / {op3}"
    elif operation == "/+":
        return f"{op1} / {op2} + {op3}"
    elif operation == "/-":
        return f"{op1} / {op2} - {op3}"
    elif operation == "+-":
        return f"{op1} + {op2} - {op3}"
    elif operation == "--":
        return f"{op1} - {op2} - {op3}"
    elif operation == "copy" or operation == "copy_from_question":
        return f"{op1}"
    elif len(operation) == 1:
        # Single operation (e.g., "+", "-", "*", "/")
        if op3 is not None:
            # 3-operand with same operation (e.g., a + b + c)
            return f"{op1} {operation} {op2} {operation} {op3}"
        return f"{op1} {operation} {op2}"
    else:
        # Unknown operation - fallback to showing the code
        if op3 is not None:
            return f"{op1} {operation} {op2} {operation} {op3}"
        return f"{op1} {operation} {op2}"


def evaluate_operation(operation: str, operands: List[int]) -> Optional[int]:
    """
    Evaluate an operation on operands.

    Args:
        operation: Operation string (e.g., "*", "+", "-", "/", "+-", "*/", etc.)
        operands: List of operand values

    Returns:
        Integer result, or None if computation fails
    """
    if len(operands) < 2:
        return None

    a, b = operands[0], operands[1]
    c = operands[2] if len(operands) > 2 else None

    try:
        if operation == "+":
            if c is not None:
                return a + b + c
            return a + b
        elif operation == "-":
            return a - b
        elif operation == "*":
            if c is not None:
                return a * b * c
            return a * b
        elif operation == "/":
            if b == 0:
                return None
            if a % b != 0:
                return None
            return a // b
        elif operation == "+-":  # a + b - c
            if c is None:
                return None
            return a + b - c
        elif operation == "--":  # a - b - c
            if c is None:
                return None
            return a - b - c
        elif operation == "*/":  # a * b / c
            if c is None or c == 0:
                return None
            if (a * b) % c != 0:
                return None
            return (a * b) // c
        elif operation == "//":  # a / b / c
            if c is None or b == 0 or c == 0:
                return None
            if a % b != 0:
                return None
            ab = a // b
            if ab % c != 0:
                return None
            return ab // c
        elif operation == "+*":  # a + b * c
            if c is None:
                return None
            return a + b * c
        elif operation == "-*":  # a - b * c
            if c is None:
                return None
            return a - b * c
        elif operation == "*+":  # a * b + c
            if c is None:
                return None
            return a * b + c
        elif operation == "*-":  # a * b - c
            if c is None:
                return None
            return a * b - c
        elif operation == "+/":  # a + b / c
            if c is None or c == 0:
                return None
            if b % c != 0:
                return None
            return a + b // c
        elif operation == "-/":  # a - b / c
            if c is None or c == 0:
                return None
            if b % c != 0:
                return None
            return a - b // c
        elif operation == "/+":  # a / b + c
            if c is None or b == 0:
                return None
            if a % b != 0:
                return None
            return a // b + c
        elif operation == "/-":  # a / b - c
            if c is None or b == 0:
                return None
            if a % b != 0:
                return None
            return a // b - c
        elif operation == "copy":
            return a
        elif operation == "copy_from_question":
            return a
        else:
            return None
    except Exception:
        return None


def create_modified_question(template: Dict, var_modifications: Dict[str, int]) -> str:
    """
    Apply variable modifications to question using var_positions.

    For word-based numbers, converts numeric values back to their word form.

    Args:
        template: Template dict with template_question and var_positions
        var_modifications: Dict mapping VAR_n -> new value

    Returns:
        Modified question string with new values substituted
    """
    question = template['template_question']
    var_metadata = template.get('var_metadata', {})

    # Apply all variable values (original + modifications)
    all_values = dict(template['variables'])
    all_values.update(var_modifications)

    # Build word replacement lookup for modified word-based numbers
    # We need to track which compound fractions have been handled
    handled_compound_fractions = set()

    # Replace variables in question
    for var_name, value in all_values.items():
        metadata = var_metadata.get(var_name)

        if metadata:
            word_type = metadata.get('word_type')
            original_text = metadata.get('original_text')

            if word_type == 'divisor':
                # Convert numeric value back to word form
                val_str = NUM_TO_DIVISOR_WORD.get(value, str(int(value)))
            elif word_type == 'multiplier':
                # Convert numeric value back to word form
                val_str = NUM_TO_MULTIPLIER_WORD.get(value, str(int(value)))
            elif word_type == 'compound_fraction':
                # For compound fractions, we replace the word form placeholder
                # Need to find the new word that matches the num/denom values
                if original_text in handled_compound_fractions:
                    continue  # Already handled this compound fraction

                paired_var = metadata.get('paired_var')
                if metadata.get('is_numerator'):
                    num = value
                    denom = all_values.get(paired_var, value)
                else:
                    denom = value
                    num = all_values.get(paired_var, value)

                # Find the matching compound fraction word
                val_str = FRACTION_TO_WORD.get((num, denom), f"{num}/{denom}")

                # Replace the uppercase word form in template
                question = question.replace(original_text.upper(), val_str)
                handled_compound_fractions.add(original_text)
                continue  # Don't do the normal replacement
            else:
                # Format value appropriately
                if isinstance(value, float) and value == int(value):
                    val_str = str(int(value))
                else:
                    val_str = str(value)
        else:
            # Regular number: format value appropriately
            if isinstance(value, float) and value == int(value):
                val_str = str(int(value))
            else:
                val_str = str(value)

        question = question.replace(var_name, val_str)

    return question


def run_inference_and_get_result(
    question: str,
    model,
    analyzer,
    step_position: int,
    num_latent: int,
    device: str,
    top_k: int = 10
) -> Tuple[Optional[int], Optional[int], Optional[List[List[str]]]]:
    """
    Run inference on modified question, return (top1_integer, rank, vocab_projection_top_k) at step position.

    Args:
        question: Modified question string
        model: The model to run inference with
        analyzer: UnifiedAnalyzer instance
        step_position: Position index in vocab projection to check
        num_latent: Number of latent tokens
        device: Device string
        top_k: Top-k tokens to extract

    Returns:
        (top1_integer, rank, vocab_projection_top_k) at the step position
        vocab_projection_top_k is a list of top-k tokens at each position
        Returns (None, None, None) if inference fails
    """
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
        return None, None, None

    # Build position indices
    if model.model_type == "codi":
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
            first_answer_pos + 1,
            first_answer_pos + 2,
            first_answer_pos + 3,
        ]
    else:
        first_answer_pos = positions['first_answer']
        position_indices = [
            positions['start_latent'],
            *[positions[f'latent_{i}'] for i in range(num_latent)],
            positions['end_latent'],
            positions['hash'],
            first_answer_pos,
            first_answer_pos + 1,
            first_answer_pos + 2,
            first_answer_pos + 3,
        ]

    # Get hidden states from final layer
    activations = result["activations"]
    if not activations:
        return None, None, None

    final_layer_idx = max(activations.keys())
    hidden_states = activations[final_layer_idx]

    # Check bounds
    seq_len = hidden_states.shape[1]
    valid_position_indices = [p for p in position_indices if p < seq_len]

    if step_position >= len(valid_position_indices):
        return None, None, None

    # Project to vocab
    hidden_at_positions = hidden_states[0, valid_position_indices, :]
    proj_result = analyzer.project_activations_to_vocab(
        hidden_at_positions.unsqueeze(0),
        top_k=top_k,
        return_probs=False
    )
    all_logits = proj_result["logits"][0]

    # Extract top-k tokens at ALL positions (for vocab projection table)
    all_top_k_indices = all_logits.argsort(dim=-1, descending=True)[:, :top_k]
    vocab_projection_top_k = []
    for pos_idx in range(len(valid_position_indices)):
        position_tokens = []
        for rank in range(top_k):
            token_id = all_top_k_indices[pos_idx, rank].item()
            token_str = model.tokenizer.decode([token_id])
            position_tokens.append(token_str)
        vocab_projection_top_k.append(position_tokens)

    # Determine answer position in list
    if model.model_type == "codi":
        answer_position_in_list = num_latent + 5
    else:
        answer_position_in_list = num_latent + 2

    # Get result at step_position
    # Use multi-token extraction for answer position (handles multi-token numbers)
    if step_position >= answer_position_in_list:
        top1_info = extract_multitoken_integer(vocab_projection_top_k, step_position)
    else:
        step_position_tokens = vocab_projection_top_k[step_position]
        top1_info = get_top1_integer(step_position_tokens)

    if top1_info is None:
        return None, None, vocab_projection_top_k

    return top1_info[0], top1_info[1], vocab_projection_top_k


def validate_step(
    step: Dict,
    tree_steps: List[Dict],
    template: Dict,
    model,
    analyzer,
    num_latent: int,
    device: str,
    n: int = 3,
    top_k: int = 10,
    required_passes: int = 2,
    max_rank: int = 1
) -> Dict:
    """
    Run n prompt modifications and verify result changes.

    Args:
        step: The step to validate
        tree_steps: All steps in the computation tree
        template: Template dict with variables
        model: Model for inference
        analyzer: UnifiedAnalyzer instance
        num_latent: Number of latent tokens
        device: Device string
        n: Number of modifications to test
        top_k: Top-k tokens to extract
        required_passes: Minimum number of validation checks that must pass (default: 2)
        max_rank: Max rank for expected result in integer tokens. 1=top-1 only, 2=top-1 or top-2, etc. (default: 1)

    Returns:
        ValidationResult dict with status and details
    """
    # Trace operand sources
    operand_sources = trace_operand_sources(step, tree_steps, template)

    # Check if step is explainable
    is_explainable = is_step_fully_explainable(operand_sources)

    # Get the step's result position in the vocab projection list
    # For coconut: position_result is the actual sequence position
    # We need to map it to the index in vocab_projection_top_k
    step_result_position = step["position_result"]

    result = {
        "step_position": step.get("position_operands"),
        "result_position": step_result_position,
        "original_result": step["result"],
        "operand_sources": {
            k: {
                "type": v["type"],
                "var": v.get("var"),
                "value": v["value"]
            } for k, v in operand_sources.items()
        },
        "modifiable_vars": [],
        "modifications_tested": [],
        "pass_count": 0,
        "fail_count": 0
    }

    if not has_any_traceable_operands(operand_sources):
        # No traceable operands at all - can't validate
        result["status"] = "unexplainable"
        return result

    # Get leaf variables that can be modified (question vars, not domain constants)
    # Sort for deterministic order across runs
    modifiable_vars = sorted(get_all_leaf_vars(operand_sources))
    result["modifiable_vars"] = modifiable_vars

    if not modifiable_vars:
        # No modifiable variables (all domain constants)
        result["status"] = "verified"  # Trivially verified - no way to test
        return result

    # Track which values we've already used for each variable
    # (for round-robin repeats, we need different values each time)
    used_values_per_var = {var: set() for var in modifiable_vars}

    # Spread modifications across variables (round-robin with repeats)
    modifications = []
    for i in range(n):
        var_to_modify = modifiable_vars[i % len(modifiable_vars)]

        # Find next valid modification value for this variable
        new_value = find_next_modification_value(
            template,
            var_to_modify,
            step["result"],
            step["operation"],
            operand_sources,
            exclude_values=used_values_per_var[var_to_modify]
        )

        if new_value is not None:
            modifications.append((var_to_modify, new_value))
            used_values_per_var[var_to_modify].add(new_value)

    if len(modifications) < n:
        # Couldn't find enough valid modifications
        result["status"] = "unverified"
        result["error"] = f"Could only find {len(modifications)} valid modifications, needed {n}"
        return result

    # Test each modification
    for var_to_modify, new_value in modifications:
        original_value = template['variables'][var_to_modify]
        original_value = int(original_value) if isinstance(original_value, float) and original_value == int(original_value) else original_value
        var_modifications = {var_to_modify: new_value}

        # Compute expected result
        expected_result = compute_expected_step_result(
            step["operation"],
            operand_sources,
            var_modifications,
            template
        )

        if expected_result is None:
            result["modifications_tested"].append({
                "var_modified": var_to_modify,
                "original_value": original_value,
                "new_value": new_value,
                "expected_result": None,
                "observed_result": None,
                "observed_rank": None,
                "passed": False,
                "error": "Failed to compute expected result"
            })
            result["fail_count"] += 1
            continue

        # Create modified question
        modified_question = create_modified_question(template, var_modifications)

        # Run inference
        observed_result, observed_rank, vocab_projection = run_inference_and_get_result(
            modified_question,
            model,
            analyzer,
            step_result_position,
            num_latent,
            device,
            top_k
        )

        # Determine answer position threshold (same logic as run_inference_and_get_result)
        if model.model_type == "codi":
            answer_position_in_list = num_latent + 5
        else:
            answer_position_in_list = num_latent + 2

        # Check if expected result matches observed
        if step_result_position >= answer_position_in_list:
            # For answer positions: multi-token integers use top-1 at each position
            # So "rank" is effectively 0, just check equality
            passed = observed_result == expected_result
        elif vocab_projection is not None and step_result_position < len(vocab_projection):
            integers_at_position = extract_integers_from_topk(vocab_projection[step_result_position])
            # Find integer rank (position among integers only, not raw vocab rank)
            integer_values = [v for v, r in integers_at_position]
            if expected_result in integer_values:
                integer_rank = integer_values.index(expected_result)
                passed = integer_rank < max_rank
            else:
                passed = False
        else:
            # Fallback to exact match if we don't have vocab projection
            passed = observed_result == expected_result

        result["modifications_tested"].append({
            "var_modified": var_to_modify,
            "original_value": original_value,
            "new_value": new_value,
            "expected_result": expected_result,
            "observed_result": observed_result,
            "observed_rank": observed_rank,
            "passed": passed,
            "max_rank": max_rank
        })

        if passed:
            result["pass_count"] += 1
        else:
            result["fail_count"] += 1

    # Require at least required_passes out of n checks to pass
    result["status"] = "verified" if result["pass_count"] >= required_passes else "unverified"
    return result


def validate_tree(
    tree: Dict,
    template: Dict,
    model,
    analyzer,
    num_latent: int,
    device: str,
    n: int = 3,
    top_k: int = 10,
    required_passes: int = 2,
    max_rank: int = 1
) -> Dict:
    """
    Validate all steps in tree, classify as verified/unverified/unexplainable.

    Args:
        tree: Computation tree dict with steps
        template: Template dict with variables
        model: Model for inference
        analyzer: UnifiedAnalyzer instance
        num_latent: Number of latent tokens
        device: Device string
        n: Number of modifications per step
        top_k: Top-k tokens to extract
        required_passes: Minimum number of validation checks that must pass (default: 2)
        max_rank: Max rank for expected result in integer tokens. 1=top-1 only, 2=top-1 or top-2, etc. (default: 1)

    Returns:
        TreeValidationResult dict
    """
    tree_steps = tree.get("steps", [])

    if not tree_steps:
        return {
            "tree_status": "unexplainable",
            "step_validations": [],
            "verified_count": 0,
            "unverified_count": 0,
            "unexplainable_count": 0,
            "error": "No steps in tree"
        }

    step_validations = []
    verified_count = 0
    unverified_count = 0
    unexplainable_count = 0

    for step in tree_steps:
        validation = validate_step(
            step,
            tree_steps,
            template,
            model,
            analyzer,
            num_latent,
            device,
            n=n,
            top_k=top_k,
            required_passes=required_passes,
            max_rank=max_rank
        )
        step_validations.append(validation)

        if validation["status"] == "verified":
            verified_count += 1
        elif validation["status"] == "unverified":
            unverified_count += 1
        else:  # unexplainable
            unexplainable_count += 1

    # Determine tree status
    if unexplainable_count > 0:
        tree_status = "unexplainable"
    elif unverified_count > 0:
        tree_status = "unverified"
    else:
        tree_status = "verified"

    return {
        "tree_status": tree_status,
        "step_validations": step_validations,
        "verified_count": verified_count,
        "unverified_count": unverified_count,
        "unexplainable_count": unexplainable_count
    }


def run_validation_checks(
    step: Dict,
    tree_steps: List[Dict],
    template: Dict,
    model,
    analyzer,
    num_latent: int,
    device: str,
    n: int = 3,
    top_k: int = 10,
    required_passes: int = 2,
    max_rank: int = 1
) -> Tuple[bool, List[Dict]]:
    """
    Run n validation checks on a step.

    Returns (is_verified, validation_details) where is_verified is True if at least
    required_passes checks pass.

    Args:
        step: The step to validate
        tree_steps: All steps in the computation tree so far
        template: Template dict with variables
        model: Model for inference
        analyzer: UnifiedAnalyzer instance
        num_latent: Number of latent tokens
        device: Device string
        n: Number of modifications to test
        top_k: Top-k tokens to extract
        required_passes: Minimum number of validation checks that must pass (default: 2)
        max_rank: Max rank for expected result in integer tokens. 1=top-1 only, 2=top-1 or top-2, etc. (default: 1)

    Returns:
        Tuple of (is_verified, list of validation check details)
    """
    validation_details = []

    # Trace operand sources
    operand_sources = trace_operand_sources(step, tree_steps, template)

    # Check if step has any traceable operands for validation
    if not has_any_traceable_operands(operand_sources):
        # No traceable operands at all - can't validate
        return False, [{"error": "no_traceable_operands", "operand_sources": operand_sources}]

    # Get leaf variables that can be modified
    # Sort for deterministic order across runs
    modifiable_vars = sorted(get_all_leaf_vars(operand_sources))

    if not modifiable_vars:
        # No modifiable variables (all domain constants) - trivially verified
        return True, [{"status": "trivially_verified", "reason": "no modifiable vars"}]

    step_result_position = step["position_result"]

    # Track which values we've already used for each variable
    # (for round-robin repeats, we need different values each time)
    used_values_per_var = {var: set() for var in modifiable_vars}

    # Collect rejections for each variable (for debugging)
    all_rejections = {}

    # Spread modifications across variables (round-robin with repeats)
    modifications = []
    for i in range(n):
        var_to_modify = modifiable_vars[i % len(modifiable_vars)]

        # Find next valid modification value for this variable, collecting rejections
        new_value, rejections = find_next_modification_value(
            template,
            var_to_modify,
            step["result"],
            step["operation"],
            operand_sources,
            exclude_values=used_values_per_var[var_to_modify],
            collect_rejections=True
        )

        if new_value is not None:
            modifications.append((var_to_modify, new_value))
            used_values_per_var[var_to_modify].add(new_value)
        else:
            # Store rejections for this variable
            if var_to_modify not in all_rejections:
                all_rejections[var_to_modify] = []
            all_rejections[var_to_modify].extend(rejections)

    if len(modifications) < n:
        return False, [{
            "error": "maxed_out_tries",
            "found": len(modifications),
            "needed": n,
            "reason": "unable to be verified -- maxed out tries",
            "modifiable_vars": modifiable_vars,
            "operand_sources": operand_sources,
            "rejections_by_var": all_rejections,  # Detailed rejection info
        }]

    # Test each modification
    passed_count = 0

    for var_to_modify, new_value in modifications:
        original_value = template['variables'][var_to_modify]
        if isinstance(original_value, float) and original_value == int(original_value):
            original_value = int(original_value)
        var_modifications = {var_to_modify: new_value}

        # Compute expected result
        expected_result = compute_expected_step_result(
            step["operation"],
            operand_sources,
            var_modifications,
            template
        )

        if expected_result is None:
            validation_details.append({
                "var_modified": var_to_modify,
                "original_value": original_value,
                "new_value": new_value,
                "expected_result": None,
                "observed_result": None,
                "passed": False,
                "error": "Failed to compute expected result"
            })
            continue

        # Create modified question
        modified_question = create_modified_question(template, var_modifications)

        # Run inference
        observed_result, observed_rank, vocab_projection = run_inference_and_get_result(
            modified_question,
            model,
            analyzer,
            step_result_position,
            num_latent,
            device,
            top_k
        )

        # Determine answer position threshold (same logic as run_inference_and_get_result)
        if model.model_type == "codi":
            answer_position_in_list = num_latent + 5
        else:
            answer_position_in_list = num_latent + 2

        # Check if expected result matches observed
        if step_result_position >= answer_position_in_list:
            # For answer positions: multi-token integers use top-1 at each position
            # So "rank" is effectively 0, just check equality
            passed = observed_result == expected_result
        elif vocab_projection is not None and step_result_position < len(vocab_projection):
            integers_at_position = extract_integers_from_topk(vocab_projection[step_result_position])
            # Find integer rank (position among integers only, not raw vocab rank)
            integer_values = [v for v, r in integers_at_position]
            if expected_result in integer_values:
                integer_rank = integer_values.index(expected_result)
                passed = integer_rank < max_rank
            else:
                passed = False
        else:
            # Fallback to exact match if we don't have vocab projection
            passed = observed_result == expected_result

        validation_details.append({
            "var_modified": var_to_modify,
            "original_value": original_value,
            "new_value": new_value,
            "expected_result": expected_result,
            "observed_result": observed_result,
            "observed_rank": observed_rank,
            "passed": passed,
            "vocab_projection_top_k": vocab_projection,
            "step_result_position": step_result_position,
            "num_latent": num_latent,
            "max_rank": max_rank,
        })

        if passed:
            passed_count += 1

    # Require at least required_passes out of n checks to pass
    is_verified = passed_count >= required_passes
    return is_verified, validation_details


def get_operand_source_priority(
    step: Dict,
    op_key: str,  # "operand1", "operand2", or "operand3"
    verified_positions: Set[int],
    explainable_positions: Set[int],
    question_numbers: Set[int],
    top_k_at_operand_pos: Set[int]
) -> int:
    """
    Return priority score for an operand's source (lower = better).

    Priority:
    1 = verified intermediate
    2 = top-k AND question number
    3 = question number only
    4 = top-k only OR intermediate that's also in top-k
    5 = unverified but explainable intermediate (not in top-k)
    6 = unexplainable intermediate (not in top-k)
    7 = other top-1 from previous position
    """
    value = step.get(op_key)
    if value is None:
        return 0  # No operand, no penalty

    is_intermediate = step.get(f"{op_key}_is_intermediate", False)
    is_question_number = step.get(f"{op_key}_is_question_number", False)
    source_pos = step.get(f"{op_key}_source_pos")

    # Check if value is in top-k at operand position
    in_top_k = value in top_k_at_operand_pos
    in_question = value in question_numbers

    # First: check for VERIFIED intermediates (highest priority)
    if is_intermediate and source_pos not in ("question", None):
        if source_pos in verified_positions:
            return 1  # Verified intermediate

    # Second: check if value is a question number (even if also an intermediate)
    # This ensures we prefer question numbers over UNVERIFIED intermediates
    is_from_question = is_question_number or source_pos == "question" or in_question

    if is_from_question:
        if in_top_k:
            return 2  # Top-k AND question number
        else:
            return 3  # Question number only

    # Third: check for intermediates - but also consider top-k status
    if is_intermediate and source_pos not in ("question", None):
        # If the intermediate is also in top-k, give it priority 4
        # (better than unexplainable intermediate priority 5/6)
        if in_top_k:
            return 4  # Intermediate that's also in top-k
        elif source_pos in explainable_positions:
            return 5  # Unverified but explainable intermediate
        else:
            return 6  # Unexplainable intermediate

    if in_top_k:
        return 4  # Top-k only

    return 7  # Other (top-1 from previous position)


def check_operand_sources_verified(
    step: Dict,
    verified_positions: Set[int],
    question_numbers: Set[int]
) -> bool:
    """
    Check if all intermediate operands in a step come from verified positions.

    Leaf operands (from question numbers) are always considered verified.

    Args:
        step: The step to check
        verified_positions: Set of position_result values that have been verified
        question_numbers: Set of numbers from the question

    Returns:
        True if all intermediate operands are from verified steps
    """
    for op_key in ["operand1", "operand2", "operand3"]:
        if step.get(op_key) is None:
            continue

        is_intermediate = step.get(f"{op_key}_is_intermediate", False)
        is_question_number = step.get(f"{op_key}_is_question_number", False)
        source_pos = step.get(f"{op_key}_source_pos")

        if is_question_number or source_pos == "question":
            # Question number operands are always "verified"
            continue

        if is_intermediate:
            # Check if the source position is verified
            if source_pos not in verified_positions:
                return False

    return True


def select_step_with_validation(
    candidates: List[Dict],
    tree_steps: List[Dict],
    template: Dict,
    verified_positions: Set[int],
    question_numbers: Set[int],
    model,
    analyzer,
    num_latent: int,
    device: str,
    validation_n: int = 3,
    top_k: int = 10,
    explainable_positions: Set[int] = None,
    top_k_at_operand_pos: Set[int] = None,
    verbose: bool = False,
    required_passes: int = 2,
    max_rank: int = 1,
    check_all_candidates: bool = False
) -> Tuple[Dict, bool, List[Dict]]:
    """
    Select best step from candidates, preferring those that validate.

    Priority for operand sources (lower = better):
    1 = verified intermediate
    2 = top-k AND question number
    3 = question number only
    4 = top-k only
    5 = unverified but explainable intermediate
    6 = unexplainable intermediate
    7 = other top-1 from previous position

    Args:
        candidates: List of candidate steps at this position
        tree_steps: Steps already selected for the tree
        template: Template dict with variables
        verified_positions: Set of result positions that have been verified
        question_numbers: Set of numbers from the question
        model: Model for inference
        analyzer: UnifiedAnalyzer instance
        num_latent: Number of latent tokens
        device: Device string
        validation_n: Number of validation checks per step
        top_k: Top-k tokens to extract
        explainable_positions: Set of positions with explainable steps
        top_k_at_operand_pos: Set of integer values in top-k at the operand position
        verbose: If True, include all candidates tried in validation details

    Returns:
        Tuple of (selected_step, is_verified, validation_details)
    """
    if not candidates:
        return None, False, []

    # Default to empty sets if not provided
    if explainable_positions is None:
        explainable_positions = set()
    if top_k_at_operand_pos is None:
        top_k_at_operand_pos = set()

    # Compute priority for each candidate
    def get_candidate_priority_info(c):
        """Get priority info for a candidate, including per-operand details."""
        operand_priorities = {}
        worst_priority = 0
        for op_key in ["operand1", "operand2", "operand3"]:
            if c.get(op_key) is None:
                continue
            op_priority = get_operand_source_priority(
                c, op_key, verified_positions, explainable_positions,
                question_numbers, top_k_at_operand_pos
            )
            operand_priorities[op_key] = op_priority
            worst_priority = max(worst_priority, op_priority)
        return worst_priority, operand_priorities

    # Sort candidates by operand source priority
    def candidate_priority(c):
        worst_priority, _ = get_candidate_priority_info(c)
        return (
            worst_priority,  # Lower = better operand sources
            1 if c.get("is_3op") else 0,  # 2-operand first
            c["avg_operand_rank"]  # Lower rank first
        )

    sorted_candidates = sorted(candidates, key=candidate_priority)

    # Track all candidates tried (for verbose mode)
    all_candidates_tried = [] if verbose else None

    # Try candidates in priority order, preferring those that validate
    best_validated = None
    best_validated_details = None
    best_non_validated = None
    best_non_validated_details = None

    for idx, candidate in enumerate(sorted_candidates):
        worst_priority, operand_priorities = get_candidate_priority_info(candidate)

        # Try to validate this candidate
        is_verified, validation_details = run_validation_checks(
            candidate, tree_steps, template, model, analyzer,
            num_latent, device, validation_n, top_k,
            required_passes=required_passes, max_rank=max_rank
        )

        # Record candidate info for verbose mode
        if verbose:
            op3 = candidate.get("operand3")
            operation = candidate['operation']
            if op3 and len(operation) == 2:
                # 3-operand: operation like "+*" means "operand1 + operand2 * operand3"
                expr = f"{candidate['operand1']} {operation[0]} {candidate['operand2']} {operation[1]} {op3}"
            else:
                expr = f"{candidate['operand1']} {operation} {candidate['operand2']}"
            # Make a JSON-serializable copy of the step (avoid sets and circular refs)
            step_copy = {k: (list(v) if isinstance(v, set) else v)
                        for k, v in candidate.items()
                        if not callable(v)}
            all_candidates_tried.append({
                "order": idx + 1,
                "expression": f"{expr} = {candidate['result']}",
                "worst_priority": worst_priority,
                "operand_priorities": operand_priorities,
                "tested": True,  # Mark as tested
                "validated": is_verified,
                "operand1_source": candidate.get("operand1_source_pos"),
                "operand2_source": candidate.get("operand2_source_pos"),
                "operand3_source": candidate.get("operand3_source_pos") if op3 else None,
                "validation_details": validation_details,  # Full validation info
                "step": step_copy,  # Full step info (serializable copy)
            })

        if is_verified and best_validated is None:
            # Track first validating candidate
            best_validated = candidate
            best_validated_details = validation_details

            if not check_all_candidates:
                # In verbose mode, add remaining untested candidates
                if verbose:
                    for remaining_idx, remaining_candidate in enumerate(sorted_candidates[idx + 1:], start=idx + 2):
                        op3 = remaining_candidate.get("operand3")
                        operation = remaining_candidate['operation']
                        if op3 and len(operation) == 2:
                            expr = f"{remaining_candidate['operand1']} {operation[0]} {remaining_candidate['operand2']} {operation[1]} {op3}"
                        else:
                            expr = f"{remaining_candidate['operand1']} {operation} {remaining_candidate['operand2']}"
                        worst_priority, operand_priorities = get_candidate_priority_info(remaining_candidate)
                        step_copy = {k: (list(v) if isinstance(v, set) else v)
                                    for k, v in remaining_candidate.items()
                                    if not callable(v)}
                        all_candidates_tried.append({
                            "order": remaining_idx,
                            "expression": f"{expr} = {remaining_candidate['result']}",
                            "worst_priority": worst_priority,
                            "operand_priorities": operand_priorities,
                            "tested": False,  # Mark as untested
                            "validated": None,  # Unknown - not tested
                            "operand1_source": remaining_candidate.get("operand1_source_pos"),
                            "operand2_source": remaining_candidate.get("operand2_source_pos"),
                            "operand3_source": remaining_candidate.get("operand3_source_pos") if op3 else None,
                            "validation_details": [],
                            "step": step_copy,
                        })
                break  # Stop testing more candidates once we find a verified one
            # If check_all_candidates is True, continue loop (don't break)

        # Track the best non-validated candidate (first one in priority order)
        if best_non_validated is None and not is_verified:
            best_non_validated = candidate
            best_non_validated_details = validation_details

    # Return best validated if found, otherwise best non-validated
    if best_validated is not None:
        if verbose:
            best_validated_details.append({"all_candidates_tried": all_candidates_tried})
        return best_validated, True, best_validated_details

    # No candidate validated - return best non-validated candidate
    if best_non_validated is not None:
        if verbose:
            best_non_validated_details.append({"all_candidates_tried": all_candidates_tried})
        return best_non_validated, False, best_non_validated_details

    # Fallback: return first candidate overall (shouldn't happen if candidates non-empty)
    _, validation_details = run_validation_checks(
        sorted_candidates[0], tree_steps, template, model, analyzer,
        num_latent, device, validation_n, top_k,
        required_passes=required_passes, max_rank=max_rank
    )
    return sorted_candidates[0], False, validation_details
