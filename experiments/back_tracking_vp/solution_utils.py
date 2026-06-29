"""Shared utilities for parsing solution strings in <<expr=result>> format."""

import re
from typing import List, Union


def to_number(s: str) -> Union[int, float, None]:
    """Convert a numeric string to int (if whole) or float.

    Returns None if the string is not a valid number.
    """
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f):
        return int(f)
    return f


def extract_all_numbers(expr_str: str) -> List[Union[int, float]]:
    """Extract all positive numbers from an expression string.

    Handles integers (16), decimals (0.5, 16.00, .5), and negatives.
    Trivial decimals (16.00) are returned as int, non-trivial (0.5) as float.

    Examples:
        "16-(3+4)" -> [16, 3, 4]
        "2*0.5"    -> [2, 0.5]
        ".5*7"     -> [0.5, 7]
        "4*4=16.00" -> [4, 4, 16]

    Args:
        expr_str: Expression string

    Returns:
        List of positive numbers (int or float)
    """
    matches = re.findall(r'-?(?:\d+(?:\.\d+)?|\.\d+)', expr_str)
    numbers = []
    for m in matches:
        val = to_number(m)
        if val is None:
            continue
        val = abs(val)
        if val > 0:
            numbers.append(val)
    return numbers
