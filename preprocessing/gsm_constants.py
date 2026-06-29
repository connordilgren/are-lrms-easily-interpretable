#!/usr/bin/env python3
"""
Shared constants and utilities for GSM8K template processing.

Used by preprocessing/prepare_gsm8k.py and experiments/forward_chaining/validation.py.
"""

import ast
import operator

# Domain knowledge constants (implicit in steps, not from question)
DOMAIN_CONSTANTS = {100, 60, 12, 7, 24, 52}

# Word numbers mapping
WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}

# Divisor words - appear as divisors (x/N) in steps
DIVISOR_WORDS = {"half": 2, "third": 3, "quarter": 4}

# Multiplier words - appear as multipliers (x*N) in steps
MULTIPLIER_WORDS = {"twice": 2, "double": 2, "triple": 3}

# Compound fractions
COMPOUND_FRACTIONS = {
    "one-half": (1, 2),
    "two-thirds": (2, 3),
    "three-quarters": (3, 4),
    "three-fourths": (3, 4),
    "three-fifths": (3, 5),
    "four-fifths": (4, 5),
    "one-third": (1, 3),
    "one-fourth": (1, 4),
    "one-fifth": (1, 5),
    "two-fifths": (2, 5),
}

# Reverse mappings: number -> word form
NUM_TO_DIVISOR_WORD = {v: k for k, v in DIVISOR_WORDS.items()}
# For multiplier, prefer "twice" over "double" for 2
NUM_TO_MULTIPLIER_WORD = {3: "triple", 2: "twice"}
# Reverse of COMPOUND_FRACTIONS: (num, denom) -> word (first occurrence wins)
FRACTION_TO_WORD = {}
for _word, _frac in COMPOUND_FRACTIONS.items():
    if _frac not in FRACTION_TO_WORD:
        FRACTION_TO_WORD[_frac] = _word


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str):
    """Safely evaluate an arithmetic expression string."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(ast.parse(expr, mode='eval').body)
