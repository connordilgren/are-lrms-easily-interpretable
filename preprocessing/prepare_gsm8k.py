#!/usr/bin/env python3
"""
Consolidated GSM8K dataset preparation script.

This script downloads and processes GSM8K data through all stages:
1. Download raw data from Internalize-CoT GitHub + MultiChain from HuggingFace
2. Convert raw GSM8K text to JSON (train/valid/test)
3. Clean test data (filter samples with non-empty steps, answer match)
4. Process multichain data (normalize, filter, deduplicate)
5. Create gsm_valid-gold-reasoning-trace_test.json (merge clean test with multichain solutions)
6. Create gsm_vocab-projection-friendly_test.json, single-token unique-numbers subset
7. Generate templates for forward_chaining experiments (used in experiments/forward_chaining)

Output files:
- data/gsm_original_train.json
- data/gsm_original_valid.json
- data/gsm_original_test.json
- data/gsm_valid-gold-reasoning-trace_test.json
- data/gsm_vocab-projection-friendly_test.json
- data/gsm_templates.json
"""

import argparse
import json
import re
import statistics
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset

# ============================================================================
# Constants
# ============================================================================

# URLs for downloading raw data
GSM8K_URLS = {
    "train": "https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt",
    "valid": "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt",
    "test": "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt",
}

MULTICHAIN_DATASET = "DJCheng/MultiChain-GSM8k-Aug-dataset"

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


# ============================================================================
# Stage 1: Download raw data
# ============================================================================

def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL with progress indication."""
    print(f"  Downloading {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded * 100 / total_size
                print(f"\r    Progress: {pct:.1f}%", end="", flush=True)
    print()


def download_gsm8k_raw(raw_dir: Path) -> dict[str, Path]:
    """Download raw GSM8K text files from Internalize-CoT GitHub."""
    print("=== Stage 1a: Downloading raw GSM8K data ===")

    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, url in GSM8K_URLS.items():
        output_path = raw_dir / f"gsm_{split}.txt"
        paths[split] = output_path
        download_file(url, output_path)

    return paths


def download_multichain() -> Dataset:
    """Download multichain data from HuggingFace (uses HF cache)."""
    print("\n=== Stage 1b: Downloading MultiChain data from HuggingFace ===")

    from datasets import load_dataset
    print(f"  Loading dataset from {MULTICHAIN_DATASET}...")
    ds = load_dataset(MULTICHAIN_DATASET, split="test")
    print(f"  Loaded {len(ds)} rows")

    return ds


# ============================================================================
# Stage 2: Convert raw GSM8K to JSON
# ============================================================================

def convert_raw_to_json(raw_paths: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    """Convert raw text files to JSON format."""
    print("\n=== Stage 2: Converting raw GSM8K to JSON ===")

    json_paths = {}
    for split, raw_path in raw_paths.items():
        with open(raw_path) as f:
            lines = f.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Format: question||steps##answer (steps are space-separated)
            # Match original gsm_icot.py behavior exactly
            question = line.split("||")[0]
            steps_str = line.split("||")[1].split("##")[0].strip()
            answer = line.split("##")[-1].strip()

            # Original behavior: always split on space, even if empty
            steps = steps_str.split(" ")

            data.append({
                "question": question,
                "steps": steps,
                "answer": answer,
            })

        output_path = output_dir / f"gsm_original_{split}.json"
        with open(output_path, "w") as f:
            json.dump(data, f)

        json_paths[split] = output_path
        print(f"  {split}: {len(data)} samples -> {output_path}")

    return json_paths


# ============================================================================
# Stage 3: Clean GSM test data
# ============================================================================

def is_steps_empty(steps: list[str]) -> bool:
    """Check if steps is effectively empty."""
    if not steps:
        return True
    if all(s.strip() == '' for s in steps):
        return True
    return False


def extract_last_result_from_steps(steps: list[str]) -> str | None:
    """Extract the result from the last step's <<expr=result>> pattern."""
    if not steps:
        return None
    last_step = steps[-1]
    matches = re.findall(r'<<[^>]*=([^>]+)>>', last_step)
    if matches:
        return matches[-1].strip()
    return None


def is_numeric_match(extracted: str | None, answer: str) -> bool:
    """Check if extracted and answer are numerically equal."""
    if extracted is None:
        return False
    try:
        return float(extracted) == float(answer)
    except (ValueError, TypeError):
        return False


def clean_gsm_test(gsm_test: list[dict]) -> tuple[list[int], dict]:
    """
    Clean GSM test dataset by filtering.

    Criteria:
    1. Non-empty "steps" field
    2. Final result of "steps" must equal the correct answer

    Returns (clean_indices, stats)
    """
    print("\n=== Stage 3: Cleaning GSM test data ===")

    total = len(gsm_test)

    # Criteria 1: Non-empty steps
    failed_criteria1 = []
    for i, item in enumerate(gsm_test):
        if is_steps_empty(item['steps']):
            failed_criteria1.append(i)

    # Criteria 2: Final result must equal answer
    failed_criteria2 = []
    for i, item in enumerate(gsm_test):
        last_result = extract_last_result_from_steps(item['steps'])
        answer = str(item['answer']).strip()

        if last_result != answer and not is_numeric_match(last_result, answer):
            failed_criteria2.append(i)

    # Get union of all failed IDs
    failed_any = sorted(set(failed_criteria1) | set(failed_criteria2))
    clean_indices = sorted(set(range(total)) - set(failed_any))

    stats = {
        'total': total,
        'failed_criteria1_empty_steps': len(failed_criteria1),
        'failed_criteria2_answer_mismatch': len(failed_criteria2),
        'failed_any': len(failed_any),
        'final_count': len(clean_indices)
    }

    print(f"  Total samples: {total}")
    print(f"  Failed criteria 1 (empty steps): {len(failed_criteria1)}")
    print(f"  Failed criteria 2 (answer mismatch): {len(failed_criteria2)}")
    print(f"  Clean samples: {len(clean_indices)}")

    return clean_indices, stats


# ============================================================================
# Stage 4: Process multichain data
# ============================================================================

def is_single_operand(s: str) -> bool:
    """Check if a string is a single operand (number) with no operators."""
    s = s.strip()
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_equation(equation_str: str) -> str:
    """Normalize an equation so the result is on the RHS."""
    match = re.search(r'<<([^>]+)>>', equation_str)
    if not match:
        return equation_str

    content = match.group(1)
    if '=' not in content:
        return equation_str

    parts = content.split('=')
    if len(parts) != 2:
        return equation_str

    left, right = parts[0].strip(), parts[1].strip()

    left_is_single = is_single_operand(left)
    right_is_single = is_single_operand(right)

    if left_is_single and not right_is_single:
        normalized = f'<<{right}={left}>>'
        return equation_str.replace(match.group(0), normalized)

    return equation_str


def normalize_solution(solution_str: str) -> str:
    """Normalize all equations in a solution string."""
    def replace_eq(m):
        return normalize_equation(m.group(0))
    return re.sub(r'<<[^>]+>>', replace_eq, solution_str)


def extract_result_from_equation(equation_str: str) -> str | None:
    """Extract the result from an equation."""
    match = re.search(r'<<([^>]+)>>', equation_str)
    if not match:
        return None

    content = match.group(1)
    if '=' not in content:
        return None

    parts = content.split('=')
    if len(parts) != 2:
        return None

    left, right = parts[0].strip(), parts[1].strip()

    left_is_single = is_single_operand(left)
    right_is_single = is_single_operand(right)

    if right_is_single and not left_is_single:
        return right
    elif left_is_single and not right_is_single:
        return left
    elif right_is_single and left_is_single:
        return right
    else:
        return None


def extract_last_result(solution_str: str) -> str | None:
    """Extract the result from the last <<...>> pattern in a solution string."""
    matches = re.findall(r'<<[^>]+>>', solution_str)
    if not matches:
        return None
    return extract_result_from_equation(matches[-1])


def normalize_commutative_expr(expr: str) -> str:
    """Normalize expression by sorting operands of commutative operators."""
    for op in ['+', '*']:
        if op in expr:
            parts = expr.split(op)
            if len(parts) == 2:
                sorted_parts = sorted(parts, key=lambda x: x.strip())
                return op.join(sorted_parts)
    return expr


def normalize_solution_commutative(solution_str: str) -> str:
    """Normalize all equations for commutative equivalence."""
    def normalize_eq(m):
        content = m.group(1)
        if '=' not in content:
            return m.group(0)
        parts = content.split('=')
        if len(parts) != 2:
            return m.group(0)
        expr, result = parts[0].strip(), parts[1].strip()
        normalized_expr = normalize_commutative_expr(expr)
        return f'<<{normalized_expr}={result}>>'

    return re.sub(r'<<([^>]+)>>', normalize_eq, solution_str)


def contains_variable(solution_str: str) -> bool:
    """Check if a solution contains variables in <<...>> patterns."""
    matches = re.findall(r'<<([^>]+)>>', solution_str)
    for content in matches:
        if re.search(r'[a-zA-Z]', content):
            return True
    return False


def process_multichain(ds: Dataset) -> list[dict]:
    """Process multichain dataset: normalize, filter, deduplicate."""
    print("\n=== Stage 4: Processing multichain data ===")

    # Process each row
    cleaned_data = []
    stats = {
        'solutions_normalized': 0,
        'gen_solutions_normalized': 0,
        'gen_solutions_dropped': 0,
        'gen_solutions_with_variables': 0,
        'gen_solutions_deduplicated': 0,
        'gen_solutions_commutative_dedup': 0,
    }

    for i in range(len(ds)):
        row = ds[i]
        answer = row['answer'].strip()

        # Normalize solution
        original_solution = row['solution']
        normalized_solution = normalize_solution(original_solution)
        if normalized_solution != original_solution:
            stats['solutions_normalized'] += 1

        # Process gen_solutions
        gen_solutions = row['gen_solutions']
        if gen_solutions is None:
            cleaned_data.append({
                'problem': row['problem'],
                'solution': normalized_solution,
                'answer': row['answer'],
                'gen_solutions': None
            })
            continue

        # Filter: final result must match answer
        filtered_gen_solutions = []
        for gen_sol in gen_solutions:
            last_result = extract_last_result(gen_sol)
            if last_result == answer or is_numeric_match(last_result, answer):
                normalized_gen = normalize_solution(gen_sol)
                filtered_gen_solutions.append(normalized_gen)
                if normalized_gen != gen_sol:
                    stats['gen_solutions_normalized'] += 1
            else:
                stats['gen_solutions_dropped'] += 1

        # Filter: remove solutions containing variables
        no_variable_gen_solutions = []
        for gen_sol in filtered_gen_solutions:
            if contains_variable(gen_sol):
                stats['gen_solutions_with_variables'] += 1
            else:
                no_variable_gen_solutions.append(gen_sol)

        filtered_gen_solutions = no_variable_gen_solutions

        # Deduplicate: exact matches (including against main solution)
        seen = {normalized_solution}
        deduplicated_gen_solutions = []
        for gen_sol in filtered_gen_solutions:
            if gen_sol not in seen:
                seen.add(gen_sol)
                deduplicated_gen_solutions.append(gen_sol)
            else:
                stats['gen_solutions_deduplicated'] += 1

        # Deduplicate: commutative equivalents
        seen_commutative = {normalize_solution_commutative(normalized_solution)}
        final_gen_solutions = []
        for gen_sol in deduplicated_gen_solutions:
            normalized_comm = normalize_solution_commutative(gen_sol)
            if normalized_comm not in seen_commutative:
                seen_commutative.add(normalized_comm)
                final_gen_solutions.append(gen_sol)
            else:
                stats['gen_solutions_commutative_dedup'] += 1

        cleaned_data.append({
            'problem': row['problem'],
            'solution': normalized_solution,
            'answer': row['answer'],
            'gen_solutions': final_gen_solutions if final_gen_solutions else None
        })

    print(f"  Solutions normalized: {stats['solutions_normalized']}")
    print(f"  Gen_solutions dropped (wrong answer): {stats['gen_solutions_dropped']}")
    print(f"  Gen_solutions dropped (variables): {stats['gen_solutions_with_variables']}")
    print(f"  Gen_solutions deduplicated (exact): {stats['gen_solutions_deduplicated']}")
    print(f"  Gen_solutions deduplicated (commutative): {stats['gen_solutions_commutative_dedup']}")

    return cleaned_data


# ============================================================================
# Stage 5: Create gsm_valid-gold-reasoning-trace_test.json
# ============================================================================

def parse_gen_solution_to_steps(gen_solution: str) -> list[str]:
    """Parse a gen_solution string into a list of steps."""
    return re.findall(r'<<[^>]+>>', gen_solution)


def create_gsm_test_clean(
    gsm_test: list[dict],
    clean_indices: list[int],
    multichain_data: list[dict],
    output_dir: Path
) -> list[dict]:
    """Create gsm_valid-gold-reasoning-trace_test.json by merging clean test with multichain solutions."""
    print("\n=== Stage 5: Creating gsm_valid-gold-reasoning-trace_test.json ===")

    cleaned_samples = []
    gen_solutions_counts = []

    for i in sorted(clean_indices):
        gsm_sample = gsm_test[i]
        multichain_sample = multichain_data[i]
        gen_solutions_raw = multichain_sample.get('gen_solutions')

        # Convert each gen_solution string to a list of steps
        if gen_solutions_raw is not None:
            gen_solutions = [parse_gen_solution_to_steps(gs) for gs in gen_solutions_raw]
            gen_solutions_counts.append(len(gen_solutions))
        else:
            gen_solutions = None
            gen_solutions_counts.append(0)

        cleaned_samples.append({
            'question': gsm_sample['question'],
            'steps': gsm_sample['steps'],
            'answer': gsm_sample['answer'],
            'gen_solutions': gen_solutions
        })

    # Save
    output_path = output_dir / "gsm_valid-gold-reasoning-trace_test.json"
    with open(output_path, "w") as f:
        json.dump(cleaned_samples, f, indent=2)

    # Stats
    if gen_solutions_counts:
        min_gen = min(gen_solutions_counts)
        max_gen = max(gen_solutions_counts)
        mean_gen = statistics.mean(gen_solutions_counts)
        median_gen = statistics.median(gen_solutions_counts)
        zero_gen_count = sum(1 for c in gen_solutions_counts if c == 0)

        print(f"  Total samples: {len(cleaned_samples)}")
        print(f"  Gen_solutions: min={min_gen}, median={median_gen}, mean={mean_gen:.2f}, max={max_gen}")
        print(f"  Samples with 0 gen_solutions: {zero_gen_count}")

    return cleaned_samples


# ============================================================================
# Stage 6: Create single-token unique-numbers subset
# ============================================================================

def extract_numbers_from_step(step_str: str) -> list[str]:
    """Extract all numbers from a step string like '<<16-3-4=9>>'."""
    return re.findall(r'\d+\.?\d*|\.\d+', step_str)


def normalize_number(num_str: str) -> float:
    """Normalize a number string to a float."""
    cleaned = num_str.strip().replace(',', '')
    return float(cleaned)


def extract_numbers_from_question(question: str) -> list[float]:
    """Extract all numbers from the question text and return as normalized floats."""
    numbers = []
    question_lower = question.lower()

    # Comma-formatted numbers
    comma_nums = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', question)
    for num in comma_nums:
        numbers.append(normalize_number(num))

    question_no_commas = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', '', question)

    # Decimals
    decimals = re.findall(r'\b(\d+\.\d+)\b', question_no_commas)
    for num in decimals:
        numbers.append(normalize_number(num))

    question_no_decimals = re.sub(r'\b\d+\.\d+\b', '', question_no_commas)

    # Integers
    integers = re.findall(r'\b(\d+)\b', question_no_decimals)
    for num in integers:
        numbers.append(normalize_number(num))

    # Divisor words
    for word, value in DIVISOR_WORDS.items():
        if re.search(r'\b' + word + r'\b', question_lower):
            numbers.append(float(value))

    # Multiplier words
    for word, value in MULTIPLIER_WORDS.items():
        if re.search(r'\b' + word + r'\b', question_lower):
            numbers.append(float(value))

    # Compound fractions
    for phrase, (num, denom) in COMPOUND_FRACTIONS.items():
        if phrase in question_lower:
            numbers.append(float(num))
            numbers.append(float(denom))

    return numbers


def extract_results_from_steps(steps: list[str]) -> list[float]:
    """Extract the result from each step and return as normalized floats."""
    results = []
    for step in steps:
        matches = re.findall(r'<<.*?=([^>]+)>>', step)
        for match in matches:
            try:
                results.append(normalize_number(match))
            except ValueError:
                pass
    return results


def extract_number_strings_from_question(question: str) -> list[str]:
    """Extract all number strings from question (for tokenization check)."""
    number_strings = []

    # Comma-formatted numbers
    number_strings.extend(re.findall(r'\b\d{1,3}(?:,\d{3})+\b', question))
    q = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', '', question)

    # Decimals
    number_strings.extend(re.findall(r'\d+\.\d+', q))
    q = re.sub(r'\d+\.\d+', '', q)

    # Plain integers
    number_strings.extend(re.findall(r'\d+', q))

    return number_strings


def create_single_token_subset(
    gsm_test_clean: list[dict],
    output_dir: Path
) -> tuple[list[dict], list[int]]:
    """
    Filter for samples where all numbers are single-token in both tokenizers.

    Returns:
        (filtered_samples, original_indices): The filtered samples and their
        indices in gsm_test_clean (for template generation).
    """
    print("\n=== Stage 6: Creating single-token unique-numbers subset ===")

    # Load tokenizers
    from transformers import AutoTokenizer

    print("  Loading GPT-2 tokenizer...")
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    print("  Loading Llama-3.2-1B-Instruct tokenizer...")
    tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    def is_multi_token(num_str: str) -> bool:
        """Check if a number tokenizes to multiple tokens in either tokenizer."""
        tokens_gpt2 = tokenizer_gpt2.encode(num_str, add_special_tokens=False)
        tokens_llama = tokenizer_llama.encode(num_str, add_special_tokens=False)
        return len(tokens_gpt2) > 1 or len(tokens_llama) > 1

    def has_multi_token_numbers_in_steps(sample: dict) -> bool:
        """Check if a sample has any multi-token numbers in its steps."""
        for step in sample["steps"]:
            numbers = extract_numbers_from_step(step)
            for num in numbers:
                if is_multi_token(num):
                    return True
        return False

    def has_multi_token_numbers_in_question(sample: dict) -> bool:
        """Check if a sample has any multi-token numbers in its question."""
        number_strings = extract_number_strings_from_question(sample["question"])
        for num in number_strings:
            if is_multi_token(num):
                return True
        return False

    def has_multi_token_numbers_in_solution(steps: list[str]) -> bool:
        """Check if a solution has any multi-token numbers."""
        for step in steps:
            numbers = extract_numbers_from_step(step)
            for num in numbers:
                if is_multi_token(num):
                    return True
        return False

    def has_non_unique_numbers(sample: dict) -> bool:
        """Check if any number appears more than once across question and results."""
        question_numbers = extract_numbers_from_question(sample["question"])
        result_numbers = extract_results_from_steps(sample["steps"])

        all_numbers = question_numbers + result_numbers
        counts = Counter(all_numbers)

        return any(count > 1 for count in counts.values())

    def has_non_unique_numbers_in_solution(question: str, steps: list[str]) -> bool:
        """Check if a solution has any non-unique numbers."""
        question_numbers = extract_numbers_from_question(question)
        result_numbers = extract_results_from_steps(steps)

        all_numbers = question_numbers + result_numbers
        counts = Counter(all_numbers)

        return any(count > 1 for count in counts.values())

    # Filter samples and track original indices
    filtered_samples = []
    original_indices = []  # Indices in gsm_test_clean
    gen_solutions_filtered_count = 0
    gen_solutions_total_count = 0

    for i, sample in enumerate(gsm_test_clean):
        has_multi_token_steps = has_multi_token_numbers_in_steps(sample)
        has_multi_token_question = has_multi_token_numbers_in_question(sample)
        has_non_unique = has_non_unique_numbers(sample)

        # Keep only samples with single-token numbers AND unique numbers
        if not has_multi_token_steps and not has_multi_token_question and not has_non_unique:
            # Filter gen_solutions for this sample
            if "gen_solutions" in sample and sample["gen_solutions"]:
                filtered_gen_solutions = []
                gen_solutions_total_count += len(sample["gen_solutions"])

                for gen_solution in sample["gen_solutions"]:
                    has_multi_token_gen = has_multi_token_numbers_in_solution(gen_solution)
                    has_non_unique_gen = has_non_unique_numbers_in_solution(
                        sample["question"], gen_solution
                    )

                    if not has_multi_token_gen and not has_non_unique_gen:
                        filtered_gen_solutions.append(gen_solution)
                    else:
                        gen_solutions_filtered_count += 1

                sample = dict(sample)  # Copy to avoid mutating original
                sample["gen_solutions"] = filtered_gen_solutions

            filtered_samples.append(sample)
            original_indices.append(i)

    # Save
    output_path = output_dir / "gsm_vocab-projection-friendly_test.json"
    with open(output_path, "w") as f:
        json.dump(filtered_samples, f, indent=2)

    print(f"  Filtered from {len(gsm_test_clean)} to {len(filtered_samples)} samples")
    print(f"  Kept {100 * len(filtered_samples) / len(gsm_test_clean):.2f}% of samples")
    if gen_solutions_total_count > 0:
        kept_gen = gen_solutions_total_count - gen_solutions_filtered_count
        print(f"  Filtered gen_solutions: removed {gen_solutions_filtered_count} of {gen_solutions_total_count}")
        print(f"  Kept {100 * kept_gen / gen_solutions_total_count:.2f}% of gen_solutions")

    return filtered_samples, original_indices


# ============================================================================
# Stage 7: Generate templates
# ============================================================================

def extract_numbers(question: str) -> list[dict]:
    """Extract numbers from a question string."""
    matches = []
    matched_positions = set()

    patterns = [
        (r'\$\d[\d,]*(?:\.\d+)?', 'currency'),
        (r'\d+(?:\.\d+)?%', 'percentage'),
        (r'\d+/\d+', 'fraction'),
        (r'\d*\.\d+', 'decimal'),
        (r'\d{1,3}(?:,\d{3})+', 'numeric'),
        (r'\d+', 'numeric'),
    ]

    for pattern, num_type in patterns:
        for match in re.finditer(pattern, question):
            start, end = match.start(), match.end()

            if any(pos in matched_positions for pos in range(start, end)):
                continue

            text = match.group()

            if num_type == 'currency':
                value = float(text.replace('$', '').replace(',', ''))
                if value == int(value):
                    value = int(value)
                matches.append({
                    'value': value,
                    'start': start + 1,
                    'end': end,
                    'text': text[1:],
                    'type': num_type,
                })
            elif num_type == 'percentage':
                value = float(text.rstrip('%'))
                if value == int(value):
                    value = int(value)
                matches.append({
                    'value': value,
                    'start': start,
                    'end': end,
                    'text': text,
                    'type': num_type,
                })
            elif num_type == 'fraction':
                parts = text.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1])
                matches.append({
                    'value': Fraction(numerator, denominator),
                    'start': start,
                    'end': end,
                    'text': text,
                    'type': num_type,
                    'fraction_parts': (numerator, denominator),
                })
            elif num_type == 'decimal':
                value = float(text)
                matches.append({
                    'value': value,
                    'start': start,
                    'end': end,
                    'text': text,
                    'type': num_type,
                })
            else:
                value = int(text.replace(',', ''))
                matches.append({
                    'value': value,
                    'start': start,
                    'end': end,
                    'text': text,
                    'type': num_type,
                })

            for pos in range(start, end):
                matched_positions.add(pos)

    # Compound fractions
    compound_pattern = r'\b(' + '|'.join(re.escape(phrase) for phrase in COMPOUND_FRACTIONS.keys()) + r')\b'
    for match in re.finditer(compound_pattern, question, re.IGNORECASE):
        start, end = match.start(), match.end()
        if any(pos in matched_positions for pos in range(start, end)):
            continue

        text = match.group()
        numerator, denominator = COMPOUND_FRACTIONS[text.lower()]

        matches.append({
            'value': (numerator, denominator),
            'start': start,
            'end': end,
            'text': text,
            'type': 'compound_fraction',
            'word_type': 'compound_fraction',
            'original_text': text.lower(),
            'numerator': numerator,
            'denominator': denominator,
        })

        for pos in range(start, end):
            matched_positions.add(pos)

    # Divisor words
    divisor_pattern = r'\b(' + '|'.join(re.escape(word) for word in DIVISOR_WORDS.keys()) + r')\b'
    for match in re.finditer(divisor_pattern, question, re.IGNORECASE):
        start, end = match.start(), match.end()
        if any(pos in matched_positions for pos in range(start, end)):
            continue

        text = match.group()
        value = DIVISOR_WORDS[text.lower()]

        matches.append({
            'value': value,
            'start': start,
            'end': end,
            'text': text,
            'type': 'word',
            'word_type': 'divisor',
            'original_text': text.lower(),
        })

        for pos in range(start, end):
            matched_positions.add(pos)

    # Multiplier words
    multiplier_pattern = r'\b(' + '|'.join(re.escape(word) for word in MULTIPLIER_WORDS.keys()) + r')\b'
    for match in re.finditer(multiplier_pattern, question, re.IGNORECASE):
        start, end = match.start(), match.end()
        if any(pos in matched_positions for pos in range(start, end)):
            continue

        text = match.group()
        value = MULTIPLIER_WORDS[text.lower()]

        matches.append({
            'value': value,
            'start': start,
            'end': end,
            'text': text,
            'type': 'word',
            'word_type': 'multiplier',
            'original_text': text.lower(),
        })

        for pos in range(start, end):
            matched_positions.add(pos)

    # Word numbers
    word_pattern = r'\b(' + '|'.join(re.escape(word) for word in WORD_NUMBERS.keys()) + r')\b'
    for match in re.finditer(word_pattern, question, re.IGNORECASE):
        start, end = match.start(), match.end()
        if any(pos in matched_positions for pos in range(start, end)):
            continue

        text = match.group()
        value = WORD_NUMBERS[text.lower()]

        matches.append({
            'value': value,
            'start': start,
            'end': end,
            'text': text,
            'type': 'word',
        })

        for pos in range(start, end):
            matched_positions.add(pos)

    matches.sort(key=lambda x: x['start'])
    return matches


def parse_step(step: str) -> tuple[str, str]:
    """Parse a step like '<<8*15=120>>' into expression and result."""
    match = re.match(r'<<(.+)=([^>]+)>>', step)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def tokenize_expression(expr: str) -> list[dict]:
    """Tokenize a math expression into numbers and operators."""
    tokens = []
    pattern = r'(\d*\.?\d+|[+\-*/()])'

    for match in re.finditer(pattern, expr):
        text = match.group()
        start = match.start()
        end = match.end()

        if text in '+-*/()':
            tokens.append({
                'type': 'operator',
                'value': text,
                'text': text,
                'start': start,
                'end': end,
            })
        else:
            if '.' in text:
                value = float(text)
            else:
                value = int(text)
            tokens.append({
                'type': 'number',
                'value': value,
                'text': text,
                'start': start,
                'end': end,
            })

    return tokens


def values_equal(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    """Check if two numeric values are equal."""
    if isinstance(a, Fraction) or isinstance(b, Fraction):
        try:
            fa = Fraction(a) if not isinstance(a, Fraction) else a
            fb = Fraction(b) if not isinstance(b, Fraction) else b
            return fa == fb
        except (ValueError, TypeError):
            pass

    try:
        return abs(float(a) - float(b)) < tolerance
    except (ValueError, TypeError):
        return False


def create_template(sample: dict, index: int) -> dict:
    """Create a template from a GSM sample."""
    question = sample['question']
    steps = sample['steps']
    answer = sample['answer']

    number_matches = extract_numbers(question)

    variables = {}
    var_metadata = {}
    var_positions = []
    var_counter = 1
    value_to_vars: dict[Any, list[str]] = {}

    for match in number_matches:
        if match['type'] == 'fraction' and 'fraction_parts' in match:
            numerator, denominator = match['fraction_parts']

            num_var = f"VAR_{var_counter}"
            var_counter += 1
            variables[num_var] = numerator

            den_var = f"VAR_{var_counter}"
            var_counter += 1
            variables[den_var] = denominator

            var_positions.append({
                'var': f"{num_var}/{den_var}",
                'start': match['start'],
                'end': match['end'],
                'text': match['text'],
                'is_fraction': True,
                'num_var': num_var,
                'den_var': den_var,
            })

            if numerator not in value_to_vars:
                value_to_vars[numerator] = []
            value_to_vars[numerator].append(num_var)

            if denominator not in value_to_vars:
                value_to_vars[denominator] = []
            value_to_vars[denominator].append(den_var)

        elif match['type'] == 'compound_fraction':
            numerator = match['numerator']
            denominator = match['denominator']

            num_var = f"VAR_{var_counter}"
            var_counter += 1
            variables[num_var] = numerator

            den_var = f"VAR_{var_counter}"
            var_counter += 1
            variables[den_var] = denominator

            var_metadata[num_var] = {
                'word_type': 'compound_fraction',
                'original_text': match['original_text'],
                'is_numerator': True,
                'paired_var': den_var,
            }
            var_metadata[den_var] = {
                'word_type': 'compound_fraction',
                'original_text': match['original_text'],
                'is_denominator': True,
                'paired_var': num_var,
            }

            var_positions.append({
                'var': match['original_text'].upper(),
                'start': match['start'],
                'end': match['end'],
                'text': match['text'],
                'is_compound_fraction': True,
                'num_var': num_var,
                'den_var': den_var,
                'original_text': match['original_text'],
            })

            if numerator not in value_to_vars:
                value_to_vars[numerator] = []
            value_to_vars[numerator].append(num_var)

            if denominator not in value_to_vars:
                value_to_vars[denominator] = []
            value_to_vars[denominator].append(den_var)

        else:
            var_name = f"VAR_{var_counter}"
            var_counter += 1

            value = match['value']
            if isinstance(value, Fraction):
                if value.denominator == 1:
                    value = int(value)
                else:
                    value = float(value)

            variables[var_name] = value
            var_position_entry = {
                'var': var_name,
                'start': match['start'],
                'end': match['end'],
                'text': match['text'],
            }

            if match.get('word_type'):
                var_metadata[var_name] = {
                    'word_type': match['word_type'],
                    'original_text': match['original_text'],
                }
                var_position_entry['word_type'] = match['word_type']
                var_position_entry['original_text'] = match['original_text']

            var_positions.append(var_position_entry)

            val_key = float(value) if not isinstance(value, int) else value
            if val_key not in value_to_vars:
                value_to_vars[val_key] = []
            value_to_vars[val_key].append(var_name)

    # Template the question
    template_question = question
    for vp in reversed(var_positions):
        template_question = (
            template_question[:vp['start']] +
            vp['var'] +
            template_question[vp['end']:]
        )

    # Parse and template steps
    template_steps = []
    original_steps = []
    results = {}
    result_counter = 1
    implicit_constants = {}
    const_counter = 1
    value_var_usage: dict[Any, int] = {}

    for step in steps:
        expr, result_str = parse_step(step)
        if not expr:
            template_steps.append(step)
            original_steps.append(step)
            continue

        original_steps.append(step)
        tokens = tokenize_expression(expr)

        template_expr_parts = []
        last_end = 0

        for token in tokens:
            if token['start'] > last_end:
                template_expr_parts.append(expr[last_end:token['start']])

            if token['type'] == 'operator':
                template_expr_parts.append(token['value'])
            else:
                num_val = token['value']
                replacement = None

                # Check if it's a previous result
                for res_name, res_val in results.items():
                    if values_equal(num_val, res_val):
                        replacement = res_name
                        break

                if replacement is None:
                    val_key = float(num_val) if not isinstance(num_val, int) else num_val

                    matched_var = None
                    if val_key in value_to_vars:
                        usage_idx = value_var_usage.get(val_key, 0)
                        vars_with_val = value_to_vars[val_key]
                        if usage_idx < len(vars_with_val):
                            matched_var = vars_with_val[usage_idx]
                            value_var_usage[val_key] = usage_idx + 1
                        elif vars_with_val:
                            matched_var = vars_with_val[0]

                    if matched_var is None and isinstance(num_val, float) and num_val == int(num_val):
                        int_key = int(num_val)
                        if int_key in value_to_vars:
                            usage_idx = value_var_usage.get(int_key, 0)
                            vars_with_val = value_to_vars[int_key]
                            if usage_idx < len(vars_with_val):
                                matched_var = vars_with_val[usage_idx]
                                value_var_usage[int_key] = usage_idx + 1
                            elif vars_with_val:
                                matched_var = vars_with_val[0]

                    if matched_var is not None:
                        replacement = matched_var

                if replacement is None:
                    int_val = int(num_val) if isinstance(num_val, float) and num_val == int(num_val) else num_val
                    if isinstance(int_val, int) and int_val in DOMAIN_CONSTANTS:
                        const_name = f"CONST_{int_val}"
                        implicit_constants[const_name] = int_val
                        replacement = const_name

                if replacement is None:
                    const_name = f"CONST_{const_counter}"
                    const_counter += 1
                    implicit_constants[const_name] = num_val
                    replacement = const_name

                template_expr_parts.append(replacement)

            last_end = token['end']

        if last_end < len(expr):
            template_expr_parts.append(expr[last_end:])

        template_expr = ''.join(template_expr_parts)

        result_name = f"RESULT_{result_counter}"
        result_counter += 1

        try:
            result_val = float(result_str)
        except ValueError:
            result_val = result_str

        results[result_name] = result_val

        val_key = float(result_val) if isinstance(result_val, (int, float)) else result_val
        if val_key not in value_to_vars:
            value_to_vars[val_key] = []
        value_to_vars[val_key].append(result_name)

        template_steps.append(f"<<{template_expr}={result_name}>>")

    if results:
        template_answer = f"RESULT_{result_counter - 1}"
    else:
        template_answer = answer

    return {
        "source_index": index,
        "original_question": question,
        "template_question": template_question,
        "original_steps": original_steps,
        "template_steps": template_steps,
        "original_answer": answer,
        "template_answer": template_answer,
        "variables": variables,
        "var_metadata": var_metadata,
        "var_positions": var_positions,
        "implicit_constants": implicit_constants,
    }


def generate_templates(
    filtered_samples: list[dict],
    original_indices: list[int],
    output_dir: Path
) -> list[dict]:
    """Generate templates for the filtered subset."""
    print("\n=== Stage 7: Generating templates ===")

    templates = []

    for i, (sample, original_index) in enumerate(zip(filtered_samples, original_indices)):
        try:
            template = create_template(sample, original_index)
            templates.append(template)
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            templates.append({
                "source_index": original_index,
                "error": str(e),
                "original_question": sample.get('question', ''),
                "original_steps": sample.get('steps', []),
                "original_answer": sample.get('answer', ''),
            })

    # Save
    output_path = output_dir / "gsm_templates.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)

    print(f"  Generated {len(templates)} templates")

    return templates


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GSM8K datasets for training and evaluation."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for generated files (default: data)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Stage 1: Download raw data
    # Use a temporary directory for raw text files (deleted after conversion)
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_paths = download_gsm8k_raw(Path(tmp_dir))
        # Stage 2: Convert raw to JSON
        json_paths = convert_raw_to_json(raw_paths, output_dir)

    # Download multichain (uses HuggingFace cache)
    multichain_ds = download_multichain()

    # Load test data for later stages
    with open(json_paths["test"]) as f:
        gsm_test = json.load(f)

    # Stage 3: Clean test data
    clean_indices, clean_stats = clean_gsm_test(gsm_test)

    # Stage 4: Process multichain data
    multichain_data = process_multichain(multichain_ds)

    # Stage 5: Create gsm_valid-gold-reasoning-trace_test.json
    gsm_test_clean = create_gsm_test_clean(
        gsm_test, clean_indices, multichain_data, output_dir
    )

    # Stage 6: Create single-token subset
    filtered_samples, original_indices = create_single_token_subset(gsm_test_clean, output_dir)

    # Stage 7: Generate templates
    templates = generate_templates(filtered_samples, original_indices, output_dir)

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print("=" * 60)
    for filename in [
        "gsm_original_train.json",
        "gsm_original_valid.json",
        "gsm_original_test.json",
        "gsm_valid-gold-reasoning-trace_test.json",
        "gsm_vocab-projection-friendly_test.json",
        "gsm_templates.json",
    ]:
        path = output_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {filename}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
