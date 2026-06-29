"""
Early Stopping Experiment for CoT, Coconut, and CODI Models.

This script measures how much of the reasoning chain is actually needed to reach
the final answer. It runs full inference, then systematically varies reasoning depth
and forces an early answer.

Supports three methods:
1. Early stopping - Force answer at each reasoning position
2. Vocabulary projection - Check if answer appears in top-k predictions
3. Rank stabilization - Find where top-k predictions stabilize

Usage:
    python -m experiments.early_stopping.run \
        --model_type coconut \
        --model_path models/gsm-coconut/checkpoint_33 \
        --dataset_path data/gsm_original_test.json \
        --num_latents 6 \
        --output_dir results/early_stopping
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch

from models.factory import ModelFactory


def get_base_model_short_name(model_id: str, model_path: str, model_type: str) -> str:
    """
    Extract a short identifier for the base model.

    For CoT/Coconut: derived from --model_id
    For CODI/Multimode: detected from the checkpoint path or model_id

    Returns: short name like 'gpt2' or 'llama32-1b'
    """
    if model_type in ["cot", "coconut"]:
        # Parse from model_id like "openai-community/gpt2" or "meta-llama/Llama-3.2-1B"
        model_id_lower = model_id.lower()
        if "gpt2" in model_id_lower:
            return "gpt2"
        elif "llama-3.2-1b" in model_id_lower or "llama32-1b" in model_id_lower:
            return "llama32-1b"
        elif "llama" in model_id_lower:
            return "llama"
        else:
            # Fallback: use last part of model_id
            return model_id.split("/")[-1].lower().replace("-", "_")
    else:
        # CODI and Multimode: detect from path or model_id
        # e.g., ".../gpt2/ep_40/..." or ".../Llama-3.2-1B-Instruct/ep_10/..."
        # Also check model_id for multimode models
        combined_lower = (model_path + " " + model_id).lower()
        if "/gpt2/" in combined_lower or "gpt2_latent" in combined_lower or "gpt2" in model_id.lower():
            return "gpt2"
        elif "llama-3.2-1b" in combined_lower or "llama1b" in combined_lower:
            return "llama32-1b"
        elif "llama" in combined_lower:
            return "llama"
        else:
            return "unknown"
from analyzers.base import UnifiedAnalyzer
from dataset_utils.base import load_dataset
from dataset_utils.adapters import DatasetAdapter


# =============================================================================
# Helper Functions
# =============================================================================

def extract_answer(text: str, delimiter: str = "###") -> str:
    """
    Extract answer from model output.

    Args:
        text: Full model output
        delimiter: Delimiter separating reasoning from answer

    Returns:
        Extracted answer string (after delimiter)
    """
    if text is None:
        return None

    # Try primary delimiter first
    if delimiter in text:
        parts = text.split(delimiter)
        if len(parts) >= 2:
            answer = parts[-1].strip()
            # Remove commas from numbers and special tokens
            answer = answer.replace(",", "")
            answer = answer.replace("<|endoftext|>", "")
            return answer.strip()

    # Fallback: Try CODI format "The answer is:"
    if "The answer is:" in text:
        parts = text.split("The answer is:")
        if len(parts) >= 2:
            answer = parts[-1].strip()
            answer = answer.replace(",", "")
            answer = answer.replace("<|endoftext|>", "")
            return answer.strip()

    # Fallback: Try just "#" for CoT
    if "#" in text:
        parts = text.split("#")
        if len(parts) >= 2:
            answer = parts[-1].strip()
            answer = answer.replace(",", "")
            answer = answer.replace("<|endoftext|>", "")
            return answer.strip()

    return None


def get_answer_token_string(answer: str, dataset_path: str) -> str:
    """
    Extract the token string to look for in vocab projection.

    Different datasets have different answer formats:
    - GSM8k: numeric answer (e.g., "42")
    - ProSQA: "A is a B" format - extract "B"
    - ProntoQA: "True" or "False"

    Args:
        answer: Full answer string
        dataset_path: Path to dataset (used to determine format)

    Returns:
        The token string to search for in top-k predictions
    """
    if answer is None:
        return ""

    if "prosqa" in dataset_path.lower():
        # Parse "A is a B" format, get "B"
        if " is a " in answer:
            return answer.split(" is a ")[-1].rstrip(".")
        else:
            return answer.strip()
    elif "gsm" in dataset_path.lower():
        # Numeric answer - use as-is
        return answer.strip()
    elif "prontoqa" in dataset_path.lower():
        # "True" or "False"
        return answer.strip()
    else:
        # Default: use full answer
        return answer.strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Answer string

    Returns:
        Normalized answer (lowercase, no spaces/commas)
    """
    if answer is None:
        return ""
    return answer.strip().lower().replace(",", "").replace(" ", "")


def find_first_match(early_results: list) -> int:
    """
    Find first reasoning position where early answer matches final answer.

    Args:
        early_results: List of early stopping results

    Returns:
        First position with match, or None if no match
    """
    for r in early_results:
        if r["matches_final"]:
            return r["num_reasoning_tokens"]
    return None


def find_stable_match(early_results: list) -> int:
    """
    Find reasoning position where answer stabilizes.

    Walks backwards from the end to find where the answer first became
    what it stayed as until the end.

    Args:
        early_results: List of early stopping results

    Returns:
        Position where answer stabilized, or None
    """
    if not early_results:
        return None

    final_answer_normalized = normalize_answer(early_results[-1]["answer"])
    stable_idx = len(early_results) - 1

    # Walk backwards to find where answer became stable
    for i in range(len(early_results) - 2, -1, -1):
        if normalize_answer(early_results[i]["answer"]) != final_answer_normalized:
            break
        stable_idx = i

    return early_results[stable_idx]["num_reasoning_tokens"]


def compute_summary(samples_results: list) -> dict:
    """
    Compute summary statistics for early stopping results.

    Args:
        samples_results: List of sample results

    Returns:
        Dictionary with summary statistics
    """
    total = len(samples_results)
    if total == 0:
        return {}

    reasoning_counts = [s["num_reasoning_tokens"] for s in samples_results]
    first_matches = [s["num_reasoning_tokens_first_match"] for s in samples_results
                     if s["num_reasoning_tokens_first_match"] is not None]
    stable_matches = [s["num_reasoning_tokens_stable_match"] for s in samples_results
                      if s["num_reasoning_tokens_stable_match"] is not None]

    summary = {
        "total_samples": total,
        "avg_reasoning_tokens": sum(reasoning_counts) / total if total > 0 else 0,
        "avg_num_reasoning_tokens_first_match": sum(first_matches) / len(first_matches) if first_matches else None,
        "avg_num_reasoning_tokens_stable_match": sum(stable_matches) / len(stable_matches) if stable_matches else None,
        "num_with_first_match": len(first_matches),
        "num_with_stable_match": len(stable_matches),
    }

    # Add step-level statistics if available (for CoT)
    if samples_results and "num_reasoning_steps" in samples_results[0]:
        step_counts = [s["num_reasoning_steps"] for s in samples_results if "num_reasoning_steps" in s]
        if step_counts:
            summary["avg_reasoning_steps"] = sum(step_counts) / len(step_counts)

    return summary


def compute_vocab_projection_summary(samples_results: list) -> dict:
    """
    Compute summary statistics for vocab projection results (token-level).

    Args:
        samples_results: List of sample results with vocab_projection_by_token field

    Returns:
        Dictionary with summary statistics
    """
    # Extract vocab projection results that exist (try new field, fall back to old)
    vocab_samples = [s.get("vocab_projection_by_token") or s.get("vocab_projection")
                     for s in samples_results
                     if (s.get("vocab_projection_by_token") or s.get("vocab_projection")) is not None]

    total = len(vocab_samples)
    if total == 0:
        return {}

    first_positions = [s["first_position_answer_in_top_k"] for s in vocab_samples
                       if s["first_position_answer_in_top_k"] is not None]
    stable_positions = [s["rank_stable_position"] for s in vocab_samples
                        if s.get("rank_stable_position") is not None]

    return {
        "total_samples": total,
        "avg_first_position_answer_in_top_k": sum(first_positions) / len(first_positions) if first_positions else None,
        "avg_rank_stable_position": sum(stable_positions) / len(stable_positions) if stable_positions else None,
        "num_with_answer_in_top_k": len(first_positions),
    }


def compute_vocab_projection_by_step_summary(samples_results: list) -> dict:
    """
    Compute summary statistics for vocab projection results (step-level, CoT only).

    Args:
        samples_results: List of sample results with vocab_projection_by_step field

    Returns:
        Dictionary with summary statistics, or empty dict if no step-level data
    """
    # Extract step-level vocab projection results that exist
    vocab_samples = [s["vocab_projection_by_step"] for s in samples_results
                     if s.get("vocab_projection_by_step") is not None]

    total = len(vocab_samples)
    if total == 0:
        return {}

    first_steps = [s["first_step_answer_in_top_k"] for s in vocab_samples
                   if s["first_step_answer_in_top_k"] is not None]
    stable_steps = [s["rank_stable_step"] for s in vocab_samples
                    if s.get("rank_stable_step") is not None]

    return {
        "total_samples": total,
        "avg_first_step_answer_in_top_k": sum(first_steps) / len(first_steps) if first_steps else None,
        "avg_rank_stable_step": sum(stable_steps) / len(stable_steps) if stable_steps else None,
        "num_with_answer_in_top_k": len(first_steps),
    }


# =============================================================================
# CoT Early Stopping
# =============================================================================

def get_delimiter_tokens(tokenizer, delimiter="###"):
    """
    Get token IDs for delimiter.

    Args:
        tokenizer: Model tokenizer
        delimiter: Delimiter string

    Returns:
        Tensor of token IDs for delimiter
    """
    tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    return torch.tensor(tokens, dtype=torch.long)


def extract_reasoning_tokens_cot(output_ids, input_ids, tokenizer):
    """
    Extract reasoning token IDs from CoT output.

    Reasoning tokens are between the input and the "###" delimiter.

    Args:
        output_ids: Full output token IDs [batch, seq_len] or [seq_len]
        input_ids: Input token IDs [batch, seq_len] or [seq_len]
        tokenizer: Model tokenizer

    Returns:
        Tensor of reasoning token IDs
    """
    # Ensure 1D
    if output_ids.dim() > 1:
        output_ids = output_ids[0]
    if input_ids.dim() > 1:
        input_ids = input_ids[0]

    # Get generated tokens (after input)
    input_len = len(input_ids)
    generated_ids = output_ids[input_len:]

    # Find "###" token
    # Try both common representations
    hash_tokens = [21017, 44386]  # Known ### token IDs in GPT2
    # Also try encoding it
    encoded_hash = tokenizer.encode("###", add_special_tokens=False)
    if encoded_hash:
        hash_tokens.extend(encoded_hash)
    hash_tokens = list(set(hash_tokens))  # Remove duplicates

    split_index = None
    for i, token_id in enumerate(generated_ids.tolist()):
        if token_id in hash_tokens:
            split_index = i
            break

    if split_index is None:
        # No ### found - treat all as reasoning
        return generated_ids

    # Return reasoning tokens (before ###)
    return generated_ids[:split_index]


def find_step_boundaries(tokenizer, reasoning_token_ids):
    """
    Find positions where newline tokens occur in the reasoning.
    Newline marks the end of each reasoning step.

    Args:
        tokenizer: Tokenizer to get newline token ID
        reasoning_token_ids: Tensor of token IDs in the reasoning portion

    Returns:
        List of positions where newline occurs (end of each step)
    """
    # Convert to list if tensor
    if isinstance(reasoning_token_ids, torch.Tensor):
        reasoning_token_ids = reasoning_token_ids.tolist()

    # Check each token's decoded text for newline character.
    # This handles tokenizers that merge newline with preceding characters
    # (e.g., Llama tokenizes ".\n" as a single token).
    boundaries = []
    for i, token_id in enumerate(reasoning_token_ids):
        decoded = tokenizer.decode([token_id])
        if "\n" in decoded:
            boundaries.append(i)
    return boundaries


def process_cot_early_stopping(model, sample, args):
    """
    Process a single CoT sample for early stopping analysis.

    Args:
        model: CoT model
        sample: Dataset sample (from DatasetAdapter with input_ids and question_tokenized)
        args: Command-line arguments

    Returns:
        Dictionary with early stopping results, or None if failed
    """
    # Get input_ids from DatasetAdapter output
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Ensure 2D tensors and move to device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # 1. Run full inference
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens
        )

    full_text = model.decode(output.output_ids)
    final_answer = extract_answer(full_text, "###")

    if final_answer is None:
        return None  # Skip samples without valid delimiter

    # 2. Extract reasoning tokens
    # Use the first element of input_ids (remove batch dimension for extraction)
    reasoning_tokens = extract_reasoning_tokens_cot(
        output.output_ids, input_ids[0], model.tokenizer
    )
    num_reasoning_tokens = len(reasoning_tokens)

    # 3. Get delimiter tokens for forced early answers
    delimiter_ids = get_delimiter_tokens(model.tokenizer, "###").to(input_ids.device)

    # 4. Try each prefix length
    early_results = []
    for k in range(num_reasoning_tokens + 1):
        # Construct: question + first k reasoning tokens + "###"
        prefix_ids = torch.cat([
            input_ids[0],  # Remove batch dim
            reasoning_tokens[:k],
            delimiter_ids
        ]).unsqueeze(0)

        # Generate answer
        with torch.no_grad():
            answer_output = model.generate(
                prefix_ids.to(model.device),
                attention_mask=torch.ones_like(prefix_ids).to(model.device),
                max_new_tokens=32  # Just the answer
            )

        answer_text = model.decode(answer_output.output_ids)
        answer = extract_answer(answer_text, "###")

        result = {
            "num_reasoning_tokens": k,
            "answer": answer,
            "matches_final": normalize_answer(answer) == normalize_answer(final_answer)
        }

        # Include full output if debug mode
        if args.debug:
            result["full_output"] = answer_text

        early_results.append(result)

    # 5. Build step-level results
    # Find step boundaries (positions where \n occurs - end of each step)
    step_boundaries = find_step_boundaries(model.tokenizer, reasoning_tokens)

    # For each complete step, record what the answer was right after that step
    step_results = []
    for step_num, boundary_pos in enumerate(step_boundaries):
        # boundary_pos is the position of the \n token (end of step)
        # Force answer right after this \n means using boundary_pos + 1 tokens
        num_tokens_after_step = boundary_pos + 1
        if num_tokens_after_step <= num_reasoning_tokens:
            # Find the early_result for this position
            early_result = early_results[num_tokens_after_step]
            step_results.append({
                "num_reasoning_steps": step_num + 1,  # 1-indexed step count
                "end_token_position": boundary_pos,   # Position of \n
                "num_reasoning_tokens": num_tokens_after_step,
                "answer": early_result["answer"],
                "matches_final": early_result["matches_final"],
            })

    # 6. Compute statistics
    question_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return {
        "question": question_text,
        "final_answer": final_answer,
        "num_reasoning_tokens": num_reasoning_tokens,
        "num_reasoning_steps": len(step_boundaries),
        "early_stopping_results": early_results,
        "step_results": step_results,
        "step_boundaries": step_boundaries,
        "num_reasoning_tokens_first_match": find_first_match(early_results),
        "num_reasoning_tokens_stable_match": find_stable_match(early_results),
    }


# =============================================================================
# Coconut Early Stopping
# =============================================================================

def process_coconut_early_stopping(model, dataset, sample_idx, max_latents, args):
    """
    Process a single Coconut sample for early stopping analysis.

    Args:
        model: Coconut model
        dataset: Full dataset
        sample_idx: Index of sample to process
        max_latents: Maximum number of latent tokens
        args: Command-line arguments

    Returns:
        Dictionary with early stopping results, or None if failed
    """
    # 1. Get final answer (with max latents)
    final_samples = DatasetAdapter.adapt_batch_for_model(
        dataset, model, [sample_idx], num_latents=max_latents
    )
    final_sample = final_samples[0]

    with torch.no_grad():
        final_output = model.generate(
            final_sample["input_ids"].to(model.device),
            final_sample["attention_mask"].to(model.device),
            max_new_tokens=args.max_new_tokens
        )

    final_text = model.decode(final_output.output_ids)
    final_answer = extract_answer(final_text, "###")

    if final_answer is None:
        return None  # Skip samples without valid delimiter

    # Get delimiter tokens
    delimiter_ids = get_delimiter_tokens(model.tokenizer, "###").to(model.device)

    # 2. Try each latent count
    early_results = []
    for k in range(max_latents + 1):
        # Prepare input with k latents (but don't generate yet)
        samples_k = DatasetAdapter.adapt_batch_for_model(
            dataset, model, [sample_idx], num_latents=k
        )
        sample_k = samples_k[0]

        # Teacher-force "###" after the latents
        # Append delimiter to the input
        sample_k_input = sample_k["input_ids"].to(model.device)
        if sample_k_input.dim() > 1:
            sample_k_input = sample_k_input.squeeze(0)

        input_with_delimiter = torch.cat([
            sample_k_input,
            delimiter_ids
        ]).unsqueeze(0)

        with torch.no_grad():
            output_k = model.generate(
                input_with_delimiter.to(model.device),
                attention_mask=torch.ones_like(input_with_delimiter).to(model.device),
                max_new_tokens=32  # Just the answer
            )

        text_k = model.decode(output_k.output_ids)
        answer_k = extract_answer(text_k, "###")

        result = {
            "num_reasoning_tokens": k,  # num latent tokens
            "answer": answer_k,
            "matches_final": normalize_answer(answer_k) == normalize_answer(final_answer)
        }

        # Include full output if debug mode
        if args.debug:
            result["full_output"] = text_k

        early_results.append(result)

    # 3. Get question text
    question_text = model.tokenizer.decode(
        dataset[sample_idx]["question_tokenized"],
        skip_special_tokens=True
    )

    return {
        "question": question_text,
        "final_answer": final_answer,
        "num_reasoning_tokens": max_latents,
        "early_stopping_results": early_results,
        "num_reasoning_tokens_first_match": find_first_match(early_results),
        "num_reasoning_tokens_stable_match": find_stable_match(early_results),
    }


# =============================================================================
# CODI Early Stopping
# =============================================================================

def process_codi_early_stopping(model, sample, max_iterations, args):
    """
    Process a single CODI sample for early stopping analysis.

    Args:
        model: CODI model
        sample: Dataset sample
        max_iterations: Maximum number of iterations
        args: Command-line arguments

    Returns:
        Dictionary with early stopping results, or None if failed
    """
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Move to device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # 1. Get final answer (with max iterations, no teacher-forcing)
    with torch.no_grad():
        final_output = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_iterations=max_iterations
        )

    final_text = model.decode(final_output.output_ids, skip_special_tokens=False)
    final_answer = extract_answer(final_text, "The answer is:")

    if final_answer is None:
        return None  # Skip samples without valid answer

    # 2. Try each iteration count with teacher-forcing
    early_results = []
    for k in range(max_iterations + 1):
        # Generate with k iterations, teacher-forcing "The answer is:" after EOT
        with torch.no_grad():
            output_k = model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=32,  # Answer tokens only
                num_iterations=k,
                teacher_force_prefix="The answer is:"  # Teacher-force delimiter after EOT
            )

        text_k = model.decode(output_k.output_ids, skip_special_tokens=False)

        answer_k = extract_answer(text_k, "The answer is:")

        result = {
            "num_reasoning_tokens": k,  # num iterations
            "answer": answer_k,
            "matches_final": normalize_answer(answer_k) == normalize_answer(final_answer)
        }

        # Include full output if debug mode
        if args.debug:
            result["full_output"] = text_k

        early_results.append(result)

    # 3. Get question text
    # Use the model's decode method which handles CODI special tokens
    # Remove the BOT token at the end before decoding
    input_ids_without_bot = input_ids[:, :-1] if input_ids[0, -1] == model.special_tokens.get("bot", -1) else input_ids
    question_text = model.tokenizer.decode(input_ids_without_bot[0], skip_special_tokens=True)

    return {
        "question": question_text,
        "final_answer": final_answer,
        "num_reasoning_tokens": max_iterations,
        "early_stopping_results": early_results,
        "num_reasoning_tokens_first_match": find_first_match(early_results),
        "num_reasoning_tokens_stable_match": find_stable_match(early_results),
    }


# =============================================================================
# Multimode Early Stopping
# =============================================================================

def get_ground_truth_answer(sample: dict, dataset_path: str) -> str:
    """
    Extract ground truth answer from sample.

    Args:
        sample: Dataset sample with answer_tokenized
        dataset_path: Path to dataset (used to determine format)

    Returns:
        Ground truth answer string
    """
    answer_tokens = sample.get("answer_tokenized")
    if answer_tokens is None:
        return None

    # Convert to list if tensor
    if hasattr(answer_tokens, 'tolist'):
        answer_tokens = answer_tokens.tolist()

    # Decode the answer (we need the tokenizer, but we can get it from context)
    # For now, return the token IDs - the caller will decode
    return answer_tokens


def process_multimode_latent_early_stopping(model, sample, max_iterations, args):
    """
    Process a single multimode sample for early stopping analysis in latent mode.

    Varies num_iterations from 0 to max_iterations and tracks both
    matches_final AND ground_truth_correct for each iteration.

    Args:
        model: Multimode model (multimode_codi or multimode_coconut)
        sample: Dataset sample
        max_iterations: Maximum number of iterations
        args: Command-line arguments

    Returns:
        Dictionary with early stopping results, or None if failed
    """
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Move to device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Get delimiter based on model type
    delimiter = "The answer is:" if model.model_type == "multimode_codi" else "###"

    # 1. Get final answer (with max iterations)
    with torch.no_grad():
        final_output = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=args.max_new_tokens,
            mode="latent",
            num_iterations=max_iterations
        )

    final_text = model.decode(final_output.output_ids, skip_special_tokens=False)
    final_answer = extract_answer(final_text, delimiter)

    if final_answer is None:
        return None  # Skip samples without valid answer

    # Get ground truth answer
    ground_truth_tokens = sample.get("answer_tokenized")
    if ground_truth_tokens is not None:
        if hasattr(ground_truth_tokens, 'tolist'):
            ground_truth_tokens = ground_truth_tokens.tolist()
        ground_truth_raw = model.tokenizer.decode(ground_truth_tokens, skip_special_tokens=True)
        # Clean up ground truth - strip delimiter prefix and whitespace
        ground_truth = extract_answer(ground_truth_raw, "###")
        if ground_truth is None:
            # Fallback: just strip the raw text
            ground_truth = ground_truth_raw.strip().replace(",", "")
    else:
        ground_truth = None

    # 2. Try each iteration count
    early_results = []
    for k in range(max_iterations + 1):
        # For multimode models, we need to re-prepare input with correct latent count
        # But the input_ids already have the latent structure, so we just vary num_iterations

        # Generate with k iterations
        with torch.no_grad():
            output_k = model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=32,  # Just the answer
                mode="latent",
                num_iterations=k
            )

        text_k = model.decode(output_k.output_ids, skip_special_tokens=False)
        answer_k = extract_answer(text_k, delimiter)

        # Check matches
        matches_final = normalize_answer(answer_k) == normalize_answer(final_answer)
        ground_truth_correct = (
            normalize_answer(answer_k) == normalize_answer(ground_truth)
            if ground_truth else None
        )

        result = {
            "num_reasoning_tokens": k,  # num iterations
            "answer": answer_k,
            "matches_final": matches_final,
            "ground_truth_correct": ground_truth_correct
        }

        # Include full output if debug mode
        if args.debug:
            result["full_output"] = text_k

        early_results.append(result)

    # 3. Get question text
    question_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return {
        "question": question_text,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "num_reasoning_tokens": max_iterations,
        "early_stopping_results": early_results,
        "num_reasoning_tokens_first_match": find_first_match(early_results),
        "num_reasoning_tokens_stable_match": find_stable_match(early_results),
    }


def process_multimode_verbalized_early_stopping(model, sample, args):
    """
    Process a single multimode sample for early stopping analysis in verbalized mode.

    Generates full CoT, then forces answer at step boundaries (newlines).
    Only produces step_results (no token-level analysis).

    Args:
        model: Multimode model (multimode_codi or multimode_coconut)
        sample: Dataset sample
        args: Command-line arguments

    Returns:
        Dictionary with early stopping results, or None if failed
    """
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Move to device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Get delimiter based on model type
    delimiter = "The answer is:" if model.model_type == "multimode_codi" else "###"

    # 1. Run full verbalized inference
    with torch.no_grad():
        final_output = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=args.max_new_tokens,
            mode="verbalized"
        )

    final_text = model.decode(final_output.output_ids, skip_special_tokens=False)
    final_answer = extract_answer(final_text, delimiter)

    if final_answer is None:
        return None  # Skip samples without valid answer

    # Get ground truth answer
    ground_truth_tokens = sample.get("answer_tokenized")
    if ground_truth_tokens is not None:
        if hasattr(ground_truth_tokens, 'tolist'):
            ground_truth_tokens = ground_truth_tokens.tolist()
        ground_truth_raw = model.tokenizer.decode(ground_truth_tokens, skip_special_tokens=True)
        # Clean up ground truth - strip delimiter prefix and whitespace
        ground_truth = extract_answer(ground_truth_raw, "###")
        if ground_truth is None:
            # Fallback: just strip the raw text
            ground_truth = ground_truth_raw.strip().replace(",", "")
    else:
        ground_truth = None

    # 2. Extract reasoning tokens (between bocot and eocot or answer)
    output_ids = final_output.output_ids
    if output_ids.dim() > 1:
        output_ids = output_ids[0]

    # Find bocot position
    bocot_id = model.special_tokens.get("bocot")
    eocot_id = model.special_tokens.get("eocot")

    bocot_positions = (output_ids == bocot_id).nonzero(as_tuple=True)[0]
    if len(bocot_positions) == 0:
        return None

    reasoning_start = bocot_positions[0].item() + 1

    # Find end of reasoning (eocot or end of sequence)
    eocot_positions = (output_ids == eocot_id).nonzero(as_tuple=True)[0]
    if len(eocot_positions) > 0:
        reasoning_end = eocot_positions[0].item()
    else:
        reasoning_end = len(output_ids)

    reasoning_tokens = output_ids[reasoning_start:reasoning_end]

    # 3. Find step boundaries (newlines)
    step_boundaries = find_step_boundaries(model.tokenizer, reasoning_tokens)
    num_reasoning_steps = len(step_boundaries)

    # 4. Build step-level results
    step_results = []
    for step_num, boundary_pos in enumerate(step_boundaries):
        # Get tokens up to and including this step
        num_tokens_after_step = boundary_pos + 1
        prefix_reasoning = reasoning_tokens[:num_tokens_after_step]

        # Construct: input + bocot + partial_reasoning + eocot
        prefix_ids = torch.cat([
            input_ids[0],
            torch.tensor([bocot_id], device=model.device),
            prefix_reasoning.to(model.device),
            torch.tensor([eocot_id], device=model.device)
        ]).unsqueeze(0)

        # Generate answer
        with torch.no_grad():
            step_output = model.generate(
                prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                max_new_tokens=32,  # Just the answer
                mode="verbalized"  # Continue in verbalized mode
            )

        step_text = model.decode(step_output.output_ids, skip_special_tokens=False)
        step_answer = extract_answer(step_text, delimiter)

        matches_final = normalize_answer(step_answer) == normalize_answer(final_answer)
        ground_truth_correct = (
            normalize_answer(step_answer) == normalize_answer(ground_truth)
            if ground_truth else None
        )

        step_results.append({
            "num_reasoning_steps": step_num + 1,
            "end_token_position": boundary_pos,
            "num_reasoning_tokens": num_tokens_after_step,
            "answer": step_answer,
            "matches_final": matches_final,
            "ground_truth_correct": ground_truth_correct
        })

    # Build early_stopping_results from step_results for compatibility
    early_results = []
    for sr in step_results:
        early_results.append({
            "num_reasoning_tokens": sr["num_reasoning_tokens"],
            "answer": sr["answer"],
            "matches_final": sr["matches_final"],
            "ground_truth_correct": sr["ground_truth_correct"]
        })

    # 5. Get question text
    question_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return {
        "question": question_text,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "num_reasoning_tokens": len(reasoning_tokens),
        "num_reasoning_steps": num_reasoning_steps,
        "early_stopping_results": early_results,
        "step_results": step_results,
        "step_boundaries": step_boundaries,
        "num_reasoning_tokens_first_match": find_first_match(early_results),
        "num_reasoning_tokens_stable_match": find_stable_match(early_results),
    }


def compute_accuracy_by_iteration(samples_results: list) -> dict:
    """
    Compute accuracy vs ground truth at each iteration count.

    Args:
        samples_results: List of sample results with ground_truth_correct field

    Returns:
        Dictionary mapping iteration count to accuracy stats:
        {k: {"accuracy": float, "count": int, "std_error": float}}
    """
    import math

    # Group results by iteration count
    by_iteration = {}

    for sample in samples_results:
        early_results = sample.get("early_stopping_results", [])
        for result in early_results:
            k = result.get("num_reasoning_tokens")
            correct = result.get("ground_truth_correct")

            if k is None or correct is None:
                continue

            if k not in by_iteration:
                by_iteration[k] = {"correct": 0, "total": 0}

            by_iteration[k]["total"] += 1
            if correct:
                by_iteration[k]["correct"] += 1

    # Compute accuracy and standard error for each k
    accuracy_by_iteration = {}
    for k, counts in sorted(by_iteration.items()):
        n = counts["total"]
        if n == 0:
            continue

        accuracy = counts["correct"] / n

        # Standard error for proportion: sqrt(p*(1-p)/n)
        if accuracy > 0 and accuracy < 1:
            std_error = math.sqrt(accuracy * (1 - accuracy) / n)
        else:
            std_error = 0.0

        accuracy_by_iteration[k] = {
            "accuracy": accuracy,
            "count": n,
            "std_error": std_error
        }

    return accuracy_by_iteration


# =============================================================================
# Vocabulary Projection
# =============================================================================

def find_reasoning_end_position(output_ids, model_type, special_tokens=None):
    """
    Find the end position of reasoning in the output.

    Args:
        output_ids: Output token IDs [batch, seq_len] or [seq_len]
        model_type: Type of model
        special_tokens: Optional dict of special token IDs

    Returns:
        End position of reasoning (exclusive)
    """
    # Ensure 1D
    if output_ids.dim() > 1:
        output_ids = output_ids[0]

    if model_type == "coconut":
        # Find <|end-latent|> token
        if special_tokens and "end_latent" in special_tokens:
            end_positions = (output_ids == special_tokens["end_latent"]).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                return end_positions[0].item()

    elif model_type == "codi":
        # Find <|eot|> token
        if special_tokens and "eot" in special_tokens:
            eot_positions = (output_ids == special_tokens["eot"]).nonzero(as_tuple=True)[0]
            if len(eot_positions) > 0:
                return eot_positions[0].item()

    elif model_type == "cot":
        # Find "###" token
        hash_tokens = [21017, 44386]
        for i, token_id in enumerate(output_ids.tolist()):
            if token_id in hash_tokens:
                return i

    # Default: return full length
    return len(output_ids)


def find_rank_stable_position(top_k_by_position):
    """
    Walk backwards from end to find where top-k ranking stabilized.

    Args:
        top_k_by_position: List of top-k token ID lists at each position

    Returns:
        First position where ranking becomes stable until end
    """
    if len(top_k_by_position) <= 1:
        return 0

    # Compare token ID lists (exact match required for stability)
    final_top_k = list(top_k_by_position[-1])
    for i in range(len(top_k_by_position) - 2, -1, -1):
        current_top_k = list(top_k_by_position[i])
        if current_top_k != final_top_k:
            return i + 1  # Stabilized at position after this
    return 0  # Was stable from the start


def process_vocab_projection_by_token(model, analyzer, sample, max_units, args, dataset=None, sample_idx=None):
    """
    Process vocabulary projection analysis for a sample (token-level).
    Analyzes ALL reasoning tokens/latents/iterations linearly.

    Args:
        model: Model instance
        analyzer: UnifiedAnalyzer instance
        sample: Dataset sample
        max_units: Maximum reasoning units (latents/iterations, or None for CoT)
        args: Command-line arguments
        dataset: Original dataset (needed for Coconut to get question_tokenized)
        sample_idx: Sample index (needed for Coconut to get question_tokenized)

    Returns:
        Dictionary with vocab projection results including debugging info, or None if failed
    """
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Ensure 2D and move to device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # 1. Run generation with activation capture
    gen_kwargs = {}
    if model.model_type == "codi" and max_units is not None:
        gen_kwargs["num_iterations"] = max_units

    try:
        result = analyzer.analyze_with_capture(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            layer_indices=None,  # Capture all layers
            **gen_kwargs
        )
    except Exception as e:
        print(f"Warning: Vocab projection failed: {e}")
        return None

    # 2. Extract reasoning region activations
    activations = result["activations"]
    if not activations:
        return None

    final_layer_idx = max(activations.keys())
    final_layer_acts = activations[final_layer_idx]  # [batch, seq_len, hidden_dim]

    if final_layer_acts is None:
        return None

    # 3. Find reasoning region boundaries
    output_ids = result["output"].output_ids
    if output_ids.dim() > 1:
        output_ids_1d = output_ids[0]
    else:
        output_ids_1d = output_ids

    # Get input length
    if input_ids.dim() > 1:
        input_len = input_ids.shape[1]
    else:
        input_len = len(input_ids)

    # Reasoning region depends on model type
    if model.model_type == "cot":
        # For CoT: reasoning is GENERATED (after input, before "###")
        reasoning_start = input_len
        reasoning_end = find_reasoning_end_position(
            output_ids,
            model.model_type,
            model.special_tokens if hasattr(model, 'special_tokens') else None
        )

    elif model.model_type == "coconut":
        # For Coconut: latents are IN THE INPUT (after question, before generation)
        # Need question length to find where latents start
        if dataset is not None and sample_idx is not None:
            question_tokens = dataset[sample_idx]["question_tokenized"]
            if hasattr(question_tokens, '__len__'):
                question_len = len(question_tokens)
            else:
                question_len = question_tokens.shape[-1] if hasattr(question_tokens, 'shape') else len(question_tokens)
            reasoning_start = question_len
            # Find <|end-latent|> token and exclude it
            reasoning_end = find_reasoning_end_position(
                input_ids,  # Use input_ids for Coconut since latents are in input
                model.model_type,
                model.special_tokens if hasattr(model, 'special_tokens') else None
            )
        else:
            print("Warning: Cannot find latent boundaries for Coconut without dataset")
            return None

    elif model.model_type == "codi":
        # For CODI: Include <|bot|> token (last token in input) and generated latent tokens
        reasoning_start = input_len - 1  # Include <|bot|> token
        reasoning_end = find_reasoning_end_position(
            output_ids,
            model.model_type,
            model.special_tokens if hasattr(model, 'special_tokens') else None
        )

    else:
        # Unknown model type
        reasoning_start = input_len
        reasoning_end = len(output_ids_1d)

    # Extract reasoning activations
    reasoning_acts = final_layer_acts[:, reasoning_start:reasoning_end, :]

    if reasoning_acts.shape[1] == 0:
        # No reasoning tokens
        return None

    # 4. Project to vocab
    try:
        vocab_result = analyzer.project_activations_to_vocab(
            reasoning_acts,
            top_k=args.top_k,
            return_probs=True
        )
    except Exception as e:
        print(f"Warning: Projection to vocab failed: {e}")
        return None

    # 5. Get final answer
    delimiter = result["output"].metadata.get("delimiter", "###") if result["output"].metadata else "###"
    final_answer = extract_answer(result["decoded"], delimiter)

    if final_answer is None:
        return None

    # 6. Check answer in top-k
    # Extract the appropriate token string based on dataset format
    # For ProSQA: "A is a B" -> look for "B"
    # For GSM8k: numeric answer
    answer_token_str = get_answer_token_string(final_answer, args.dataset_path)

    try:
        comparison = analyzer.compare_with_answer(
            reasoning_acts,
            answer_token_str,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"Warning: Answer comparison failed: {e}")
        comparison = {
            "first_appearance": None,
            "positions": [],
            "ranks": []
        }

    # 7. Find rank stabilization
    top_k_indices = vocab_result["top_k_indices"]  # [batch, seq_len, top_k]
    top_k_by_position = [top_k_indices[0, i, :].tolist() for i in range(top_k_indices.shape[1])]
    rank_stable_pos = find_rank_stable_position(top_k_by_position)

    # 8. Build debugging info (results by position)
    results_by_position = []
    num_positions = reasoning_acts.shape[1]

    for pos in range(num_positions):
        # Get top-k token IDs at this position
        top_k_ids = top_k_indices[0, pos, :].tolist()

        # Decode to text
        top_k_tokens = [model.tokenizer.decode([tid]) for tid in top_k_ids]

        # Check if answer is in top-k at this position
        answer_in_topk = pos in comparison["positions"]
        rank = comparison["ranks"][comparison["positions"].index(pos)] if answer_in_topk else None

        results_by_position.append({
            "position": pos,
            "answer_in_top_k": answer_in_topk,
            "rank": rank,
            "top_k_tokens": top_k_tokens
        })

    return {
        "first_position_answer_in_top_k": comparison["first_appearance"],
        "positions_with_answer": comparison["positions"],
        "ranks": comparison["ranks"],
        "rank_stable_position": rank_stable_pos,
        "num_reasoning_positions": num_positions,
        "results_by_position": results_by_position  # NEW: debugging info
    }


def process_vocab_projection_by_step(model, analyzer, sample, step_boundaries, args):
    """
    Process vocabulary projection analysis for a sample (step-level, CoT only).
    Analyzes hidden states ONLY at step boundaries (newline positions).

    Args:
        model: Model instance
        analyzer: UnifiedAnalyzer instance
        sample: Dataset sample
        step_boundaries: List of token positions where steps end (newline positions)
        args: Command-line arguments

    Returns:
        Dictionary with step-level vocab projection results, or None if failed
    """
    if not step_boundaries:
        return None

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Ensure 2D and move to device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # 1. Run generation with activation capture
    try:
        result = analyzer.analyze_with_capture(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            layer_indices=None  # Capture all layers
        )
    except Exception as e:
        print(f"Warning: Step-level vocab projection failed: {e}")
        return None

    # 2. Extract reasoning region activations
    activations = result["activations"]
    if not activations:
        return None

    final_layer_idx = max(activations.keys())
    final_layer_acts = activations[final_layer_idx]  # [batch, seq_len, hidden_dim]

    if final_layer_acts is None:
        return None

    # 3. Find reasoning region boundaries
    output_ids = result["output"].output_ids
    if output_ids.dim() > 1:
        output_ids_1d = output_ids[0]
    else:
        output_ids_1d = output_ids

    # Get input length
    if input_ids.dim() > 1:
        input_len = input_ids.shape[1]
    else:
        input_len = len(input_ids)

    # Reasoning region depends on model type
    if model.model_type == "cot":
        # For CoT: reasoning is GENERATED (after input, before "###")
        reasoning_start = input_len
        reasoning_end = find_reasoning_end_position(
            output_ids,
            model.model_type,
            model.special_tokens if hasattr(model, 'special_tokens') else None
        )

    elif model.model_type == "coconut":
        # For Coconut: latents are IN THE INPUT (after question, before generation)
        # Need question length to find where latents start
        if dataset is not None and sample_idx is not None:
            question_tokens = dataset[sample_idx]["question_tokenized"]
            if hasattr(question_tokens, '__len__'):
                question_len = len(question_tokens)
            else:
                question_len = question_tokens.shape[-1] if hasattr(question_tokens, 'shape') else len(question_tokens)
            reasoning_start = question_len
            # Find <|end-latent|> token and exclude it
            reasoning_end = find_reasoning_end_position(
                input_ids,  # Use input_ids for Coconut since latents are in input
                model.model_type,
                model.special_tokens if hasattr(model, 'special_tokens') else None
            )
        else:
            print("Warning: Cannot find latent boundaries for Coconut without dataset")
            return None

    elif model.model_type == "codi":
        # For CODI: Include <|bot|> token (last token in input) and generated latent tokens
        reasoning_start = input_len - 1  # Include <|bot|> token
        reasoning_end = find_reasoning_end_position(
            output_ids,
            model.model_type,
            model.special_tokens if hasattr(model, 'special_tokens') else None
        )

    else:
        # Unknown model type
        reasoning_start = input_len
        reasoning_end = len(output_ids_1d)

    # Extract reasoning activations
    reasoning_acts = final_layer_acts[:, reasoning_start:reasoning_end, :]

    if reasoning_acts.shape[1] == 0:
        return None

    # 4. Get final answer
    delimiter = result["output"].metadata.get("delimiter", "###") if result["output"].metadata else "###"
    final_answer = extract_answer(result["decoded"], delimiter)

    if final_answer is None:
        return None

    # Extract the appropriate token string based on dataset format
    # For ProSQA: "A is a B" -> look for "B"
    # For GSM8k: numeric answer
    answer_token_str = get_answer_token_string(final_answer, args.dataset_path)

    # 5. Project to vocab ONLY at step boundaries
    num_steps = len(step_boundaries)
    results_by_step = []
    steps_with_answer = []
    ranks = []
    first_step = None
    top_k_by_step = []

    for step_num, boundary_pos in enumerate(step_boundaries):
        # boundary_pos is relative to reasoning start
        if boundary_pos >= reasoning_acts.shape[1]:
            continue  # Skip if boundary is out of range

        # Get hidden state at step boundary
        hidden_state = reasoning_acts[:, boundary_pos:boundary_pos+1, :]  # [1, 1, hidden]

        # Project to vocab
        try:
            vocab_result = analyzer.project_activations_to_vocab(
                hidden_state,
                top_k=args.top_k,
                return_probs=True
            )
        except Exception as e:
            print(f"Warning: Step {step_num+1} projection failed: {e}")
            continue

        # Get top-k token IDs
        top_k_indices = vocab_result["top_k_indices"]  # [1, 1, top_k]
        top_k_ids = top_k_indices[0, 0, :].tolist()
        top_k_by_step.append(top_k_ids)

        # Decode to text
        top_k_tokens = [model.tokenizer.decode([tid]) for tid in top_k_ids]

        # Check if answer is in top-k at this step
        try:
            comparison = analyzer.compare_with_answer(
                hidden_state,
                answer_token_str,
                top_k=args.top_k
            )

            answer_in_topk = comparison["first_appearance"] is not None
            rank = comparison["ranks"][0] if answer_in_topk else None

            if answer_in_topk:
                if first_step is None:
                    first_step = step_num + 1  # 1-indexed
                steps_with_answer.append(step_num + 1)
                ranks.append(rank)

        except Exception as e:
            print(f"Warning: Step {step_num+1} answer comparison failed: {e}")
            answer_in_topk = False
            rank = None

        results_by_step.append({
            "step": step_num + 1,  # 1-indexed
            "position": boundary_pos,  # Token position relative to reasoning start
            "answer_in_top_k": answer_in_topk,
            "rank": rank,
            "top_k_tokens": top_k_tokens
        })

    # 6. Find rank stabilization at step level
    rank_stable_step = find_rank_stable_position(top_k_by_step)
    if rank_stable_step is not None and len(results_by_step) > 0:
        rank_stable_step = results_by_step[min(rank_stable_step, len(results_by_step)-1)]["step"]

    return {
        "first_step_answer_in_top_k": first_step,
        "steps_with_answer": steps_with_answer,
        "ranks": ranks,
        "rank_stable_step": rank_stable_step,
        "num_reasoning_steps": num_steps,
        "results_by_step": results_by_step  # Debugging info
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(args):
    """
    Main experiment runner - runs early stopping and vocab projection.

    Args:
        args: Command-line arguments
    """
    # 1. Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Early Stopping Experiment")
    print("=" * 70)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {device}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("")

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Load model using ModelFactory
    print(f"Loading {args.model_type} model...")
    model_kwargs = {}
    if args.model_type in ["cot", "coconut", "codi"]:
        model_kwargs["model_id"] = args.model_id
    if args.model_type == "codi":
        model_kwargs["num_latent"] = args.num_latents
    if args.model_type in ["multimode_codi", "multimode_coconut"]:
        model_kwargs["model_id"] = args.model_id
        model_kwargs["num_latent"] = args.num_latents

    model = ModelFactory.create(
        model_type=args.model_type,
        model_path=args.model_path,
        device=str(device),
        **model_kwargs
    )

    # 4. Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    dataset = load_dataset(
        dataset_path=args.dataset_path,
        tokenizer=model.tokenizer,
        max_size=args.max_samples
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # 5. Determine max reasoning units per model type
    if args.model_type == "coconut":
        max_units = args.num_latents
        print(f"Max latent tokens: {max_units}")
    elif args.model_type == "codi":
        max_units = args.num_latents  # num_iterations
        print(f"Max iterations: {max_units}")
    elif args.model_type in ["multimode_codi", "multimode_coconut"]:
        max_units = args.num_latents  # num_iterations
        print(f"Max iterations: {max_units}")
        print(f"Mode: {args.mode}")
    else:  # cot
        max_units = None  # Will be determined per sample
        print(f"Max reasoning tokens: determined per sample")

    # 6. Create analyzer
    analyzer = UnifiedAnalyzer(model)

    # 7. Process each sample
    print(f"\nProcessing samples...")
    results = []
    num_skipped_early_stop = 0
    num_skipped_vocab_proj = 0

    num_samples = len(dataset)

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc=f"Processing {args.model_type}"):
            # Run early stopping analysis
            if args.model_type == "cot":
                # Prepare sample using DatasetAdapter
                samples = DatasetAdapter.adapt_batch_for_model(
                    dataset, model, [idx]
                )
                sample = samples[0]
                early_stop_result = process_cot_early_stopping(model, sample, args)

            elif args.model_type == "coconut":
                early_stop_result = process_coconut_early_stopping(
                    model, dataset, idx, max_units, args
                )

            elif args.model_type == "codi":
                # Prepare sample
                samples = DatasetAdapter.adapt_batch_for_model(
                    dataset, model, [idx], num_iterations=max_units
                )
                sample = samples[0]
                early_stop_result = process_codi_early_stopping(
                    model, sample, max_units, args
                )

            elif args.model_type in ["multimode_codi", "multimode_coconut"]:
                # Prepare sample with mode
                samples = DatasetAdapter.adapt_batch_for_model(
                    dataset, model, [idx], num_iterations=max_units, mode=args.mode
                )
                sample = samples[0]

                if args.mode == "latent":
                    early_stop_result = process_multimode_latent_early_stopping(
                        model, sample, max_units, args
                    )
                elif args.mode == "verbalized":
                    early_stop_result = process_multimode_verbalized_early_stopping(
                        model, sample, args
                    )
                else:  # direct mode - no reasoning, skip
                    continue

            if early_stop_result is None:
                num_skipped_early_stop += 1
                continue

            # Run vocab projection analysis (if enabled)
            if args.skip_vocab_projection:
                vocab_proj_by_token = None
                vocab_proj_by_step = None
            else:
                # Get sample for vocab projection
                if args.model_type == "cot":
                    samples = DatasetAdapter.adapt_batch_for_model(
                        dataset, model, [idx]
                    )
                    sample_vp = samples[0]
                elif args.model_type == "coconut":
                    samples = DatasetAdapter.adapt_batch_for_model(
                        dataset, model, [idx], num_latents=max_units
                    )
                    sample_vp = samples[0]
                elif args.model_type == "codi":
                    samples = DatasetAdapter.adapt_batch_for_model(
                        dataset, model, [idx], num_iterations=max_units
                    )
                    sample_vp = samples[0]
                else:  # multimode models
                    samples = DatasetAdapter.adapt_batch_for_model(
                        dataset, model, [idx], num_iterations=max_units, mode=args.mode
                    )
                    sample_vp = samples[0]

                # Token-level vocab projection (all models)
                vocab_proj_by_token = process_vocab_projection_by_token(
                    model, analyzer, sample_vp, max_units, args,
                    dataset=dataset, sample_idx=idx
                )

                if vocab_proj_by_token is None:
                    num_skipped_vocab_proj += 1

                # Step-level vocab projection (CoT only)
                vocab_proj_by_step = None
                if args.model_type == "cot" and early_stop_result.get("step_boundaries"):
                    vocab_proj_by_step = process_vocab_projection_by_step(
                        model, analyzer, sample_vp,
                        early_stop_result["step_boundaries"],
                        args
                    )

            # Combine results
            result = {
                "sample_idx": idx,
                **early_stop_result,
                "vocab_projection_by_token": vocab_proj_by_token,
                "vocab_projection_by_step": vocab_proj_by_step
            }
            results.append(result)

    print(f"\nProcessed {len(results)} samples")
    print(f"  Skipped (early stopping): {num_skipped_early_stop}")
    print(f"  Skipped (vocab projection): {num_skipped_vocab_proj}")

    # 8. Compute summary statistics
    early_stop_summary = compute_summary(results)
    vocab_proj_by_token_summary = compute_vocab_projection_summary(results)
    vocab_proj_by_step_summary = compute_vocab_projection_by_step_summary(results)

    # Compute accuracy by iteration for multimode models
    accuracy_by_iteration = None
    if args.model_type in ["multimode_codi", "multimode_coconut"]:
        accuracy_by_iteration = compute_accuracy_by_iteration(results)

    # 9. Build output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.dirname(args.model_path)) if "/" in args.model_path else args.model_path
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    base_model_name = get_base_model_short_name(args.model_id, args.model_path, args.model_type)

    output = {
        "timestamp": timestamp,
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "config": {
            "base_model": base_model_name,
            "num_latents": args.num_latents if args.model_type in ["coconut", "codi", "multimode_codi", "multimode_coconut"] else None,
            "mode": args.mode if args.model_type in ["multimode_codi", "multimode_coconut"] else None,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "max_samples": args.max_samples,
            "device": str(device)
        },
        "samples": results,
        "summary": {
            "early_stopping": early_stop_summary,
            "vocab_projection_by_token": vocab_proj_by_token_summary,
            "vocab_projection_by_step": vocab_proj_by_step_summary,
            "accuracy_by_iteration": accuracy_by_iteration
        }
    }

    # 10. Save output
    # k value is the top_k argument
    k_value = args.top_k

    # Include mode in filename for multimode models
    if args.model_type in ["multimode_codi", "multimode_coconut"]:
        output_filename = f"early_stopping_{args.model_type}_{args.mode}_{base_model_name}_{model_name}_{dataset_name}_k{k_value}_{timestamp}.json"
    else:
        output_filename = f"early_stopping_{args.model_type}_{base_model_name}_{model_name}_{dataset_name}_k{k_value}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # 11. Print summaries
    print(f"\n{'=' * 70}")
    print("EARLY STOPPING SUMMARY")
    print('=' * 70)
    print(f"  Total samples: {early_stop_summary.get('total_samples', 0)}")
    print(f"  Avg reasoning tokens: {early_stop_summary.get('avg_reasoning_tokens', 0):.2f}")
    first_match = early_stop_summary.get('avg_num_reasoning_tokens_first_match')
    stable_match = early_stop_summary.get('avg_num_reasoning_tokens_stable_match')
    print(f"  Avg first match: {first_match:.2f}" if first_match else "  Avg first match: N/A")
    print(f"  Avg stable match: {stable_match:.2f}" if stable_match else "  Avg stable match: N/A")

    print(f"\n{'=' * 70}")
    print("VOCAB PROJECTION SUMMARY (BY TOKEN)")
    print('=' * 70)
    print(f"  Total samples: {vocab_proj_by_token_summary.get('total_samples', 0)}")
    first_pos = vocab_proj_by_token_summary.get('avg_first_position_answer_in_top_k')
    stable_pos = vocab_proj_by_token_summary.get('avg_rank_stable_position')
    print(f"  Avg first position answer in top-k: {first_pos:.2f}" if first_pos else "  Avg first position answer in top-k: N/A")
    print(f"  Avg rank stable position: {stable_pos:.2f}" if stable_pos else "  Avg rank stable position: N/A")
    print(f"  Num with answer in top-k: {vocab_proj_by_token_summary.get('num_with_answer_in_top_k', 0)}")
    print('=' * 70)

    # Print step-level vocab projection summary if available (CoT only)
    if vocab_proj_by_step_summary is not None:
        print(f"\n{'=' * 70}")
        print("VOCAB PROJECTION SUMMARY (BY STEP - CoT only)")
        print('=' * 70)
        print(f"  Total samples: {vocab_proj_by_step_summary.get('total_samples', 0)}")
        first_step = vocab_proj_by_step_summary.get('avg_first_step_answer_in_top_k')
        stable_step = vocab_proj_by_step_summary.get('avg_rank_stable_step')
        print(f"  Avg first step answer in top-k: {first_step:.2f}" if first_step else "  Avg first step answer in top-k: N/A")
        print(f"  Avg rank stable step: {stable_step:.2f}" if stable_step else "  Avg rank stable step: N/A")
        print(f"  Num with answer in top-k: {vocab_proj_by_step_summary.get('num_with_answer_in_top_k', 0)}")
        print('=' * 70)

    # Print accuracy by iteration for multimode models
    if accuracy_by_iteration:
        print(f"\n{'=' * 70}")
        print("ACCURACY BY ITERATION (vs Ground Truth)")
        print('=' * 70)
        for k, stats in sorted(accuracy_by_iteration.items()):
            acc = stats['accuracy'] * 100
            n = stats['count']
            se = stats['std_error'] * 100
            print(f"  {k} iterations: {acc:.1f}% ± {se:.1f}% (n={n})")
        print('=' * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Early Stopping Experiment for CoT, Coconut, and CODI Models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["cot", "coconut", "codi", "multimode_codi", "multimode_coconut"],
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="latent",
        choices=["direct", "verbalized", "latent"],
        help="Inference mode for multimode models (default: latent)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/early_stopping",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=6,
        help="Number of latent tokens (Coconut) or iterations (CODI)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k tokens for vocab projection analysis"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID for tokenizer (CoT/Coconut)"
    )
    parser.add_argument(
        "--skip_vocab_projection",
        action="store_true",
        help="Skip vocabulary projection analysis (faster)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include full model outputs in results for debugging"
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
