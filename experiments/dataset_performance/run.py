"""
Dataset Performance Evaluation

Evaluate any model (CoT, Coconut, CODI) on any dataset and compute accuracy.

Usage:
    python -m experiments.dataset_performance.run \
        --model_type coconut \
        --model_path models/gsm-coconut/checkpoint_33 \
        --dataset_path data/gsm_original_test.json \
        --output_dir results/dataset_performance \
        --max_samples 100
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

from models.factory import ModelFactory
from dataset_utils.base import load_dataset
from dataset_utils.adapters import DatasetAdapter
from experiments.dataset_performance.count_tokens import COUNTER_FNS


def get_base_model_short_name(model_id: str, model_path: str, model_type: str) -> str:
    """
    Extract a short identifier for the base model.

    For CoT/Coconut: derived from --model_id
    For CODI: detected from the checkpoint path

    Returns: short name like 'gpt2' or 'llama32-1b'
    """
    if model_type in ["cot", "coconut", "no_cot", "multimode_coconut"]:
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
        # CODI: detect from path. Handles both local checkpoints
        # (e.g. ".../gpt2/ep_40/...") and HuggingFace repo IDs
        # (e.g. "zen-E/CODI-gpt2").
        path_lower = model_path.lower()
        if "gpt2" in path_lower or "gpt-2" in path_lower:
            return "gpt2"
        elif "llama-3.2-1b" in path_lower or "llama32-1b" in path_lower or "llama1b" in path_lower:
            return "llama32-1b"
        elif "llama" in path_lower:
            return "llama"
        else:
            return "unknown"


def extract_answer(text: str, delimiter: str = "###") -> str:
    """
    Extract answer from model output.

    Args:
        text: Full model output
        delimiter: Delimiter separating reasoning from answer
                  Can be "###" (Coconut/CoT) or "The answer is:" (CODI)

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
            # Remove commas from numbers
            answer = answer.replace(",", "")
            return answer

    # Fallback: Try CODI format "The answer is:"
    if "The answer is:" in text:
        parts = text.split("The answer is:")
        if len(parts) >= 2:
            answer = parts[-1].strip()
            # Remove commas from numbers
            answer = answer.replace(",", "")
            return answer

    return None


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


def compute_accuracy(predictions: list, ground_truths: list) -> dict:
    """
    Compute accuracy metrics.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Dictionary with accuracy metrics
    """
    assert len(predictions) == len(ground_truths)

    correct = 0
    total = len(predictions)
    failed_extractions = 0

    for pred, gt in zip(predictions, ground_truths):
        # Normalize both
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        if pred is None:
            failed_extractions += 1
            continue

        if pred_norm == gt_norm:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    extraction_rate = 1.0 - (failed_extractions / total) if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failed_extractions": failed_extractions,
        "extraction_rate": extraction_rate
    }


def evaluate_model(
    model,
    dataset,
    max_samples: int = None,
    max_new_tokens: int = 256,
    num_latents: int = 6,  # For Coconut and CODI
    include_latent_markers: bool = True,  # For Coconut ablation
    mode: str = "latent",  # For multimode_codi
    device: str = "cuda"
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: Loaded model instance
        dataset: Loaded dataset
        max_samples: Maximum samples to evaluate
        max_new_tokens: Max tokens to generate
        num_latents: Number of latent tokens/iterations (Coconut and CODI)
        include_latent_markers: Whether to include <|start-latent|>, <|latent|>, <|end-latent|>
            for Coconut models. If False, runs inference without latent markers (no-CoT style).
        device: Device to run on

    Returns:
        Dictionary with results
    """
    model_type = model.model_type
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    mode_str = ""
    if model_type == "coconut" and not include_latent_markers:
        mode_str = " (no-latent-markers mode)"
    elif model_type in ["multimode_codi", "multimode_coconut"]:
        mode_str = f" (mode={mode})"
    print(f"\nEvaluating {model_type} model{mode_str} on {num_samples} samples...")

    # Adapt dataset for model
    model_kwargs = {}
    if model_type == "coconut":
        model_kwargs["num_latents"] = num_latents
        model_kwargs["include_latent_markers"] = include_latent_markers
    elif model_type == "codi":
        model_kwargs["num_iterations"] = num_latents
    elif model_type in ["multimode_codi", "multimode_coconut"]:
        model_kwargs["num_iterations"] = num_latents
        model_kwargs["mode"] = mode

    adapted_samples = DatasetAdapter.adapt_batch_for_model(
        dataset=dataset,
        model=model,
        indices=range(num_samples),
        **model_kwargs
    )

    # Run inference
    results = []
    predictions = []
    ground_truths = []

    for sample in tqdm(adapted_samples, desc="Running inference"):
        # Get ground truth answer
        gt_answer = model.tokenizer.decode(
            sample["answer_tokenized"],
            skip_special_tokens=True
        )
        # Extract just the answer part (after ###)
        gt_answer = extract_answer(gt_answer, delimiter="###")
        if gt_answer is None:
            gt_answer = model.tokenizer.decode(
                sample["answer_tokenized"],
                skip_special_tokens=True
            ).strip()

        # Run inference
        generation_kwargs = {}
        if model_type == "codi" and "num_iterations" in sample:
            generation_kwargs["num_iterations"] = sample["num_iterations"]
        elif model_type in ["multimode_codi", "multimode_coconut"]:
            generation_kwargs["num_iterations"] = num_latents
            generation_kwargs["mode"] = mode

        output = model.generate(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )

        # Decode output
        skip_special = output.metadata.get("skip_special_tokens", True)
        output_text = model.decode(output.output_ids, skip_special_tokens=skip_special)

        # Extract predicted answer
        delimiter = output.metadata.get("delimiter", "###")
        pred_answer = extract_answer(output_text, delimiter=delimiter)

        # Store results
        predictions.append(pred_answer)
        ground_truths.append(gt_answer)

        results.append({
            "idx": sample["idx"],
            "output": output_text,
            "predicted_answer": pred_answer,
            "ground_truth": gt_answer,
            "correct": normalize_answer(pred_answer) == normalize_answer(gt_answer)
        })

    # Compute metrics
    metrics = compute_accuracy(predictions, ground_truths)

    return {
        "metrics": metrics,
        "results": results,
        "num_samples": num_samples,
        "model_type": model_type
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on dataset"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["cot", "coconut", "codi", "no_cot", "multimode_codi", "multimode_coconut"],
        help="Type of model to evaluate"
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
        default="results/dataset_performance",
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
        help="Maximum number of samples to evaluate (default: all)"
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
        help="Number of latent tokens/iterations (Coconut and CODI)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID for tokenizer (CoT/Coconut)"
    )
    parser.add_argument(
        "--no_latent_markers",
        action="store_true",
        help="(Coconut only) Run inference without <|start-latent|>, <|latent|>, <|end-latent|> tokens. "
             "Used for ablation to test if latent markers are doing meaningful work."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="latent",
        choices=["direct", "verbalized", "latent"],
        help="Inference mode for multimode_codi (default: latent)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 60)
    print("Dataset Performance Evaluation")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples or 'all'}")
    if args.model_type == "coconut":
        print(f"No latent markers: {args.no_latent_markers}")

    # Load model
    print(f"\nLoading {args.model_type} model...")
    model_kwargs = {}
    if args.model_type in ["codi", "multimode_codi", "multimode_coconut"]:
        model_kwargs["num_latent"] = args.num_latents
    if args.model_type in ["cot", "coconut", "no_cot", "codi", "multimode_codi", "multimode_coconut"]:
        model_kwargs["model_id"] = args.model_id

    model = ModelFactory.create(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
        **model_kwargs
    )

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    dataset = load_dataset(
        dataset_path=args.dataset_path,
        tokenizer=model.tokenizer,
        max_size=args.max_samples
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Run evaluation
    eval_results = evaluate_model(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_latents=args.num_latents,
        include_latent_markers=not args.no_latent_markers,
        mode=args.mode,
        device=args.device
    )

    # Count tokens for each result
    # multimode_codi and multimode_coconut use the same counter as codi
    token_counter_type = args.model_type if args.model_type not in ["multimode_codi", "multimode_coconut"] else "codi"
    count_fn = COUNTER_FNS[token_counter_type]
    total_reasoning_tokens = 0
    total_answer_tokens = 0
    for result in eval_results["results"]:
        reasoning, answer = count_fn(result["output"], model.tokenizer)
        result["reasoning_tokens"] = reasoning
        result["answer_tokens"] = answer
        total_reasoning_tokens += reasoning
        total_answer_tokens += answer

    n_samples = len(eval_results["results"])
    token_metrics = {
        "avg_reasoning_tokens": total_reasoning_tokens / n_samples,
        "avg_answer_tokens": total_answer_tokens / n_samples,
        "avg_total_output_tokens": (total_reasoning_tokens + total_answer_tokens) / n_samples,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_answer_tokens": total_answer_tokens,
        "total_output_tokens": total_reasoning_tokens + total_answer_tokens,
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics = eval_results["metrics"]
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Extraction rate: {metrics['extraction_rate']:.2%}")
    print(f"Failed extractions: {metrics['failed_extractions']}")
    print(f"\nToken Usage:")
    print(f"  Avg reasoning tokens: {token_metrics['avg_reasoning_tokens']:.1f}")
    print(f"  Avg answer tokens: {token_metrics['avg_answer_tokens']:.1f}")
    print(f"  Avg total output tokens: {token_metrics['avg_total_output_tokens']:.1f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.dirname(args.model_path))
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    base_model_name = get_base_model_short_name(args.model_id, args.model_path, args.model_type)
    # Add suffix for no-latent-markers mode or multimode modes
    mode_suffix = "_no_latent_markers" if args.no_latent_markers else ""
    if args.model_type in ["multimode_codi", "multimode_coconut"]:
        mode_suffix = f"_{args.mode}"
    output_filename = f"{args.model_type}{mode_suffix}_{base_model_name}_{model_name}_{dataset_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    output_data = {
        "timestamp": timestamp,
        "config": {
            "model_type": args.model_type,
            "model_id": args.model_id,
            "base_model": base_model_name,
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "num_latents": args.num_latents,
            "no_latent_markers": args.no_latent_markers,
            "mode": args.mode if args.model_type in ["multimode_codi", "multimode_coconut"] else None,
            "device": args.device
        },
        "metrics": metrics,
        "token_metrics": token_metrics,
        "results": eval_results["results"]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
