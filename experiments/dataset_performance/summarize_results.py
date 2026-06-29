"""
Summarize dataset performance results into Table 1 CSV format.

Reads JSON result files from the results directory and generates a CSV
matching the Table 1 format in the paper:

    Method, Base Model, GSM8k-Aug Acc (%), GSM8k-Aug # Tok, ...

Usage:
    python -m experiments.dataset_performance.summarize_results [--results_dir PATH]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


# Display names for methods and models
METHOD_DISPLAY_NAMES = {
    "no_cot": "No-CoT",
    "cot": "CoT",
    "coconut": "Coconut",
    "codi": "CODI",
}

BASE_MODEL_DISPLAY_NAMES = {
    "gpt2": "GPT-2 Small",
    "llama32-1b": "Llama-3.2-1B-Instruct",
    "llama": "Llama",
}

DATASET_DISPLAY_NAMES = {
    "gsm8k": "GSM8k-Aug",
    "gsm_original_test": "GSM8k-Aug",
    "gsm_test": "GSM8k-Aug",
    "prontoqa": "PrOntoQA",
    "prontoqa_test": "PrOntoQA",
    "prosqa": "ProsQA",
    "prosqa_test": "ProsQA",
}

# Canonical dataset order
DATASET_ORDER = ["GSM8k-Aug", "PrOntoQA", "ProsQA"]

# Method order for Table 1
METHOD_ORDER = ["No-CoT", "CoT", "Coconut", "CODI"]

# Base model order
BASE_MODEL_ORDER = ["GPT-2 Small", "Llama-3.2-1B-Instruct"]


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to canonical form."""
    name_lower = name.lower().replace("_", "").replace("-", "")
    if "gsm" in name_lower:
        return "GSM8k-Aug"
    elif "prontoqa" in name_lower:
        return "PrOntoQA"
    elif "prosqa" in name_lower:
        return "ProsQA"
    return name


def normalize_method_name(model_type: str) -> str:
    """Normalize method name to canonical form."""
    return METHOD_DISPLAY_NAMES.get(model_type, model_type)


def normalize_base_model_name(base_model: str) -> str:
    """Normalize base model name to canonical form."""
    base_lower = base_model.lower()
    if "gpt2" in base_lower or "gpt-2" in base_lower:
        return "GPT-2 Small"
    elif "llama" in base_lower:
        return "Llama-3.2-1B-Instruct"
    return BASE_MODEL_DISPLAY_NAMES.get(base_model, base_model)


def find_result_files(results_dir: str) -> list:
    """Find all JSON result files in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"WARNING: Results directory not found: {results_dir}")
        return []

    # Find all JSON files (excluding multimode subdirectory for Table 1)
    json_files = []
    for f in results_path.glob("*.json"):
        # Skip multimode files (they have _direct, _verbalized, _latent suffixes)
        if any(mode in f.stem for mode in ["_direct_", "_verbalized_", "_latent_"]):
            continue
        json_files.append(f)

    return sorted(json_files)


def load_result_file(filepath: Path) -> dict:
    """Load and parse a result JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not load {filepath}: {e}")
        return None


def infer_base_model(config: dict) -> str:
    """Infer base model from config fields."""
    # Try explicit base_model field first. Treat "unknown" as
    # non-authoritative so we fall through to model_id/model_path
    # inference (e.g. for HuggingFace checkpoints).
    base_model = config.get("base_model")
    if base_model and base_model not in ("None", "unknown"):
        return base_model

    # Try model_id
    model_id = config.get("model_id", "")
    if model_id:
        model_id_lower = model_id.lower()
        if "gpt2" in model_id_lower or "gpt-2" in model_id_lower:
            return "gpt2"
        elif "llama" in model_id_lower:
            return "llama32-1b"

    # Try model_path
    model_path = config.get("model_path", "")
    if model_path:
        path_lower = model_path.lower()
        if "gpt2" in path_lower or "/gpt2/" in path_lower:
            return "gpt2"
        elif "llama" in path_lower:
            return "llama32-1b"

    # Default to gpt2 for older result files
    return "gpt2"


def extract_metrics(data: dict) -> dict:
    """Extract relevant metrics from result data."""
    if data is None:
        return None

    config = data.get("config", {})
    metrics = data.get("metrics", {})
    token_metrics = data.get("token_metrics", {})

    # Determine model type (exclude multimode for Table 1)
    model_type = config.get("model_type", "")
    if model_type.startswith("multimode"):
        return None

    # Extract base model
    base_model = infer_base_model(config)

    # Extract dataset name from path
    dataset_path = config.get("dataset_path", "")
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Handle missing token_metrics (older result files)
    avg_total_tokens = None
    if token_metrics:
        avg_total_tokens = token_metrics.get("avg_total_output_tokens")

    return {
        "method": normalize_method_name(model_type),
        "base_model": normalize_base_model_name(base_model),
        "dataset": normalize_dataset_name(dataset_name),
        "accuracy": metrics.get("accuracy", 0.0),
        "avg_total_tokens": avg_total_tokens,
    }


def aggregate_results(results_dir: str) -> dict:
    """
    Aggregate results from all JSON files.

    Returns:
        dict: {(method, base_model): {dataset: {"acc": float, "tok": float}}}
    """
    aggregated = defaultdict(lambda: defaultdict(dict))

    files = find_result_files(results_dir)
    print(f"Found {len(files)} result files")

    for filepath in files:
        data = load_result_file(filepath)
        metrics = extract_metrics(data)

        if metrics is None:
            continue

        key = (metrics["method"], metrics["base_model"])
        dataset = metrics["dataset"]

        # Store metrics (take latest if multiple files for same config)
        aggregated[key][dataset] = {
            "acc": metrics["accuracy"],
            "tok": metrics["avg_total_tokens"],
        }

    return aggregated


def generate_csv(aggregated: dict, output_path: str):
    """Generate Table 1 CSV from aggregated results."""
    # Build header
    headers = ["Method", "Base Model"]
    for dataset in DATASET_ORDER:
        headers.append(f"{dataset} Acc (%)")
        headers.append(f"{dataset} # Tok")

    # Build rows (all rows for one base model before the next)
    rows = []
    for base_model in BASE_MODEL_ORDER:
        for method in METHOD_ORDER:
            key = (method, base_model)
            if key not in aggregated:
                continue

            row = [method, base_model]
            for dataset in DATASET_ORDER:
                data = aggregated[key].get(dataset, {})
                acc = data.get("acc")
                tok = data.get("tok")

                # Format accuracy as percentage with 1 decimal
                acc_str = f"{acc * 100:.1f}" if acc is not None else "-"
                tok_str = f"{tok:.1f}" if tok is not None else "-"

                row.append(acc_str)
                row.append(tok_str)

            rows.append(row)

    # Write CSV
    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

    print(f"Saved: {output_path}")


def print_table(aggregated: dict):
    """Print a formatted table to console."""
    # Column widths
    method_w = 12
    model_w = 25
    acc_w = 10
    tok_w = 8

    # Header
    header = f"{'Method':<{method_w}} {'Base Model':<{model_w}}"
    for dataset in DATASET_ORDER:
        header += f" {dataset:>{acc_w + tok_w + 1}}"
    print(header)

    # Sub-header
    subheader = f"{'':<{method_w}} {'':<{model_w}}"
    for _ in DATASET_ORDER:
        subheader += f" {'Acc (%)':>{acc_w}} {'# Tok':>{tok_w}}"
    print(subheader)
    print("-" * len(subheader))

    # Rows (all rows for one base model before the next)
    for base_model in BASE_MODEL_ORDER:
        for method in METHOD_ORDER:
            key = (method, base_model)
            if key not in aggregated:
                continue

            row = f"{method:<{method_w}} {base_model:<{model_w}}"
            for dataset in DATASET_ORDER:
                data = aggregated[key].get(dataset, {})
                acc = data.get("acc")
                tok = data.get("tok")

                acc_str = f"{acc * 100:.1f}" if acc is not None else "-"
                tok_str = f"{tok:.1f}" if tok is not None else "-"

                row += f" {acc_str:>{acc_w}} {tok_str:>{tok_w}}"

            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize dataset performance results into Table 1 CSV"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/dataset_performance",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: {results_dir}/table1_summary.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.results_dir, "table1_summary.csv")

    print("=" * 60)
    print("Summarizing Dataset Performance Results")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output CSV: {output_path}")
    print("")

    # Aggregate results
    aggregated = aggregate_results(args.results_dir)

    if not aggregated:
        print("WARNING: No valid results found")
        return

    # Print table to console
    print("\nTable 1 Summary:")
    print("-" * 60)
    print_table(aggregated)
    print("-" * 60)
    print("")

    # Generate CSV
    generate_csv(aggregated, output_path)


if __name__ == "__main__":
    main()
