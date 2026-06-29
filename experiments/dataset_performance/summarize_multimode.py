"""
Summarize multimode (Figure 3 / Table 10) results into a CSV.

Produces a CSV version of figure3_multimode.png so the per-mode accuracies can
be compared directly against Table 10 in the paper. Reads the SAME data the
dumbbell plot uses (via plot_multimode_dumbbell.load_data), so the CSV and the
figure are always consistent.

Table layout (matches Table 10, minus the published parenthetical results):

    LRM, Base Model, Reasoning Mode, GSM8k-Aug, PrOntoQA, ProsQA

where each (LRM, Base Model) block has three reasoning-mode rows:
    direct     -> No-CoT
    verbalized -> CoT
    latent     -> <LRM>   (i.e. "Coconut" or "CODI")

Usage:
    python -m experiments.dataset_performance.summarize_multimode [--results_dir PATH] [--output PATH]
"""

import argparse
import csv
import os
from pathlib import Path

from experiments.dataset_performance.plot_multimode_dumbbell import (
    DATASETS,
    MODEL_COMBOS,
    load_data,
)

# Display names for the base model column (Table 10 capitalization).
BASE_MODEL_DISPLAY = {
    "GPT-2 small": "GPT-2 Small",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
}

# (mode-data key in load_data, reasoning-mode label). The latent label is the
# LRM name itself, filled in per row.
MODE_ROWS = [
    ("DA", "No-CoT"),
    ("VCoT", "CoT"),
    ("Latent", None),  # None -> use the LRM name (Coconut / CODI)
]


def format_value(val):
    """Format an accuracy (already a percentage) to one decimal, or '-'."""
    return f"{val:.1f}" if val is not None else "-"


def build_rows(data):
    """Build Table 10 rows from the loaded multimode data."""
    rows = []
    for latent_model, base_model in MODEL_COMBOS:
        base_display = BASE_MODEL_DISPLAY.get(base_model, base_model)
        for mode_key, mode_label in MODE_ROWS:
            label = mode_label if mode_label is not None else latent_model
            row = [latent_model, base_display, label]
            for dataset in DATASETS:
                vals = data.get((latent_model, base_model, dataset), {})
                row.append(format_value(vals.get(mode_key)))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize multimode (Table 10) results into a CSV"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing result files (default: results/dataset_performance)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: {results_dir}/figure3_multimode.csv)",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent.parent / "results" / "dataset_performance"

    output_path = args.output or os.path.join(str(results_dir), "figure3_multimode.csv")

    print("=" * 60)
    print("Summarizing Multimode (Table 10) Results")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output CSV: {output_path}")
    print("")

    data = load_data(results_dir)
    if not data:
        print("WARNING: No multimode results found")
        return

    headers = ["LRM", "Base Model", "Reasoning Mode"] + DATASETS
    rows = build_rows(data)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Echo a readable table to the console.
    print("\nTable 10 Summary:")
    print("-" * 72)
    print("{:<8} {:<22} {:<10} {:>10} {:>9} {:>8}".format(*headers))
    print("-" * 72)
    for row in rows:
        print("{:<8} {:<22} {:<10} {:>10} {:>9} {:>8}".format(*row))
    print("-" * 72)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
