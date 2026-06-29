"""
Dumbbell plot comparing Latent CoT performance against Direct Answer and Verbalized CoT
across three datasets: GSM8k-Aug, PrOntoQA, and ProsQA.

Usage:
    python -m experiments.dataset_performance.plot_multimode_dumbbell [--results_dir PATH]

The script can read data from either:
1. A CSV file (multimode_results.csv) - for manually curated results
2. JSON result files - auto-detected from the results directory
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project colors
COLOR_GREEN = "#2ecc71"  # Latent wins vs No-CoT
COLOR_RED = "#e74c3c"    # Latent loses vs No-CoT
COLOR_BLUE = "#3498db"   # Latent wins vs Explicit reasoning
COLOR_ORANGE = "#e67e22" # Latent loses vs Explicit reasoning
COLOR_GREY = "#888888"   # Tie

# Font settings for 5.5 inch wide figure
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
})

# Dataset column names in CSV
DATASETS = ["GSM8k-Aug", "PrOntoQA", "ProsQA"]

# Model combinations (in display order)
MODEL_COMBOS = [
    ("Coconut", "GPT-2 small"),
    ("Coconut", "Llama-3.2-1B-Instruct"),
    ("CODI", "GPT-2 small"),
    ("CODI", "Llama-3.2-1B-Instruct"),
]

# Display labels for model combinations
MODEL_LABELS = {
    ("Coconut", "GPT-2 small"): "Coconut + GPT-2",
    ("Coconut", "Llama-3.2-1B-Instruct"): "Coconut + Llama",
    ("CODI", "GPT-2 small"): "CODI + GPT-2",
    ("CODI", "Llama-3.2-1B-Instruct"): "CODI + Llama",
}


def parse_percentage(val):
    """Parse percentage string to float, handling empty values."""
    if pd.isna(val) or val == "" or val == "-":
        return None
    if isinstance(val, str):
        val = val.strip().rstrip('%')
        if val == "":
            return None
        return float(val)
    return float(val)


def load_data_from_csv(csv_path):
    """Load and parse the multimode results CSV.

    Returns:
        dict: {(latent_model, base_model, dataset): {'DA': float, 'VCoT': float, 'Latent': float}}
    """
    df = pd.read_csv(csv_path)

    # Clean column names (handle BOM and whitespace)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    data = {}

    for _, row in df.iterrows():
        latent_model = row.get('Latent Reasoning Model', '')
        base_model = row.get('Base Model', '')
        mode = row.get('Reasoning Mode', '')

        if pd.isna(latent_model) or latent_model == '':
            continue

        for dataset in DATASETS:
            val = parse_percentage(row.get(dataset, ''))

            key = (latent_model, base_model, dataset)
            if key not in data:
                data[key] = {'DA': None, 'VCoT': None, 'Latent': None}

            if mode == 'Direct Answer':
                data[key]['DA'] = val
            elif mode == 'Verbalized CoT':
                data[key]['VCoT'] = val
            elif mode == 'Latent CoT':
                data[key]['Latent'] = val

    return data


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


def normalize_base_model_name(base_model: str) -> str:
    """Normalize base model name to canonical form."""
    base_lower = base_model.lower()
    if "gpt2" in base_lower or "gpt-2" in base_lower:
        return "GPT-2 small"
    elif "llama" in base_lower:
        return "Llama-3.2-1B-Instruct"
    return base_model


def normalize_latent_model_name(model_type: str) -> str:
    """Normalize model type to latent model name."""
    if "coconut" in model_type.lower():
        return "Coconut"
    elif "codi" in model_type.lower():
        return "CODI"
    return model_type


def load_data_from_json(results_dir: Path):
    """Load multimode results from JSON files.

    Auto-detects multimode result files by looking for _direct_, _verbalized_, _latent_ in filename.

    Returns:
        dict: {(latent_model, base_model, dataset): {'DA': float, 'VCoT': float, 'Latent': float}}
    """
    data = {}

    # Find all multimode JSON files
    json_files = list(results_dir.glob("multimode_*.json"))

    # Also check subdirectory
    subdir = results_dir / "multimode_codi"
    if subdir.exists():
        json_files.extend(subdir.glob("*.json"))

    print(f"Found {len(json_files)} multimode result files")

    for filepath in json_files:
        try:
            with open(filepath) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Could not load {filepath}: {e}")
            continue

        config = result.get("config", {})
        metrics = result.get("metrics", {})

        model_type = config.get("model_type", "")
        if not model_type.startswith("multimode"):
            continue

        mode = config.get("mode", "")
        base_model = config.get("base_model", "")
        dataset_path = config.get("dataset_path", "")
        accuracy = metrics.get("accuracy", 0.0)

        # Normalize names
        latent_model = normalize_latent_model_name(model_type)
        base_model_norm = normalize_base_model_name(base_model)
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_norm = normalize_dataset_name(dataset_name)

        # Create key
        key = (latent_model, base_model_norm, dataset_norm)
        if key not in data:
            data[key] = {'DA': None, 'VCoT': None, 'Latent': None}

        # Map mode to data key
        if mode == "direct":
            data[key]['DA'] = accuracy * 100  # Convert to percentage
        elif mode == "verbalized":
            data[key]['VCoT'] = accuracy * 100
        elif mode == "latent":
            data[key]['Latent'] = accuracy * 100

    return data


def load_data(results_dir: Path):
    """Load data from either CSV or JSON files.

    Prefers CSV if it exists, otherwise auto-detects from JSON files.
    """
    csv_path = results_dir / "multimode_results.csv"

    if csv_path.exists():
        print(f"Loading from CSV: {csv_path}")
        return load_data_from_csv(csv_path)
    else:
        print("CSV not found, loading from JSON files...")
        return load_data_from_json(results_dir)


def plot_dumbbell(data, output_dir):
    """Create the 1x3 dumbbell plot."""
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2), sharey=True)

    line_thickness = 1.5
    dot_size = 15
    bar_offset = 0.15  # Vertical offset for upper/lower bars

    # Use consistent y-positions across all datasets
    y_positions = list(range(len(MODEL_COMBOS)))
    y_labels = [MODEL_LABELS[combo] for combo in MODEL_COMBOS]

    # Compute per-dataset x-axis limits
    dataset_limits = {}
    for dataset in DATASETS:
        all_vals = []
        for latent_model, base_model in MODEL_COMBOS:
            key = (latent_model, base_model, dataset)
            vals = data.get(key, {})
            for v in [vals.get('Latent'), vals.get('DA'), vals.get('VCoT')]:
                if v is not None:
                    all_vals.append(v)
        if all_vals:
            min_val, max_val = min(all_vals), max(all_vals)
            # Add padding (15% of range on each side, minimum 2 points)
            data_range = max_val - min_val
            padding = max(data_range * 0.15, 2)
            # Allow upper limit to go past 100 for visual breathing room
            upper_limit = max_val + padding if max_val < 98 else 102
            dataset_limits[dataset] = (max(0, min_val - padding), upper_limit)
        else:
            dataset_limits[dataset] = (0, 100)

    # Manual override for PrOntoQA
    dataset_limits["PrOntoQA"] = (92, 100.5)

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]

        for y_pos, (latent_model, base_model) in enumerate(MODEL_COMBOS):
            key = (latent_model, base_model, dataset)
            vals = data.get(key, {})

            latent = vals.get('Latent')
            da = vals.get('DA')
            vcot = vals.get('VCoT')

            # Skip if no Latent CoT data
            if latent is None:
                continue

            # Upper bar: Latent vs Direct Answer (solid line)
            # Note: y-axis is inverted, so negative offset appears above
            if da is not None:
                if latent > da:
                    color = COLOR_GREEN
                elif latent < da:
                    color = COLOR_RED
                else:
                    color = COLOR_GREY
                ax.plot([latent, da], [y_pos - bar_offset, y_pos - bar_offset],
                        color=color, linewidth=line_thickness, linestyle='-',
                        solid_capstyle='round', zorder=2)
                # Endpoint marker (vertical dash)
                ax.scatter([da], [y_pos - bar_offset], color=color, s=20,
                          marker='|', linewidths=line_thickness, zorder=3)

            # Lower bar: Latent vs Verbalized CoT (dashed line)
            if vcot is not None:
                if latent > vcot:
                    color = COLOR_BLUE
                elif latent < vcot:
                    color = COLOR_ORANGE
                else:
                    color = COLOR_GREY
                ax.plot([latent, vcot], [y_pos + bar_offset, y_pos + bar_offset],
                        color=color, linewidth=line_thickness, linestyle='--',
                        dashes=(4, 2), solid_capstyle='round', zorder=2)
                # Endpoint marker (vertical dash)
                ax.scatter([vcot], [y_pos + bar_offset], color=color, s=20,
                          marker='|', linewidths=line_thickness, zorder=3)

            # White dot with dark outline at Latent CoT accuracy (centered)
            ax.scatter([latent], [y_pos], color='white', s=dot_size,
                      edgecolors='#333333', linewidths=1, zorder=4)

        # Configure axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(dataset, fontweight='bold')
        ax.set_xlim(dataset_limits[dataset])
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # First model combo at top
        ax.set_axisbelow(True)

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', markerfacecolor='white',
               markeredgecolor='#333333', markersize=4, markeredgewidth=0.8,
               linestyle='None', label='Latent reasoning accuracy'),
        Line2D([0], [0], color=COLOR_GREY, linewidth=line_thickness,
               linestyle='-', label='tie'),
        Line2D([0], [0], color=COLOR_GREEN, linewidth=line_thickness,
               linestyle='-', label='vs No-CoT, latent wins'),
        Line2D([0], [0], color=COLOR_RED, linewidth=line_thickness,
               linestyle='-', label='vs No-CoT, latent loses'),
        Line2D([0], [0], color=COLOR_BLUE, linewidth=line_thickness,
               linestyle='--', dashes=(3, 1.5), label='vs Explicit reasoning, latent wins'),
        Line2D([0], [0], color=COLOR_ORANGE, linewidth=line_thickness,
               linestyle='--', dashes=(3, 1.5), label='vs Explicit reasoning, latent loses'),
    ]

    # Place legend below the figure, shifted right to center under middle subplot
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.55, -0.02), ncol=3, frameon=True,
               fancybox=True, shadow=False, columnspacing=0.8, handletextpad=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)  # Make room for legend

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, "figure3_multimode.png")
    pdf_path = os.path.join(output_dir, "figure3_multimode.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    print(f"Saved: {png_path}")

    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 multimode dumbbell plot"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing result files (default: results/dataset_performance)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results_dir)",
    )
    args = parser.parse_args()

    # Determine paths
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent.parent / "results" / "dataset_performance"

    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    print("=" * 60)
    print("Generating Figure 3: Multimode Dumbbell Plot")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("")

    # Load data
    data = load_data(results_dir)

    if not data:
        print("ERROR: No multimode results found")
        return

    # Print summary of loaded data
    print(f"\nLoaded data for {len(data)} model/dataset combinations:")
    for key, vals in sorted(data.items()):
        latent_model, base_model, dataset = key
        da = f"{vals['DA']:.1f}%" if vals['DA'] is not None else "-"
        vcot = f"{vals['VCoT']:.1f}%" if vals['VCoT'] is not None else "-"
        latent = f"{vals['Latent']:.1f}%" if vals['Latent'] is not None else "-"
        print(f"  {latent_model} + {base_model} on {dataset}: DA={da}, VCoT={vcot}, Latent={latent}")

    # Generate plot
    print("")
    plot_dumbbell(data, output_dir)

    print("\nDumbbell plot generation complete.")


if __name__ == "__main__":
    main()
