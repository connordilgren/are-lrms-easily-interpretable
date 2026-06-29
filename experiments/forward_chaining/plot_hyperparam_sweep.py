"""Plotting script for hyperparameter sweep visualization.

Creates a 3x3 subplot grid showing tree found/verified rates across different
required_passes (rp) and max_rank (mr) values for both Coconut and CODI models.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Colors for bars
COLOR_CORRECT = "#2ecc71"    # Green for correct answers
COLOR_INCORRECT = "#e74c3c"  # Red for incorrect answers

# Multi-LLM colors (matching back_tracking_vp/plot_results.py)
COLOR_GPT2_COCONUT = "#2ecc71"   # Green
COLOR_GPT2_CODI = "#e74c3c"      # Red
COLOR_LLAMA_COCONUT = "#3498db"  # Blue
COLOR_LLAMA_CODI = "#f39c12"     # Orange


def load_results(results_path):
    """Load results.json from the given path."""
    with open(results_path, "r") as f:
        return json.load(f)


def compute_tree_verified_metrics(data):
    """Compute tree found and verified percentages for correct/incorrect samples."""
    samples = data["per_sample"]

    correct_samples = [s for s in samples if s["answer_correct"]]
    incorrect_samples = [s for s in samples if not s["answer_correct"]]

    # For correct samples
    correct_found = sum(1 for s in correct_samples if s["tree_found"])
    correct_verified = sum(1 for s in correct_samples if s.get("tree_verified", False))

    # For incorrect samples
    incorrect_found = sum(1 for s in incorrect_samples if s["tree_found"])
    incorrect_verified = sum(1 for s in incorrect_samples if s.get("tree_verified", False))

    n_correct = len(correct_samples)
    n_incorrect = len(incorrect_samples)

    return {
        "correct_verified_pct": (correct_verified / n_correct * 100) if n_correct > 0 else 0,
        "correct_found_not_verified_pct": ((correct_found - correct_verified) / n_correct * 100) if n_correct > 0 else 0,
        "incorrect_verified_pct": (incorrect_verified / n_incorrect * 100) if n_incorrect > 0 else 0,
        "incorrect_found_not_verified_pct": ((incorrect_found - incorrect_verified) / n_incorrect * 100) if n_incorrect > 0 else 0,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
    }


def get_result_path(results_dir, model_type, rp, mr, base_llm="gpt2"):
    """Build path to result file for given hyperparameters."""
    folder_name = f"{model_type}_{base_llm}_gsm_vocab-projection-friendly_test_yes-question-tokens_rp{rp}_mr{mr}"
    return os.path.join(results_dir, folder_name, "results.json")


def plot_single_cell(ax, rp, mr, all_metrics, base_llm_display, include_title=True, include_legend=False):
    """Plot a single cell (one rp/mr combination) on the given axes."""
    # 4 bar positions: Coconut Correct, Coconut Incorrect, CODI Correct, CODI Incorrect
    x = np.array([0, 1, 2.5, 3.5])
    bar_width = 0.7

    # Get metrics for both models
    metrics_coconut = all_metrics["coconut"][(rp, mr)]
    metrics_codi = all_metrics["codi"][(rp, mr)]

    # Build labels - always show all 4 labels
    labels = []
    for model_type, metrics in [("coconut", metrics_coconut), ("codi", metrics_codi)]:
        model_label = "Coconut" if model_type == "coconut" else "CODI"
        if metrics is not None:
            labels.append(f"{model_label}\nCorrect\n(n={metrics['n_correct']})")
            labels.append(f"{model_label}\nIncorrect\n(n={metrics['n_incorrect']})")
        else:
            labels.append(f"{model_label}\nCorrect\n(no data)")
            labels.append(f"{model_label}\nIncorrect\n(no data)")

    # Plot bars for each model
    for model_idx, (model_type, metrics) in enumerate([("coconut", metrics_coconut), ("codi", metrics_codi)]):
        if metrics is None:
            continue

        # Bar positions for this model (2 bars each)
        x_model = x[model_idx * 2:(model_idx + 1) * 2]

        # Verified values (bottom of stack)
        verified_vals = [metrics["correct_verified_pct"], metrics["incorrect_verified_pct"]]

        # Found but not verified (top of stack)
        found_not_verified_vals = [
            metrics["correct_found_not_verified_pct"],
            metrics["incorrect_found_not_verified_pct"]
        ]

        colors = [COLOR_CORRECT, COLOR_INCORRECT]

        # Bottom bars (verified) - solid
        ax.bar(x_model, verified_vals, bar_width, color=colors, edgecolor="black", linewidth=0.5)

        # Top bars (found but not verified) - hatched
        ax.bar(x_model, found_not_verified_vals, bar_width, bottom=verified_vals,
               color=colors, edgecolor="black", linewidth=0.5, hatch="///")

        # Value labels
        for i, (v, f) in enumerate(zip(verified_vals, found_not_verified_vals)):
            total = v + f
            # Label for verified segment (centered in verified bar)
            if v > 5:  # Only show if segment is big enough
                ax.text(x_model[i], v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
            # Label for total (above bar)
            if total > 0:
                ax.text(x_model[i], total + 2, f"{total:.1f}%", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    # Subplot formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 4.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical separator between models
    ax.axvline(x=1.75, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    if include_title:
        ax.set_title(f"rp={rp}, mr={mr} - {base_llm_display}", fontsize=10, fontweight="bold")

    if include_legend:
        legend_elements = [
            Patch(facecolor=COLOR_CORRECT, edgecolor="black", label="Correct & Verified"),
            Patch(facecolor=COLOR_INCORRECT, edgecolor="black", label="Incorrect & Verified"),
            Patch(facecolor=COLOR_CORRECT, edgecolor="black", hatch="///", label="Correct & Found (not verified)"),
            Patch(facecolor=COLOR_INCORRECT, edgecolor="black", hatch="///", label="Incorrect & Found (not verified)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=6)


def plot_combined_cell(ax, rp, mr, metrics_by_llm):
    """Plot a single cell with both GPT-2 and Llama results side by side.

    Args:
        ax: matplotlib axes
        rp: required_passes value
        mr: max_rank value
        metrics_by_llm: dict mapping base_llm -> {model_type -> {(rp, mr) -> metrics}}
    """
    # 8 bar positions: GPT-2 (Coconut Correct/Incorrect, CODI Correct/Incorrect),
    #                  Llama (Coconut Correct/Incorrect, CODI Correct/Incorrect)
    # Group spacing: within model=1, between models=1.5, between LLMs=2.5
    x = np.array([0, 1, 2.5, 3.5,  # GPT-2: Coconut C/I, CODI C/I
                  6.5, 7.5, 9, 10])  # Llama: Coconut C/I, CODI C/I
    bar_width = 0.7

    base_llm_configs = [
        ("gpt2", "GPT-2", 0),
        ("llama32-1b", "Llama", 4),
    ]

    labels = []

    for base_llm, llm_display, x_offset in base_llm_configs:
        all_metrics = metrics_by_llm.get(base_llm, {})
        metrics_coconut = all_metrics.get("coconut", {}).get((rp, mr))
        metrics_codi = all_metrics.get("codi", {}).get((rp, mr))

        # Build labels for this LLM
        for model_type, metrics in [("coconut", metrics_coconut), ("codi", metrics_codi)]:
            model_label = "Coconut" if model_type == "coconut" else "CODI"
            if metrics is not None:
                labels.append(f"{llm_display}\n{model_label}\nCorrect\n(n={metrics['n_correct']})")
                labels.append(f"{llm_display}\n{model_label}\nIncorrect\n(n={metrics['n_incorrect']})")
            else:
                labels.append(f"{llm_display}\n{model_label}\nCorrect\n(no data)")
                labels.append(f"{llm_display}\n{model_label}\nIncorrect\n(no data)")

        # Plot bars for each model type
        for model_idx, (model_type, metrics) in enumerate([("coconut", metrics_coconut), ("codi", metrics_codi)]):
            if metrics is None:
                continue

            # Bar positions for this model (2 bars each)
            x_model = x[x_offset + model_idx * 2:x_offset + (model_idx + 1) * 2]

            # Verified values (bottom of stack)
            verified_vals = [metrics["correct_verified_pct"], metrics["incorrect_verified_pct"]]

            # Found but not verified (top of stack)
            found_not_verified_vals = [
                metrics["correct_found_not_verified_pct"],
                metrics["incorrect_found_not_verified_pct"]
            ]

            colors = [COLOR_CORRECT, COLOR_INCORRECT]

            # Bottom bars (verified) - solid
            ax.bar(x_model, verified_vals, bar_width, color=colors, edgecolor="black", linewidth=0.5)

            # Top bars (found but not verified) - hatched
            ax.bar(x_model, found_not_verified_vals, bar_width, bottom=verified_vals,
                   color=colors, edgecolor="black", linewidth=0.5, hatch="///")

            # Value labels
            for i, (v, f) in enumerate(zip(verified_vals, found_not_verified_vals)):
                total = v + f
                # Label for verified segment (centered in verified bar)
                if v > 5:  # Only show if segment is big enough
                    ax.text(x_model[i], v / 2, f"{v:.1f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")
                # Label for total (above bar)
                if total > 0:
                    ax.text(x_model[i], total + 2, f"{total:.1f}%", ha="center", va="bottom",
                            fontsize=7, fontweight="bold")

    # Subplot formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 10.6)
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical separators between models within each LLM
    ax.axvline(x=1.75, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(x=8.25, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add vertical separator between LLMs (thicker)
    ax.axvline(x=5, color="black", linestyle="-", linewidth=1, alpha=0.7)

    ax.set_title(f"required_passes={rp}, max_rank={mr}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Tree Found/Verified (%)", fontsize=9)


def load_all_metrics(results_dir, base_llm, rp_values, mr_values, model_types):
    """Load all metrics for a given base LLM.

    Returns:
        dict mapping model_type -> {(rp, mr) -> metrics}
    """
    all_metrics = {}
    for model_type in model_types:
        all_metrics[model_type] = {}
        for mr in mr_values:
            for rp in rp_values:
                path = get_result_path(results_dir, model_type, rp, mr, base_llm)
                if os.path.exists(path):
                    data = load_results(path)
                    all_metrics[model_type][(rp, mr)] = compute_tree_verified_metrics(data)
                    print(f"Loaded: {model_type} ({base_llm}) rp={rp} mr={mr}")
                else:
                    print(f"Missing: {model_type} ({base_llm}) rp={rp} mr={mr}")
                    all_metrics[model_type][(rp, mr)] = None
    return all_metrics


def plot_hyperparam_sweep(results_dir, output_dir, base_llm="gpt2", all_metrics=None):
    """Create 3x3 subplot grid showing metrics across rp and mr values for both models."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    rp_values = [1, 2, 3]
    mr_values = [1, 2, 3]
    model_types = ["coconut", "codi"]

    # Human-readable name for the base LLM
    base_llm_display = {
        "gpt2": "GPT-2",
        "llama32-1b": "Llama 3.2-1B",
    }.get(base_llm, base_llm)

    # Load data if not provided
    if all_metrics is None:
        all_metrics = load_all_metrics(results_dir, base_llm, rp_values, mr_values, model_types)

    # Create each subplot in the grid
    for row_idx, mr in enumerate(mr_values):
        for col_idx, rp in enumerate(rp_values):
            ax = axes[row_idx, col_idx]
            plot_single_cell(ax, rp, mr, all_metrics, base_llm_display, include_title=False, include_legend=False)

            # Only show y-axis label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"max_rank = {mr}", fontsize=10, fontweight="bold")
            else:
                ax.set_yticklabels([])

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(f"required_passes = {rp}", fontsize=10, fontweight="bold")

    # Create shared legend
    legend_elements = [
        Patch(facecolor=COLOR_CORRECT, edgecolor="black", label="Correct & Verified"),
        Patch(facecolor=COLOR_INCORRECT, edgecolor="black", label="Incorrect & Verified"),
        Patch(facecolor=COLOR_CORRECT, edgecolor="black", hatch="///", label="Correct & Found (not verified)"),
        Patch(facecolor=COLOR_INCORRECT, edgecolor="black", hatch="///", label="Incorrect & Found (not verified)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    # Overall title
    fig.suptitle(f"Hyperparameter Sweep: Tree Found/Verified Rates (Coconut vs CODI) - {base_llm_display}",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)

    # Save combined grid plot
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, f"hyperparam_sweep_{base_llm}")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    return all_metrics


def plot_heatmaps(output_dir, metrics_by_llm, rp_values, mr_values):
    """Create 2x4 heatmap grid showing verified rates for each LLM×model×correctness condition.

    Args:
        output_dir: Directory to save plots
        metrics_by_llm: dict mapping base_llm -> {model_type -> {(rp, mr) -> metrics}}
        rp_values: List of required_passes values
        mr_values: List of max_rank values
    """
    base_llm_configs = [
        ("gpt2", "GPT-2"),
        ("llama32-1b", "Llama 3.2-1B-Instruct"),
    ]

    # Define the 4 conditions (columns): (model_type, correctness_key, title)
    conditions = [
        ("coconut", "correct_verified_pct", "Coconut Correct"),
        ("coconut", "incorrect_verified_pct", "Coconut Incorrect"),
        ("codi", "correct_verified_pct", "CODI Correct"),
        ("codi", "incorrect_verified_pct", "CODI Incorrect"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.75))

    # Fixed color scale from 0 to 100
    vmin = 0
    vmax = 100

    im = None  # Will store the last image for colorbar

    for row_idx, (base_llm, llm_display) in enumerate(base_llm_configs):
        all_metrics = metrics_by_llm.get(base_llm, {})

        for col_idx, (model_type, metric_key, col_title) in enumerate(conditions):
            ax = axes[row_idx, col_idx]

            # Build the heatmap matrix (rows=mr, cols=rp)
            matrix = np.zeros((len(mr_values), len(rp_values)))
            for mr_idx, mr in enumerate(mr_values):
                for rp_idx, rp in enumerate(rp_values):
                    metrics = all_metrics.get(model_type, {}).get((rp, mr))
                    if metrics is not None:
                        matrix[mr_idx, rp_idx] = metrics[metric_key]
                    else:
                        matrix[mr_idx, rp_idx] = np.nan

            # Create heatmap (flip so row 0 is at bottom)
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax,
                           origin="lower")

            # Add value annotations
            for mr_idx in range(len(mr_values)):
                for rp_idx in range(len(rp_values)):
                    val = matrix[mr_idx, rp_idx]
                    if not np.isnan(val):
                        # Use white text on dark cells, black on light
                        text_color = "white" if val > (vmax - vmin) * 0.6 + vmin else "black"
                        ax.text(rp_idx, mr_idx, f"{val:.0f}%", ha="center", va="center",
                                fontsize=6, color=text_color)

            # Axis labels
            ax.set_xticks(range(len(rp_values)))
            ax.set_xticklabels(rp_values, fontsize=6)
            ax.set_yticks(range(len(mr_values)))
            ax.set_yticklabels(mr_values, fontsize=6)

            # X-axis label only on bottom row
            if row_idx == 1:
                ax.set_xlabel("Required Passes", fontsize=7)

            # Column titles only on top row
            if row_idx == 0:
                ax.set_title(col_title, fontsize=7)

            # Row label (LLM name) on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{llm_display}\n\nMax Rank", fontsize=7)
            else:
                ax.set_ylabel("")

    # Add shared colorbar to the right
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Verified Rate of Reasoning Trace (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "heatmap_verified")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_forward_chaining_results(output_dir, metrics_by_llm, rp_values):
    """Create 2x3 subplot showing forward chaining results.

    Rows: GPT-2, Llama
    Columns: required_passes = 1, 2, 3
    Uses max_rank = 1 for all subplots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 2.75))

    # LLM-specific color mappings for Coconut/CODI
    llm_colors = {
        "gpt2": {"coconut": COLOR_GPT2_COCONUT, "codi": COLOR_GPT2_CODI},
        "llama32-1b": {"coconut": COLOR_LLAMA_COCONUT, "codi": COLOR_LLAMA_CODI},
    }

    base_llm_configs = [
        ("gpt2", "GPT-2"),
        ("llama32-1b", "Llama 3.2-1B-Instruct"),
    ]

    mr = 1  # Fixed max_rank = 1

    for row_idx, (base_llm, llm_display) in enumerate(base_llm_configs):
        all_metrics = metrics_by_llm.get(base_llm, {})
        colors_for_llm = llm_colors[base_llm]

        for col_idx, rp in enumerate(rp_values):
            ax = axes[row_idx, col_idx]

            # Get metrics for both models
            metrics_coconut = all_metrics.get("coconut", {}).get((rp, mr))
            metrics_codi = all_metrics.get("codi", {}).get((rp, mr))

            # 4 bar positions: Coconut Correct, Coconut Incorrect, CODI Correct, CODI Incorrect
            x = np.array([0, 1, 2.5, 3.5])
            bar_width = 0.7

            # Build labels
            labels = []
            for model_type, metrics in [("coconut", metrics_coconut), ("codi", metrics_codi)]:
                model_label = "Coconut" if model_type == "coconut" else "CODI"
                if metrics is not None:
                    labels.append(f"{model_label}\nCorrect")
                    labels.append(f"{model_label}\nIncorrect")
                else:
                    labels.append(f"{model_label}\nCorrect")
                    labels.append(f"{model_label}\nIncorrect")

            # Plot bars for each model
            for model_idx, (model_type, metrics) in enumerate([("coconut", metrics_coconut), ("codi", metrics_codi)]):
                if metrics is None:
                    continue

                # Bar positions for this model (2 bars each)
                x_model = x[model_idx * 2:(model_idx + 1) * 2]

                # Verified values (bottom of stack)
                verified_vals = [metrics["correct_verified_pct"], metrics["incorrect_verified_pct"]]

                # Found but not verified (top of stack)
                found_not_verified_vals = [
                    metrics["correct_found_not_verified_pct"],
                    metrics["incorrect_found_not_verified_pct"]
                ]

                # Use LLM-specific model color; correct=full alpha, incorrect=lighter
                model_color = colors_for_llm[model_type]
                alphas = [1.0, 0.6]  # Correct=full, Incorrect=lighter

                # Bottom bars (verified) - solid
                for i, (xpos, val) in enumerate(zip(x_model, verified_vals)):
                    ax.bar(xpos, val, bar_width, color=model_color, alpha=alphas[i],
                           edgecolor="black", linewidth=0.3)

                # Top bars (found but not verified) - hatched
                for i, (xpos, val, bottom) in enumerate(zip(x_model, found_not_verified_vals, verified_vals)):
                    ax.bar(xpos, val, bar_width, bottom=bottom, color=model_color, alpha=alphas[i],
                           edgecolor="black", linewidth=0.3, hatch="///")

                # Value labels
                for i, (v, f) in enumerate(zip(verified_vals, found_not_verified_vals)):
                    total = v + f
                    # Label for verified segment (centered in verified bar)
                    if v > 8:  # Only show if segment is big enough
                        ax.text(x_model[i], v / 2, f"{v:.0f}%", ha="center", va="center",
                                fontsize=4, fontweight="bold", color="white")
                    # Label for total (above bar)
                    if total > 0:
                        ax.text(x_model[i], total + 3, f"{total:.0f}%", ha="center", va="bottom",
                                fontsize=4, fontweight="bold")

            # Subplot formatting
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=4)
            ax.set_ylim(0, 115)
            ax.set_xlim(-0.6, 4.1)
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis='y', labelsize=4)

            # Add vertical separator between models
            ax.axvline(x=1.75, color="gray", linestyle="--", linewidth=0.3, alpha=0.5)

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(f"Required Passes = {rp}", fontsize=6, fontweight="bold")

            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{llm_display}", fontsize=6)
            else:
                ax.set_yticklabels([])

    # Create shared legend with LLM-specific colors and verification status
    legend_elements = [
        Patch(facecolor=COLOR_GPT2_COCONUT, edgecolor="black", label="GPT-2 + Coconut, Verified"),
        Patch(facecolor=COLOR_GPT2_COCONUT, edgecolor="black", hatch="///", label="GPT-2 + Coconut, Found"),
        Patch(facecolor=COLOR_GPT2_CODI, edgecolor="black", label="GPT-2 + CODI, Verified"),
        Patch(facecolor=COLOR_GPT2_CODI, edgecolor="black", hatch="///", label="GPT-2 + CODI, Found"),
        Patch(facecolor=COLOR_LLAMA_COCONUT, edgecolor="black", label="Llama + Coconut, Verified"),
        Patch(facecolor=COLOR_LLAMA_COCONUT, edgecolor="black", hatch="///", label="Llama + Coconut, Found"),
        Patch(facecolor=COLOR_LLAMA_CODI, edgecolor="black", label="Llama + CODI, Verified"),
        Patch(facecolor=COLOR_LLAMA_CODI, edgecolor="black", hatch="///", label="Llama + CODI, Found"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=4,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.92)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "forward_chaining_results")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_forward_chaining_results_2x4(output_dir, metrics_by_llm, rp_values):
    """Create 2x4 subplot showing forward chaining results.

    Rows: GPT-2, Llama
    Columns: Coconut Correct, Coconut Incorrect, CODI Correct, CODI Incorrect
    Each cell has 3 grouped bars for required_passes = 1, 2, 3
    Each bar shows found rate (hatched) with verified rate (solid).
    Uses max_rank = 1 for all subplots.
    """
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2))

    # LLM-specific color mappings for Coconut/CODI
    llm_colors = {
        "gpt2": {"coconut": COLOR_GPT2_COCONUT, "codi": COLOR_GPT2_CODI},
        "llama32-1b": {"coconut": COLOR_LLAMA_COCONUT, "codi": COLOR_LLAMA_CODI},
    }

    base_llm_configs = [
        ("gpt2", "GPT-2 small"),
        ("llama32-1b", "Llama 3.2-1B"),
    ]

    # Define the 4 conditions (columns): (model_type, correctness_key, found_key, title)
    conditions = [
        ("coconut", "correct_verified_pct", "correct_found_not_verified_pct", "Coconut Correct"),
        ("coconut", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "Coconut Incorrect"),
        ("codi", "correct_verified_pct", "correct_found_not_verified_pct", "CODI Correct"),
        ("codi", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "CODI Incorrect"),
    ]

    mr = 1  # Fixed max_rank = 1
    bar_width = 0.6
    x = np.arange(len(rp_values))  # Bar positions for rp = 1, 2, 3

    for row_idx, (base_llm, llm_display) in enumerate(base_llm_configs):
        all_metrics = metrics_by_llm.get(base_llm, {})
        colors_for_llm = llm_colors[base_llm]

        for col_idx, (model_type, verified_key, found_key, col_title) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            model_color = colors_for_llm[model_type]

            # Determine alpha: full for "Correct" columns, lighter for "Incorrect"
            is_incorrect = "Incorrect" in col_title
            alpha = 0.6 if is_incorrect else 1.0

            # Gather data for each rp
            verified_vals = []
            found_vals = []  # Total found = verified + found_not_verified
            for rp in rp_values:
                metrics = all_metrics.get(model_type, {}).get((rp, mr))
                if metrics is not None:
                    verified_vals.append(metrics[verified_key])
                    found_vals.append(metrics[verified_key] + metrics[found_key])
                else:
                    verified_vals.append(0)
                    found_vals.append(0)

            # Plot solid bars (verified only)
            ax.bar(x, verified_vals, bar_width, color=model_color, alpha=alpha,
                   edgecolor="black", linewidth=0.3)

            # Add horizontal reference line for found rate (use average or first value)
            avg_found = np.mean(found_vals)
            if avg_found > 0:
                ax.axhline(y=avg_found, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                # Annotate once per cell (on the right side)
                # Position text slightly lower if found rate is very high to avoid clipping
                y_offset = 0 if avg_found > 95 else 2
                ax.text(len(rp_values) - 0.7, avg_found + y_offset, f"Found: {avg_found:.0f}%",
                        ha="right", va="bottom", fontsize=5, style="italic")

            # Value labels for verified bars
            for i, v in enumerate(verified_vals):
                if v > 10:
                    # Large enough - put inside bar
                    ax.text(x[i], v / 2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=5, fontweight="bold", color="black")
                elif v > 0:
                    # Too small - put on top of bar
                    ax.text(x[i], v + 2, f"{v:.0f}%", ha="center", va="bottom",
                            fontsize=5, fontweight="bold", color="black")

            # Subplot formatting
            ax.set_xticks(x)
            ax.set_xticklabels(rp_values, fontsize=6)
            ax.set_ylim(0, 115)
            ax.set_xlim(-0.5, len(rp_values) - 0.5)
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis='y', labelsize=5)

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(col_title, fontsize=7)

            # X-axis label only on bottom row
            if row_idx == 1:
                ax.set_xlabel("Required Passes", fontsize=6)

            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{llm_display}", fontsize=7)
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "figure6_forward_chaining")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_forward_chaining_results_single(output_dir, metrics_by_llm, rp_values):
    """Create a single plot with 8 clusters showing forward chaining results.

    8 clusters: GPT-2 (Coconut Correct/Incorrect, CODI Correct/Incorrect),
                Llama (Coconut Correct/Incorrect, CODI Correct/Incorrect)
    Each cluster has 3 bars for required_passes = 1, 2, 3.
    Each bar shows found rate (hatched) with verified rate (solid).
    Uses max_rank = 1 for all.
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    # LLM-specific color mappings for Coconut/CODI
    llm_colors = {
        "gpt2": {"coconut": COLOR_GPT2_COCONUT, "codi": COLOR_GPT2_CODI},
        "llama32-1b": {"coconut": COLOR_LLAMA_COCONUT, "codi": COLOR_LLAMA_CODI},
    }

    # Define the 8 configurations: (base_llm, model_type, correctness_key, found_key, label, is_incorrect)
    configurations = [
        ("gpt2", "coconut", "correct_verified_pct", "correct_found_not_verified_pct", "GPT-2\nCoconut\nCorrect", False),
        ("gpt2", "coconut", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "GPT-2\nCoconut\nIncorrect", True),
        ("gpt2", "codi", "correct_verified_pct", "correct_found_not_verified_pct", "GPT-2\nCODI\nCorrect", False),
        ("gpt2", "codi", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "GPT-2\nCODI\nIncorrect", True),
        ("llama32-1b", "coconut", "correct_verified_pct", "correct_found_not_verified_pct", "Llama\nCoconut\nCorrect", False),
        ("llama32-1b", "coconut", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "Llama\nCoconut\nIncorrect", True),
        ("llama32-1b", "codi", "correct_verified_pct", "correct_found_not_verified_pct", "Llama\nCODI\nCorrect", False),
        ("llama32-1b", "codi", "incorrect_verified_pct", "incorrect_found_not_verified_pct", "Llama\nCODI\nIncorrect", True),
    ]

    mr = 1  # Fixed max_rank = 1
    n_rp = len(rp_values)
    bar_width = 0.25
    cluster_width = n_rp * bar_width + 0.3  # Width of each cluster with some padding

    cluster_labels = []
    cluster_centers = []

    for cluster_idx, (base_llm, model_type, verified_key, found_key, label, is_incorrect) in enumerate(configurations):
        all_metrics = metrics_by_llm.get(base_llm, {})
        model_color = llm_colors[base_llm][model_type]
        alpha = 0.6 if is_incorrect else 1.0

        # Calculate x positions for this cluster
        cluster_start = cluster_idx * cluster_width
        x_positions = [cluster_start + i * bar_width for i in range(n_rp)]
        cluster_centers.append(cluster_start + (n_rp - 1) * bar_width / 2)
        cluster_labels.append(label)

        # Gather data for each rp
        for i, rp in enumerate(rp_values):
            metrics = all_metrics.get(model_type, {}).get((rp, mr))
            if metrics is not None:
                verified_val = metrics[verified_key]
                found_not_verified_val = metrics[found_key]
            else:
                verified_val = 0
                found_not_verified_val = 0

            # Plot bars - bottom solid (verified), top hatched (found not verified)
            ax.bar(x_positions[i], verified_val, bar_width, color=model_color, alpha=alpha,
                   edgecolor="black", linewidth=0.3)
            ax.bar(x_positions[i], found_not_verified_val, bar_width, bottom=verified_val,
                   color=model_color, alpha=alpha, edgecolor="black", linewidth=0.3, hatch="///")

            # Value labels
            total = verified_val + found_not_verified_val
            if verified_val > 8:
                ax.text(x_positions[i], verified_val / 2, f"{verified_val:.0f}%", ha="center", va="center",
                        fontsize=4, fontweight="bold", color="white")
            if total > 0:
                ax.text(x_positions[i], total + 2, f"{total:.0f}%", ha="center", va="bottom",
                        fontsize=4, fontweight="bold")

    # Add vertical separators between GPT-2 and Llama groups
    separator_x = 4 * cluster_width - 0.15
    ax.axvline(x=separator_x, color="black", linestyle="-", linewidth=1, alpha=0.7)

    # Subplot formatting
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(cluster_labels, fontsize=5)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.3, 8 * cluster_width - 0.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylabel("Tree Found/Verified (%)", fontsize=7)

    # Create legend for rp values and verification status
    legend_elements = [
        Patch(facecolor="gray", edgecolor="black", label="Verified"),
        Patch(facecolor="gray", edgecolor="black", hatch="///", label="Found (not verified)"),
    ]
    # Add small text explaining rp bar positions
    ax.text(0.98, 0.98, "Bars L→R: rp=1, 2, 3", transform=ax.transAxes,
            fontsize=5, ha="right", va="top", style="italic")

    ax.legend(handles=legend_elements, loc="upper left", fontsize=5)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "forward_chaining_results_single")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_forward_chaining_results_lines(output_dir, metrics_by_llm, rp_values):
    """Create a line graph showing verified rates across rp values.

    X-axis: required_passes (1, 2, 3)
    8 lines: one for each LLM × model × correctness configuration.
    Uses max_rank = 1 for all.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # LLM-specific color mappings for Coconut/CODI
    llm_colors = {
        "gpt2": {"coconut": COLOR_GPT2_COCONUT, "codi": COLOR_GPT2_CODI},
        "llama32-1b": {"coconut": COLOR_LLAMA_COCONUT, "codi": COLOR_LLAMA_CODI},
    }

    # Define the 8 configurations: (base_llm, model_type, verified_key, label, is_incorrect)
    configurations = [
        ("gpt2", "coconut", "correct_verified_pct", "GPT-2 Coconut Correct", False),
        ("gpt2", "coconut", "incorrect_verified_pct", "GPT-2 Coconut Incorrect", True),
        ("gpt2", "codi", "correct_verified_pct", "GPT-2 CODI Correct", False),
        ("gpt2", "codi", "incorrect_verified_pct", "GPT-2 CODI Incorrect", True),
        ("llama32-1b", "coconut", "correct_verified_pct", "Llama Coconut Correct", False),
        ("llama32-1b", "coconut", "incorrect_verified_pct", "Llama Coconut Incorrect", True),
        ("llama32-1b", "codi", "correct_verified_pct", "Llama CODI Correct", False),
        ("llama32-1b", "codi", "incorrect_verified_pct", "Llama CODI Incorrect", True),
    ]

    mr = 1  # Fixed max_rank = 1

    # Markers for different configurations
    markers = ["o", "s", "^", "D", "o", "s", "^", "D"]

    for idx, (base_llm, model_type, verified_key, label, is_incorrect) in enumerate(configurations):
        all_metrics = metrics_by_llm.get(base_llm, {})
        model_color = llm_colors[base_llm][model_type]
        linestyle = "--" if is_incorrect else "-"
        marker = markers[idx]

        # Gather data for each rp
        verified_vals = []
        for rp in rp_values:
            metrics = all_metrics.get(model_type, {}).get((rp, mr))
            if metrics is not None:
                verified_vals.append(metrics[verified_key])
            else:
                verified_vals.append(None)

        ax.plot(rp_values, verified_vals, color=model_color, linestyle=linestyle,
                marker=marker, markersize=5, linewidth=1.5, label=label)

    # Formatting
    ax.set_xlabel("Required Passes", fontsize=9)
    ax.set_ylabel("Verified Rate (%)", fontsize=9)
    ax.set_xticks(rp_values)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)

    # Legend outside the plot on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "forward_chaining_results_lines")
    for ext in [".png", ".pdf"]:
        path = base_path + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_individual(output_dir, metrics_by_llm, rp_values, mr_values):
    """Create individual plots for each (rp, mr) with both LLMs side by side."""
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    # Create shared legend elements
    legend_elements = [
        Patch(facecolor=COLOR_CORRECT, edgecolor="black", label="Correct & Verified"),
        Patch(facecolor=COLOR_INCORRECT, edgecolor="black", label="Incorrect & Verified"),
        Patch(facecolor=COLOR_CORRECT, edgecolor="black", hatch="///", label="Correct & Found (not verified)"),
        Patch(facecolor=COLOR_INCORRECT, edgecolor="black", hatch="///", label="Incorrect & Found (not verified)"),
    ]

    for mr in mr_values:
        for rp in rp_values:
            fig_single, ax_single = plt.subplots(figsize=(10, 5))
            plot_combined_cell(ax_single, rp, mr, metrics_by_llm)

            # Add legend below the plot
            fig_single.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8,
                              bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.18)

            base_path_single = os.path.join(individual_dir, f"hyperparam_sweep_rp{rp}_mr{mr}")
            for ext in [".png", ".pdf"]:
                path = base_path_single + ext
                fig_single.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved combined individual: rp={rp}, mr={mr}")
            plt.close(fig_single)


def main():
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter sweep results for forward chaining experiments"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/forward_chaining",
        help="Base directory containing result folders (default: results/forward_chaining)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/forward_chaining/plots",
        help="Output directory for plots (default: results/forward_chaining/plots)"
    )
    parser.add_argument(
        "--base_llm",
        type=str,
        nargs="+",
        default=["gpt2", "llama32-1b"],
        help="Base LLM(s) to plot (default: both gpt2 and llama32-1b)"
    )
    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base LLMs: {args.base_llm}")

    rp_values = [1, 2, 3]
    mr_values = [1, 2, 3]
    model_types = ["coconut", "codi"]

    # Load metrics for all LLMs
    metrics_by_llm = {}
    for base_llm in args.base_llm:
        print(f"\n{'='*50}")
        print(f"Loading metrics for: {base_llm}")
        print(f"{'='*50}")
        all_metrics = load_all_metrics(args.results_dir, base_llm, rp_values, mr_values, model_types)
        metrics_by_llm[base_llm] = all_metrics

    # Create Figure 6
    print(f"\n{'='*50}")
    print("Creating Figure 6")
    print(f"{'='*50}")
    plot_forward_chaining_results_2x4(args.output_dir, metrics_by_llm, rp_values)

    print("\nDone!")


if __name__ == "__main__":
    main()
