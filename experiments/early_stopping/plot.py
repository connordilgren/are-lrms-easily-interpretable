"""
Plot early stopping results for CoT, Coconut, and CODI models.

Generates CDF plots and bar charts comparing early stopping metrics.
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Global font size settings for higher resolution plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def get_normalized_first_match(data):
    """
    Get normalized first_match positions (first_match / num_reasoning_tokens).
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        first_match = sample.get("num_reasoning_tokens_first_match")
        total = sample.get("num_reasoning_tokens")
        if first_match is not None and total is not None and total > 0:
            percentages.append(first_match / total * 100)
    return percentages


def get_normalized_stable_match(data):
    """
    Get normalized stable_match positions (stable_match / num_reasoning_tokens).
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        stable_match = sample.get("num_reasoning_tokens_stable_match")
        total = sample.get("num_reasoning_tokens")
        if stable_match is not None and total is not None and total > 0:
            percentages.append(stable_match / total * 100)
    return percentages


def get_normalized_answer_in_topk(data):
    """
    Get normalized positions where answer first appears in top-k (token-level).
    Uses vocab_projection_by_token data.
    Returns list of percentages in [0, 100].

    Note: first_pos is 0-indexed. We use the index directly as the numerator.
    Position 0 (first) -> 0/num_tokens = 0%, Position 6 (last for 6 tokens) -> 6/6 = 100%.
    """
    percentages = []
    for sample in data["samples"]:
        vp = sample.get("vocab_projection_by_token")
        num_tokens = sample.get("num_reasoning_tokens")
        if vp is not None and num_tokens is not None and num_tokens > 0:
            first_pos = vp.get("first_position_answer_in_top_k")
            if first_pos is not None:
                percentages.append(first_pos / num_tokens * 100)
    return percentages


def get_normalized_rank_stable(data):
    """
    Get normalized positions where top-k ranking stops changing (token-level).
    Uses vocab_projection_by_token data.
    Returns list of percentages in [0, 100].

    Note: rank_stable is 0-indexed. We use the index directly as the numerator.
    Position 0 (first) -> 0/num_tokens = 0%, Position 6 (last for 6 tokens) -> 6/6 = 100%.
    """
    percentages = []
    for sample in data["samples"]:
        vp = sample.get("vocab_projection_by_token")
        num_tokens = sample.get("num_reasoning_tokens")
        if vp is not None and num_tokens is not None and num_tokens > 0:
            rank_stable = vp.get("rank_stable_position")
            if rank_stable is not None:
                percentages.append(rank_stable / num_tokens * 100)
    return percentages


def get_step_first_match(data):
    """
    Get normalized first_match by step (for CoT only).
    Uses step_results to find which step first matches the final answer.
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        step_results = sample.get("step_results", [])
        num_steps = sample.get("num_reasoning_steps")
        if not step_results or not num_steps or num_steps == 0:
            continue
        # Find first step where answer matches final
        first_match_step = None
        for step_result in step_results:
            if step_result.get("matches_final"):
                first_match_step = step_result.get("num_reasoning_steps")
                break
        if first_match_step is not None:
            percentages.append(first_match_step / num_steps * 100)
    return percentages


def get_step_stable_match(data):
    """
    Get normalized stable_match by step (for CoT only).
    Finds the step where the answer stabilizes to the final answer.
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        step_results = sample.get("step_results", [])
        num_steps = sample.get("num_reasoning_steps")
        if not step_results or not num_steps or num_steps == 0:
            continue
        # Find stable step - walk backwards to find where answer became stable
        final_answer = step_results[-1].get("answer") if step_results else None
        stable_step = num_steps
        for i in range(len(step_results) - 1, -1, -1):
            if step_results[i].get("answer") != final_answer:
                break
            stable_step = step_results[i].get("num_reasoning_steps")
        percentages.append(stable_step / num_steps * 100)
    return percentages


def get_step_answer_in_topk(data):
    """
    Get normalized step positions where answer first appears in top-k (CoT only).
    Uses vocab_projection_by_step data.
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        vp_step = sample.get("vocab_projection_by_step")
        if vp_step is None:
            continue

        first_step = vp_step.get("first_step_answer_in_top_k")
        num_steps = vp_step.get("num_reasoning_steps")

        if first_step is not None and num_steps is not None and num_steps > 0:
            percentages.append(first_step / num_steps * 100)

    return percentages


def get_step_rank_stable(data):
    """
    Get normalized step positions where top-k ranking stabilizes (CoT only).
    Uses vocab_projection_by_step data.
    Returns list of percentages in [0, 100].
    """
    percentages = []
    for sample in data["samples"]:
        vp_step = sample.get("vocab_projection_by_step")
        if vp_step is None:
            continue

        rank_stable = vp_step.get("rank_stable_step")
        num_steps = vp_step.get("num_reasoning_steps")

        if rank_stable is not None and num_steps is not None and num_steps > 0:
            percentages.append(rank_stable / num_steps * 100)

    return percentages


def plot_cdf(ax, values, label, color, linestyle="-"):
    """Plot CDF on given axis. Y-axis is in percent (0-100)."""
    if not values:
        return
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100  # Convert to percent
    ax.plot(sorted_vals, cdf, label=label, color=color, linestyle=linestyle, linewidth=2)


def save_plot(output_dir, name, base_llm=None):
    """Save current figure as PNG and PDF."""
    if base_llm:
        name = f"{name}_{base_llm}"
    plt.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / name} (.png, .pdf)")


def create_cdf_plot_by_token(results_files, output_dir, base_llm=None):
    """
    Create CDF plot comparing CoT, Coconut, and CODI by token/latent/iteration.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        "cot": "#3498db",      # Blue
        "coconut": "#2ecc71",  # Green
        "codi": "#e74c3c"      # Red
    }

    for model_type, filepath in results_files.items():
        data = load_results(filepath)

        # Plot first_match (solid)
        fractions_first = get_normalized_first_match(data)
        if fractions_first:
            plot_cdf(ax, fractions_first,
                    f"{model_type.upper()} (first match, n={len(fractions_first)})",
                    colors[model_type], linestyle="-")

        # Plot stable_match (dashed)
        fractions_stable = get_normalized_stable_match(data)
        if fractions_stable:
            plot_cdf(ax, fractions_stable,
                    f"{model_type.upper()} (stable match, n={len(fractions_stable)})",
                    colors[model_type], linestyle="--")

    ax.set_xlabel("% of Reasoning Units Used\n(CoT: tokens, Coconut: latents, CODI: iterations)")
    ax.set_ylabel("Cumulative % Matching Model's Final Answer")
    ax.set_title("Early Stopping by Token/Latent/Iteration", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.axvline(x=100, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_cdf_by_token", base_llm)
    plt.close()


def create_cdf_plot_by_step(results_files, output_dir, base_llm=None):
    """
    Create CDF plot with CoT by steps, Coconut by latents, CODI by iterations.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    # CoT - use step-based
    if "cot" in results_files:
        data_cot = load_results(results_files["cot"])
        fractions_first = get_step_first_match(data_cot)
        if fractions_first:
            plot_cdf(ax, fractions_first,
                    f"COT (first match, by step, n={len(fractions_first)})",
                    colors["cot"], linestyle="-")
        fractions_stable = get_step_stable_match(data_cot)
        if fractions_stable:
            plot_cdf(ax, fractions_stable,
                    f"COT (stable match, by step, n={len(fractions_stable)})",
                    colors["cot"], linestyle="--")

    # Coconut - use token-based (no steps)
    if "coconut" in results_files:
        data_coconut = load_results(results_files["coconut"])
        fractions_first = get_normalized_first_match(data_coconut)
        if fractions_first:
            plot_cdf(ax, fractions_first,
                    f"COCONUT (first match, by latent, n={len(fractions_first)})",
                    colors["coconut"], linestyle="-")
        fractions_stable = get_normalized_stable_match(data_coconut)
        if fractions_stable:
            plot_cdf(ax, fractions_stable,
                    f"COCONUT (stable match, by latent, n={len(fractions_stable)})",
                    colors["coconut"], linestyle="--")

    # CODI - use iteration-based (no steps)
    if "codi" in results_files:
        data_codi = load_results(results_files["codi"])
        fractions_first = get_normalized_first_match(data_codi)
        if fractions_first:
            plot_cdf(ax, fractions_first,
                    f"CODI (first match, by iteration, n={len(fractions_first)})",
                    colors["codi"], linestyle="-")
        fractions_stable = get_normalized_stable_match(data_codi)
        if fractions_stable:
            plot_cdf(ax, fractions_stable,
                    f"CODI (stable match, by iteration, n={len(fractions_stable)})",
                    colors["codi"], linestyle="--")

    ax.set_xlabel("% of Reasoning Used\n(CoT: steps, Coconut: latents, CODI: iterations)")
    ax.set_ylabel("Cumulative % Matching Model's Final Answer")
    ax.set_title("Early Stopping by Step/Latent/Iteration", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.axvline(x=100, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_cdf_by_step", base_llm)
    plt.close()


def create_bar_chart_by_token(results_files, output_dir, base_llm=None):
    """
    Create 2x2 grouped bar chart comparing metrics across all 3 models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = [
        ("Force Stop\nFirst Match", get_normalized_first_match),
        ("Force Stop\nStable Match", get_normalized_stable_match),
        ("Vocab Proj\nAnswer Appearance", get_normalized_answer_in_topk),
        ("Vocab Proj\nRank Stability", get_normalized_rank_stable),
    ]

    model_order = ["cot", "coconut", "codi"]
    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    # Get k value from one of the data files
    first_file = next(iter(results_files.values()))
    data = load_results(first_file)
    k_value = data.get("config", {}).get("top_k", 10)

    bar_width = 0.25
    x = np.arange(1)  # Single dataset (GSM)

    for idx, (metric_name, metric_fn) in enumerate(metrics):
        ax = axes[idx]

        means = []
        stds = []
        labels = []

        for model_type in model_order:
            if model_type in results_files:
                data = load_results(results_files[model_type])
                fracs = metric_fn(data)
                if fracs:
                    means.append(np.mean(fracs))
                    stds.append(np.std(fracs))
                else:
                    means.append(0)
                    stds.append(0)
                labels.append(model_type.upper())
            else:
                means.append(0)
                stds.append(0)
                labels.append(model_type.upper())

        # Create bars
        x_positions = x + np.arange(len(model_order)) * bar_width - bar_width
        for i, (mean, std, label, model_type) in enumerate(zip(means, stds, labels, model_order)):
            if model_type in results_files:
                ax.bar(x_positions[i], mean, bar_width,
                       yerr=std, label=label,
                       color=colors[model_type], alpha=0.8,
                       capsize=5, error_kw={"linewidth": 1.5})
                # Add percentage label to the right side of bar to avoid error bar
                ax.text(x_positions[i] + bar_width * 0.1, mean + 3, f'{mean:.0f}%',
                       ha='left', va='center', fontweight='bold')

        ax.set_ylabel("% of Reasoning Used")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        # Add k value to x-axis label for vocab projection plots
        if idx >= 2:  # Vocab projection plots
            ax.set_xticklabels([f"GSM\nk={k_value}"])
        else:
            ax.set_xticklabels(["GSM"])
        ax.set_ylim(0, 110)  # Extended to 110 to accommodate percentage labels
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(loc="lower right")

    plt.suptitle("Early Stopping Metrics (CoT: by token)", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_by_token", base_llm)
    plt.close()


def create_bar_chart_by_step(results_files, output_dir, base_llm=None):
    """
    Create 2x2 bar chart with CoT using step-level metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Define metrics - CoT uses step-based, others use token-based
    metrics = [
        ("Force Stop\nFirst Match", {
            "cot": get_step_first_match,
            "coconut": get_normalized_first_match,
            "codi": get_normalized_first_match
        }),
        ("Force Stop\nStable Match", {
            "cot": get_step_stable_match,
            "coconut": get_normalized_stable_match,
            "codi": get_normalized_stable_match
        }),
        ("Vocab Proj\nAnswer Appearance", {
            "cot": get_step_answer_in_topk,  # Step-level for CoT
            "coconut": get_normalized_answer_in_topk,
            "codi": get_normalized_answer_in_topk
        }),
        ("Vocab Proj\nRank Stability", {
            "cot": get_step_rank_stable,  # Step-level for CoT
            "coconut": get_normalized_rank_stable,
            "codi": get_normalized_rank_stable
        }),
    ]

    model_order = ["cot", "coconut", "codi"]
    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    # Get k value from one of the data files
    first_file = next(iter(results_files.values()))
    data = load_results(first_file)
    k_value = data.get("config", {}).get("top_k", 10)

    bar_width = 0.25
    x = np.arange(1)  # Single dataset (GSM)

    for idx, (metric_name, metric_fns) in enumerate(metrics):
        ax = axes[idx]

        means = []
        stds = []
        labels = []

        for model_type in model_order:
            if model_type in results_files:
                data = load_results(results_files[model_type])
                metric_fn = metric_fns[model_type]
                fracs = metric_fn(data)
                if fracs:
                    means.append(np.mean(fracs))
                    stds.append(np.std(fracs))
                else:
                    means.append(0)
                    stds.append(0)
                labels.append(model_type.upper())
            else:
                means.append(0)
                stds.append(0)
                labels.append(model_type.upper())

        # Create bars
        x_positions = x + np.arange(len(model_order)) * bar_width - bar_width
        for i, (mean, std, label, model_type) in enumerate(zip(means, stds, labels, model_order)):
            if model_type in results_files:
                ax.bar(x_positions[i], mean, bar_width,
                       yerr=std, label=label,
                       color=colors[model_type], alpha=0.8,
                       capsize=5, error_kw={"linewidth": 1.5})
                # Add percentage label to the right side of bar to avoid error bar
                ax.text(x_positions[i] + bar_width * 0.1, mean + 3, f'{mean:.0f}%',
                       ha='left', va='center', fontweight='bold')

        ax.set_ylabel("% of Reasoning Used")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        # Add k value to x-axis label for vocab projection plots
        if idx >= 2:  # Vocab projection plots
            ax.set_xticklabels([f"GSM\nk={k_value}"])
        else:
            ax.set_xticklabels(["GSM"])
        ax.set_ylim(0, 110)  # Extended to 110 to accommodate percentage labels
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(loc="lower right")

    plt.suptitle("Early Stopping Metrics (CoT: by step)", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_by_step", base_llm)
    plt.close()


def print_summary_table(results_files):
    """Print summary statistics for all models."""
    print("\n" + "=" * 60)
    print("EARLY STOPPING SUMMARY")
    print("=" * 60)

    for model_type, filepath in results_files.items():
        data = load_results(filepath)
        summary = data["summary"]
        es = summary["early_stopping"]
        vp = summary["vocab_projection_by_token"]

        total_tokens = es.get("avg_reasoning_tokens", 0)
        first_match = es.get("avg_num_reasoning_tokens_first_match")
        stable_match = es.get("avg_num_reasoning_tokens_stable_match")
        vocab_first = vp.get("avg_first_position_answer_in_top_k")

        print(f"\n{model_type.upper()}:")
        print(f"  Samples: {es['total_samples']}")
        print(f"  Avg reasoning units: {total_tokens:.2f}")

        if first_match is not None:
            print(f"  First match: {first_match:.2f} ({first_match/total_tokens*100:.1f}%)")
        else:
            print(f"  First match: N/A")

        if stable_match is not None:
            print(f"  Stable match: {stable_match:.2f} ({stable_match/total_tokens*100:.1f}%)")
        else:
            print(f"  Stable match: N/A")

        if vocab_first is not None:
            print(f"  Vocab proj first: {vocab_first:.2f}")
        else:
            print(f"  Vocab proj first: N/A")


def print_summary_table_multi_dataset(datasets):
    """Print summary table for multiple datasets."""
    for dataset_name, results_files in datasets.items():
        print(f"\n{dataset_name} Results:")
        print("="*80)
        print_summary_table(results_files)


def create_cdf_plot_by_token_multi(datasets, output_dir, base_llm=None):
    """Create CDF plot with multiple datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    linestyles = {
        "GSM8k": "-",
        "ProsQA": "--",
        "ProntoQA": ":"
    }

    for dataset_name, results_files in datasets.items():
        for model_type in ["cot", "coconut", "codi"]:
            if model_type in results_files:
                data = load_results(results_files[model_type])
                fracs = get_normalized_answer_in_topk(data)
                if fracs:
                    plot_cdf(ax, fracs,
                           f"{model_type.upper()} {dataset_name} (n={len(fracs)})",
                           colors[model_type], linestyle=linestyles.get(dataset_name, "-"))

    ax.set_xlabel("% of Reasoning Used (Token-Level)")
    ax.set_ylabel("Cumulative % with Answer in Top-K")
    ax.set_title("Vocabulary Projection: Answer Appearance", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_cdf_by_token", base_llm)
    plt.close()


def create_cdf_plot_by_step_multi(datasets, output_dir, base_llm=None):
    """Create CDF plot by step for multiple datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    linestyles = {
        "GSM8k": {"-": "-", "--": ":"},
        "ProsQA": {"-": "--", "--": "-."},
        "ProntoQA": {"-": (0, (3, 1, 1, 1)), "--": (0, (1, 1))}
    }

    for dataset_name, results_files in datasets.items():
        # CoT - use step-based
        if "cot" in results_files:
            data_cot = load_results(results_files["cot"])
            fractions_first = get_step_first_match(data_cot)
            if fractions_first:
                plot_cdf(ax, fractions_first,
                        f"COT {dataset_name} (first match, n={len(fractions_first)})",
                        colors["cot"], linestyle=linestyles[dataset_name]["-"])
            fractions_stable = get_step_stable_match(data_cot)
            if fractions_stable:
                plot_cdf(ax, fractions_stable,
                        f"COT {dataset_name} (stable match, n={len(fractions_stable)})",
                        colors["cot"], linestyle=linestyles[dataset_name]["--"])

        # Coconut/CODI - use token-based
        for model_type in ["coconut", "codi"]:
            if model_type in results_files:
                data = load_results(results_files[model_type])
                fractions_first = get_normalized_first_match(data)
                if fractions_first:
                    plot_cdf(ax, fractions_first,
                            f"{model_type.upper()} {dataset_name} (first, n={len(fractions_first)})",
                            colors[model_type], linestyle=linestyles[dataset_name]["-"])
                fractions_stable = get_normalized_stable_match(data)
                if fractions_stable:
                    plot_cdf(ax, fractions_stable,
                            f"{model_type.upper()} {dataset_name} (stable, n={len(fractions_stable)})",
                            colors[model_type], linestyle=linestyles[dataset_name]["--"])

    ax.set_xlabel("% of Reasoning Used")
    ax.set_ylabel("Cumulative % Matching Final Answer")
    ax.set_title("Early Stopping by Step/Latent/Iteration", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_cdf_by_step", base_llm)
    plt.close()


def create_bar_chart_by_token_multi(datasets, output_dir, base_llm=None):
    """Create bar chart comparing metrics across datasets."""
    import re

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("Force Stop\nFirst Match", get_normalized_first_match, False),
        ("Force Stop\nStable Match", get_normalized_stable_match, False),
        ("Vocab Proj\nAnswer Appearance", get_normalized_answer_in_topk, True),
        ("Vocab Proj\nRank Stability", get_normalized_rank_stable, True),
    ]

    model_order = ["cot", "coconut", "codi"]
    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25

    # Extract k values from filenames for each dataset
    k_values = {}
    for dataset_name, results_files in datasets.items():
        # Get k from any file in this dataset
        first_file = next(iter(results_files.values()))
        filename = str(first_file.name)
        k_match = re.search(r'_k(\d+)_', filename)
        if k_match:
            k_values[dataset_name] = k_match.group(1)
        else:
            k_values[dataset_name] = None

    for idx, (metric_name, metric_fn, show_k) in enumerate(metrics):
        ax = axes[idx]

        for model_idx, model_type in enumerate(model_order):
            means = []
            stds = []

            for dataset_name in dataset_names:
                results_files = datasets[dataset_name]
                if model_type in results_files:
                    data = load_results(results_files[model_type])
                    fracs = metric_fn(data)
                    if fracs:
                        means.append(np.mean(fracs))
                        stds.append(np.std(fracs))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (model_idx - 1) * bar_width
            ax.bar(x + offset, means, bar_width,
                  label=model_type.upper(),
                  color=colors[model_type],
                  yerr=stds, capsize=3, alpha=0.8)

        ax.set_ylabel("% of Reasoning Used")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)

        # Add k value to x-axis labels for vocab projection metrics
        if show_k:
            labels = [f"{name}\nk={k_values[name]}" if k_values[name] else name
                     for name in dataset_names]
        else:
            labels = dataset_names
        ax.set_xticklabels(labels)

        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    plt.suptitle("Early Stopping Metrics by Dataset (Token-Level)", fontweight="bold", y=0.995)
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_by_token", base_llm)
    plt.close()


def create_bar_chart_by_step_multi(datasets, output_dir, base_llm=None):
    """Create bar chart by step for multiple datasets."""
    import re

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Metrics: (name, token_metric_fn, step_metric_fn, show_k)
    metrics = [
        ("Force Stop\nFirst Match", get_normalized_first_match, get_step_first_match, False),
        ("Force Stop\nStable Match", get_normalized_stable_match, get_step_stable_match, False),
        ("Vocab Proj\nAnswer Appearance", get_normalized_answer_in_topk, get_step_answer_in_topk, True),
        ("Vocab Proj\nRank Stability", get_normalized_rank_stable, get_step_rank_stable, True),
    ]

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25

    # Extract k values from filenames for each dataset
    k_values = {}
    for dataset_name, results_files in datasets.items():
        # Get k from any file in this dataset
        first_file = next(iter(results_files.values()))
        filename = str(first_file.name)
        k_match = re.search(r'_k(\d+)_', filename)
        if k_match:
            k_values[dataset_name] = k_match.group(1)
        else:
            k_values[dataset_name] = None

    for idx, (metric_name, token_metric_fn, step_metric_fn, show_k) in enumerate(metrics):
        ax = axes[idx]

        # CoT uses step-based metrics
        cot_means = []
        cot_stds = []
        for dataset_name in dataset_names:
            results_files = datasets[dataset_name]
            if "cot" in results_files:
                data = load_results(results_files["cot"])
                fracs = step_metric_fn(data)
                if fracs:
                    cot_means.append(np.mean(fracs))
                    cot_stds.append(np.std(fracs))
                else:
                    cot_means.append(0)
                    cot_stds.append(0)
            else:
                cot_means.append(0)
                cot_stds.append(0)

        ax.bar(x - bar_width, cot_means, bar_width,
              label="COT (by step)",
              color=colors["cot"],
              yerr=cot_stds, capsize=3, alpha=0.8)

        # Coconut and CODI use token-based metrics
        for model_idx, model_type in enumerate(["coconut", "codi"]):
            means = []
            stds = []

            for dataset_name in dataset_names:
                results_files = datasets[dataset_name]
                if model_type in results_files:
                    data = load_results(results_files[model_type])
                    fracs = token_metric_fn(data)
                    if fracs:
                        means.append(np.mean(fracs))
                        stds.append(np.std(fracs))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = model_idx * bar_width
            label_suffix = "(by token)" if model_type == "coconut" else "(by iteration)"
            ax.bar(x + offset, means, bar_width,
                  label=f"{model_type.upper()} {label_suffix}",
                  color=colors[model_type],
                  yerr=stds, capsize=3, alpha=0.8)

        ax.set_ylabel("% of Reasoning Used")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)

        # Add k value to x-axis labels for vocab projection metrics
        if show_k:
            labels = [f"{name}\nk={k_values[name]}" if k_values[name] else name
                     for name in dataset_names]
        else:
            labels = dataset_names
        ax.set_xticklabels(labels)

        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    plt.suptitle("Early Stopping Metrics by Dataset\n(CoT: by step, Coconut: by token, CODI: by iteration)",
                fontweight="bold", y=0.995)
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_by_step", base_llm)
    plt.close()


def create_bar_chart_force_stop_by_step_multi(datasets, output_dir, base_llm=None):
    """Create 1x2 bar chart for force stop metrics (CoT by step)."""
    import re

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Force stop metrics only
    metrics = [
        ("First Match", get_normalized_first_match, get_step_first_match, False),
        ("Stable Match", get_normalized_stable_match, get_step_stable_match, False),
    ]

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25

    for idx, (metric_name, token_metric_fn, step_metric_fn, show_k) in enumerate(metrics):
        ax = axes[idx]

        # CoT uses step-based metrics
        cot_means = []
        cot_stds = []
        for dataset_name in dataset_names:
            results_files = datasets[dataset_name]
            if "cot" in results_files:
                data = load_results(results_files["cot"])
                fracs = step_metric_fn(data)
                if fracs:
                    cot_means.append(np.mean(fracs))
                    cot_stds.append(np.std(fracs))
                else:
                    cot_means.append(0)
                    cot_stds.append(0)
            else:
                cot_means.append(0)
                cot_stds.append(0)

        ax.bar(x - bar_width, cot_means, bar_width,
              label="Explicit Reasoning",
              color=colors["cot"],
              yerr=cot_stds, capsize=3, alpha=0.8)
        # Add value labels for CoT bars
        for i, mean in enumerate(cot_means):
            if mean > 0:
                ax.text(x[i] - bar_width, mean + 1, f'{mean:.0f}%',
                       ha='center', va='bottom', fontweight='bold',
                       alpha=1.0,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor='none'),
                       zorder=5)

        # Coconut and CODI use token-based metrics
        for model_idx, model_type in enumerate(["coconut", "codi"]):
            means = []
            stds = []

            for dataset_name in dataset_names:
                results_files = datasets[dataset_name]
                if model_type in results_files:
                    data = load_results(results_files[model_type])
                    fracs = token_metric_fn(data)
                    if fracs:
                        means.append(np.mean(fracs))
                        stds.append(np.std(fracs))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = model_idx * bar_width
            display_label = "Coconut" if model_type == "coconut" else model_type.upper()
            ax.bar(x + offset, means, bar_width,
                  label=display_label,
                  color=colors[model_type],
                  yerr=stds, capsize=3, alpha=0.8)
            # Add value labels
            for i, mean in enumerate(means):
                if mean > 0:
                    ax.text(x[i] + offset, mean + 1, f'{mean:.0f}%',
                           ha='center', va='bottom', fontweight='bold',
                           alpha=1.0,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                     edgecolor='none'),
                           zorder=5)

        if idx == 0:
            ax.set_ylabel("Percent of Reasoning to Output Final Answer")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        # Rename datasets for display
        display_names = {"GSM8k": "GSM8k-Aug", "ProntoQA": "PrOntoQA"}
        display_labels = [display_names.get(name, name) for name in dataset_names]
        ax.set_xticklabels(display_labels)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')

    # plt.suptitle("Early Stopping Experiment",
    #             fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_force_stop_by_step", base_llm)
    plt.close()


def create_bar_chart_force_stop_stacked_combined(
    gpt2_datasets,
    llama_datasets,
    output_dir,
):
    """
    Create 1x2 subplot comparing GPT-2 (left) and Llama (right) results using stacked bar charts.

    Each bar is stacked:
    - Bottom (solid): first match mean
    - Top (hatched '///'): stable match mean - first match mean
    - Error bar on top using stable match std

    Args:
        gpt2_datasets: Dict of {"GSM8k": {"cot": Path, "coconut": Path, "codi": Path}, ...}
        llama_datasets: Same structure for Llama
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharey=True)

    colors = {
        "cot": "#3498db",      # Blue
        "coconut": "#2ecc71",  # Green
        "codi": "#e74c3c"      # Red
    }

    model_order = ["cot", "coconut", "codi"]
    model_display_labels = {
        "cot": "Explicit Reasoning",
        "coconut": "Coconut",
        "codi": "CODI"
    }

    # Rename datasets for display
    display_names = {"GSM8k": "GSM8k-Aug", "ProsQA": "ProsQA", "ProntoQA": "PrOntoQA"}

    # Explicit order: GSM8k-Aug, ProsQA, PrOntoQA
    dataset_order = ["GSM8k", "ProsQA", "ProntoQA"]

    all_llms = [
        ("GPT-2 Small", gpt2_datasets),
        ("Llama-3.2-1B-Instruct", llama_datasets)
    ]

    bar_width = 0.25

    # Collect data for CSV export
    csv_rows = []

    for ax_idx, (llm_name, datasets) in enumerate(all_llms):
        ax = axes[ax_idx]

        dataset_names = [d for d in dataset_order if d in datasets]
        n_ds = len(dataset_names)
        x = np.arange(n_ds)

        for model_idx, model_type in enumerate(model_order):
            first_means = []
            stable_means = []
            stable_stds = []

            for dataset_name in dataset_names:
                results_files = datasets[dataset_name]
                if model_type in results_files:
                    data = load_results(results_files[model_type])

                    # CoT uses step-based metrics, others use token-based
                    if model_type == "cot":
                        first_fracs = get_step_first_match(data)
                        stable_fracs = get_step_stable_match(data)
                    else:
                        first_fracs = get_normalized_first_match(data)
                        stable_fracs = get_normalized_stable_match(data)

                    if first_fracs:
                        first_mean = np.mean(first_fracs)
                        first_std = np.std(first_fracs)
                        first_means.append(first_mean)
                    else:
                        first_mean = 0
                        first_std = 0
                        first_means.append(0)

                    if stable_fracs:
                        stable_mean = np.mean(stable_fracs)
                        stable_std = np.std(stable_fracs)
                        stable_means.append(stable_mean)
                        stable_stds.append(stable_std)
                    else:
                        stable_mean = 0
                        stable_std = 0
                        stable_means.append(0)
                        stable_stds.append(0)

                    # Add row to CSV data (round to nearest tenth)
                    csv_rows.append({
                        "llm": llm_name,
                        "dataset": dataset_name,
                        "model": model_display_labels[model_type],
                        "first_match_mean": round(first_mean, 1),
                        "first_match_std": round(first_std, 1),
                        "stable_match_mean": round(stable_mean, 1),
                        "stable_match_std": round(stable_std, 1),
                    })
                else:
                    first_means.append(0)
                    stable_means.append(0)
                    stable_stds.append(0)

            # Calculate bar positions
            offset = (model_idx - 1) * bar_width
            positions = x + offset

            # Convert to arrays for element-wise operations
            first_means = np.array(first_means)
            stable_means = np.array(stable_means)
            stable_stds = np.array(stable_stds)

            # Calculate the "additional" portion (stable - first)
            additional = stable_means - first_means
            additional = np.maximum(additional, 0)  # Ensure non-negative

            # Plot bottom bar (first match, solid)
            ax.bar(positions, first_means, bar_width,
                   label=model_display_labels[model_type] if ax_idx == 1 else "",
                   color=colors[model_type],
                   alpha=0.9)

            # Plot top bar (additional portion, hatched)
            ax.bar(positions, additional, bar_width,
                   bottom=first_means,
                   color=colors[model_type],
                   alpha=0.9,
                   hatch='///',
                   edgecolor='white',
                   linewidth=0.5)

            # Add error bars on top of the stacked bar (at stable match height)
            ax.errorbar(positions, stable_means, yerr=stable_stds,
                        fmt='none', capsize=2, color='black',
                        elinewidth=0.8, capthick=0.8)

            # Add labels at top of hatched bars
            for pos, stable_val in zip(positions, stable_means):
                # Show stable match % at top of hatched bar
                ax.text(pos, stable_val + 1, f'{stable_val:.0f}%',
                        ha='center', va='bottom', fontweight='normal',
                        fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  edgecolor='none', alpha=1.0),
                        zorder=5)

        # Set axis properties
        ax.set_title(llm_name, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(name, name) for name in dataset_names], fontsize=7)
        ax.set_ylim(0, 110)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        if ax_idx == 0:
            ax.set_ylabel("% of RT to Output Final Answer", fontsize=7)
        if ax_idx == 1:
            ax.legend(loc='upper right', framealpha=1.0, facecolor='white', fontsize=5)

    plt.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.12, wspace=0.08)

    # Save plot
    output_path = Path(output_dir)
    plt.savefig(output_path / "early_stopping_bar_force_stop_stacked_combined.png",
                dpi=600)
    plt.savefig(output_path / "early_stopping_bar_force_stop_stacked_combined.pdf")
    print(f"Saved: {output_path / 'early_stopping_bar_force_stop_stacked_combined'} (.png, .pdf)")

    # Save CSV with the plot data
    csv_path = output_path / "early_stopping_bar_force_stop_stacked_combined.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "llm", "dataset", "model",
            "first_match_mean", "first_match_std",
            "stable_match_mean", "stable_match_std"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_path}")

    plt.close()


def create_bar_chart_force_stop_by_token_multi(datasets, output_dir, base_llm=None):
    """Create 1x2 bar chart for force stop metrics (all models by token)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Force stop metrics only
    metrics = [
        ("Force Stop\nFirst Match", get_normalized_first_match),
        ("Force Stop\nStable Match", get_normalized_stable_match),
    ]

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25
    model_order = ["cot", "coconut", "codi"]

    for idx, (metric_name, metric_fn) in enumerate(metrics):
        ax = axes[idx]

        # All models use token-based metrics
        for model_idx, model_type in enumerate(model_order):
            means = []
            stds = []

            for dataset_name in dataset_names:
                results_files = datasets[dataset_name]
                if model_type in results_files:
                    data = load_results(results_files[model_type])
                    fracs = metric_fn(data)
                    if fracs:
                        means.append(np.mean(fracs))
                        stds.append(np.std(fracs))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (model_idx - 1) * bar_width
            label = f"{model_type.upper()} (by token)" if model_type == "cot" else model_type.upper()
            ax.bar(x + offset, means, bar_width,
                  label=label,
                  color=colors[model_type],
                  yerr=stds, capsize=3, alpha=0.8)

        ax.set_ylabel("% of Reasoning Used")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')

    plt.suptitle("Early Stopping Experiment: Force Stop Metrics", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_force_stop_by_token", base_llm)
    plt.close()


def get_position_distribution(data, metric_fn_name):
    """
    Get distribution of positions where answer appears or rank stabilizes.

    Returns dict with bins: 'early' (0-33%), 'middle' (33-67%), 'late' (67-100%), 'never'
    Each value is the count of samples in that bin.
    """
    bins = {'early': 0, 'middle': 0, 'late': 0, 'never': 0}

    for sample in data["samples"]:
        vp = sample.get("vocab_projection_by_token")
        num_tokens = sample.get("num_reasoning_tokens")

        if vp is None or num_tokens is None or num_tokens == 0:
            bins['never'] += 1
            continue

        # Get the position based on metric type
        if metric_fn_name == "answer":
            pos = vp.get("first_position_answer_in_top_k")
        else:  # rank_stable
            pos = vp.get("rank_stable_position")

        if pos is None:
            bins['never'] += 1
        else:
            # Calculate percentage (0-100)
            pct = pos / num_tokens * 100

            if pct < 33.33:
                bins['early'] += 1
            elif pct < 66.67:
                bins['middle'] += 1
            else:
                bins['late'] += 1

    return bins


def get_answer_coverage(data):
    """
    Get percentage of samples where answer appears in top-k at ANY position.

    Returns: percentage (0-100)
    """
    total_samples = 0
    samples_with_answer = 0

    for sample in data["samples"]:
        vp = sample.get("vocab_projection_by_token")
        if vp is None:
            continue

        total_samples += 1
        first_pos = vp.get("first_position_answer_in_top_k")

        if first_pos is not None:
            samples_with_answer += 1

    if total_samples == 0:
        return 0

    return (samples_with_answer / total_samples) * 100


def create_bar_chart_answer_coverage_multi(datasets, output_dir, base_llm=None):
    """Create bar chart showing % of samples where answer appears in top-k."""
    import re

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25
    model_order = ["cot", "coconut", "codi"]

    # Extract k values
    k_values = {}
    for dataset_name, results_files in datasets.items():
        first_file = next(iter(results_files.values()))
        filename = str(first_file.name)
        k_match = re.search(r'_k(\d+)_', filename)
        if k_match:
            k_values[dataset_name] = k_match.group(1)
        else:
            k_values[dataset_name] = None

    # Plot bars for each model
    for model_idx, model_type in enumerate(model_order):
        coverages = []

        for dataset_name in dataset_names:
            results_files = datasets[dataset_name]
            if model_type in results_files:
                data = load_results(results_files[model_type])
                coverage = get_answer_coverage(data)
                coverages.append(coverage)
            else:
                coverages.append(0)

        offset = (model_idx - 1) * bar_width
        ax.bar(x + offset, coverages, bar_width,
              label=model_type.upper(),
              color=colors[model_type],
              alpha=0.8)

    # Set labels
    labels = [f"{name}\nk={k_values[name]}" if k_values[name] else name
             for name in dataset_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Samples with Answer in Top-K", fontsize=11)
    ax.set_title("Vocab Projection: Answer Coverage", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_answer_coverage", base_llm)
    plt.close()


def create_bar_chart_rank_stability_multi(datasets, output_dir, base_llm=None):
    """Create bar chart for rank stability metric (mean + std)."""
    import re

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.25
    model_order = ["cot", "coconut", "codi"]

    # Extract k values
    k_values = {}
    for dataset_name, results_files in datasets.items():
        first_file = next(iter(results_files.values()))
        filename = str(first_file.name)
        k_match = re.search(r'_k(\d+)_', filename)
        if k_match:
            k_values[dataset_name] = k_match.group(1)
        else:
            k_values[dataset_name] = None

    # Plot bars for each model
    for model_idx, model_type in enumerate(model_order):
        means = []
        stds = []

        for dataset_name in dataset_names:
            results_files = datasets[dataset_name]
            if model_type in results_files:
                data = load_results(results_files[model_type])
                fracs = get_normalized_rank_stable(data)
                if fracs:
                    means.append(np.mean(fracs))
                    stds.append(np.std(fracs))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        offset = (model_idx - 1) * bar_width
        ax.bar(x + offset, means, bar_width,
              label=model_type.upper(),
              color=colors[model_type],
              yerr=stds, capsize=3, alpha=0.8)

    # Set labels
    labels = [f"{name}\nk={k_values[name]}" if k_values[name] else name
             for name in dataset_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Reasoning Used")
    ax.set_title("Vocab Projection: Rank Stability", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_rank_stability", base_llm)
    plt.close()


def create_bar_chart_vocab_proj_stacked_multi(datasets, output_dir, base_llm=None):
    """Create 1x2 stacked bar chart for vocab projection metrics."""
    import re

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = [
        ("Vocab Proj: Answer Appearance", "answer"),
        ("Vocab Proj: Rank Stability", "rank_stable"),
    ]

    colors = {
        "cot": "#3498db",
        "coconut": "#2ecc71",
        "codi": "#e74c3c"
    }

    # Hatching patterns for bins
    hatches = {
        'early': None,      # Solid
        'middle': '///',    # Light hatching
        'late': 'xxx',      # Dense hatching
        'never': '\\\\\\'   # Diagonal stripes
    }

    bin_labels = {
        'early': '0-33%',
        'middle': '33-67%',
        'late': '67-100%',
        'never': 'Never'
    }

    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    model_order = ["cot", "coconut", "codi"]

    # Extract k values from filenames
    k_values = {}
    for dataset_name, results_files in datasets.items():
        first_file = next(iter(results_files.values()))
        filename = str(first_file.name)
        k_match = re.search(r'_k(\d+)_', filename)
        if k_match:
            k_values[dataset_name] = k_match.group(1)
        else:
            k_values[dataset_name] = None

    for metric_idx, (metric_name, metric_type) in enumerate(metrics):
        ax = axes[metric_idx]

        x = np.arange(n_datasets * len(model_order))
        bar_width = 0.8

        # For each dataset and model, get distribution
        bar_positions = []
        bar_index = 0

        for dataset_idx, dataset_name in enumerate(dataset_names):
            results_files = datasets[dataset_name]

            for model_type in model_order:
                if model_type not in results_files:
                    bar_index += 1
                    continue

                data = load_results(results_files[model_type])
                bins = get_position_distribution(data, metric_type)

                total = sum(bins.values())
                if total == 0:
                    bar_index += 1
                    continue

                # Convert to percentages
                bin_pcts = {k: v / total * 100 for k, v in bins.items()}

                # Stack the bars
                bottom = 0
                for bin_name in ['early', 'middle', 'late', 'never']:
                    height = bin_pcts[bin_name]

                    ax.bar(bar_index, height, bar_width,
                          bottom=bottom,
                          color=colors[model_type],
                          hatch=hatches[bin_name],
                          edgecolor='black',
                          linewidth=0.5,
                          alpha=0.8)
                    bottom += height

                bar_index += 1

        # Set x-axis labels
        x_positions = []
        x_labels = []

        for dataset_idx, dataset_name in enumerate(dataset_names):
            results_files = datasets[dataset_name]

            # Calculate center position for this dataset's group
            n_models = len([m for m in model_order if m in results_files])
            start_pos = dataset_idx * len(model_order)
            center_pos = start_pos + (len(model_order) - 1) / 2

            x_positions.append(center_pos)
            k_val = k_values[dataset_name]
            label = f"{dataset_name}\nk={k_val}" if k_val else dataset_name
            x_labels.append(label)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)

        # Add model labels
        for dataset_idx, dataset_name in enumerate(dataset_names):
            results_files = datasets[dataset_name]
            for model_idx, model_type in enumerate(model_order):
                if model_type in results_files:
                    pos = dataset_idx * len(model_order) + model_idx
                    ax.text(pos, -8, model_type.upper(),
                           ha='center', va='top', rotation=0)

        ax.set_ylabel("% of Samples", fontsize=11)
        ax.set_title(metric_name, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend (only for first subplot)
        if metric_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', edgecolor='black', hatch=hatches['early'],
                     label=f'Early ({bin_labels["early"]})'),
                Patch(facecolor='gray', edgecolor='black', hatch=hatches['middle'],
                     label=f'Middle ({bin_labels["middle"]})'),
                Patch(facecolor='gray', edgecolor='black', hatch=hatches['late'],
                     label=f'Late ({bin_labels["late"]})'),
                Patch(facecolor='gray', edgecolor='black', hatch=hatches['never'],
                     label=bin_labels['never'])
            ]
            ax.legend(handles=legend_elements, loc='upper left', title='Position Bins')

    plt.suptitle("Vocabulary Projection: Distribution of Answer Position", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_vocab_proj", base_llm)
    plt.close()


def get_pct_uses_latent_reasoning(data, metric_name):
    """
    Get percentage of samples where the specified metric is not zero.

    Args:
        data: Loaded results data
        metric_name: Either "num_reasoning_tokens_first_match" or "num_reasoning_tokens_stable_match"

    Returns:
        Percentage (0-100) of samples where metric != 0
    """
    total = 0
    nonzero = 0
    for sample in data["samples"]:
        val = sample.get(metric_name)
        if val is not None:
            total += 1
            if val != 0:
                nonzero += 1
    if total == 0:
        return 0
    return (nonzero / total) * 100


def create_bar_chart_latent_by_correctness(
    early_stopping_files, performance_files, labels, output_dir, base_llm=None
):
    """
    Create grouped bar chart showing % using non-zero reasoning tokens,
    grouped by correctness (correct vs incorrect).

    Args:
        early_stopping_files: List of paths to early stopping result files
        performance_files: List of paths to dataset performance files
        labels: List of labels for each dataset/model
        output_dir: Directory to save the plot
        base_llm: Base LLM name for output filename
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    bar_width = 0.35

    correct_pcts = []
    incorrect_pcts = []
    correct_ns = []
    incorrect_ns = []

    for es_file, perf_file in zip(early_stopping_files, performance_files):
        # Load both files
        es_data = load_results(es_file)
        perf_data = load_results(perf_file)

        # Build lookup for correctness by index
        correctness_by_idx = {}
        for result in perf_data["results"]:
            correctness_by_idx[result["idx"]] = result["correct"]

        # Count samples by correctness and reasoning usage
        correct_total = 0
        correct_nonzero = 0
        incorrect_total = 0
        incorrect_nonzero = 0

        for sample in es_data["samples"]:
            sample_idx = sample["sample_idx"]
            stable_match = sample.get("num_reasoning_tokens_stable_match")

            if stable_match is None:
                continue
            if sample_idx not in correctness_by_idx:
                continue

            correct = correctness_by_idx[sample_idx]
            uses_reasoning = (stable_match != 0)

            if correct:
                correct_total += 1
                if uses_reasoning:
                    correct_nonzero += 1
            else:
                incorrect_total += 1
                if uses_reasoning:
                    incorrect_nonzero += 1

        # Calculate percentages
        correct_pct = (correct_nonzero / correct_total * 100) if correct_total > 0 else 0
        incorrect_pct = (incorrect_nonzero / incorrect_total * 100) if incorrect_total > 0 else 0

        correct_pcts.append(correct_pct)
        incorrect_pcts.append(incorrect_pct)
        correct_ns.append(correct_total)
        incorrect_ns.append(incorrect_total)

    # Create bars
    bars1 = ax.bar(x - bar_width/2, correct_pcts, bar_width, label='Correct', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + bar_width/2, incorrect_pcts, bar_width, label='Incorrect', color='#e74c3c', alpha=0.8)

    # Add percentage and n labels on top of bars
    for i, (bar, pct, n) in enumerate(zip(bars1, correct_pcts, correct_ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
               f'{pct:.1f}%\n(n={n})', ha='center', va='bottom', fontweight='bold')

    for i, (bar, pct, n) in enumerate(zip(bars2, incorrect_pcts, incorrect_ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
               f'{pct:.1f}%\n(n={n})', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("% Uses >=1 Latent Reasoning Token")
    ax.set_xlabel("Dataset / Model")
    ax.set_title("Latent Reasoning Usage by Correctness (Stable Match)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, max(max(correct_pcts), max(incorrect_pcts)) + 25)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_latent_by_correctness", base_llm)
    plt.close()


def create_confusion_matrix_latent_vs_correctness(
    early_stopping_files, performance_files, labels, output_dir, base_llm=None
):
    """
    Create 3x2 subplot of confusion matrices showing relationship between
    latent reasoning usage and correctness.

    Args:
        early_stopping_files: List of paths to early stopping result files
        performance_files: List of paths to dataset performance files
        labels: List of labels for each dataset/model
        output_dir: Directory to save the plot
        base_llm: Base LLM name for output filename
    """
    from matplotlib.colors import LinearSegmentedColormap

    n_datasets = len(labels)
    n_rows = (n_datasets + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    axes = axes.flatten()

    # Create a green colormap
    greens = LinearSegmentedColormap.from_list('greens', ['#f0fff0', '#2ecc71'])

    for idx, (es_file, perf_file, label) in enumerate(zip(early_stopping_files, performance_files, labels)):
        ax = axes[idx]

        # Load both files
        es_data = load_results(es_file)
        perf_data = load_results(perf_file)

        # Build lookup for correctness by index
        correctness_by_idx = {}
        for result in perf_data["results"]:
            correctness_by_idx[result["idx"]] = result["correct"]

        # Build confusion matrix counts
        # Rows: Zero, Non-Zero (reasoning tokens)
        # Cols: Correct, Incorrect
        matrix = np.zeros((2, 2), dtype=int)

        for sample in es_data["samples"]:
            sample_idx = sample["sample_idx"]
            stable_match = sample.get("num_reasoning_tokens_stable_match")

            if stable_match is None:
                continue
            if sample_idx not in correctness_by_idx:
                continue

            correct = correctness_by_idx[sample_idx]
            is_zero = (stable_match == 0)

            row = 0 if is_zero else 1  # 0=Zero, 1=Non-Zero
            col = 0 if correct else 1  # 0=Correct, 1=Incorrect

            matrix[row, col] += 1

        # Calculate percentages for annotation
        total = matrix.sum()
        if total > 0:
            pct_matrix = matrix / total * 100
        else:
            pct_matrix = matrix.astype(float)

        # Create heatmap using imshow
        im = ax.imshow(matrix, cmap=greens, aspect='auto')

        # Add grid lines
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_yticklabels(['Zero', 'Non-Zero'])

        # Add text annotations (count and percentage)
        for i in range(2):
            for j in range(2):
                count = matrix[i, j]
                pct = pct_matrix[i, j]
                ax.text(j, i - 0.1, f'{count}',
                       ha='center', va='center', fontweight='bold')
                ax.text(j, i + 0.2, f'({pct:.1f}%)',
                       ha='center', va='center', color='#555555')

        ax.set_xlabel('Correctness')
        ax.set_ylabel('Reasoning Tokens Used')
        ax.set_title(label, fontweight='bold')

        # Add cell borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    # Hide unused axes
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Latent Reasoning Usage vs Correctness (Stable Match)", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_confusion_latent_vs_correct", base_llm)
    plt.close()


def create_bar_chart_pct_uses_latent(result_files_with_labels, output_dir, base_llm=None):
    """
    Create 2x1 bar chart showing % of samples that use >=1 latent reasoning token.

    Top subplot: based on num_reasoning_tokens_first_match != 0
    Bottom subplot: based on num_reasoning_tokens_stable_match != 0

    Args:
        result_files_with_labels: List of (filepath, label) tuples
        output_dir: Directory to save the plot
        base_llm: Base LLM name for output filename
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    metrics = [
        ("First Match", "num_reasoning_tokens_first_match"),
        ("Stable Match", "num_reasoning_tokens_stable_match"),
    ]

    labels = [label for _, label in result_files_with_labels]
    x = np.arange(len(labels))
    bar_width = 0.6

    color = "#2ecc71"  # Green for Coconut

    for idx, (metric_label, metric_name) in enumerate(metrics):
        ax = axes[idx]

        percentages = []
        for filepath, label in result_files_with_labels:
            data = load_results(filepath)
            pct = get_pct_uses_latent_reasoning(data, metric_name)
            percentages.append(pct)

        bars = ax.bar(x, percentages, bar_width, color=color, alpha=0.8)

        # Add percentage labels on top of bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel("% Uses >=1 Latent Reasoning Token")
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Percent Use Any Latent Reasoning Tokens, Coconut", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "early_stopping_bar_pct_uses_latent", base_llm)
    plt.close()


def create_accuracy_by_iteration_plot(data, output_dir, base_llm=None, model_name=""):
    """
    Create bar chart showing accuracy vs iteration count.

    Args:
        data: Loaded results data with accuracy_by_iteration in summary
        output_dir: Directory to save plot
        base_llm: Base LLM name for filename
        model_name: Model name for title/filename
    """
    accuracy_data = data.get("summary", {}).get("accuracy_by_iteration")
    if not accuracy_data:
        print(f"No accuracy_by_iteration data found for {model_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = sorted([int(k) for k in accuracy_data.keys()])
    accuracies = [accuracy_data[str(k)]["accuracy"] * 100 for k in iterations]
    std_errors = [accuracy_data[str(k)]["std_error"] * 100 for k in iterations]
    counts = [accuracy_data[str(k)]["count"] for k in iterations]

    # Create bars
    bars = ax.bar(
        range(len(iterations)),
        accuracies,
        yerr=std_errors,
        capsize=5,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Color first bar (0 iterations) gray, rest blue
    for i, bar in enumerate(bars):
        if iterations[i] == 0:
            bar.set_color('#888888')  # Gray
        else:
            bar.set_color('#3498db')  # Blue

    # Add value labels on top of bars
    for i, (acc, n) in enumerate(zip(accuracies, counts)):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs Iterations: {model_name}", fontweight="bold")
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels(iterations)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    filename_base = f"accuracy_by_iteration_{model_name.replace(' ', '_').lower()}"
    save_plot(output_dir, filename_base, base_llm)
    plt.close()


def create_accuracy_by_iteration_multi(datasets, output_dir, base_llm=None, model_types=None):
    """
    Create combined accuracy vs iteration plot for multiple datasets.

    Args:
        datasets: Dict of {dataset_name: {model_type: filepath}}
        output_dir: Directory to save plot
        base_llm: Base LLM name for filename
        model_types: List of model types to include (e.g., ["multimode_codi", "multimode_coconut"])
    """
    if model_types is None:
        model_types = ["multimode_codi", "multimode_coconut"]

    # Collect all data
    all_data = {}
    for dataset_name, results_files in datasets.items():
        for model_type in model_types:
            if model_type in results_files:
                data = load_results(results_files[model_type])
                accuracy_data = data.get("summary", {}).get("accuracy_by_iteration")
                if accuracy_data:
                    key = f"{dataset_name} ({model_type})"
                    all_data[key] = accuracy_data

    if not all_data:
        print("No accuracy_by_iteration data found")
        return

    # Create subplots
    n_plots = len(all_data)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, (key, accuracy_data) in enumerate(all_data.items()):
        ax = axes[idx]

        iterations = sorted([int(k) for k in accuracy_data.keys()])
        accuracies = [accuracy_data[str(k)]["accuracy"] * 100 for k in iterations]
        std_errors = [accuracy_data[str(k)]["std_error"] * 100 for k in iterations]

        bars = ax.bar(
            range(len(iterations)),
            accuracies,
            yerr=std_errors,
            capsize=3,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Color first bar gray, rest blue
        for i, bar in enumerate(bars):
            if iterations[i] == 0:
                bar.set_color('#888888')
            else:
                bar.set_color('#3498db')

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(key, fontweight="bold")
        ax.set_xticks(range(len(iterations)))
        ax.set_xticklabels(iterations)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(len(all_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Ground Truth Accuracy by Iteration Count", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(output_dir, "accuracy_by_iteration_multi", base_llm)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot early stopping results for CoT, Coconut, and CODI models"
    )
    parser.add_argument("--results_dir", default="results/early_stopping",
                        help="Directory containing results JSON files")
    parser.add_argument("--base_llm", required=True,
                        help="Base LLM name (e.g., 'gpt2', 'Llama-3.2-1B-Instruct') for output filenames")

    # GSM8k dataset files
    parser.add_argument("--gsm_cot_file", help="GSM8k CoT results file")
    parser.add_argument("--gsm_coconut_file", help="GSM8k Coconut results file")
    parser.add_argument("--gsm_codi_file", help="GSM8k CODI results file")

    # ProsQA dataset files
    parser.add_argument("--prosqa_cot_file", help="ProsQA CoT results file")
    parser.add_argument("--prosqa_coconut_file", help="ProsQA Coconut results file")
    parser.add_argument("--prosqa_codi_file", help="ProsQA CODI results file")

    # ProntoQA dataset files
    parser.add_argument("--prontoqa_cot_file", help="ProntoQA CoT results file")
    parser.add_argument("--prontoqa_coconut_file", help="ProntoQA Coconut results file")
    parser.add_argument("--prontoqa_codi_file", help="ProntoQA CODI results file")

    # Multimode CODI dataset files
    parser.add_argument("--gsm_multimode_codi_file", help="GSM8k Multimode CODI results file")
    parser.add_argument("--prosqa_multimode_codi_file", help="ProsQA Multimode CODI results file")
    parser.add_argument("--prontoqa_multimode_codi_file", help="ProntoQA Multimode CODI results file")

    # Multimode Coconut dataset files
    parser.add_argument("--gsm_multimode_coconut_file", help="GSM8k Multimode Coconut results file")
    parser.add_argument("--prosqa_multimode_coconut_file", help="ProsQA Multimode Coconut results file")
    parser.add_argument("--prontoqa_multimode_coconut_file", help="ProntoQA Multimode Coconut results file")

    # Combined LLM plot - GPT-2 files
    parser.add_argument("--gpt2_gsm_cot_file", help="GPT-2 GSM8k CoT results file")
    parser.add_argument("--gpt2_gsm_coconut_file", help="GPT-2 GSM8k Coconut results file")
    parser.add_argument("--gpt2_gsm_codi_file", help="GPT-2 GSM8k CODI results file")
    parser.add_argument("--gpt2_prosqa_cot_file", help="GPT-2 ProsQA CoT results file")
    parser.add_argument("--gpt2_prosqa_coconut_file", help="GPT-2 ProsQA Coconut results file")
    parser.add_argument("--gpt2_prosqa_codi_file", help="GPT-2 ProsQA CODI results file")
    parser.add_argument("--gpt2_prontoqa_cot_file", help="GPT-2 ProntoQA CoT results file")
    parser.add_argument("--gpt2_prontoqa_coconut_file", help="GPT-2 ProntoQA Coconut results file")
    parser.add_argument("--gpt2_prontoqa_codi_file", help="GPT-2 ProntoQA CODI results file")

    # Combined LLM plot - Llama files
    parser.add_argument("--llama_gsm_cot_file", help="Llama GSM8k CoT results file")
    parser.add_argument("--llama_gsm_coconut_file", help="Llama GSM8k Coconut results file")
    parser.add_argument("--llama_gsm_codi_file", help="Llama GSM8k CODI results file")
    parser.add_argument("--llama_prosqa_cot_file", help="Llama ProsQA CoT results file")
    parser.add_argument("--llama_prosqa_coconut_file", help="Llama ProsQA Coconut results file")
    parser.add_argument("--llama_prosqa_codi_file", help="Llama ProsQA CODI results file")
    parser.add_argument("--llama_prontoqa_cot_file", help="Llama ProntoQA CoT results file")
    parser.add_argument("--llama_prontoqa_coconut_file", help="Llama ProntoQA Coconut results file")
    parser.add_argument("--llama_prontoqa_codi_file", help="Llama ProntoQA CODI results file")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load results organized by dataset
    datasets = {}

    # GSM8k results
    gsm_files = {}
    if args.gsm_cot_file:
        gsm_files["cot"] = results_dir / args.gsm_cot_file
    if args.gsm_coconut_file:
        gsm_files["coconut"] = results_dir / args.gsm_coconut_file
    if args.gsm_codi_file:
        gsm_files["codi"] = results_dir / args.gsm_codi_file
    if gsm_files:
        datasets["GSM8k"] = gsm_files

    # ProsQA results
    prosqa_files = {}
    if args.prosqa_cot_file:
        prosqa_files["cot"] = results_dir / args.prosqa_cot_file
    if args.prosqa_coconut_file:
        prosqa_files["coconut"] = results_dir / args.prosqa_coconut_file
    if args.prosqa_codi_file:
        prosqa_files["codi"] = results_dir / args.prosqa_codi_file
    if prosqa_files:
        datasets["ProsQA"] = prosqa_files

    # ProntoQA results
    prontoqa_files = {}
    if args.prontoqa_cot_file:
        prontoqa_files["cot"] = results_dir / args.prontoqa_cot_file
    if args.prontoqa_coconut_file:
        prontoqa_files["coconut"] = results_dir / args.prontoqa_coconut_file
    if args.prontoqa_codi_file:
        prontoqa_files["codi"] = results_dir / args.prontoqa_codi_file
    if args.prontoqa_multimode_codi_file:
        prontoqa_files["multimode_codi"] = results_dir / args.prontoqa_multimode_codi_file
    if args.prontoqa_multimode_coconut_file:
        prontoqa_files["multimode_coconut"] = results_dir / args.prontoqa_multimode_coconut_file
    if prontoqa_files:
        datasets["ProntoQA"] = prontoqa_files

    # Add multimode files to GSM8k
    if args.gsm_multimode_codi_file:
        if "GSM8k" not in datasets:
            datasets["GSM8k"] = {}
        datasets["GSM8k"]["multimode_codi"] = results_dir / args.gsm_multimode_codi_file
    if args.gsm_multimode_coconut_file:
        if "GSM8k" not in datasets:
            datasets["GSM8k"] = {}
        datasets["GSM8k"]["multimode_coconut"] = results_dir / args.gsm_multimode_coconut_file

    # Add multimode files to ProsQA
    if args.prosqa_multimode_codi_file:
        if "ProsQA" not in datasets:
            datasets["ProsQA"] = {}
        datasets["ProsQA"]["multimode_codi"] = results_dir / args.prosqa_multimode_codi_file
    if args.prosqa_multimode_coconut_file:
        if "ProsQA" not in datasets:
            datasets["ProsQA"] = {}
        datasets["ProsQA"]["multimode_coconut"] = results_dir / args.prosqa_multimode_coconut_file

    # Build combined LLM datasets if any combined arguments are provided
    gpt2_datasets = {}
    llama_datasets = {}

    # GPT-2 GSM8k
    gsm_gpt2 = {}
    if args.gpt2_gsm_cot_file:
        gsm_gpt2["cot"] = results_dir / args.gpt2_gsm_cot_file
    if args.gpt2_gsm_coconut_file:
        gsm_gpt2["coconut"] = results_dir / args.gpt2_gsm_coconut_file
    if args.gpt2_gsm_codi_file:
        gsm_gpt2["codi"] = results_dir / args.gpt2_gsm_codi_file
    if gsm_gpt2:
        gpt2_datasets["GSM8k"] = gsm_gpt2

    # GPT-2 ProsQA
    prosqa_gpt2 = {}
    if args.gpt2_prosqa_cot_file:
        prosqa_gpt2["cot"] = results_dir / args.gpt2_prosqa_cot_file
    if args.gpt2_prosqa_coconut_file:
        prosqa_gpt2["coconut"] = results_dir / args.gpt2_prosqa_coconut_file
    if args.gpt2_prosqa_codi_file:
        prosqa_gpt2["codi"] = results_dir / args.gpt2_prosqa_codi_file
    if prosqa_gpt2:
        gpt2_datasets["ProsQA"] = prosqa_gpt2

    # GPT-2 ProntoQA
    prontoqa_gpt2 = {}
    if args.gpt2_prontoqa_cot_file:
        prontoqa_gpt2["cot"] = results_dir / args.gpt2_prontoqa_cot_file
    if args.gpt2_prontoqa_coconut_file:
        prontoqa_gpt2["coconut"] = results_dir / args.gpt2_prontoqa_coconut_file
    if args.gpt2_prontoqa_codi_file:
        prontoqa_gpt2["codi"] = results_dir / args.gpt2_prontoqa_codi_file
    if prontoqa_gpt2:
        gpt2_datasets["ProntoQA"] = prontoqa_gpt2

    # Llama GSM8k
    gsm_llama = {}
    if args.llama_gsm_cot_file:
        gsm_llama["cot"] = results_dir / args.llama_gsm_cot_file
    if args.llama_gsm_coconut_file:
        gsm_llama["coconut"] = results_dir / args.llama_gsm_coconut_file
    if args.llama_gsm_codi_file:
        gsm_llama["codi"] = results_dir / args.llama_gsm_codi_file
    if gsm_llama:
        llama_datasets["GSM8k"] = gsm_llama

    # Llama ProsQA
    prosqa_llama = {}
    if args.llama_prosqa_cot_file:
        prosqa_llama["cot"] = results_dir / args.llama_prosqa_cot_file
    if args.llama_prosqa_coconut_file:
        prosqa_llama["coconut"] = results_dir / args.llama_prosqa_coconut_file
    if args.llama_prosqa_codi_file:
        prosqa_llama["codi"] = results_dir / args.llama_prosqa_codi_file
    if prosqa_llama:
        llama_datasets["ProsQA"] = prosqa_llama

    # Llama ProntoQA
    prontoqa_llama = {}
    if args.llama_prontoqa_cot_file:
        prontoqa_llama["cot"] = results_dir / args.llama_prontoqa_cot_file
    if args.llama_prontoqa_coconut_file:
        prontoqa_llama["coconut"] = results_dir / args.llama_prontoqa_coconut_file
    if args.llama_prontoqa_codi_file:
        prontoqa_llama["codi"] = results_dir / args.llama_prontoqa_codi_file
    if prontoqa_llama:
        llama_datasets["ProntoQA"] = prontoqa_llama

    # Check if we have combined LLM data (even if no regular datasets)
    has_combined_data = gpt2_datasets and llama_datasets

    if not datasets and not has_combined_data:
        print("No results files specified!")
        print("\nUsage:")
        print("  python -m experiments.early_stopping.plot \\")
        print("    --base_llm <base_llm_name> \\")
        print("    --gsm_cot_file <gsm_cot_results.json> \\")
        print("    --gsm_coconut_file <gsm_coconut_results.json> \\")
        print("    --gsm_codi_file <gsm_codi_results.json> \\")
        print("    --prosqa_cot_file <prosqa_cot_results.json> \\")
        print("    --prosqa_coconut_file <prosqa_coconut_results.json> \\")
        print("    --prosqa_codi_file <prosqa_codi_results.json> \\")
        print("    --prontoqa_cot_file <prontoqa_cot_results.json> \\")
        print("    --prontoqa_coconut_file <prontoqa_coconut_results.json> \\")
        print("    --prontoqa_codi_file <prontoqa_codi_results.json>")
        return

    base_llm = args.base_llm

    # Generate combined plot if both GPT-2 and Llama datasets are provided
    if has_combined_data:
        print("Generating combined GPT-2 + Llama stacked bar chart...")
        create_bar_chart_force_stop_stacked_combined(gpt2_datasets, llama_datasets, results_dir)

    # Skip regular plots if no regular datasets
    if not datasets:
        print("\nDone!")
        return

    # Print summary
    print_summary_table_multi_dataset(datasets)

    # Generate plots
    print("\nGenerating plots...")
    create_cdf_plot_by_token_multi(datasets, results_dir, base_llm)
    create_cdf_plot_by_step_multi(datasets, results_dir, base_llm)

    # New separated plots
    create_bar_chart_force_stop_by_token_multi(datasets, results_dir, base_llm)
    create_bar_chart_force_stop_by_step_multi(datasets, results_dir, base_llm)
    create_bar_chart_answer_coverage_multi(datasets, results_dir, base_llm)
    create_bar_chart_rank_stability_multi(datasets, results_dir, base_llm)

    # Keep old combined plots for backward compatibility
    create_bar_chart_by_token_multi(datasets, results_dir, base_llm)
    create_bar_chart_by_step_multi(datasets, results_dir, base_llm)

    # Create accuracy by iteration plots for multimode models
    has_multimode = any(
        "multimode_codi" in files or "multimode_coconut" in files
        for files in datasets.values()
    )
    if has_multimode:
        print("Generating accuracy by iteration plots...")
        create_accuracy_by_iteration_multi(datasets, results_dir, base_llm)

        # Also create individual plots for each multimode file
        for dataset_name, results_files in datasets.items():
            for model_type in ["multimode_codi", "multimode_coconut"]:
                if model_type in results_files:
                    data = load_results(results_files[model_type])
                    model_name = f"{dataset_name}_{model_type}"
                    create_accuracy_by_iteration_plot(data, results_dir, base_llm, model_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
