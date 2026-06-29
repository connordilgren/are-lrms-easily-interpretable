"""Stacked bar chart: correct + incorrect breakdown across 4 metric groups.

Each bar totals 100% of all samples, showing correct/incorrect x found/QT-lift/not-found.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

def get_subdirs(base_llm, dataset_name='gsm_valid-gold-reasoning-trace_test'):
    """Generate subdirectory name patterns based on base_llm and dataset_name."""
    return {
        ("coconut", "no_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens",
        ("coconut", "yes_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens",
        ("codi", "no_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens",
        ("codi", "yes_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens",
    }


# Default subdirectory name patterns (for backwards compatibility)
DEFAULT_DATASET_NAME = 'gsm_valid-gold-reasoning-trace_test'
SUBDIRS = get_subdirs('gpt2', DEFAULT_DATASET_NAME)

METRIC_LABELS = [
    "Primary\nSolution",
    "Any Gold Label\nSolution",
    "One Random\nSolution",
    "Best of Five\nRandom Solutions",
]

# Colors for 6-segment bars (correct/incorrect split)
COLOR_CORRECT_FOUND = "#1a7a2e"       # dark green
COLOR_CORRECT_QT_LIFT = "#7dcea0"     # light green
COLOR_CORRECT_NOT_FOUND = "#f0c040"   # yellow-amber

COLOR_INCORRECT_FOUND = "#a02020"     # dark red
COLOR_INCORRECT_QT_LIFT = "#e88888"   # light red
COLOR_INCORRECT_NOT_FOUND = "#999999" # gray

# Colors for 3-segment bars (baseline, no correct/incorrect split)
COLOR_BASELINE_FOUND = "#2060b0"      # blue
COLOR_BASELINE_QT_LIFT = "#7eb8e0"    # light blue
COLOR_BASELINE_NOT_FOUND = "#999999"  # gray


def load_json_data(output_dir, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Load 4 results.json files and return per_sample lists keyed by (model, qt)."""
    subdirs = get_subdirs(base_llm, dataset_name)
    data = {}
    for key, subdir in subdirs.items():
        path = os.path.join(output_dir, subdir, "results.json")
        with open(path) as f:
            raw = json.load(f)
        data[key] = raw["per_sample"]
    return data


def _match_samples(no_qt_samples, yes_qt_samples):
    """Match samples across no-QT and yes-QT runs by sample_idx.

    Returns list of (no_qt_sample, yes_qt_sample) tuples for matched indices.
    """
    yes_qt_by_idx = {s["sample_idx"]: s for s in yes_qt_samples}
    matched = []
    for s in no_qt_samples:
        idx = s["sample_idx"]
        if idx in yes_qt_by_idx:
            matched.append((s, yes_qt_by_idx[idx]))
    return matched


def compute_6_segments(no_qt_samples, yes_qt_samples, field):
    """Compute 6-segment percentages for a field with correct/incorrect split.

    Segments:
        correct_found:        answer_correct=T and field=T in no-QT
        correct_qt_lift:      answer_correct=T and field=F in no-QT but T in yes-QT
        correct_not_found:    answer_correct=T and field=F in yes-QT
        incorrect_found:      answer_correct=F and field=T in no-QT
        incorrect_qt_lift:    answer_correct=F and field=F in no-QT but T in yes-QT
        incorrect_not_found:  answer_correct=F and field=F in yes-QT
    """
    matched = _match_samples(no_qt_samples, yes_qt_samples)
    n = len(matched)

    counts = {
        "correct_found": 0,
        "correct_qt_lift": 0,
        "correct_not_found": 0,
        "incorrect_found": 0,
        "incorrect_qt_lift": 0,
        "incorrect_not_found": 0,
    }

    for no_qt, yes_qt in matched:
        correct = no_qt["answer_correct"]
        found_no_qt = no_qt[field]
        found_yes_qt = yes_qt[field]

        prefix = "correct" if correct else "incorrect"
        if found_no_qt:
            counts[f"{prefix}_found"] += 1
        elif found_yes_qt:
            counts[f"{prefix}_qt_lift"] += 1
        else:
            counts[f"{prefix}_not_found"] += 1

    return {k: v / n * 100 for k, v in counts.items()}


def compute_3_segments(no_qt_samples, yes_qt_samples, field):
    """Compute 3-segment percentages for a baseline field (no correct/incorrect split).

    Baseline fields are only populated for correct samples, so incorrect samples
    always count as not found.

    Segments:
        found:      field=T in no-QT
        qt_lift:    field=F in no-QT but T in yes-QT
        not_found:  field=F in yes-QT
    """
    matched = _match_samples(no_qt_samples, yes_qt_samples)
    n = len(matched)

    counts = {"found": 0, "qt_lift": 0, "not_found": 0}

    for no_qt, yes_qt in matched:
        found_no_qt = no_qt["baseline"][field]
        found_yes_qt = yes_qt["baseline"][field]

        if found_no_qt:
            counts["found"] += 1
        elif found_yes_qt:
            counts["qt_lift"] += 1
        else:
            counts["not_found"] += 1

    return {k: v / n * 100 for k, v in counts.items()}


def plot_stacked_bar(output_dir, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Build and save the stacked bar chart."""
    data = load_json_data(output_dir, base_llm, dataset_name)

    # Metric configs: (label, field, is_baseline, baseline_field_name)
    metric_configs = [
        ("Primary\nSolution", "primary_found", False, None),
        ("Any Gold Label\nSolution", "any_gt_found", False, None),
        ("One Random\nSolution", None, True, "baseline_1_found"),
        ("Best of Five\nRandom Solutions", None, True, "baseline_5_found"),
    ]

    models = ["coconut", "codi"]
    model_labels = {"coconut": "Coconut", "codi": "CODI"}

    # Compute all segments
    all_segments = {}  # (metric_idx, model) -> segment dict
    for mi, (_, field, is_baseline, baseline_field) in enumerate(metric_configs):
        for model in models:
            no_qt = data[(model, "no_qt")]
            yes_qt = data[(model, "yes_qt")]
            if is_baseline:
                all_segments[(mi, model)] = compute_3_segments(no_qt, yes_qt, baseline_field)
            else:
                all_segments[(mi, model)] = compute_6_segments(no_qt, yes_qt, field)

    # Print verification totals
    print("Verification — bar totals:")
    for mi, (label, _, _, _) in enumerate(metric_configs):
        for model in models:
            segs = all_segments[(mi, model)]
            total = sum(segs.values())
            label_clean = label.replace("\n", " ")
            print(f"  {label_clean} / {model_labels[model]}: {total:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    n_groups = len(metric_configs)
    n_models = len(models)
    bar_width = 0.3
    group_positions = np.arange(n_groups)

    for model_i, model in enumerate(models):
        offset = (model_i - 0.5) * bar_width
        for mi in range(n_groups):
            segs = all_segments[(mi, model)]
            x = group_positions[mi] + offset
            is_baseline = metric_configs[mi][2]

            if is_baseline:
                # 3-segment bar
                segment_order = ["found", "qt_lift", "not_found"]
                colors = [COLOR_BASELINE_FOUND, COLOR_BASELINE_QT_LIFT, COLOR_BASELINE_NOT_FOUND]
            else:
                # 6-segment bar
                segment_order = [
                    "correct_found", "correct_qt_lift", "correct_not_found",
                    "incorrect_found", "incorrect_qt_lift", "incorrect_not_found",
                ]
                colors = [
                    COLOR_CORRECT_FOUND, COLOR_CORRECT_QT_LIFT, COLOR_CORRECT_NOT_FOUND,
                    COLOR_INCORRECT_FOUND, COLOR_INCORRECT_QT_LIFT, COLOR_INCORRECT_NOT_FOUND,
                ]

            bottom = 0
            for seg_name, color in zip(segment_order, colors):
                val = segs[seg_name]
                ax.bar(x, val, bar_width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                # Value label (skip < 1%)
                if val >= 1.0:
                    ax.text(x, bottom + val / 2, f"{val:.1f}%",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold", color="white")
                bottom += val

    # X-axis: model labels on the tick marks, group labels above
    tick_positions = []
    tick_labels = []
    for mi in range(n_groups):
        for model_i, model in enumerate(models):
            offset = (model_i - 0.5) * bar_width
            tick_positions.append(group_positions[mi] + offset)
            tick_labels.append(model_labels[model])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    # Group labels centered above each pair of bars
    for mi, mc in enumerate(metric_configs):
        ax.text(group_positions[mi], 105, mc[0], ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Y-axis
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percent of All Samples", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    ax.set_title(
        "Gold Label Representation in Vocabulary Projections\n"
        "Correct + Incorrect Samples",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Build legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CORRECT_FOUND, edgecolor="white",
                       label="Correct + Found excl. QT"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CORRECT_QT_LIFT, edgecolor="white",
                       label="Correct + Found incl. QT (lift)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CORRECT_NOT_FOUND, edgecolor="white",
                       label="Correct + Not found"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_INCORRECT_FOUND, edgecolor="white",
                       label="Incorrect + Found excl. QT"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_INCORRECT_QT_LIFT, edgecolor="white",
                       label="Incorrect + Found incl. QT (lift)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_INCORRECT_NOT_FOUND, edgecolor="white",
                       label="Incorrect / Baseline + Not found"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_BASELINE_FOUND, edgecolor="white",
                       label="Baseline Found excl. QT"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_BASELINE_QT_LIFT, edgecolor="white",
                       label="Baseline Found incl. QT (lift)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=4, framealpha=0.9)

    fig.subplots_adjust(bottom=0.22)
    path = os.path.join(output_dir, f"back_tracking_vp_summary_correct-and-incorrect_stacked_bar_{base_llm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar chart: correct + incorrect breakdown"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/back_tracking_vp",
        help="Directory containing result subdirectories",
    )
    parser.add_argument(
        "--base_llm", type=str, default="gpt2",
        help="Base LLM short name for directory matching",
    )
    parser.add_argument(
        "--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
        help="Dataset name stem for subdirectory matching",
    )
    args = parser.parse_args()
    plot_stacked_bar(args.output_dir, args.base_llm, args.dataset_name)


if __name__ == "__main__":
    main()
