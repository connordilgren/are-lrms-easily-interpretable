"""Plot GT representation analysis results comparing Coconut vs CODI models."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global font size settings for higher resolution plots
# (scaled up ~2x to compensate for larger figure sizes like 24x10)
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
})


def get_subdirs(base_llm, dataset_name='gsm_valid-gold-reasoning-trace_test'):
    """Generate subdirectory name patterns based on base_llm and dataset_name."""
    return {
        ("coconut", "no_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens",
        ("coconut", "yes_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens",
        ("codi", "no_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens",
        ("codi", "yes_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens",
    }


def get_subdirs_incorrect(base_llm, dataset_name='gsm_valid-gold-reasoning-trace_test'):
    """Generate incorrect subdirectory name patterns based on base_llm and dataset_name."""
    return {
        ("coconut", "no_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens_incorrect_predictions",
        ("coconut", "yes_qt"): f"coconut_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens_incorrect_predictions",
        ("codi", "no_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_no-question-tokens_incorrect_predictions",
        ("codi", "yes_qt"): f"codi_{base_llm}_{dataset_name}_k10_no-baseline-require-answer_yes-question-tokens_incorrect_predictions",
    }


# Default subdirectory name patterns (for backwards compatibility)
DEFAULT_DATASET_NAME = 'gsm_valid-gold-reasoning-trace_test'
SUBDIRS = get_subdirs('gpt2', DEFAULT_DATASET_NAME)

METRICS = ["Primary Found %", "Any GT Found %", "Base-1 %", "Base-5 %"]
METRIC_LABELS = [
    "Primary\nSolution",
    "Any Gold\nReasoning Trace",
    "One Random\nSolution",
    "Best of Five\nRandom Solutions",
]

COLOR_COCONUT = "#2ecc71"  # Green (matches early_stopping plots)
COLOR_CODI = "#e74c3c"     # Red (matches early_stopping plots)

# Multi-LLM colors (for 1x2 combined plot with all 4 model+LLM combinations)
COLOR_GPT2_COCONUT = "#2ecc71"   # Green
COLOR_GPT2_CODI = "#e74c3c"      # Red
COLOR_LLAMA_COCONUT = "#3498db"  # Blue
COLOR_LLAMA_CODI = "#f39c12"     # Orange

LLM_DISPLAY_NAMES = {
    'gpt2': 'GPT-2',
    'llama32-1b': 'Llama 3.2-1B',
}


def load_data(output_dir, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Load all 4 summary CSVs into a dict keyed by (model, condition)."""
    subdirs = get_subdirs(base_llm, dataset_name)
    data = {}
    for key, subdir in subdirs.items():
        path = os.path.join(output_dir, subdir, "summary.csv")
        df = pd.read_csv(path)
        data[key] = df
    return data


def get_all_row(df):
    """Extract the 'All' summary row from a dataframe."""
    return df[df["Steps"] == "All"].iloc[0]


def _plot_overall_single(data, output_dir, cond, title, filename):
    """Plot a single overall summary bar chart for one condition."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(METRICS))
    bar_width = 0.25

    coconut_row = get_all_row(data[("coconut", cond)])
    codi_row = get_all_row(data[("codi", cond)])

    coconut_vals = [coconut_row[m] for m in METRICS]
    codi_vals = [codi_row[m] for m in METRICS]

    bars1 = ax.bar(x - bar_width / 2, coconut_vals, bar_width,
                   label=f"Coconut (n={int(coconut_row['Correct'])})",
                   color=COLOR_COCONUT, alpha=0.8)
    bars2 = ax.bar(x + bar_width / 2, codi_vals, bar_width,
                   label=f"CODI (n={int(codi_row['Correct'])})",
                   color=COLOR_CODI, alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of Samples where\nGold Label Solution Found")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_overall_summary(data, output_dir, base_llm='gpt2'):
    """Figure 1a/1b: Separate bar charts for each condition."""
    _plot_overall_single(data, output_dir, "no_qt",
                         "Gold Label Representation\nWithout Question Tokens",
                         f"overall_summary_without_question_tokens_{base_llm}.png")
    _plot_overall_single(data, output_dir, "yes_qt",
                         "Gold Label Representation\nWith Question Tokens",
                         f"overall_summary_with_question_tokens_{base_llm}.png")


def compute_baseline_in_topk(output_dir, base_llm='gpt2', model_id=None, dataset_name=DEFAULT_DATASET_NAME):
    """Compute baseline answer-in-topk percentages from results.json for correct samples.

    For each correct sample with baselines, checks whether the baseline's final
    answer token appears anywhere in the stored top-k vocab projections.

    Returns:
        dict: model -> {'base_1_in_topk_pct': float, 'base_5_in_topk_pct': float}
    """
    from transformers import AutoTokenizer
    from experiments.back_tracking_vp.analyze_gt_representation import parse_solution
    from experiments.back_tracking_vp.summarize_incorrect_gt_representation import _answer_in_topk

    tokenizer_id = model_id if model_id else base_llm
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    subdirs = get_subdirs(base_llm, dataset_name)
    result = {}
    for model in ['coconut', 'codi']:
        subdir = subdirs[(model, 'no_qt')]
        results_path = os.path.join(output_dir, subdir, 'results.json')

        print(f"  Loading {results_path}...")
        with open(results_path, 'r') as f:
            data = json.load(f)

        correct_with_baselines = 0
        base_1_in_topk_count = 0
        base_5_in_topk_count = 0

        for sample in data['per_sample']:
            if not sample['answer_correct']:
                continue
            baselines_data = sample.get('baseline', {}).get('baselines', [])
            if not baselines_data:
                continue

            correct_with_baselines += 1
            vp = sample.get('vocab_projection_top_k', [])

            sample_1_topk = False
            sample_5_topk = False
            for i, baseline in enumerate(baselines_data):
                steps = parse_solution(baseline['solution'])
                if not steps:
                    continue
                if _answer_in_topk(steps, vp, tokenizer):
                    if i == 0:
                        sample_1_topk = True
                    sample_5_topk = True

            if sample_1_topk:
                base_1_in_topk_count += 1
            if sample_5_topk:
                base_5_in_topk_count += 1

        if correct_with_baselines > 0:
            b1_pct = base_1_in_topk_count / correct_with_baselines * 100
            b5_pct = base_5_in_topk_count / correct_with_baselines * 100
        else:
            b1_pct = b5_pct = 0.0

        result[model] = {'base_1_in_topk_pct': b1_pct, 'base_5_in_topk_pct': b5_pct}
        print(f"  {model}: n={correct_with_baselines}, "
              f"Base-1 in topk={b1_pct:.1f}%, Base-5 in topk={b5_pct:.1f}%")

    return result


def plot_overall_stacked(data, output_dir, base_llm='gpt2'):
    """Combined stacked bar chart: base = no QT, stacked = QT lift."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(METRICS))
    bar_width = 0.3

    for model, color, offset_sign, label in [
        ("coconut", COLOR_COCONUT, -1, "Coconut"),
        ("codi", COLOR_CODI, +1, "CODI"),
    ]:
        no_qt_row = get_all_row(data[(model, "no_qt")])
        yes_qt_row = get_all_row(data[(model, "yes_qt")])

        base_vals = [no_qt_row[m] for m in METRICS]
        lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
        incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]

        offset = offset_sign * bar_width / 2

        # Layer 1: excluding question tokens (solid fill)
        ax.bar(x + offset, base_vals, bar_width,
               label=f"{label} — excluding question tokens",
               color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

        # Layer 2: including question tokens (hatched)
        ax.bar(x + offset, lift_vals, bar_width,
               bottom=base_vals,
               label=f"{label} — including question tokens",
               color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

        # Labels for GT metrics (base value + total)
        for i in range(2):
            bar_x = x[i] + offset
            ax.text(bar_x, base_vals[i] + 1,
                    f"{base_vals[i]:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        # Total (incl-QT) labels for all metrics
        for i in range(len(METRICS)):
            bar_x = x[i] + offset
            ax.text(bar_x, incl_qt[i] + 1,
                    f"{incl_qt[i]:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    # No title — kept clean for paper figures
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of Correct Samples where\nGold Label Solution Found")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=2)

    fig.tight_layout()
    path = os.path.join(output_dir, f"back_tracking_vp_overall_summary_correct_samples_{base_llm}.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
    pdf_path = path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def _plot_per_step_single(data, output_dir, cond, title, filename):
    """Plot a single per-step line chart for one condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = list(range(1, 6))  # 1-5 only

    lines_config = [
        ("coconut", "Primary Found %", COLOR_COCONUT, "-", "o", "Coconut — Primary"),
        ("coconut", "Any GT Found %", COLOR_COCONUT, "--", "s", "Coconut — Any Gold Label"),
        ("codi", "Primary Found %", COLOR_CODI, "-", "o", "CODI — Primary"),
        ("codi", "Any GT Found %", COLOR_CODI, "--", "s", "CODI — Any Gold Label"),
    ]

    for model, metric, color, ls, marker, label in lines_config:
        df = data[(model, cond)]
        step_df = df[df["Steps"].astype(str).str.isdigit()]
        step_df = step_df[step_df["Steps"].astype(int).isin(steps)]
        vals = [step_df[step_df["Steps"].astype(int) == s][metric].values[0] for s in steps]
        ax.plot(steps, vals, color=color, linestyle=ls, marker=marker,
                markersize=6, linewidth=2, label=label)

    # Sample count annotations along x-axis
    coconut_df = data[("coconut", cond)]
    step_df = coconut_df[coconut_df["Steps"].astype(str).str.isdigit()]
    sample_counts = []
    correct_coconut = []
    correct_codi = []
    for s in steps:
        row = step_df[step_df["Steps"].astype(int) == s].iloc[0]
        sample_counts.append(int(row["Samples"]))
        correct_coconut.append(int(row["Correct"]))
        codi_df = data[("codi", cond)]
        codi_step = codi_df[codi_df["Steps"].astype(str).str.isdigit()]
        correct_codi.append(int(codi_step[codi_step["Steps"].astype(int) == s].iloc[0]["Correct"]))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"n={sample_counts[i]}\n({correct_coconut[i]}/{correct_codi[i]})"
                         for i in range(len(steps))])
    ax2.set_xlabel("Samples (Coconut correct / CODI correct)")

    ax.set_xlabel("Number of Solution Steps")
    ax.set_ylabel("Percent of Samples where\nGold Label Solution Found")
    ax.set_xticks(steps)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(title, fontweight="bold", pad=35)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_per_step(data, output_dir, base_llm='gpt2'):
    """Figure 2a/2b: Separate per-step line charts for each condition."""
    _plot_per_step_single(data, output_dir, "no_qt",
                          "Per-Step Gold Label Representation\nWithout Question Tokens",
                          f"per_step_without_question_tokens_{base_llm}.png")
    _plot_per_step_single(data, output_dir, "yes_qt",
                          "Per-Step Gold Label Representation\nWith Question Tokens",
                          f"per_step_with_question_tokens_{base_llm}.png")


def _get_step_vals(data, model, cond, metric, steps):
    """Extract per-step metric values."""
    df = data[(model, cond)]
    step_df = df[df["Steps"].astype(str).str.isdigit()]
    step_df = step_df[step_df["Steps"].astype(int).isin(steps)]
    return [step_df[step_df["Steps"].astype(int) == s][metric].values[0] for s in steps]


def plot_back_tracking_vp_per_step_combined_lines(data, output_dir, base_llm='gpt2'):
    """Per-step line charts, one plot per metric, with both QT conditions."""
    # Use smaller fonts for this smaller figure (12x7 vs 24x10)
    with plt.rc_context({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}):
        _plot_back_tracking_vp_per_step_combined_lines_impl(data, output_dir, base_llm)

def _plot_back_tracking_vp_per_step_combined_lines_impl(data, output_dir, base_llm='gpt2'):
    """Implementation of per-step line charts."""
    steps = list(range(1, 6))

    metrics = [
        ("Primary Found %", "Primary Solution",
         f"back_tracking_vp_per_step_primary_{base_llm}.png"),
        ("Any GT Found %", "Any Gold Reasoning Trace",
         f"back_tracking_vp_per_step_any_gold_label_{base_llm}.png"),
    ]

    # Get sample counts per step for each model
    def _get_step_counts(model):
        df = data[(model, "no_qt")]
        step_df = df[df["Steps"].astype(str).str.isdigit()]
        step_df = step_df[step_df["Steps"].astype(int).isin(steps)]
        return [int(step_df[step_df["Steps"].astype(int) == s].iloc[0]["Correct"])
                for s in steps]

    coconut_n = _get_step_counts("coconut")
    codi_n = _get_step_counts("codi")

    for metric, metric_label, filename in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))

        all_series = {}
        for model, color, label in [
            ("coconut", COLOR_COCONUT, "Coconut"),
            ("codi", COLOR_CODI, "CODI"),
        ]:
            base_vals = _get_step_vals(data, model, "no_qt", metric, steps)
            yes_vals = _get_step_vals(data, model, "yes_qt", metric, steps)

            ax.plot(steps, base_vals, color=color, linestyle="-", marker="o",
                    markersize=8, linewidth=2.5,
                    label=f"{label} — excl. question tokens")
            ax.plot(steps, yes_vals, color=color, linestyle="--", marker="s",
                    markersize=8, linewidth=2.5, dashes=(6, 3),
                    label=f"{label} — incl. question tokens")

            all_series[(model, "no_qt")] = base_vals
            all_series[(model, "yes_qt")] = yes_vals

        # Add value labels with manual placement per series/step
        # Placement: (dx, dy, ha, va) for each step index 0-4
        above = (0, 4, "center", "bottom")
        below = (0, -4, "center", "top")
        right = (0.12, 3, "left", "center")

        right_low = (0.07, 0, "left", "center")
        right_up = (0.07, 1.5, "left", "center")
        right_down = (0.07, -1.5, "left", "center")

        placements = {
            ("coconut", "yes_qt"): [above, above, above, above, above],
            ("coconut", "no_qt"):  [below, below, below, above, right_up],
            ("codi", "yes_qt"):    [above, above, above, below, right_low],
            ("codi", "no_qt"):     [above, above, above, above, right_down],
        }

        for key, offsets in placements.items():
            vals = all_series[key]
            color = COLOR_COCONUT if key[0] == "coconut" else COLOR_CODI
            for i, s in enumerate(steps):
                dx, dy, ha, va = offsets[i]
                ax.text(s + dx, vals[i] + dy, f"{vals[i]:.0f}",
                        ha=ha, va=va, fontweight="bold", color=color)

        ax.set_xlabel("Number of Solution Steps\n(Coconut samples / CODI samples)")
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s}\n({coconut_n[i]} / {codi_n[i]})"
                            for i, s in enumerate(steps)])
        ax.set_ylim(0, 115)
        ax.set_ylabel("Percent of Correct Samples where\nGold Reasoning Trace Found")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        fig.tight_layout()
        basename = filename.replace(".png", "")
        for ext in [".png", ".pdf"]:
            path = os.path.join(output_dir, basename + ext)
            fig.savefig(path, dpi=600, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


def plot_per_step_any_gold_multi_llm_1x2(output_dir, base_llms=['gpt2', 'llama32-1b'], dataset_name=DEFAULT_DATASET_NAME):
    """1x2 subplot: one panel per LLM, showing 'Any GT Found %' per step."""
    # Load data for all LLMs
    all_data = {}
    for llm in base_llms:
        all_data[llm] = load_data(output_dir, llm, dataset_name)

    # Use smaller fonts for this figure
    with plt.rc_context({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

        steps = list(range(1, 6))
        metric = "Any GT Found %"

        for ax_idx, llm in enumerate(base_llms):
            ax = axes[ax_idx]
            data = all_data[llm]
            llm_name = LLM_DISPLAY_NAMES.get(llm, llm)

            # Get sample counts per step for each model
            def _get_step_counts(model):
                df = data[(model, "no_qt")]
                step_df = df[df["Steps"].astype(str).str.isdigit()]
                step_df = step_df[step_df["Steps"].astype(int).isin(steps)]
                return [int(step_df[step_df["Steps"].astype(int) == s].iloc[0]["Correct"])
                        for s in steps]

            coconut_n = _get_step_counts("coconut")
            codi_n = _get_step_counts("codi")

            for model, color, label in [
                ("coconut", COLOR_COCONUT, "Coconut"),
                ("codi", COLOR_CODI, "CODI"),
            ]:
                base_vals = _get_step_vals(data, model, "no_qt", metric, steps)
                yes_vals = _get_step_vals(data, model, "yes_qt", metric, steps)

                ax.plot(steps, base_vals, color=color, linestyle="-", marker="o",
                        markersize=8, linewidth=2.5,
                        label=f"{label} — excl. question tokens")
                ax.plot(steps, yes_vals, color=color, linestyle="--", marker="s",
                        markersize=8, linewidth=2.5, dashes=(6, 3),
                        label=f"{label} — incl. question tokens")

            ax.set_xlabel("Number of Solution Steps\n(Coconut samples / CODI samples)")
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{s}\n({coconut_n[i]} / {codi_n[i]})"
                                for i, s in enumerate(steps)])
            ax.set_ylim(0, 115)
            ax.set_title(llm_name, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend()

        axes[0].set_ylabel("Percent of Correct Samples where\nGold Reasoning Trace Found")

        fig.tight_layout()
        base = os.path.join(output_dir, "back_tracking_vp_per_step_any_gold_label_multi_llm_1x2")
        for ext in [".png", ".pdf"]:
            path = base + ext
            fig.savefig(path, dpi=600, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


def plot_per_step_any_gold_multi_llm_combined(output_dir, base_llms=['gpt2', 'llama32-1b'], dataset_name=DEFAULT_DATASET_NAME):
    """Single plot with all 4 model+LLM combinations for 'Any GT Found %' per step."""
    # Load data for all LLMs
    all_data = {}
    for llm in base_llms:
        all_data[llm] = load_data(output_dir, llm, dataset_name)

    # Use smaller fonts for this figure
    with plt.rc_context({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12}):
        fig, ax = plt.subplots(figsize=(14, 8))

        steps = list(range(1, 6))
        metric = "Any GT Found %"

        # Define the 4 model+LLM combinations with distinct colors (matching 1x2 bar plot)
        combinations = [
            ('gpt2', 'coconut', COLOR_GPT2_COCONUT, 'GPT-2 Coconut'),
            ('gpt2', 'codi', COLOR_GPT2_CODI, 'GPT-2 CODI'),
            ('llama32-1b', 'coconut', COLOR_LLAMA_COCONUT, 'Llama 3.2-1B Coconut'),
            ('llama32-1b', 'codi', COLOR_LLAMA_CODI, 'Llama 3.2-1B CODI'),
        ]

        for llm, model, color, label in combinations:
            if llm not in all_data:
                continue
            data = all_data[llm]

            base_vals = _get_step_vals(data, model, "no_qt", metric, steps)
            yes_vals = _get_step_vals(data, model, "yes_qt", metric, steps)

            ax.plot(steps, base_vals, color=color, linestyle="-", marker="o",
                    markersize=8, linewidth=2.5,
                    label=f"{label} — excl. QT")
            ax.plot(steps, yes_vals, color=color, linestyle="--", marker="s",
                    markersize=8, linewidth=2.5, dashes=(6, 3),
                    label=f"{label} — incl. QT")

        ax.set_xlabel("Number of Solution Steps")
        ax.set_xticks(steps)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Percent of Correct Samples where\nGold Reasoning Trace Found")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", ncol=2)

        fig.tight_layout()
        base = os.path.join(output_dir, "back_tracking_vp_per_step_any_gold_label_multi_llm_combined")
        for ext in [".png", ".pdf"]:
            path = base + ext
            fig.savefig(path, dpi=600, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


def plot_back_tracking_vp_per_step_combined_bars(data, output_dir, base_llm='gpt2'):
    """Combined per-step stacked bar chart, one panel per metric."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    steps = list(range(1, 6))
    x = np.arange(len(steps))
    bar_width = 0.3

    metrics = [("Primary Found %", "Primary Solution"),
               ("Any GT Found %", "Any Gold Label Solution")]

    for ax, (metric, metric_label) in zip(axes, metrics):
        for model, color, offset_sign, label in [
            ("coconut", COLOR_COCONUT, -1, "Coconut"),
            ("codi", COLOR_CODI, +1, "CODI"),
        ]:
            base_vals = _get_step_vals(data, model, "no_qt", metric, steps)
            yes_vals = _get_step_vals(data, model, "yes_qt", metric, steps)
            lift_vals = [y - b for y, b in zip(yes_vals, base_vals)]

            offset = offset_sign * bar_width / 2

            # Base bars (excl QT) — solid
            ax.bar(x + offset, base_vals, bar_width,
                   label=f"{label} — excluding question tokens",
                   color=color, alpha=0.8,
                   edgecolor="white", linewidth=0.5)

            # Base value labels just above solid bars
            for i in range(len(steps)):
                ax.text(x[i] + offset, base_vals[i] + 1,
                        f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

            # Lift bars (incl QT) — hatched, stacked
            bars_lift = ax.bar(x + offset, lift_vals, bar_width,
                               bottom=base_vals,
                               label=f"{label} — including question tokens",
                               color=color, alpha=0.4,
                               hatch="///", edgecolor=color, linewidth=0.5)

            # Total value label on top
            for i, bar in enumerate(bars_lift):
                total = base_vals[i] + lift_vals[i]
                ax.text(bar.get_x() + bar.get_width() / 2, total + 1,
                        f"{total:.0f}%", ha="center", va="bottom", fontweight="bold")

        ax.set_xlabel("Number of Solution Steps")
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.set_ylim(0, 115)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(metric_label, fontweight="bold")
        ax.legend()

    axes[0].set_ylabel("Percent of Correct Samples where\nGold Label Solution Found", fontsize=11)

    fig.suptitle("Per-Step Gold Label Representation in\nVocabulary Projections of Latent Reasoning Tokens",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"back_tracking_vp_per_step_combined_bars_{base_llm}.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_question_token_lift(data, output_dir, base_llm='gpt2'):
    """Figure 3: Bar chart showing improvement from adding question tokens."""
    fig, ax = plt.subplots(figsize=(8, 5))

    lift_metrics = ["Primary Found %", "Any GT Found %"]
    lift_labels = ["Primary Found %", "Any GT Found %"]

    coconut_lift = []
    codi_lift = []
    for m in lift_metrics:
        coconut_no = get_all_row(data[("coconut", "no_qt")])[m]
        coconut_yes = get_all_row(data[("coconut", "yes_qt")])[m]
        codi_no = get_all_row(data[("codi", "no_qt")])[m]
        codi_yes = get_all_row(data[("codi", "yes_qt")])[m]
        coconut_lift.append(coconut_yes - coconut_no)
        codi_lift.append(codi_yes - codi_no)

    x = np.arange(len(lift_metrics))
    bar_width = 0.25

    bars1 = ax.bar(x - bar_width / 2, coconut_lift, bar_width,
                   label="Coconut", color=COLOR_COCONUT, alpha=0.8)
    bars2 = ax.bar(x + bar_width / 2, codi_lift, bar_width,
                   label="CODI", color=COLOR_CODI, alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"+{h:.1f}pp", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lift_labels)
    ax.set_ylabel("Improvement (percentage points)")
    ax.set_title("Effect of Adding Question Tokens", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, f"question_token_lift_{base_llm}.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


SUBDIRS_INCORRECT = get_subdirs_incorrect('gpt2', DEFAULT_DATASET_NAME)


def _load_incorrect_data(output_dir, csv_name, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Load incorrect-sample summary CSVs. Returns None if any file is missing."""
    subdirs_incorrect = get_subdirs_incorrect(base_llm, dataset_name)
    data = {}
    for key, subdir in subdirs_incorrect.items():
        path = os.path.join(output_dir, subdir, csv_name)
        if not os.path.exists(path):
            return None
        data[key] = pd.read_csv(path)
    return data


def _draw_combined_panel(ax, data_correct, data_incorrect, bar_width=0.3):
    """Draw stacked bars for correct or incorrect panel.

    Args:
        ax: matplotlib axis to draw on
        data_correct: dict with (model, cond) -> DataFrame for correct samples
        data_incorrect: dict with (model, cond) -> DataFrame for incorrect samples, or None
        bar_width: width of each bar
    """
    x = np.arange(len(METRICS))

    for model, color, offset_sign, label in [
        ("coconut", COLOR_COCONUT, -1, "Coconut"),
        ("codi", COLOR_CODI, +1, "CODI"),
    ]:
        no_qt_row = get_all_row(data_correct[(model, "no_qt")])
        yes_qt_row = get_all_row(data_correct[(model, "yes_qt")])

        base_vals = [no_qt_row[m] for m in METRICS]
        lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
        incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]

        offset = offset_sign * bar_width / 2

        # Layer 1: solid
        ax.bar(x + offset, base_vals, bar_width,
               label=f"{label} — excl. question tokens",
               color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

        # Layer 2: hatched
        ax.bar(x + offset, lift_vals, bar_width,
               bottom=base_vals,
               label=f"{label} — incl. question tokens",
               color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

        # Labels
        for i in range(len(METRICS)):
            bar_x = x[i] + offset
            # Base (excl-QT) label for Primary and Any GT
            if i < 2:
                ax.text(bar_x, base_vals[i] + 1,
                        f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")
            # Top (incl-QT) label for all
            ax.text(bar_x, incl_qt[i] + 1,
                    f"{incl_qt[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.grid(True, alpha=0.3, axis="y")


def _draw_panel_for_sample_type(ax, data, sample_type, bar_width=0.3):
    """Draw stacked bars for a single panel (correct or incorrect samples).

    Args:
        ax: matplotlib axis to draw on
        data: dict with (model, cond) -> DataFrame
        sample_type: 'correct' or 'incorrect' (for title)
        bar_width: width of each bar
    """
    x = np.arange(len(METRICS))

    for model, color, offset_sign, label in [
        ("coconut", COLOR_COCONUT, -1, "Coconut"),
        ("codi", COLOR_CODI, +1, "CODI"),
    ]:
        no_qt_row = get_all_row(data[(model, "no_qt")])
        yes_qt_row = get_all_row(data[(model, "yes_qt")])

        base_vals = [no_qt_row[m] for m in METRICS]
        lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
        incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]

        offset = offset_sign * bar_width / 2

        # Layer 1: solid
        ax.bar(x + offset, base_vals, bar_width,
               label=f"{label} — excl. question tokens",
               color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

        # Layer 2: hatched
        ax.bar(x + offset, lift_vals, bar_width,
               bottom=base_vals,
               label=f"{label} — incl. question tokens",
               color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

        # Labels
        for i in range(len(METRICS)):
            bar_x = x[i] + offset
            # Base (excl-QT) label for Primary and Any GT
            if i < 2:
                ax.text(bar_x, base_vals[i] + 1,
                        f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")
            # Top (incl-QT) label for all
            ax.text(bar_x, incl_qt[i] + 1,
                    f"{incl_qt[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.grid(True, alpha=0.3, axis="y")


def plot_combined(data_correct, output_dir, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Side-by-side correct vs incorrect stacked bar chart with shared y-axis."""
    # Load incorrect data
    data_inc_all = _load_incorrect_data(output_dir, "summary_all_incorrect.csv", base_llm, dataset_name)
    if data_inc_all is None:
        print("Skipping combined plot: incorrect-prediction summaries not yet generated.")
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

    x = np.arange(len(METRICS))
    bar_width = 0.3

    for ax, panel in [(ax_l, "correct"), (ax_r, "incorrect")]:
        for model, color, offset_sign, label in [
            ("coconut", COLOR_COCONUT, -1, "Coconut"),
            ("codi", COLOR_CODI, +1, "CODI"),
        ]:
            if panel == "correct":
                no_qt_row = get_all_row(data_correct[(model, "no_qt")])
                yes_qt_row = get_all_row(data_correct[(model, "yes_qt")])

                base_vals = [no_qt_row[m] for m in METRICS]
                lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
                incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]
            else:
                no_qt_all = get_all_row(data_inc_all[(model, "no_qt")])
                yes_qt_all = get_all_row(data_inc_all[(model, "yes_qt")])

                base_vals = [no_qt_all[m] for m in METRICS]
                lift_vals = [yes_qt_all[m] - no_qt_all[m] for m in METRICS]
                incl_qt = [yes_qt_all[m] for m in METRICS]

            offset = offset_sign * bar_width / 2

            # Layer 1: solid
            ax.bar(x + offset, base_vals, bar_width,
                   label=f"{label} — excl. question tokens",
                   color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

            # Layer 2: hatched
            ax.bar(x + offset, lift_vals, bar_width,
                   bottom=base_vals,
                   label=f"{label} — incl. question tokens",
                   color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

            # Labels
            for i in range(len(METRICS)):
                bar_x = x[i] + offset
                # Base (excl-QT) label for Primary and Any GT
                if i < 2:
                    ax.text(bar_x, base_vals[i] + 1,
                            f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")
                # Top (incl-QT) label for all
                ax.text(bar_x, incl_qt[i] + 1,
                        f"{incl_qt[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS)
        ax.grid(True, alpha=0.3, axis="y")

    ax_l.set_ylim(0, 110)
    ax_l.set_ylabel("Percent of Samples where\nGold Reasoning Trace Found")
    ax_l.set_title("Correct Samples", fontweight="bold")
    ax_r.set_title("Incorrect Samples", fontweight="bold")

    # Single shared legend inside the right panel
    handles, labels = ax_l.get_legend_handles_labels()
    ax_r.legend(handles, labels, ncol=1, loc="upper right")

    fig.tight_layout()
    base = os.path.join(output_dir, f"back_tracking_vp_overall_summary_combined_{base_llm}")
    for ext in [".png", ".pdf"]:
        path = base + ext
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_multi_llm_2x2(output_dir, base_llms=['gpt2', 'llama32-1b'], dataset_name=DEFAULT_DATASET_NAME):
    """2x2 subplot: rows=LLMs, columns=correct/incorrect."""
    # Load data for all LLMs
    all_data_correct = {}
    all_data_incorrect = {}

    for llm in base_llms:
        all_data_correct[llm] = load_data(output_dir, llm, dataset_name)
        inc_data = _load_incorrect_data(output_dir, "summary_all_incorrect.csv", llm, dataset_name)
        if inc_data is None:
            print(f"Skipping 2x2 multi-LLM plot: incorrect data not found for {llm}")
            return
        all_data_incorrect[llm] = inc_data

    fig, axes = plt.subplots(2, 2, figsize=(24, 16), sharey=True)

    x = np.arange(len(METRICS))
    bar_width = 0.3

    for row_idx, llm in enumerate(base_llms):
        llm_name = LLM_DISPLAY_NAMES.get(llm, llm)
        data_correct = all_data_correct[llm]
        data_incorrect = all_data_incorrect[llm]

        for col_idx, (panel_type, data) in enumerate([
            ("correct", data_correct),
            ("incorrect", data_incorrect),
        ]):
            ax = axes[row_idx, col_idx]

            for model, color, offset_sign, label in [
                ("coconut", COLOR_COCONUT, -1, "Coconut"),
                ("codi", COLOR_CODI, +1, "CODI"),
            ]:
                no_qt_row = get_all_row(data[(model, "no_qt")])
                yes_qt_row = get_all_row(data[(model, "yes_qt")])

                base_vals = [no_qt_row[m] for m in METRICS]
                lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
                incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]

                offset = offset_sign * bar_width / 2

                # Layer 1: solid
                ax.bar(x + offset, base_vals, bar_width,
                       label=f"{label} — excl. question tokens",
                       color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

                # Layer 2: hatched
                ax.bar(x + offset, lift_vals, bar_width,
                       bottom=base_vals,
                       label=f"{label} — incl. question tokens",
                       color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

                # Labels
                for i in range(len(METRICS)):
                    bar_x = x[i] + offset
                    if i < 2:
                        ax.text(bar_x, base_vals[i] + 1,
                                f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")
                    ax.text(bar_x, incl_qt[i] + 1,
                            f"{incl_qt[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(METRIC_LABELS)
            ax.grid(True, alpha=0.3, axis="y")

            # Title for each panel
            sample_type = "Correct" if panel_type == "correct" else "Incorrect"
            ax.set_title(f"{llm_name} — {sample_type} Samples", fontweight="bold")

            # Y-axis label only on left column
            if col_idx == 0:
                ax.set_ylabel("Percent of Samples where\nGold Reasoning Trace Found")

    # Set y-axis limit
    axes[0, 0].set_ylim(0, 110)

    # Single shared legend from last panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = os.path.join(output_dir, "back_tracking_vp_overall_summary_combined_multi_llm_2x2")
    for ext in [".png", ".pdf"]:
        path = base + ext
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_multi_llm_1x4(output_dir, base_llms=['gpt2', 'llama32-1b'], dataset_name=DEFAULT_DATASET_NAME):
    """1x4 subplot: GPT-2 Correct, GPT-2 Incorrect, Llama Correct, Llama Incorrect."""
    # Load data for all LLMs
    all_data_correct = {}
    all_data_incorrect = {}

    for llm in base_llms:
        all_data_correct[llm] = load_data(output_dir, llm, dataset_name)
        inc_data = _load_incorrect_data(output_dir, "summary_all_incorrect.csv", llm, dataset_name)
        if inc_data is None:
            print(f"Skipping 1x4 multi-LLM plot: incorrect data not found for {llm}")
            return
        all_data_incorrect[llm] = inc_data

    fig, axes = plt.subplots(1, 4, figsize=(40, 10), sharey=True)

    x = np.arange(len(METRICS))
    bar_width = 0.3

    panel_idx = 0
    for llm in base_llms:
        llm_name = LLM_DISPLAY_NAMES.get(llm, llm)
        data_correct = all_data_correct[llm]
        data_incorrect = all_data_incorrect[llm]

        for panel_type, data in [("correct", data_correct), ("incorrect", data_incorrect)]:
            ax = axes[panel_idx]

            for model, color, offset_sign, label in [
                ("coconut", COLOR_COCONUT, -1, "Coconut"),
                ("codi", COLOR_CODI, +1, "CODI"),
            ]:
                no_qt_row = get_all_row(data[(model, "no_qt")])
                yes_qt_row = get_all_row(data[(model, "yes_qt")])

                base_vals = [no_qt_row[m] for m in METRICS]
                lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in METRICS]
                incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(METRICS))]

                offset = offset_sign * bar_width / 2

                # Layer 1: solid
                ax.bar(x + offset, base_vals, bar_width,
                       label=f"{label} — excl. question tokens",
                       color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

                # Layer 2: hatched
                ax.bar(x + offset, lift_vals, bar_width,
                       bottom=base_vals,
                       label=f"{label} — incl. question tokens",
                       color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

                # Labels
                for i in range(len(METRICS)):
                    bar_x = x[i] + offset
                    if i < 2:
                        ax.text(bar_x, base_vals[i] + 1,
                                f"{base_vals[i]:.0f}%", ha="center", va="bottom", fontweight="bold")
                    ax.text(bar_x, incl_qt[i] + 1,
                            f"{incl_qt[i]:.0f}%", ha="center", va="bottom", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(METRIC_LABELS)
            ax.grid(True, alpha=0.3, axis="y")

            sample_type = "Correct" if panel_type == "correct" else "Incorrect"
            ax.set_title(f"{llm_name} {sample_type}", fontweight="bold")

            panel_idx += 1

    # Y-axis label and limit
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Percent of Samples where\nGold Reasoning Trace Found")

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    base = os.path.join(output_dir, "back_tracking_vp_overall_summary_combined_multi_llm_1x4")
    for ext in [".png", ".pdf"]:
        path = base + ext
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_multi_llm_1x2(output_dir, base_llms=['gpt2', 'llama32-1b'], dataset_name=DEFAULT_DATASET_NAME):
    """1x2 subplot: Correct vs Incorrect, with 4 bar groups per metric (GPT-2/Llama × Coconut/CODI)."""
    # Load data for all LLMs
    all_data_correct = {}
    all_data_incorrect = {}

    for llm in base_llms:
        all_data_correct[llm] = load_data(output_dir, llm, dataset_name)
        inc_data = _load_incorrect_data(output_dir, "summary_all_incorrect.csv", llm, dataset_name)
        if inc_data is None:
            print(f"Skipping 1x2 multi-LLM plot: incorrect data not found for {llm}")
            return
        all_data_incorrect[llm] = inc_data

    _plot_combined_multi_llm_1x2_impl(all_data_correct, all_data_incorrect, output_dir, base_llms)


def _plot_combined_multi_llm_1x2_impl(all_data_correct, all_data_incorrect, output_dir, base_llms):
    """Implementation of combined multi-LLM plot with correct/incorrect as subplots."""
    # Metrics for each subplot
    metrics_per_group = ["Primary Found %", "Any GT Found %", "Base-5 %"]
    labels_per_group = ["Primary\nGold RT", "Any\nGold RT", "Best of 5\nOther Solutions"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharey=True)

    n_metrics = len(metrics_per_group)
    x = np.arange(n_metrics)
    bar_width = 0.22  # Bar width for 4 groups per position

    # Define the 4 model+LLM combinations with distinct colors
    combinations = [
        ('gpt2', 'coconut', COLOR_GPT2_COCONUT, 'GPT-2 + Coconut'),
        ('gpt2', 'codi', COLOR_GPT2_CODI, 'GPT-2 + CODI'),
        ('llama32-1b', 'coconut', COLOR_LLAMA_COCONUT, 'Llama 3.2-1B + Coconut'),
        ('llama32-1b', 'codi', COLOR_LLAMA_CODI, 'Llama 3.2-1B + CODI'),
    ]

    for ax_idx, (sample_type, all_data, title) in enumerate([
        ("correct", all_data_correct, "Correct Instances"),
        ("incorrect", all_data_incorrect, "Incorrect Instances"),
    ]):
        ax = axes[ax_idx]

        for combo_idx, (llm, model, color, label) in enumerate(combinations):
            if llm not in all_data:
                continue
            data = all_data[llm]

            no_qt_row = get_all_row(data[(model, "no_qt")])
            yes_qt_row = get_all_row(data[(model, "yes_qt")])

            base_vals = [no_qt_row[m] for m in metrics_per_group]
            lift_vals = [yes_qt_row[m] - no_qt_row[m] for m in metrics_per_group]
            incl_qt = [base_vals[i] + lift_vals[i] for i in range(len(metrics_per_group))]

            # Position bars: center the 4 groups around each x position
            offset = (combo_idx - 1.5) * bar_width

            # Layer 1: solid (excl QT) - only add label on first subplot, first metric
            label_solid = label if (ax_idx == 1 and combo_idx == 0) else (label if ax_idx == 1 else None)
            label_solid = label if ax_idx == 1 else None
            ax.bar(x + offset, base_vals, bar_width,
                   label=label_solid,
                   color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

            # Layer 2: hatched (incl QT) - no legend entry
            ax.bar(x + offset, lift_vals, bar_width,
                   bottom=base_vals,
                   color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

            # Value labels
            for metric_idx in range(n_metrics):
                bar_x = x[metric_idx] + offset
                # Base (excl. QT) label - only for Primary and Any GT (not Best of 5)
                if metric_idx < 2:
                    ax.text(bar_x, base_vals[metric_idx] + 1,
                            f"{base_vals[metric_idx]:.0f}%", ha="center", va="bottom",
                            fontsize=5, fontweight="normal")
                # Top (incl. QT) label
                ax.text(bar_x, incl_qt[metric_idx] + 1,
                        f"{incl_qt[metric_idx]:.0f}%", ha="center", va="bottom",
                        fontsize=5, fontweight="normal")

        # Set subplot properties
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_per_group, fontsize=7)
        ax.set_ylim(0, 110)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        if ax_idx == 0:
            ax.set_ylabel("% of Instances with Gold RT Found", fontsize=7)
        if ax_idx == 1:
            ax.legend(loc='upper right', framealpha=1.0, facecolor='white', fontsize=5)

    plt.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.12, wspace=0.08)
    base = os.path.join(output_dir, "back_tracking_vp_overall_summary_combined_multi_llm")
    for ext in [".png", ".pdf"]:
        path = base + ext
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GT representation analysis results")
    parser.add_argument("--output_dir", type=str, default="results/back_tracking_vp",
                        help="Directory containing result subdirectories")
    parser.add_argument("--base_llm", type=str, default="gpt2",
                        help="Base LLM short name for directory matching")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Full HF model ID for tokenizer (defaults to base_llm)")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Dataset name stem for subdirectory matching")
    parser.add_argument("--multi_llm", action="store_true",
                        help="Generate multi-LLM combined plots (GPT-2 + Llama)")
    args = parser.parse_args()

    if args.multi_llm:
        plot_combined_multi_llm_1x2(args.output_dir, dataset_name=args.dataset_name)
        print("Multi-LLM plot generated.")
    else:
        # Standard single-LLM plots
        data = load_data(args.output_dir, args.base_llm, args.dataset_name)
        plot_overall_summary(data, args.output_dir, args.base_llm)
        plot_overall_stacked(data, args.output_dir, args.base_llm)
        plot_combined(data, args.output_dir, args.base_llm, args.dataset_name)
        plot_per_step(data, args.output_dir, args.base_llm)
        plot_back_tracking_vp_per_step_combined_lines(data, args.output_dir, args.base_llm)
        plot_back_tracking_vp_per_step_combined_bars(data, args.output_dir, args.base_llm)
        plot_question_token_lift(data, args.output_dir, args.base_llm)
        print("All plots generated.")


if __name__ == "__main__":
    main()
