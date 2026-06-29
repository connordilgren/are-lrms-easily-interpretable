"""Plot GT representation analysis results for incorrect predictions."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_subdirs(base_llm, dataset_name='gsm_valid-gold-reasoning-trace_test'):
    """Generate subdirectory name patterns based on base_llm and dataset_name."""
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
    "Any Gold Label\nSolution",
    "One Random\nSolution",
    "Best of Five\nRandom Solutions",
]

COLOR_COCONUT = "#2ecc71"
COLOR_CODI = "#e74c3c"


def load_data(output_dir, csv_name, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Load summary CSVs into a dict keyed by (model, condition)."""
    subdirs = get_subdirs(base_llm, dataset_name)
    data = {}
    for key, subdir in subdirs.items():
        path = os.path.join(output_dir, subdir, csv_name)
        df = pd.read_csv(path)
        data[key] = df
    return data


def get_all_row(df):
    return df[df["Steps"] == "All"].iloc[0]


def _plot_overall_single(data, output_dir, cond, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(METRICS))
    bar_width = 0.25

    coconut_row = get_all_row(data[("coconut", cond)])
    codi_row = get_all_row(data[("codi", cond)])

    coconut_vals = [coconut_row[m] for m in METRICS]
    codi_vals = [codi_row[m] for m in METRICS]

    bars1 = ax.bar(x - bar_width / 2, coconut_vals, bar_width,
                   label=f"Coconut (n={int(coconut_row['Incorrect'])})",
                   color=COLOR_COCONUT, alpha=0.8)
    bars2 = ax.bar(x + bar_width / 2, codi_vals, bar_width,
                   label=f"CODI (n={int(codi_row['Incorrect'])})",
                   color=COLOR_CODI, alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_overall_summary(data, output_dir, suffix, base_llm='gpt2'):
    _plot_overall_single(data, output_dir, "no_qt",
                         "Gold Label Representation (Incorrect Samples)\nWithout Question Tokens",
                         f"overall_summary_without_question_tokens_incorrect_samples{suffix}_{base_llm}.png")
    _plot_overall_single(data, output_dir, "yes_qt",
                         "Gold Label Representation (Incorrect Samples)\nWith Question Tokens",
                         f"overall_summary_with_question_tokens_incorrect_samples{suffix}_{base_llm}.png")


def plot_overall_stacked(data, output_dir, suffix, base_llm='gpt2'):
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

        offset = offset_sign * bar_width / 2

        ax.bar(x + offset, base_vals, bar_width,
               label=f"{label} — excluding question tokens",
               color=color, alpha=0.8,
               edgecolor="white", linewidth=0.5)

        bars_lift = ax.bar(x + offset, lift_vals, bar_width,
                           bottom=base_vals,
                           label=f"{label} — including question tokens",
                           color=color, alpha=0.4,
                           hatch="///", edgecolor=color, linewidth=0.5)

        for i in range(len(METRICS)):
            bar_x = x[i] + offset
            total = base_vals[i] + lift_vals[i]
            # Skip base label when lift is thin to avoid overlap
            if lift_vals[i] > 3:
                ax.text(bar_x, base_vals[i] + 1,
                        f"{base_vals[i]:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
            ax.text(bar_x, total + 1,
                    f"{total:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_title("Gold Label Representation (Incorrect Samples)\nVocabulary Projections of Latent Reasoning Tokens",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, f"back_tracking_vp_overall_summary_incorrect_samples{suffix}_{base_llm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def _get_step_vals(data, model, cond, metric, steps):
    df = data[(model, cond)]
    step_df = df[df["Steps"].astype(str).str.isdigit()]
    step_df = step_df[step_df["Steps"].astype(int).isin(steps)]
    return [step_df[step_df["Steps"].astype(int) == s][metric].values[0] for s in steps]


def _plot_per_step_single(data, output_dir, cond, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = list(range(1, 6))

    lines_config = [
        ("coconut", "Primary Found %", COLOR_COCONUT, "-", "o", "Coconut — Primary"),
        ("coconut", "Any GT Found %", COLOR_COCONUT, "--", "s", "Coconut — Any Gold Label"),
        ("codi", "Primary Found %", COLOR_CODI, "-", "o", "CODI — Primary"),
        ("codi", "Any GT Found %", COLOR_CODI, "--", "s", "CODI — Any Gold Label"),
    ]

    for model, metric, color, ls, marker, label in lines_config:
        vals = _get_step_vals(data, model, cond, metric, steps)
        ax.plot(steps, vals, color=color, linestyle=ls, marker=marker,
                markersize=6, linewidth=2, label=label)

    # Sample count annotations
    coconut_df = data[("coconut", cond)]
    step_df = coconut_df[coconut_df["Steps"].astype(str).str.isdigit()]
    incorrect_coconut = []
    incorrect_codi = []
    for s in steps:
        row = step_df[step_df["Steps"].astype(int) == s].iloc[0]
        incorrect_coconut.append(int(row["Incorrect"]))
        codi_df = data[("codi", cond)]
        codi_step = codi_df[codi_df["Steps"].astype(str).str.isdigit()]
        incorrect_codi.append(int(codi_step[codi_step["Steps"].astype(int) == s].iloc[0]["Incorrect"]))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"n={incorrect_coconut[i]}/{incorrect_codi[i]}"
                         for i in range(len(steps))], fontsize=7)
    ax2.set_xlabel("Incorrect samples (Coconut / CODI)", fontsize=8)

    ax.set_xlabel("Number of Solution Steps", fontsize=12)
    ax.set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=12)
    ax.set_xticks(steps)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=35)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_per_step(data, output_dir, suffix, base_llm='gpt2'):
    _plot_per_step_single(data, output_dir, "no_qt",
                          "Per-Step Gold Label Representation (Incorrect)\nWithout Question Tokens",
                          f"per_step_without_question_tokens_incorrect_samples{suffix}_{base_llm}.png")
    _plot_per_step_single(data, output_dir, "yes_qt",
                          "Per-Step Gold Label Representation (Incorrect)\nWith Question Tokens",
                          f"per_step_with_question_tokens_incorrect_samples{suffix}_{base_llm}.png")


def plot_per_step_combined_lines(data, output_dir, suffix, base_llm='gpt2'):
    steps = list(range(1, 6))

    metrics = [
        ("Primary Found %", "Primary Solution",
         f"back_tracking_vp_per_step_primary_incorrect_samples{suffix}_{base_llm}.png"),
        ("Any GT Found %", "Any Gold Label Solution",
         f"back_tracking_vp_per_step_any_gold_label_incorrect_samples{suffix}_{base_llm}.png"),
    ]

    for metric, metric_label, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for model, color, label in [
            ("coconut", COLOR_COCONUT, "Coconut"),
            ("codi", COLOR_CODI, "CODI"),
        ]:
            base_vals = _get_step_vals(data, model, "no_qt", metric, steps)
            yes_vals = _get_step_vals(data, model, "yes_qt", metric, steps)

            ax.plot(steps, base_vals, color=color, linestyle="-", marker="o",
                    markersize=6, linewidth=2,
                    label=f"{label} — excluding question tokens")
            ax.plot(steps, yes_vals, color=color, linestyle="--", marker="s",
                    markersize=6, linewidth=2,
                    label=f"{label} — including question tokens")

        ax.set_xlabel("Number of Solution Steps", fontsize=12)
        ax.set_xticks(steps)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
        ax.set_title(f"Per-Step Gold Label Representation (Incorrect) — {metric_label}\n"
                     f"Vocabulary Projections of Latent Reasoning Tokens",
                     fontsize=14, fontweight="bold")

        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def plot_per_step_combined_bars(data, output_dir, suffix, base_llm='gpt2'):
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

            ax.bar(x + offset, base_vals, bar_width,
                   label=f"{label} — excluding question tokens",
                   color=color, alpha=0.8,
                   edgecolor="white", linewidth=0.5)

            for i in range(len(steps)):
                ax.text(x[i] + offset, base_vals[i] + 1,
                        f"{base_vals[i]:.0f}%", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

            bars_lift = ax.bar(x + offset, lift_vals, bar_width,
                               bottom=base_vals,
                               label=f"{label} — including question tokens",
                               color=color, alpha=0.4,
                               hatch="///", edgecolor=color, linewidth=0.5)

            for i, bar in enumerate(bars_lift):
                total = base_vals[i] + lift_vals[i]
                ax.text(bar.get_x() + bar.get_width() / 2, total + 1,
                        f"{total:.0f}%", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

        ax.set_xlabel("Number of Solution Steps", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.set_ylim(0, 115)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(metric_label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=11)

    fig.suptitle("Per-Step Gold Label Representation (Incorrect Samples)\nVocabulary Projections of Latent Reasoning Tokens",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"back_tracking_vp_per_step_combined_bars_incorrect_samples{suffix}_{base_llm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_question_token_lift(data, output_dir, suffix, base_llm='gpt2'):
    fig, ax = plt.subplots(figsize=(8, 5))

    coconut_lift = []
    codi_lift = []
    for m in METRICS:
        coconut_no = get_all_row(data[("coconut", "no_qt")])[m]
        coconut_yes = get_all_row(data[("coconut", "yes_qt")])[m]
        codi_no = get_all_row(data[("codi", "no_qt")])[m]
        codi_yes = get_all_row(data[("codi", "yes_qt")])[m]
        coconut_lift.append(coconut_yes - coconut_no)
        codi_lift.append(codi_yes - codi_no)

    x = np.arange(len(METRICS))
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
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("Improvement (percentage points)", fontsize=11)
    ax.set_title("Effect of Adding Question Tokens (Incorrect Samples)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, f"question_token_lift_incorrect_samples{suffix}_{base_llm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_overall_stacked_with_topk(output_dir, base_llm='gpt2', dataset_name=DEFAULT_DATASET_NAME):
    """Plot overall stacked bars: excl-QT base + incl-QT lift."""
    data_all = load_data(output_dir, "summary_all_incorrect.csv", base_llm, dataset_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(METRICS))
    bar_width = 0.3

    for model, color, offset_sign, label in [
        ("coconut", COLOR_COCONUT, -1, "Coconut"),
        ("codi", COLOR_CODI, +1, "CODI"),
    ]:
        no_qt_all = get_all_row(data_all[(model, "no_qt")])
        yes_qt_all = get_all_row(data_all[(model, "yes_qt")])

        layer1 = [no_qt_all[m] for m in METRICS]
        layer2 = [yes_qt_all[m] - no_qt_all[m] for m in METRICS]
        incl_qt = [yes_qt_all[m] for m in METRICS]

        offset = offset_sign * bar_width / 2

        # Layer 1: excluding question tokens (solid)
        ax.bar(x + offset, layer1, bar_width,
               label=f"{label} — excluding question tokens",
               color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

        # Layer 2: including question tokens (hatched)
        ax.bar(x + offset, layer2, bar_width,
               bottom=layer1,
               label=f"{label} — including question tokens",
               color=color, alpha=0.55, hatch="///", edgecolor=color, linewidth=0.5)

        # Labels
        for i in range(len(METRICS)):
            bar_x = x[i] + offset
            if layer2[i] > 3:
                ax.text(bar_x, layer1[i] + 1,
                        f"{layer1[i]:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
            ax.text(bar_x, incl_qt[i] + 1,
                    f"{incl_qt[i]:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of Incorrect Samples where\nGold Label Solution Found", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    fig.tight_layout()
    path = os.path.join(output_dir, f"back_tracking_vp_overall_summary_incorrect_samples_{base_llm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    pdf_path = path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT representation results for incorrect predictions"
    )
    parser.add_argument("--output_dir", type=str, default="results/back_tracking_vp",
                        help="Directory containing result subdirectories")
    parser.add_argument("--base_llm", type=str, default="gpt2",
                        help="Base LLM short name for directory matching")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Dataset name stem for subdirectory matching")
    args = parser.parse_args()

    for csv_name, suffix in [
        ("summary_all_incorrect.csv", "_all"),
        ("summary_gt_in_topk.csv", "_gt_in_topk"),
    ]:
        print(f"\n=== Plotting from {csv_name} ===")
        data = load_data(args.output_dir, csv_name, args.base_llm, args.dataset_name)
        plot_overall_summary(data, args.output_dir, suffix, args.base_llm)
        plot_overall_stacked(data, args.output_dir, suffix, args.base_llm)
        plot_per_step(data, args.output_dir, suffix, args.base_llm)
        plot_per_step_combined_lines(data, args.output_dir, suffix, args.base_llm)
        plot_per_step_combined_bars(data, args.output_dir, suffix, args.base_llm)
        plot_question_token_lift(data, args.output_dir, suffix, args.base_llm)

    print("\n=== Plotting with top-k not-found categories ===")
    plot_overall_stacked_with_topk(args.output_dir, args.base_llm, args.dataset_name)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
