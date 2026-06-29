#!/bin/bash
# ============================================================================
# Gold Reasoning Trace Backtracking Experiment - Main Pipeline
# ============================================================================
#
# Runs the backtracking experiment on Coconut and CODI models, analyzing how
# gold reasoning traces are represented in vocabulary projections.
#
# Generates:
#   - results/back_tracking_vp/*/results.json    (per-sample results)
#   - results/back_tracking_vp/*/summary.csv     (aggregate statistics)
#   - results/back_tracking_vp/figure5_backtracking.png/pdf  (Figure 5)
#
# To visualize specific samples after running this script, use:
#   bash experiments/back_tracking_vp/visualize_samples.sh --help
#
# Usage:
#   bash experiments/back_tracking_vp/run_all.sh [OPTIONS]
#
# Options:
#   --config PATH        Path to config file (default: model_paths.yaml)
#   --output-dir PATH    Output directory (default: results/back_tracking_vp)
#   --max-samples N      Limit samples per evaluation (for testing)
#   --skip-gpt2          Skip GPT-2 model evaluations
#   --skip-llama         Skip Llama model evaluations
#   --skip-analysis      Skip model runs (use existing results.json files)
#   --skip-plots         Skip plot generation
#   --dry-run            Print commands without executing
#
# ============================================================================

set -e  # Exit on error

# Default configuration
# Use PWD after cd rather than BASH_SOURCE (which breaks under sbatch)
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/experiments/back_tracking_vp"
CONFIG_PATH="$PROJECT_ROOT/model_paths.yaml"
OUTPUT_DIR="results/back_tracking_vp"
MAX_SAMPLES=""
TOP_K=10
NUM_LATENT=6
DEVICE="cuda"
SKIP_GPT2=false
SKIP_LLAMA=false
SKIP_ANALYSIS=false
SKIP_PLOTS=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --skip-gpt2)
            SKIP_GPT2=true
            shift
            ;;
        --skip-llama)
            SKIP_LLAMA=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v python &> /dev/null; then
    echo "ERROR: python not found"
    exit 1
fi

if ! python -c "import yaml" &> /dev/null; then
    echo "ERROR: PyYAML not installed. Run: pip install pyyaml"
    exit 1
fi

# Check config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    echo "Copy model_paths.yaml.example to model_paths.yaml and edit paths"
    exit 1
fi

# Check device availability
if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        echo "WARNING: CUDA not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

# Print configuration
echo "============================================================================"
echo "Gold Reasoning Trace Backtracking Experiment"
echo "============================================================================"
echo "Configuration:"
echo "  Config file:    $CONFIG_PATH"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Max samples:    ${MAX_SAMPLES:-all}"
echo "  Top-k:          $TOP_K"
echo "  Num latent:     $NUM_LATENT"
echo "  Device:         $DEVICE"
echo "  Skip GPT-2:     $SKIP_GPT2"
echo "  Skip Llama:     $SKIP_LLAMA"
echo "  Skip analysis:  $SKIP_ANALYSIS"
echo "  Skip plots:     $SKIP_PLOTS"
echo "  Dry run:        $DRY_RUN"
echo "============================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Dataset for backtracking experiment (uses valid gold reasoning trace set)
DATASET_PATH="data/gsm_valid-gold-reasoning-trace_test.json"

# Helper function to build results subdirectory name
results_subdir() {
    local model_type=$1  # coconut | codi
    local base_llm=$2    # gpt2 | llama32-1b
    local qt_flag=$3     # no-question-tokens | yes-question-tokens
    echo "${model_type}_${base_llm}_gsm_valid-gold-reasoning-trace_test_k${TOP_K}_no-baseline-require-answer_${qt_flag}"
}

# Helper function to run analysis
run_analysis() {
    local model_type="$1"
    local model_path="$2"
    local model_id="$3"
    local base_llm="$4"
    local include_qt="$5"

    local cmd="python -m experiments.back_tracking_vp.analyze_gt_representation"
    cmd="$cmd --model_type $model_type"
    cmd="$cmd --model_path \"$model_path\""
    cmd="$cmd --model_id \"$model_id\""
    cmd="$cmd --dataset_path \"$DATASET_PATH\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --top_k $TOP_K"
    cmd="$cmd --num_latent $NUM_LATENT"
    cmd="$cmd --base_llm $base_llm"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --no_baseline_require_answer"

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi

    if [[ "$include_qt" == "true" ]]; then
        cmd="$cmd --include_question_tokens"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $cmd"
    else
        echo "Running: $cmd"
        eval "$cmd"
    fi
}

# ============================================================================
# STEP 1: Run GT representation analysis
# ============================================================================
if [[ "$SKIP_ANALYSIS" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "STEP 1: GT REPRESENTATION ANALYSIS"
    echo "============================================================================"

    # Export environment variables for Python heredocs
    export CONFIG_PATH SKIP_GPT2 SKIP_LLAMA

    # Run the analysis commands
    while IFS= read -r line; do
        if [[ "$line" == ANALYSIS_CMD:* ]]; then
            cmd_data="${line#ANALYSIS_CMD:}"
            IFS='|' read -r model_type model_path model_id base_llm include_qt <<< "$cmd_data"
            run_analysis "$model_type" "$model_path" "$model_id" "$base_llm" "$include_qt"
            echo ""
        else
            echo "$line"
        fi
    done < <(CONFIG_PATH="$CONFIG_PATH" SKIP_GPT2="$SKIP_GPT2" SKIP_LLAMA="$SKIP_LLAMA" python3 << 'PYTHON_SCRIPT'
import yaml
import os

config_path = os.environ.get('CONFIG_PATH')
skip_gpt2 = os.environ.get('SKIP_GPT2') == 'true'
skip_llama = os.environ.get('SKIP_LLAMA') == 'true'

with open(config_path) as f:
    config = yaml.safe_load(f)

base_models = []
if not skip_gpt2:
    base_models.append(('gpt2', 'gpt2'))
if not skip_llama:
    base_models.append(('llama', 'llama32-1b'))

model_types = ['coconut', 'codi']

eval_count = 0
total_evals = 0
for config_key, base_llm in base_models:
    for model_type in model_types:
        path = config.get(config_key, {}).get('gsm8k', {}).get(model_type)
        if path and path != 'null':
            total_evals += 2

print(f"Total analysis runs: {total_evals}")
print("")

for config_key, base_llm in base_models:
    model_config = config.get(config_key, {})
    model_id = model_config.get('model_id', 'openai-community/gpt2')

    for model_type in model_types:
        model_path = model_config.get('gsm8k', {}).get(model_type)

        if not model_path or model_path == 'null':
            continue

        for include_qt in ['false', 'true']:
            eval_count += 1
            qt_label = "with question tokens" if include_qt == 'true' else "without question tokens"
            print(f"\n[{eval_count}/{total_evals}] {model_type.upper()} - {base_llm} - {qt_label}")
            print("-" * 50)
            print(f"ANALYSIS_CMD:{model_type}|{model_path}|{model_id}|{base_llm}|{include_qt}")

PYTHON_SCRIPT
    )
else
    echo "(Skipping analysis - using existing results.json files)"
fi

# ============================================================================
# STEP 2: Analyze incorrect predictions
# ============================================================================
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "STEP 2: INCORRECT PREDICTION ANALYSIS"
    echo "============================================================================"

    for base_llm in gpt2 llama32-1b; do
        if [[ "$base_llm" == "gpt2" && "$SKIP_GPT2" == "true" ]]; then
            continue
        fi
        if [[ "$base_llm" == "llama32-1b" && "$SKIP_LLAMA" == "true" ]]; then
            continue
        fi

        for model_type in coconut codi; do
            for qt in no-question-tokens yes-question-tokens; do
                subdir=$(results_subdir "$model_type" "$base_llm" "$qt")
                results_json="${OUTPUT_DIR}/${subdir}/results.json"

                if [[ ! -f "$results_json" ]]; then
                    echo "  SKIP (not found): $results_json"
                    continue
                fi

                echo "  Analyzing incorrect: $subdir"
                cmd="python -m experiments.back_tracking_vp.analyze_incorrect_predictions"
                cmd="$cmd --results_json \"$results_json\""
                cmd="$cmd --base_llm $base_llm"
                cmd="$cmd --num_visualizations 0"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "  [DRY RUN] $cmd"
                else
                    eval "$cmd"
                fi
            done
        done
    done

    echo ""
    echo "============================================================================"
    echo "STEP 3: INCORRECT PREDICTION SUMMARIES"
    echo "============================================================================"

    for base_llm in gpt2 llama32-1b; do
        if [[ "$base_llm" == "gpt2" && "$SKIP_GPT2" == "true" ]]; then
            continue
        fi
        if [[ "$base_llm" == "llama32-1b" && "$SKIP_LLAMA" == "true" ]]; then
            continue
        fi

        for model_type in coconut codi; do
            for qt in no-question-tokens yes-question-tokens; do
                subdir=$(results_subdir "$model_type" "$base_llm" "$qt")
                results_json="${OUTPUT_DIR}/${subdir}/results.json"

                if [[ ! -f "$results_json" ]]; then
                    echo "  SKIP (not found): $results_json"
                    continue
                fi

                echo "  Summarizing: $subdir"
                cmd="python -m experiments.back_tracking_vp.summarize_incorrect_gt_representation"
                cmd="$cmd --results_json \"$results_json\""
                cmd="$cmd --base_llm $base_llm"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "  [DRY RUN] $cmd"
                else
                    eval "$cmd"
                fi
            done
        done
    done
fi

# ============================================================================
# STEP 4: Generate plots (Figure 5)
# ============================================================================
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "STEP 4: GENERATING FIGURE 5 AND SUMMARY PLOTS"
    echo "============================================================================"

    # Generate single-LLM plots
    for base_llm in gpt2 llama32-1b; do
        if [[ "$base_llm" == "gpt2" && "$SKIP_GPT2" == "true" ]]; then
            continue
        fi
        if [[ "$base_llm" == "llama32-1b" && "$SKIP_LLAMA" == "true" ]]; then
            continue
        fi

        echo "Generating plots for $base_llm..."

        # Correct prediction plots
        cmd="python -m experiments.back_tracking_vp.plot_results --output_dir \"$OUTPUT_DIR\" --base_llm $base_llm"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] $cmd"
        else
            eval "$cmd"
        fi

        # Incorrect prediction plots
        cmd="python -m experiments.back_tracking_vp.plot_incorrect_results --output_dir \"$OUTPUT_DIR\" --base_llm $base_llm"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] $cmd"
        else
            eval "$cmd" 2>/dev/null || echo "  (skipped - incorrect data may not exist)"
        fi
    done

    # Generate multi-LLM combined plots if both LLMs were run
    if [[ "$SKIP_GPT2" != "true" && "$SKIP_LLAMA" != "true" ]]; then
        echo ""
        echo "Generating multi-LLM combined plots..."
        cmd="python -m experiments.back_tracking_vp.plot_results --output_dir \"$OUTPUT_DIR\" --multi_llm"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] $cmd"
        else
            eval "$cmd"
        fi
    fi

    # Copy to standard figure names
    if [[ "$DRY_RUN" != "true" ]]; then
        # Figure 5: Combined correct/incorrect plot
        if [[ -f "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_multi_llm.png" ]]; then
            cp "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_multi_llm.png" "$OUTPUT_DIR/figure5_backtracking.png"
            cp "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_multi_llm.pdf" "$OUTPUT_DIR/figure5_backtracking.pdf"
            echo "Created: $OUTPUT_DIR/figure5_backtracking.png/pdf"
        elif [[ -f "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_gpt2.png" ]]; then
            cp "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_gpt2.png" "$OUTPUT_DIR/figure5_backtracking.png"
            cp "$OUTPUT_DIR/back_tracking_vp_overall_summary_combined_gpt2.pdf" "$OUTPUT_DIR/figure5_backtracking.pdf"
            echo "Created: $OUTPUT_DIR/figure5_backtracking.png/pdf"
        fi
    fi
fi

# ============================================================================
# COMPLETE
# ============================================================================
echo ""
echo "============================================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  Results JSON:  $OUTPUT_DIR/*/results.json"
echo "  Summaries:     $OUTPUT_DIR/*/summary.csv"
echo "  Figure 5:      $OUTPUT_DIR/figure5_backtracking.png/pdf"
echo ""
echo "To visualize specific samples, run:"
echo "  bash experiments/back_tracking_vp/visualize_samples.sh \\"
echo "      --results-json $OUTPUT_DIR/<subdir>/results.json \\"
echo "      --sample-indices 0 1 2"
echo ""
