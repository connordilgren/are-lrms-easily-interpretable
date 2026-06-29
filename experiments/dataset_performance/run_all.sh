#!/bin/bash
# ============================================================================
# Dataset Performance Evaluation - Full Pipeline
# ============================================================================
#
# Evaluates all models and generates:
#   - results/dataset_performance/table1_summary.csv  (Table 1)
#   - results/dataset_performance/figure3_multimode.png/pdf (Figure 3)
#
# Usage:
#   bash experiments/dataset_performance/run_all.sh [OPTIONS]
#
# Options:
#   --config PATH      Path to config file (default: model_paths.yaml in project root)
#   --output-dir PATH  Output directory (default: results/dataset_performance)
#   --max-samples N    Maximum samples per evaluation (default: all)
#   --skip-standard    Skip standard model evaluations (Table 1)
#   --skip-multimode   Skip multimode model evaluations (Figure 3)
#   --skip-summary     Skip CSV and plot generation
#   --dry-run          Print commands without executing
#
# ============================================================================

set -e  # Exit on error

# Default configuration
# Use PWD after cd rather than BASH_SOURCE (which breaks under sbatch)
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/experiments/dataset_performance"
CONFIG_PATH="$PROJECT_ROOT/model_paths.yaml"
OUTPUT_DIR="results/dataset_performance"
MAX_SAMPLES=""
MAX_NEW_TOKENS=256
NUM_LATENTS=6
DEVICE="cuda"
SKIP_STANDARD=false
SKIP_MULTIMODE=false
SKIP_SUMMARY=false
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
        --skip-standard)
            SKIP_STANDARD=true
            shift
            ;;
        --skip-multimode)
            SKIP_MULTIMODE=true
            shift
            ;;
        --skip-summary)
            SKIP_SUMMARY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -35 "$0" | tail -30
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
echo "Dataset Performance Evaluation - Full Pipeline"
echo "============================================================================"
echo "Configuration:"
echo "  Config file:    $CONFIG_PATH"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Max samples:    ${MAX_SAMPLES:-all}"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Num latents:    $NUM_LATENTS"
echo "  Device:         $DEVICE"
echo "  Skip standard:  $SKIP_STANDARD"
echo "  Skip multimode: $SKIP_MULTIMODE"
echo "  Skip summary:   $SKIP_SUMMARY"
echo "  Dry run:        $DRY_RUN"
echo "============================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Helper function to run evaluation
run_eval() {
    local model_type="$1"
    local model_path="$2"
    local model_id="$3"
    local dataset_path="$4"
    local mode="$5"  # Optional: for multimode models

    # Build command
    local cmd="python -m experiments.dataset_performance.run"
    cmd="$cmd --model_type $model_type"
    cmd="$cmd --model_path \"$model_path\""
    cmd="$cmd --model_id \"$model_id\""
    cmd="$cmd --dataset_path \"$dataset_path\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --num_latents $NUM_LATENTS"
    cmd="$cmd --max_new_tokens $MAX_NEW_TOKENS"
    cmd="$cmd --device $DEVICE"

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi

    if [[ -n "$mode" ]]; then
        cmd="$cmd --mode $mode"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $cmd"
    else
        echo "Running: $cmd"
        eval "$cmd"
    fi
}

# Parse YAML config and run evaluations
# We use a Python script to parse YAML and generate shell commands
parse_and_run() {
    python3 << 'PYTHON_SCRIPT'
import yaml
import os
import sys

config_path = os.environ.get('CONFIG_PATH')
output_dir = os.environ.get('OUTPUT_DIR')
skip_standard = os.environ.get('SKIP_STANDARD') == 'true'
skip_multimode = os.environ.get('SKIP_MULTIMODE') == 'true'
dry_run = os.environ.get('DRY_RUN') == 'true'

with open(config_path) as f:
    config = yaml.safe_load(f)

datasets = config.get('datasets', {
    'gsm8k': 'data/gsm_original_test.json',
    'prontoqa': 'data/prontoqa_test.json',
    'prosqa': 'data/prosqa_test.json',
})

# Standard model types (Table 1)
standard_methods = ['no_cot', 'cot', 'coconut', 'codi']

# Multimode model types (Figure 3)
multimode_methods = ['multimode_coconut', 'multimode_codi']
modes = ['direct', 'verbalized', 'latent']

# Count total evaluations
eval_count = 0
total_evals = 0

# Count standard evaluations
if not skip_standard:
    for base_model in ['gpt2', 'llama']:
        for dataset in datasets.keys():
            for method in standard_methods:
                path = config.get(base_model, {}).get(dataset, {}).get(method)
                if path and path != 'null':
                    total_evals += 1

# Count multimode evaluations
if not skip_multimode:
    for base_model in ['gpt2', 'llama']:
        for dataset in datasets.keys():
            for method in multimode_methods:
                path = config.get(base_model, {}).get(dataset, {}).get(method)
                if path and path != 'null':
                    total_evals += 3  # 3 modes per multimode model

print(f"Total evaluations to run: {total_evals}")
print("")

# Run standard evaluations
if not skip_standard:
    print("=" * 60)
    print("STANDARD MODEL EVALUATIONS (Table 1)")
    print("=" * 60)

    for base_model in ['gpt2', 'llama']:
        model_config = config.get(base_model, {})
        model_id = model_config.get('model_id', 'openai-community/gpt2')

        for dataset, dataset_path in datasets.items():
            for method in standard_methods:
                model_path = model_config.get(dataset, {}).get(method)

                if not model_path or model_path == 'null':
                    continue

                eval_count += 1
                print(f"\n[{eval_count}/{total_evals}] {method.upper()} - {base_model} - {dataset}")
                print("-" * 40)

                # Output the command for the shell to execute
                print(f"EVAL_CMD:{method}|{model_path}|{model_id}|{dataset_path}|")

# Run multimode evaluations
if not skip_multimode:
    print("")
    print("=" * 60)
    print("MULTIMODE MODEL EVALUATIONS (Figure 3)")
    print("=" * 60)

    for base_model in ['gpt2', 'llama']:
        model_config = config.get(base_model, {})
        model_id = model_config.get('model_id', 'openai-community/gpt2')

        for dataset, dataset_path in datasets.items():
            for method in multimode_methods:
                model_path = model_config.get(dataset, {}).get(method)

                if not model_path or model_path == 'null':
                    continue

                for mode in modes:
                    eval_count += 1
                    print(f"\n[{eval_count}/{total_evals}] {method.upper()} ({mode}) - {base_model} - {dataset}")
                    print("-" * 40)

                    # Output the command for the shell to execute
                    print(f"EVAL_CMD:{method}|{model_path}|{model_id}|{dataset_path}|{mode}")

PYTHON_SCRIPT
}

# Export environment variables for Python script
export CONFIG_PATH OUTPUT_DIR SKIP_STANDARD SKIP_MULTIMODE DRY_RUN

# Run the parsing and evaluation loop
# We capture Python output and execute EVAL_CMD lines
while IFS= read -r line; do
    if [[ "$line" == EVAL_CMD:* ]]; then
        # Parse the command
        cmd_data="${line#EVAL_CMD:}"
        IFS='|' read -r model_type model_path model_id dataset_path mode <<< "$cmd_data"

        # Run the evaluation
        run_eval "$model_type" "$model_path" "$model_id" "$dataset_path" "$mode"

        echo ""
    else
        echo "$line"
    fi
done < <(parse_and_run)

# Generate summary CSV and plots
if [[ "$SKIP_SUMMARY" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "GENERATING SUMMARY OUTPUTS"
    echo "============================================================================"

    # Generate Table 1 CSV
    echo ""
    echo "Generating Table 1 summary CSV..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] python -m experiments.dataset_performance.summarize_results --results_dir \"$OUTPUT_DIR\""
    else
        python -m experiments.dataset_performance.summarize_results --results_dir "$OUTPUT_DIR"
    fi

    # Generate Figure 3 dumbbell plot
    echo ""
    echo "Generating Figure 3 multimode dumbbell plot..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] python -m experiments.dataset_performance.plot_multimode_dumbbell --results_dir \"$OUTPUT_DIR\""
    else
        python -m experiments.dataset_performance.plot_multimode_dumbbell --results_dir "$OUTPUT_DIR"
    fi

    # Generate Table 10 (CSV version of Figure 3) for numeric comparison
    echo ""
    echo "Generating Table 10 multimode summary CSV..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] python -m experiments.dataset_performance.summarize_multimode --results_dir \"$OUTPUT_DIR\""
    else
        python -m experiments.dataset_performance.summarize_multimode --results_dir "$OUTPUT_DIR"
    fi
fi

echo ""
echo "============================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  Results JSON:  $OUTPUT_DIR/*.json"
echo "  Table 1 CSV:   $OUTPUT_DIR/table1_summary.csv"
echo "  Figure 3:      $OUTPUT_DIR/figure3_multimode.png"
echo "                 $OUTPUT_DIR/figure3_multimode.pdf"
echo "  Table 10 CSV:  $OUTPUT_DIR/figure3_multimode.csv"
echo ""
