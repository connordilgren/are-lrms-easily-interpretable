#!/bin/bash
# ============================================================================
# Early Stopping Experiment - Full Pipeline
# ============================================================================
#
# Runs the early stopping experiment on all models and generates:
#   - results/early_stopping/figure2_early_stopping.png/pdf  (Figure 2)
#   - results/early_stopping/table11_early_stopping.csv      (Table 11)
#
# Usage:
#   bash experiments/early_stopping/run_all.sh [OPTIONS]
#
# Options:
#   --config PATH      Path to config file (default: model_paths.yaml in project root)
#   --output-dir PATH  Output directory (default: results/early_stopping)
#   --max-samples N    Maximum samples per evaluation (default: all)
#   --skip-gpt2        Skip GPT-2 model evaluations
#   --skip-llama       Skip Llama model evaluations
#   --skip-plots       Skip plot generation
#   --dry-run          Print commands without executing
#
# ============================================================================

set -e  # Exit on error

# Default configuration
# Use PWD after cd rather than BASH_SOURCE (which breaks under sbatch)
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/experiments/early_stopping"
CONFIG_PATH="$PROJECT_ROOT/model_paths.yaml"
OUTPUT_DIR="results/early_stopping"
MAX_SAMPLES=""
MAX_NEW_TOKENS=256
NUM_LATENTS=6
DEVICE="cuda"
SKIP_GPT2=false
SKIP_LLAMA=false
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
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -20
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
echo "Early Stopping Experiment - Full Pipeline"
echo "============================================================================"
echo "Configuration:"
echo "  Config file:    $CONFIG_PATH"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Max samples:    ${MAX_SAMPLES:-all}"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Num latents:    $NUM_LATENTS"
echo "  Device:         $DEVICE"
echo "  Skip GPT-2:     $SKIP_GPT2"
echo "  Skip Llama:     $SKIP_LLAMA"
echo "  Skip plots:     $SKIP_PLOTS"
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
    local top_k="$5"

    # Build command
    local cmd="python -m experiments.early_stopping.run"
    cmd="$cmd --model_type $model_type"
    cmd="$cmd --model_path \"$model_path\""
    cmd="$cmd --model_id \"$model_id\""
    cmd="$cmd --dataset_path \"$dataset_path\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --max_new_tokens $MAX_NEW_TOKENS"
    cmd="$cmd --top_k $top_k"
    cmd="$cmd --device $DEVICE"

    # Add num_latents for coconut and codi
    if [[ "$model_type" == "coconut" || "$model_type" == "codi" ]]; then
        cmd="$cmd --num_latents $NUM_LATENTS"
    fi

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $cmd"
    else
        echo "Running: $cmd"
        eval "$cmd"
    fi
}

# Parse YAML config and run evaluations
parse_and_run() {
    python3 << 'PYTHON_SCRIPT'
import yaml
import os

config_path = os.environ.get('CONFIG_PATH')
skip_gpt2 = os.environ.get('SKIP_GPT2') == 'true'
skip_llama = os.environ.get('SKIP_LLAMA') == 'true'

with open(config_path) as f:
    config = yaml.safe_load(f)

# Early stopping uses gsm_valid-gold-reasoning-trace_test.json for GSM8k
datasets = {
    'gsm8k': ('data/gsm_valid-gold-reasoning-trace_test.json', 10),
    'prontoqa': ('data/prontoqa_test.json', 5),
    'prosqa': ('data/prosqa_test.json', 5),
}

# Model types for early stopping (cot, coconut, codi only)
model_types = ['cot', 'coconut', 'codi']

# Count total evaluations
eval_count = 0
total_evals = 0

base_models = []
if not skip_gpt2:
    base_models.append('gpt2')
if not skip_llama:
    base_models.append('llama')

for base_model in base_models:
    for dataset in datasets.keys():
        for method in model_types:
            path = config.get(base_model, {}).get(dataset, {}).get(method)
            if path and path != 'null':
                total_evals += 1

print(f"Total evaluations to run: {total_evals}")
print("")

print("=" * 60)
print("EARLY STOPPING EVALUATIONS")
print("=" * 60)

for base_model in base_models:
    model_config = config.get(base_model, {})
    model_id = model_config.get('model_id', 'openai-community/gpt2')

    for dataset, (dataset_path, top_k) in datasets.items():
        for method in model_types:
            model_path = model_config.get(dataset, {}).get(method)

            if not model_path or model_path == 'null':
                continue

            eval_count += 1
            print(f"\n[{eval_count}/{total_evals}] {method.upper()} - {base_model} - {dataset}")
            print("-" * 40)

            # Output the command for the shell to execute
            print(f"EVAL_CMD:{method}|{model_path}|{model_id}|{dataset_path}|{top_k}")

PYTHON_SCRIPT
}

# Export environment variables for Python script
export CONFIG_PATH SKIP_GPT2 SKIP_LLAMA

# Run the parsing and evaluation loop
while IFS= read -r line; do
    if [[ "$line" == EVAL_CMD:* ]]; then
        # Parse the command
        cmd_data="${line#EVAL_CMD:}"
        IFS='|' read -r model_type model_path model_id dataset_path top_k <<< "$cmd_data"

        # Run the evaluation
        run_eval "$model_type" "$model_path" "$model_id" "$dataset_path" "$top_k"

        echo ""
    else
        echo "$line"
    fi
done < <(parse_and_run)

# Generate plots
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "GENERATING FIGURE 2 AND TABLE 11"
    echo "============================================================================"

    # Find latest result files using Python
    find_and_plot() {
        python3 << 'PYTHON_SCRIPT'
import os
import glob
from pathlib import Path

output_dir = os.environ.get('OUTPUT_DIR', 'results/early_stopping')
skip_gpt2 = os.environ.get('SKIP_GPT2') == 'true'
skip_llama = os.environ.get('SKIP_LLAMA') == 'true'
dry_run = os.environ.get('DRY_RUN') == 'true'

def find_latest(pattern, exclude_pattern=None):
    """Find the most recent file matching a pattern, optionally excluding another pattern."""
    files = glob.glob(str(Path(output_dir) / pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in f]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

file_args = []

# GPT-2 files (exclude files with 'llama' in the name)
if not skip_gpt2:
    gpt2_patterns = {
        'gpt2_gsm_cot': 'early_stopping_cot_*gsm*k10*.json',
        'gpt2_gsm_coconut': 'early_stopping_coconut_*gsm*k10*.json',
        'gpt2_gsm_codi': 'early_stopping_codi_*gsm*k10*.json',
        'gpt2_prosqa_cot': 'early_stopping_cot_*prosqa*k5*.json',
        'gpt2_prosqa_coconut': 'early_stopping_coconut_*prosqa*k5*.json',
        'gpt2_prosqa_codi': 'early_stopping_codi_*prosqa*k5*.json',
        'gpt2_prontoqa_cot': 'early_stopping_cot_*prontoqa*k5*.json',
        'gpt2_prontoqa_coconut': 'early_stopping_coconut_*prontoqa*k5*.json',
        'gpt2_prontoqa_codi': 'early_stopping_codi_*prontoqa*k5*.json',
    }

    for key, pattern in gpt2_patterns.items():
        filepath = find_latest(pattern, exclude_pattern='llama')
        if filepath:
            filename = os.path.basename(filepath)
            file_args.append(f"--{key}_file {filename}")
        else:
            print(f"WARNING: No file found for {key}")

# Llama files (must contain 'llama' in the name)
if not skip_llama:
    llama_patterns = {
        'llama_gsm_cot': 'early_stopping_cot_llama*gsm*k10*.json',
        'llama_gsm_coconut': 'early_stopping_coconut_llama*gsm*k10*.json',
        'llama_gsm_codi': 'early_stopping_codi_llama*gsm*k10*.json',
        'llama_prosqa_cot': 'early_stopping_cot_llama*prosqa*k5*.json',
        'llama_prosqa_coconut': 'early_stopping_coconut_llama*prosqa*k5*.json',
        'llama_prosqa_codi': 'early_stopping_codi_llama*prosqa*k5*.json',
        'llama_prontoqa_cot': 'early_stopping_cot_llama*prontoqa*k5*.json',
        'llama_prontoqa_coconut': 'early_stopping_coconut_llama*prontoqa*k5*.json',
        'llama_prontoqa_codi': 'early_stopping_codi_llama*prontoqa*k5*.json',
    }

    for key, pattern in llama_patterns.items():
        filepath = find_latest(pattern)
        if filepath:
            filename = os.path.basename(filepath)
            file_args.append(f"--{key}_file {filename}")
        else:
            print(f"WARNING: No file found for {key}")

if not file_args:
    print("ERROR: No result files found in " + output_dir)
    exit(1)

# Build plot command
cmd = f"python -m experiments.early_stopping.plot --base_llm combined --results_dir {output_dir}"
cmd += " " + " ".join(file_args)

print(f"PLOT_CMD:{cmd}")

PYTHON_SCRIPT
    }

    # Export additional variables
    export OUTPUT_DIR DRY_RUN

    # Run plotting
    while IFS= read -r line; do
        if [[ "$line" == PLOT_CMD:* ]]; then
            cmd="${line#PLOT_CMD:}"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY RUN] $cmd"
            else
                echo "Running: $cmd"
                eval "$cmd"
            fi
        elif [[ "$line" == ERROR:* ]]; then
            echo "$line"
            exit 1
        else
            echo "$line"
        fi
    done < <(find_and_plot)

    # Rename output files to match paper naming convention
    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ -f "$OUTPUT_DIR/early_stopping_bar_force_stop_stacked_combined.png" ]]; then
            mv "$OUTPUT_DIR/early_stopping_bar_force_stop_stacked_combined.png" "$OUTPUT_DIR/figure2_early_stopping.png"
            mv "$OUTPUT_DIR/early_stopping_bar_force_stop_stacked_combined.pdf" "$OUTPUT_DIR/figure2_early_stopping.pdf"
            echo "Saved: $OUTPUT_DIR/figure2_early_stopping.png/pdf"
        fi
        if [[ -f "$OUTPUT_DIR/early_stopping_bar_force_stop_stacked_combined.csv" ]]; then
            mv "$OUTPUT_DIR/early_stopping_bar_force_stop_stacked_combined.csv" "$OUTPUT_DIR/table11_early_stopping.csv"
            echo "Saved: $OUTPUT_DIR/table11_early_stopping.csv"
        fi
    fi
fi

echo ""
echo "============================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  Results JSON:  $OUTPUT_DIR/early_stopping_*.json"
echo "  Figure 2:      $OUTPUT_DIR/figure2_early_stopping.png"
echo "                 $OUTPUT_DIR/figure2_early_stopping.pdf"
echo "  Table 11:      $OUTPUT_DIR/table11_early_stopping.csv"
echo ""
