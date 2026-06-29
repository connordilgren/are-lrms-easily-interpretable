#!/bin/bash
# ============================================================================
# Forward Chaining Experiment - Main Pipeline
# ============================================================================
#
# Discovers computation trees from vocabulary projections by forward chaining.
# For each sample, finds valid arithmetic steps and chains them to form trees
# ending at the model's predicted answer.
#
# Generates:
#   - results/forward_chaining/*/results.json    (per-sample results)
#   - results/forward_chaining/figure6_forward_chaining.png/pdf  (Figure 6)
#
# Usage:
#   bash experiments/forward_chaining/run_all.sh [OPTIONS]
#
# Options:
#   --config PATH              Path to config file (default: model_paths.yaml)
#   --output-dir PATH          Output directory (default: results/forward_chaining)
#   --max-samples N            Limit samples per evaluation (for testing)
#   --model-types T [T...]     Only run these model types: coconut, codi (default: both)
#   --base-llms L [L...]       Only run these base LLMs: gpt2, llama32-1b (default: both)
#   --required-passes N [N...] Only run these rp values: 1, 2, 3 (default: all)
#   --force                    Re-run even if results.json already exists
#   --skip-plots               Skip plot generation
#   --dry-run                  Print commands without executing
#
# ============================================================================

set -e  # Exit on error

# Default configuration
# Use PWD after cd rather than BASH_SOURCE (which breaks under sbatch)
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/experiments/forward_chaining"
CONFIG_PATH="$PROJECT_ROOT/model_paths.yaml"
OUTPUT_DIR="results/forward_chaining"
MAX_SAMPLES=""
TOP_K=10
NUM_LATENT=6
DEVICE="cuda"
MODEL_TYPES="coconut codi"
BASE_LLMS="gpt2 llama32-1b"
REQUIRED_PASSES="1 2 3"
FORCE=false
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
        --model-types)
            MODEL_TYPES=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODEL_TYPES="$MODEL_TYPES $1"
                shift
            done
            MODEL_TYPES="${MODEL_TYPES# }"
            ;;
        --base-llms)
            BASE_LLMS=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                BASE_LLMS="$BASE_LLMS $1"
                shift
            done
            BASE_LLMS="${BASE_LLMS# }"
            ;;
        --required-passes)
            REQUIRED_PASSES=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                REQUIRED_PASSES="$REQUIRED_PASSES $1"
                shift
            done
            REQUIRED_PASSES="${REQUIRED_PASSES# }"
            ;;
        --force)
            FORCE=true
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
            head -33 "$0" | tail -28
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
echo "Forward Chaining Experiment"
echo "============================================================================"
echo "Configuration:"
echo "  Config file:      $CONFIG_PATH"
echo "  Output dir:       $OUTPUT_DIR"
echo "  Max samples:      ${MAX_SAMPLES:-all}"
echo "  Top-k:            $TOP_K"
echo "  Num latent:       $NUM_LATENT"
echo "  Device:           $DEVICE"
echo "  Model types:      $MODEL_TYPES"
echo "  Base LLMs:        $BASE_LLMS"
echo "  Required passes:  $REQUIRED_PASSES"
echo "  Force re-run:     $FORCE"
echo "  Skip plots:       $SKIP_PLOTS"
echo "  Dry run:          $DRY_RUN"
echo "============================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Dataset for forward chaining experiment (uses vocab-projection-friendly set)
DATASET_PATH="data/gsm_vocab-projection-friendly_test.json"

# Helper function to build results subdirectory name
results_subdir() {
    local model_type=$1  # coconut | codi
    local base_llm=$2    # gpt2 | llama32-1b
    local rp=$3          # required_passes
    local mr=$4          # max_rank
    echo "${model_type}_${base_llm}_gsm_vocab-projection-friendly_test_yes-question-tokens_rp${rp}_mr${mr}"
}

# Helper function to run forward chaining analysis
run_forward_chaining() {
    local model_type="$1"
    local model_path="$2"
    local model_id="$3"
    local base_llm="$4"
    local required_passes="$5"
    local max_rank="$6"

    local subdir
    subdir=$(results_subdir "$model_type" "$base_llm" "$required_passes" "$max_rank")
    local results_file="$OUTPUT_DIR/$subdir/results.json"

    if [[ "$FORCE" != "true" && -f "$results_file" ]]; then
        echo "Skipping (results exist): $subdir"
        return 0
    fi

    local cmd="python -m experiments.forward_chaining.run"
    cmd="$cmd --model_type $model_type"
    cmd="$cmd --model_path \"$model_path\""
    cmd="$cmd --model_id \"$model_id\""
    cmd="$cmd --dataset_path \"$DATASET_PATH\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --top_k $TOP_K"
    cmd="$cmd --num_latent $NUM_LATENT"
    cmd="$cmd --base_llm $base_llm"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --include_question_tokens"
    cmd="$cmd --validate"
    cmd="$cmd --validation_required_passes $required_passes"
    cmd="$cmd --validation_max_rank $max_rank"

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

# ============================================================================
# STEP 1: Run Forward Chaining Analysis
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: FORWARD CHAINING ANALYSIS"
echo "============================================================================"

# Export environment variables for Python heredoc
export CONFIG_PATH MODEL_TYPES BASE_LLMS REQUIRED_PASSES

# Run the analysis commands
while IFS= read -r line; do
    if [[ "$line" == ANALYSIS_CMD:* ]]; then
        cmd_data="${line#ANALYSIS_CMD:}"
        IFS='|' read -r model_type model_path model_id base_llm required_passes max_rank <<< "$cmd_data"
        run_forward_chaining "$model_type" "$model_path" "$model_id" "$base_llm" "$required_passes" "$max_rank"
        echo ""
    else
        echo "$line"
    fi
done < <(python3 << 'PYTHON_SCRIPT'
import yaml
import os

config_path = os.environ.get('CONFIG_PATH')
allowed_model_types = os.environ.get('MODEL_TYPES', 'coconut codi').split()
allowed_base_llms   = os.environ.get('BASE_LLMS', 'gpt2 llama32-1b').split()
allowed_rp          = [int(x) for x in os.environ.get('REQUIRED_PASSES', '1 2 3').split()]

with open(config_path) as f:
    config = yaml.safe_load(f)

all_base_models = [('gpt2', 'gpt2'), ('llama', 'llama32-1b')]
base_models = [(ck, bl) for ck, bl in all_base_models if bl in allowed_base_llms]
model_types = [mt for mt in ['coconut', 'codi'] if mt in allowed_model_types]
required_passes_values = [rp for rp in [1, 2, 3] if rp in allowed_rp]
max_rank = 1  # Fixed for Figure 6

total_evals = 0
for config_key, base_llm in base_models:
    for model_type in model_types:
        path = config.get(config_key, {}).get('gsm8k', {}).get(model_type)
        if path and path != 'null':
            total_evals += len(required_passes_values)

print(f"Total analysis runs: {total_evals}")
print("")

eval_count = 0
for config_key, base_llm in base_models:
    model_config = config.get(config_key, {})
    model_id = model_config.get('model_id', 'openai-community/gpt2')

    for model_type in model_types:
        model_path = model_config.get('gsm8k', {}).get(model_type)

        if not model_path or model_path == 'null':
            continue

        for rp in required_passes_values:
            eval_count += 1
            print(f"\n[{eval_count}/{total_evals}] {model_type.upper()} - {base_llm} - rp={rp}, mr=1")
            print("-" * 50)
            print(f"ANALYSIS_CMD:{model_type}|{model_path}|{model_id}|{base_llm}|{rp}|1")

PYTHON_SCRIPT
)

# ============================================================================
# STEP 2: Generate Plots (Figure 6)
# ============================================================================
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo ""
    echo "============================================================================"
    echo "STEP 2: GENERATING FIGURE 6"
    echo "============================================================================"

    # Build base_llm argument from whatever was run
    BASE_LLM_ARGS="$BASE_LLMS"

    if [[ -n "$BASE_LLM_ARGS" ]]; then
        cmd="python -m experiments.forward_chaining.plot_hyperparam_sweep"
        cmd="$cmd --results_dir \"$OUTPUT_DIR\""
        cmd="$cmd --output_dir \"$OUTPUT_DIR\""
        cmd="$cmd --base_llm $BASE_LLM_ARGS"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] $cmd"
        else
            echo "Running: $cmd"
            eval "$cmd"
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
echo "  Figure 6:      $OUTPUT_DIR/figure6_forward_chaining.png/pdf"
echo ""
