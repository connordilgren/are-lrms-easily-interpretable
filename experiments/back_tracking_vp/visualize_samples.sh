#!/bin/bash
# ============================================================================
# Gold Reasoning Trace Backtracking - Sample Visualization
# ============================================================================
#
# Generates HTML visualizations for specific samples from backtracking results.
# Run the main experiment first with run_all.sh to generate results.json files.
#
# Usage:
#   bash experiments/back_tracking_vp/visualize_samples.sh [OPTIONS]
#
# Required:
#   --results-json PATH    Path to results.json from run_all.sh
#
# Options:
#   --sample-indices N...  Specific sample indices to visualize (space-separated)
#   --num-found N          Number of "GT found" samples to visualize (default: 10)
#   --num-not-found N      Number of "GT not found" samples to visualize (default: 10)
#   --dry-run              Print commands without executing
#
# Examples:
#   # Visualize specific samples by index
#   bash experiments/back_tracking_vp/visualize_samples.sh \
#       --results-json results/back_tracking_vp/coconut_gpt2_.../results.json \
#       --sample-indices 0 5 10 15
#
#   # Auto-select samples (10 found, 10 not found)
#   bash experiments/back_tracking_vp/visualize_samples.sh \
#       --results-json results/back_tracking_vp/coconut_gpt2_.../results.json
#
#   # List available results files
#   ls results/back_tracking_vp/*/results.json
#
# ============================================================================

set -e  # Exit on error

# Default configuration
RESULTS_JSON=""
SAMPLE_INDICES=""
NUM_FOUND=10
NUM_NOT_FOUND=10
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-json)
            RESULTS_JSON="$2"
            shift 2
            ;;
        --sample-indices)
            shift
            # Collect all following arguments until next flag
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SAMPLE_INDICES="$SAMPLE_INDICES $1"
                shift
            done
            ;;
        --num-found)
            NUM_FOUND="$2"
            shift 2
            ;;
        --num-not-found)
            NUM_NOT_FOUND="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -38 "$0" | tail -33
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RESULTS_JSON" ]]; then
    echo "ERROR: --results-json is required"
    echo ""
    echo "Available results files:"
    ls results/back_tracking_vp/*/results.json 2>/dev/null || echo "  (none found - run run_all.sh first)"
    echo ""
    echo "Usage: bash experiments/back_tracking_vp/visualize_samples.sh --results-json PATH [OPTIONS]"
    exit 1
fi

if [[ ! -f "$RESULTS_JSON" ]]; then
    echo "ERROR: Results file not found: $RESULTS_JSON"
    echo ""
    echo "Run the main experiment first:"
    echo "  bash experiments/back_tracking_vp/run_all.sh"
    exit 1
fi

# Check for required tools
if ! command -v python &> /dev/null; then
    echo "ERROR: python not found"
    exit 1
fi

# Print configuration
echo "============================================================================"
echo "Gold Reasoning Trace Backtracking - Sample Visualization"
echo "============================================================================"
echo "Configuration:"
echo "  Results JSON:    $RESULTS_JSON"
if [[ -n "$SAMPLE_INDICES" ]]; then
    echo "  Sample indices:  $SAMPLE_INDICES"
else
    echo "  Num found:       $NUM_FOUND"
    echo "  Num not found:   $NUM_NOT_FOUND"
fi
echo "  Dry run:         $DRY_RUN"
echo "============================================================================"
echo ""

# Build command
cmd="python -m experiments.back_tracking_vp.backtrace_computation_tree"
cmd="$cmd --results_json \"$RESULTS_JSON\""

if [[ -n "$SAMPLE_INDICES" ]]; then
    cmd="$cmd --sample_indices $SAMPLE_INDICES"
else
    cmd="$cmd --num_found $NUM_FOUND"
    cmd="$cmd --num_not_found $NUM_NOT_FOUND"
fi

# Execute
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] $cmd"
else
    echo "Running visualization..."
    eval "$cmd"
fi

# Show output location
RESULTS_DIR="$(dirname "$RESULTS_JSON")"
echo ""
echo "============================================================================"
echo "VISUALIZATION COMPLETE"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  $RESULTS_DIR/visualizations/*.html"
echo ""
echo "Open in browser:"
echo "  open $RESULTS_DIR/visualizations/"
echo ""
