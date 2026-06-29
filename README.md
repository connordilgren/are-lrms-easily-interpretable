# are-lrms-easily-interpretable
This is the repository for "Are Latent Reasoning Models Easily Interpretable?" ([arXiv](https://arxiv.org/abs/2604.04902))

## Getting Started

Clone repo:

```bash
git clone https://github.com/connordilgren/are-lrms-easily-interpretable.git
cd are-lrms-easily-interpretable
```

Setup environment:

```bash
conda create --name lrm_interp python=3.12
conda activate lrm_interp
pip install -r requirements.txt
```

## Data Preparation

### ProsQA and PrOntoQA

These datasets are included in `data/`. See the [Coconut repository](https://github.com/facebookresearch/coconut) and [PrOntoQA repository](https://github.com/asaparov/prontoqa) for generation details.

### GSM8K-Aug Datasets

We use three versions of the GSM8k-Aug test set:

1. gsm_original_test.json: This set is the original GSM8K test set, used to calculate model performance.
2. gsm_valid-gold-reasoning-trace_test.json: This set filters out instances in gsm_original_test.json where the result of the last reasoning step is not equal to the correct answer. The early stopping and backtracking experiments use this set.
3. gsm_vocab-projection-friendly_test.json: This set flters for instances in gsm_valid-gold-reasoning-trace_test.json that use unique, single-token numbers (with respect to the GPT-2 and Llama-3.2-1B-Instruct tokenizers) as operands and intermediate results in the gold reasoning trace.

The gsm_valid-gold-reasoning-trace_test.json and gsm_vocab-projection-friendly_test.json files include additional valid solutions from the [MultiChain-GSM8k-Aug-dataset](https://huggingface.co/datasets/DJCheng/MultiChain-GSM8k-Aug-dataset).

Generate all GSM8K datasets into `data/` using:

```bash
python preprocessing/prepare_gsm8k.py
```

## Model Training

### Configuration

After training a model, edit `model_paths.yaml` in the project root to specify your trained model checkpoint paths:

```yaml
gpt2:
  model_id: "openai-community/gpt2"
  gsm8k:
    no_cot: "checkpoints/gpt2_gsm8k_no_cot/checkpoint_XX"
    cot: "checkpoints/gsm-cot/checkpoint_33"
    coconut: "checkpoints/gsm-coconut/checkpoint_33"
    codi: "zen-E/CODI-gpt2"
    multimode_coconut: "checkpoints/gpt2_gsm_multimode/checkpoint_41"
    multimode_codi: "checkpoints/codi_trained_models/gsm8k_gpt2.../pytorch_model.bin"
  # ... etc for prontoqa, prosqa

llama:
  model_id: "meta-llama/Llama-3.2-1B-Instruct"
  # ... same structure
```

### No-CoT, CoT, and Coconut Models

We train No-CoT, CoT, and Coconut models using the [Coconut](https://github.com/facebookresearch/coconut) training framework.

#### Setup

1. Clone the Coconut repository at the commit we used:
   ```bash
   git clone https://github.com/facebookresearch/coconut.git
   cd coconut
   git checkout 27273cb8cca4bb763c041a63b036d0c3b7cbbb48
   ```

2. Install dependencies following the Coconut repository instructions.

3. Copy our args files into the Coconut repo:
   ```bash
   cp /path/to/this-repo/args_coconut-cot-no_cot/*.yaml /path/to/coconut/args
   ```

4. Link or copy datasets into Coconut's data directory:
   ```bash
   ln -s /path/to/this-repo/data /path/to/coconut/data
   ```

#### Training Commands

Training is done via `torchrun`. From the Coconut repository directory:

`torchrun --nnodes 1 --nproc_per_node 4 run.py args/{base_llm}_{dataset}_{reasoning_method}.yaml`

Where:
- `base_llm`: `gpt2`, `llama32-1b`
- `dataset`: `gsm8k`, `prosqa`, `prontoqa`
- `reasoning_method`: `no_cot`, `cot`, `coconut`

Trained models are saved to `checkpoints/` within the Coconut repository.

Note that these commands assume 4 * A100 (80GB) GPUs. You may change the corresponding arguments in the config file (batch_size_training, gradient_accumulation_steps) and nproc_per_node when launching the run, to adapt your resources.

Also note that these Coconut args files load from a CoT checkpoint, so you should complete the corresponding CoT model training first:
- `gpt2_gsm8k_coconut.yaml` (requires `gpt2_gsm8k_cot`)
- `llama32-1b_gsm8k_coconut.yaml` (requires `llama32-1b_gsm8k_cot`)
- `llama32-1b_prontoqa_coconut.yaml` (requires `llama32-1b_prontoqa_cot`)
- `llama32-1b_prosqa_coconut.yaml` (requires `llama32-1b_prosqa_cot`)

### Coconut Multi-mode Models

Multi-mode Coconut models can perform latent reasoning, explicit (CoT) reasoning, or direct answer depending on input tokens. Training these requires modifications to the Coconut repository.

#### Setup

1. First, complete the Coconut setup above (clone repo, install dependencies, link data).

2. Copy our modified training files into the Coconut repo (these replace the originals):
   ```bash
   cp /path/to/this-repo/src_coconut_multimode/coconut.py /path/to/coconut/
   cp /path/to/this-repo/src_coconut_multimode/dataset.py /path/to/coconut/
   cp /path/to/this-repo/src_coconut_multimode/run.py /path/to/coconut/
   cp -r /path/to/this-repo/src_coconut_multimode/utils /path/to/coconut/
   ```

3. Copy our multi-mode config files:
   ```bash
   cp /path/to/this-repo/args_coconut_multimode/*.yaml /path/to/coconut/args/
   ```

#### Training Commands

From the Coconut repository directory:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/{base_llm}_{dataset}_multimode.yaml
```

Where:
- `base_llm`: `gpt2`, `llama32-1b`
- `dataset`: `gsm8k`, `prosqa`, `prontoqa`

These commands assume 4 × A100 (80GB) GPUs. Adjust `batch_size_training`, `gradient_accumulation_steps`, and `nproc_per_node` to adapt to your resources.

#### Key Configuration Parameters

The multimode configs include:
- `multimode: True` - Enable multi-reasoning mode training
- `alpha: 1.0` - Latent loss weight
- `beta: 1.0` - Verbalized loss weight
- `gamma: 1.0` - Direct loss weight

Total loss = α × L_latent + β × L_verbalized + γ × L_direct

### CODI Models

We train CODI models using the [CODI](https://github.com/zhenyi4/codi) repository.

For CODI + GPT-2 + GSM8k, we use the authors' pretrained checkpoint available at [zen-E/CODI-gpt2](https://huggingface.co/zen-E/CODI-gpt2).

For the remaining 5 CODI models:

#### Setup

1. Clone the CODI repository at the commit we used:
   ```bash
   git clone https://github.com/zhenyi4/codi.git
   cd codi
   git checkout 2c2314662c63e9f482ebc46614ffe9af17a241e5
   ```

2. Install dependencies following the CODI repository instructions.

3. Copy our training scripts into the CODI repo:
   ```bash
   cp /path/to/this-repo/args_codi/*.sh /path/to/codi/scripts/
   ```

4. Link or copy datasets into CODI's data directory:
   ```bash
   ln -s /path/to/this-repo/data /path/to/codi/data
   ```

#### Training Commands

From the CODI repository directory, run the training scripts:

```bash
bash scripts/train_{base_llm}_{dataset}_codi.sh
```

Where:
- `base_llm`: `gpt2`, `llama32-1b`
- `dataset`: `gsm8k`, `prosqa`, `prontoqa`

Note: There is no `train_gpt2_gsm8k_codi.sh` since we use the authors' pretrained checkpoint for that configuration.

These scripts assume 1 GPU. You may adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to adapt to your resources while maintaining the same effective batch size (128).

### CODI Multi-mode Models

Multi-mode CODI models can perform latent reasoning, explicit (CoT) reasoning, or direct answer depending on input tokens. Training these requires modifications to the CODI repository.

#### Setup

1. First, complete the CODI setup above (clone repo, install dependencies, link data).

2. Copy our modified training files into the CODI repo:
   ```bash
   cp /path/to/this-repo/src_codi_multimode/src/model.py /path/to/codi/src/
   cp /path/to/this-repo/src_codi_multimode/train.py /path/to/codi/
   cp /path/to/this-repo/src_codi_multimode/test.py /path/to/codi/
   ```

3. Copy our multi-mode training scripts:
   ```bash
   cp /path/to/this-repo/args_codi_multimode/*.sh /path/to/codi/scripts/
   ```

#### Training Commands

From the CODI repository directory:

```bash
bash scripts/train_{base_llm}_{dataset}_codi_multimode.sh
```

Where:
- `base_llm`: `gpt2`, `llama32-1b`
- `dataset`: `gsm8k`, `prosqa`, `prontoqa`

These scripts assume 1 GPU. You may adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to adapt to your resources while maintaining the same effective batch size (128).

## Dataset Performance

Evaluate all trained models on GSM8k-Aug, PrOntoQA, and ProsQA test sets.

Set any path to `null` in the config file (`model_paths.yaml`) to skip that model during evaluation.

### Running Evaluations

Run all evaluations and generate outputs:

```bash
# Dry-run to verify configuration
bash experiments/dataset_performance/run_all.sh --dry-run

# Run all evaluations
bash experiments/dataset_performance/run_all.sh
```

Options:
- `--config PATH` - Custom config file path (default: `model_paths.yaml`)
- `--output-dir PATH` - Output directory (default: `results/dataset_performance`)
- `--max-samples N` - Limit samples per evaluation (for testing)
- `--skip-standard` - Skip standard model evaluations (Table 1)
- `--skip-multimode` - Skip multimode evaluations (Figure 3)
- `--skip-summary` - Skip CSV and plot generation
- `--dry-run` - Print commands without executing

### Outputs

- `results/dataset_performance/*.json` - Individual evaluation results
- `results/dataset_performance/table1_summary.csv` - Table 1 accuracy summary
- `results/dataset_performance/figure3_multimode.png/pdf` - Figure 3 dumbbell plot
- `results/dataset_performance/figure3_multimode.csv` - Table 10 multimode accuracies (CSV version of Figure 3)

### Individual Scripts

You can also run components individually:

```bash
# Single model evaluation
python -m experiments.dataset_performance.run \
    --model_type coconut \
    --model_path checkpoints/gsm-coconut/checkpoint_33 \
    --dataset_path data/gsm_original_test.json

# Generate Table 1 CSV from existing results
python -m experiments.dataset_performance.summarize_results

# Generate Figure 3 plot from existing results
python -m experiments.dataset_performance.plot_multimode_dumbbell

# Generate Table 10 multimode CSV from existing results
python -m experiments.dataset_performance.summarize_multimode
```

## Early Stopping Experiment

Measures how much of the reasoning trace is needed to reach the final answer. For each sample, we vary the amount of reasoning (tokens for CoT, latent tokens for Coconut, iterations for CODI) and check when the model first outputs its final answer.

### Running Evaluations

Run all evaluations and generate outputs:

```bash
# Dry-run to verify configuration
bash experiments/early_stopping/run_all.sh --dry-run

# Run all evaluations
bash experiments/early_stopping/run_all.sh
```

Options:
- `--config PATH` - Custom config file path (default: `model_paths.yaml`)
- `--output-dir PATH` - Output directory (default: `results/early_stopping`)
- `--max-samples N` - Limit samples per evaluation (for testing)
- `--skip-gpt2` - Skip GPT-2 model evaluations
- `--skip-llama` - Skip Llama model evaluations
- `--skip-plots` - Skip plot generation
- `--dry-run` - Print commands without executing

### Outputs

- `results/early_stopping/*.json` - Individual evaluation results
- `results/early_stopping/figure2_early_stopping.png/pdf` - Figure 2 (stacked bar chart)
- `results/early_stopping/table11_early_stopping.csv` - Table 11 (first/stable match percentages)

### Individual Scripts

You can also run components individually:

```bash
# Single model evaluation
python -m experiments.early_stopping.run \
    --model_type coconut \
    --model_path checkpoints/gsm-coconut/checkpoint_33 \
    --dataset_path data/gsm_valid-gold-reasoning-trace_test.json \
    --top_k 10

# Generate Figure 2 and Table 11 from existing results
python -m experiments.early_stopping.plot \
    --base_llm combined \
    --results_dir results/early_stopping \
    --gpt2_gsm_cot_file <filename> \
    --gpt2_gsm_coconut_file <filename> \
    --gpt2_gsm_codi_file <filename> \
    --gpt2_prosqa_cot_file <filename> \
    --gpt2_prosqa_coconut_file <filename> \
    --gpt2_prosqa_codi_file <filename> \
    --gpt2_prontoqa_cot_file <filename> \
    --gpt2_prontoqa_coconut_file <filename> \
    --gpt2_prontoqa_codi_file <filename> \
    --llama_gsm_cot_file <filename> \
    --llama_gsm_coconut_file <filename> \
    --llama_gsm_codi_file <filename> \
    --llama_prosqa_cot_file <filename> \
    --llama_prosqa_coconut_file <filename> \
    --llama_prosqa_codi_file <filename> \
    --llama_prontoqa_cot_file <filename> \
    --llama_prontoqa_coconut_file <filename> \
    --llama_prontoqa_codi_file <filename>
```

## Gold Reasoning Trace Backtracking Experiment

Analyzes how gold reasoning traces are represented in the vocabulary projections of latent reasoning tokens. For each sample, we search the top-k vocabulary projections at each latent position to find operands and intermediate results from the gold solution.

### Running the Experiment

Run the main experiment to generate results and Figure 5:

```bash
# Dry-run to verify configuration
bash experiments/back_tracking_vp/run_all.sh --dry-run

# Run all evaluations
bash experiments/back_tracking_vp/run_all.sh
```

Options:
- `--config PATH` - Custom config file path (default: `model_paths.yaml`)
- `--output-dir PATH` - Output directory (default: `results/back_tracking_vp`)
- `--max-samples N` - Limit samples per evaluation (for testing)
- `--skip-gpt2` - Skip GPT-2 model evaluations
- `--skip-llama` - Skip Llama model evaluations
- `--skip-analysis` - Skip model runs (use existing results.json files)
- `--skip-plots` - Skip plot generation
- `--dry-run` - Print commands without executing

### Visualizing Specific Samples

After running the main experiment, visualize specific samples with HTML output:

```bash
# List available results files
ls results/back_tracking_vp/*/results.json

# Visualize specific samples by index
bash experiments/back_tracking_vp/visualize_samples.sh \
    --results-json results/back_tracking_vp/<subdir>/results.json \
    --sample-indices 0 5 10 15

# Auto-select samples (10 found, 10 not found)
bash experiments/back_tracking_vp/visualize_samples.sh \
    --results-json results/back_tracking_vp/<subdir>/results.json
```

Options:
- `--results-json PATH` - Path to results.json from run_all.sh (required)
- `--sample-indices N...` - Specific sample indices to visualize
- `--num-found N` - Number of "GT found" samples to auto-select (default: 10)
- `--num-not-found N` - Number of "GT not found" samples to auto-select (default: 10)
- `--dry-run` - Print commands without executing

### Outputs

- `results/back_tracking_vp/*/results.json` - Per-sample evaluation results
- `results/back_tracking_vp/*/summary.csv` - Aggregate statistics
- `results/back_tracking_vp/*/visualizations/*.html` - Sample visualizations (Figure 4 style)
- `results/back_tracking_vp/figure5_backtracking.png/pdf` - Figure 5 (aggregate results)

## Forward Chaining Experiment

Discovers computation trees by forward chaining through vocabulary projections.
For each sample, finds valid arithmetic steps and chains them to form trees
ending at the model's predicted answer.

### Running the Experiment

Run the main experiment to generate results and Figure 6:

```bash
# Dry-run to verify configuration
bash experiments/forward_chaining/run_all.sh --dry-run

# Run all evaluations
bash experiments/forward_chaining/run_all.sh
```

Options:
- `--config PATH` - Custom config file path (default: `model_paths.yaml`)
- `--output-dir PATH` - Output directory (default: `results/forward_chaining`)
- `--max-samples N` - Limit samples per evaluation (for testing)
- `--model-types T [T...]` - Only run these model types: `coconut`, `codi` (default: both)
- `--base-llms L [L...]` - Only run these base LLMs: `gpt2`, `llama32-1b` (default: both)
- `--required-passes N [N...]` - Only run these rp values: `1`, `2`, `3` (default: all)
- `--force` - Re-run even if results already exist (default: skips completed runs)
- `--skip-plots` - Skip plot generation
- `--dry-run` - Print commands without executing

Runs that already have a `results.json` are skipped automatically, so re-running after a timeout resumes from where it left off.

### Outputs

- `results/forward_chaining/*/results.json` - Per-sample evaluation results
- `results/forward_chaining/figure6_forward_chaining.png/pdf` - Figure 6

## Citation

If you use this code base in your research, please cite our paper with the following BibTeX entry:

```bibtex
@misc{dilgren2026latentreasoningmodelseasily,
      title={Are Latent Reasoning Models Easily Interpretable?},
      author={Connor Dilgren and Sarah Wiegreffe},
      year={2026},
      eprint={2604.04902},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.04902},
}
```

## License

This code is released under the MIT License.

