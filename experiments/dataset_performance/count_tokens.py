"""Count output tokens (reasoning + answer) for CoT, Coconut, and CoDi result files.

Replicates the token counting logic from run_save_eval.py:
- CoT: reasoning tokens = tokenize the text chain-of-thought; answer tokens = tokenize "### <answer>" + EOS
- Coconut: reasoning tokens = count of <|latent|> tokens; answer tokens = same as CoT
- CoDi: reasoning tokens = count of <|latent|> tokens; answer tokens = tokenize "The answer is: <answer>" + EOS
"""

import json
import os
from transformers import AutoTokenizer


def count_tokens_cot(output_text, tokenizer):
    """CoT: reasoning is text between question (first line) and ###."""
    lines = output_text.split("\n")
    after_question = "\n".join(lines[1:])

    if "###" in after_question:
        idx = after_question.index("###")
        cot_output = after_question[:idx]
        answer_output = after_question[idx:]  # includes "### ..."
    else:
        cot_output = after_question
        answer_output = ""

    reasoning_tokens = len(tokenizer.encode(cot_output, add_special_tokens=False))
    answer_tokens = len(tokenizer.encode(answer_output, add_special_tokens=False)) + 1  # +1 for EOS
    return reasoning_tokens, answer_tokens


def count_tokens_coconut(output_text, tokenizer):
    """Coconut: reasoning is count of <|latent|> tokens; answer after ###."""
    reasoning_tokens = output_text.count("<|latent|>")

    if "###" in output_text:
        idx = output_text.index("###")
        answer_output = output_text[idx:]  # includes "### ..."
    else:
        answer_output = ""

    answer_tokens = len(tokenizer.encode(answer_output, add_special_tokens=False)) + 1  # +1 for EOS
    return reasoning_tokens, answer_tokens


def count_tokens_codi(output_text, tokenizer):
    """CoDi: reasoning is count of <|latent|> tokens; answer starts with 'The answer is:'."""
    reasoning_tokens = output_text.count("<|latent|>")

    if "The answer is:" in output_text:
        idx = output_text.index("The answer is:")
        answer_output = output_text[idx:]  # includes "The answer is: ..."
    else:
        answer_output = ""

    answer_tokens = len(tokenizer.encode(answer_output, add_special_tokens=False)) + 1  # +1 for EOS
    return reasoning_tokens, answer_tokens


def count_tokens_no_cot(output_text, tokenizer):
    """No-CoT: no reasoning tokens, just the answer after ###."""
    reasoning_tokens = 0

    if "###" in output_text:
        idx = output_text.index("###")
        answer_output = output_text[idx:]  # includes "### ..."
    else:
        answer_output = ""

    answer_tokens = len(tokenizer.encode(answer_output, add_special_tokens=False)) + 1  # +1 for EOS
    return reasoning_tokens, answer_tokens


def count_tokens_multimode_codi(output_text, tokenizer):
    """
    Multimode CODI token counting.

    - Direct/Verbalized: reasoning = text between <|bocot|>/<|eocot|> or count of explicit CoT tokens
    - Latent: reasoning = count of <|latent|> placeholders (or <|bocot|> placeholders)
    - Answer = text after <|eocot|> or "The answer is:"
    """
    # Count latent placeholders (used in latent mode)
    latent_count = output_text.count("<|latent|>") + output_text.count("<|bocot|>")

    # For verbalized mode, count actual CoT tokens between <|bocot|> and <|eocot|>
    if "<|bocot|>" in output_text and "<|eocot|>" in output_text:
        bocot_idx = output_text.index("<|bocot|>") + len("<|bocot|>")
        eocot_idx = output_text.index("<|eocot|>")
        cot_text = output_text[bocot_idx:eocot_idx].strip()
        # If there's actual text (not just latent markers), count those tokens
        cot_text_clean = cot_text.replace("<|latent|>", "").replace("<|bocot|>", "").strip()
        if cot_text_clean:
            reasoning_tokens = len(tokenizer.encode(cot_text_clean, add_special_tokens=False))
        else:
            # Latent mode: count placeholders
            reasoning_tokens = latent_count
    else:
        # Fallback: just count latent markers
        reasoning_tokens = latent_count

    # Extract answer (after <|eocot|> or "The answer is:")
    if "<|eocot|>" in output_text:
        idx = output_text.index("<|eocot|>") + len("<|eocot|>")
        answer_output = output_text[idx:]
    elif "The answer is:" in output_text:
        idx = output_text.index("The answer is:")
        answer_output = output_text[idx:]
    else:
        answer_output = ""

    answer_tokens = len(tokenizer.encode(answer_output, add_special_tokens=False)) + 1  # +1 for EOS
    return reasoning_tokens, answer_tokens


COUNTER_FNS = {
    "cot": count_tokens_cot,
    "coconut": count_tokens_coconut,
    "codi": count_tokens_codi,
    "no_cot": count_tokens_no_cot,
    "multimode_codi": count_tokens_multimode_codi,
}

RESULT_FILES = [
    "results/dataset_performance/cot_gsm-cot_gsm_test_20251231_165037.json",
    "results/dataset_performance/cot_prontoqa-cot_prontoqa_test_20260128_180753.json",
    "results/dataset_performance/cot_prosqa-cot_prosqa_test_20260128_180203.json",
    "results/dataset_performance/coconut_gsm-coconut_gsm_test_20251231_164515.json",
    "results/dataset_performance/coconut_prontoqa-coconut_prontoqa_test_20260128_180833.json",
    "results/dataset_performance/coconut_prosqa-coconut_prosqa_test_20260128_180251.json",
    "results/dataset_performance/codi_zen-E_gsm_test_20260101_233242.json",
    "results/dataset_performance/codi_seed_11_prontoqa_test_20260128_180408.json",
    "results/dataset_performance/codi_seed_11_prosqa_test_20260128_221532.json",
]


def process_file(filepath, tokenizer=None):
    with open(filepath) as f:
        data = json.load(f)

    # Use tokenizer from config if not provided
    if tokenizer is None:
        model_id = data["config"].get("model_id", "openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_type = data["config"]["model_type"]
    count_fn = COUNTER_FNS[model_type]

    total_reasoning = 0
    total_answer = 0
    n = len(data["results"])

    for result in data["results"]:
        reasoning, answer = count_fn(result["output"], tokenizer)
        total_reasoning += reasoning
        total_answer += answer

    total_output = total_reasoning + total_answer
    return {
        "file": os.path.basename(filepath),
        "model_type": model_type,
        "dataset": os.path.basename(data["config"]["dataset_path"]).replace(".json", ""),
        "n_samples": n,
        "accuracy": data["metrics"]["accuracy"],
        "avg_reasoning_tokens": total_reasoning / n,
        "avg_answer_tokens": total_answer / n,
        "avg_total_output_tokens": total_output / n,
        "total_reasoning_tokens": total_reasoning,
        "total_answer_tokens": total_answer,
        "total_output_tokens": total_output,
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rows = []
    for relpath in RESULT_FILES:
        filepath = os.path.join(root, relpath)
        if not os.path.exists(filepath):
            print(f"WARNING: {relpath} not found, skipping")
            continue
        rows.append(process_file(filepath))

    # Print summary table
    header = f"{'Model':<10} {'Dataset':<16} {'N':>5} {'Acc':>6} {'Avg Reasoning':>15} {'Avg Answer':>12} {'Avg Total':>11}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['model_type']:<10} {r['dataset']:<16} {r['n_samples']:>5} "
            f"{r['accuracy']:>6.1%} {r['avg_reasoning_tokens']:>15.1f} "
            f"{r['avg_answer_tokens']:>12.1f} {r['avg_total_output_tokens']:>11.1f}"
        )
    print(sep)


if __name__ == "__main__":
    main()
