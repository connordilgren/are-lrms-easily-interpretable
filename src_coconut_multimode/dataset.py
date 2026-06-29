# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional
import os

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    sample_token_counts=None,
):
    """
    for inference / eval

    Args:
        sample_token_counts: Optional dict mapping sample_idx to exact number of latent tokens to use.
                           If provided, overrides the stage-based calculation for those samples.
    """

    def process_dataset(sample):

        sample_idx = sample["idx"]

        # Check if we have an exact token count for this sample
        if sample_token_counts is not None and sample_idx in sample_token_counts:
            k = sample_token_counts[sample_idx]
        else:
            # Use original stage-based calculation
            if configs.pad_latent_to_max:
                max_latent_stage = configs.max_latent_stage
            else:
                max_latent_stage = min(
                    configs.max_latent_stage, len(sample["steps_tokenized"])
                )

            k = min(max_latent_stage, scheduled_stage)
            k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    """for training"""

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset


def load_cot_token_counts(cot_results_path):
    """Load CoT evaluation results and create a mapping from sample_idx to reasoning token count."""
    if not os.path.exists(cot_results_path):
        print(f"Warning: CoT results file not found at {cot_results_path}. Using default token limits.")
        return {}
    
    try:
        with open(cot_results_path, 'r') as f:
            cot_data = json.load(f)
        
        token_counts = {}
        for result in cot_data.get('detailed_results', []):
            sample_idx = result['sample_idx']
            gt_tokens = result['model_reasoning_tokens']
            token_counts[sample_idx] = gt_tokens
            
        print(f"Loaded CoT token counts for {len(token_counts)} samples from {cot_results_path}")
        return token_counts
        
    except Exception as e:
        print(f"Error loading CoT results: {e}. Using default token limits.")
        return {}


def get_multimode_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    eot_id,
    bocot_id,
    eocot_id,
    latent_id,
    shuffle=False,
    use_eot=True,
):
    """
    For multi-mode training. Creates three versions of each sample:
    - Direct: mask prompt + [eot_id] + <|eocot|>, train on answer
    - Verbalized: mask prompt + [eot_id] + <|bocot|>, train on full CoT + <|eocot|> + answer
    - Latent: mask prompt + [eot_id] + <|bocot|> + latent tokens, train on remaining CoT + <|eocot|> + answer

    Args:
        use_eot: If True (Llama), include eot_id in sequences. If False (GPT-2), skip eot_id.
    """

    def process_dataset(sample):
        question_tokens = sample["question_tokenized"]
        steps_tokenized = sample["steps_tokenized"]
        answer_tokens = sample["answer_tokenized"]

        # Flatten all steps for verbalized mode
        all_steps_flat = list(itertools.chain.from_iterable(steps_tokenized))

        # Determine latent stage (same logic as get_cot_latent_dataset)
        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(steps_tokenized) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(steps_tokenized), configs.max_latent_stage
                )
        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        n_latent_tokens *= configs.c_thought

        # Remaining steps after latent tokens
        remaining_steps_flat = list(
            itertools.chain.from_iterable(steps_tokenized[n_skip_steps:])
        )

        # Control tokens after prompt depend on use_eot
        # GPT-2: 1 control token (no eot)
        # Llama: 2 control tokens (eot + mode token)
        eot_tokens = [eot_id] if use_eot else []
        n_control_tokens = 1 + (1 if use_eot else 0)  # mode token + optional eot

        # === Direct mode ===
        # Format: {prompt}[eot]<|eocot|>{answer}
        # Mask: prompt + [eot] + <|eocot|>
        # Train on: answer
        direct_input_ids = (
            question_tokens + eot_tokens + [eocot_id] + answer_tokens
        )
        direct_mask_len = len(question_tokens) + n_control_tokens  # prompt + [eot] + eocot
        direct_labels = (
            [-100] * direct_mask_len + direct_input_ids[direct_mask_len:]
        )

        # === Verbalized mode ===
        # Format: {prompt}[eot]<|bocot|>{full_cot}<|eocot|>{answer}
        # Mask: prompt + [eot] + <|bocot|>
        # Train on: full CoT + <|eocot|> + answer
        verbalized_input_ids = (
            question_tokens + eot_tokens + [bocot_id] + all_steps_flat + [eocot_id] + answer_tokens
        )
        verbalized_mask_len = len(question_tokens) + n_control_tokens  # prompt + [eot] + bocot
        verbalized_labels = (
            [-100] * verbalized_mask_len + verbalized_input_ids[verbalized_mask_len:]
        )

        # === Latent mode ===
        # Format: {prompt}[eot]<|bocot|><|latent|>...<|latent|>{remaining_cot}<|eocot|>{answer}
        # Mask: prompt + [eot] + <|bocot|> + latent tokens
        # Train on: remaining CoT + <|eocot|> + answer
        latent_input_ids = (
            question_tokens
            + eot_tokens
            + [bocot_id]
            + [latent_id] * n_latent_tokens
            + remaining_steps_flat
            + [eocot_id]
            + answer_tokens
        )
        latent_mask_len = len(question_tokens) + n_control_tokens + n_latent_tokens  # prompt + [eot] + bocot + latents
        latent_labels = (
            [-100] * latent_mask_len + latent_input_ids[latent_mask_len:]
        )

        return {
            "direct_input_ids": direct_input_ids,
            "direct_labels": direct_labels,
            "direct_attention_mask": [1] * len(direct_input_ids),
            "verbalized_input_ids": verbalized_input_ids,
            "verbalized_labels": verbalized_labels,
            "verbalized_attention_mask": [1] * len(verbalized_input_ids),
            "latent_input_ids": latent_input_ids,
            "latent_labels": latent_labels,
            "latent_attention_mask": [1] * len(latent_input_ids),
            "idx": sample["idx"],
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset


@dataclass
class MultiModeCollator:
    """
    Collator for multi-mode training. Pads each mode independently.
    Handles latent mode KV-cache-friendly padding (same as MyCollator).
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        batch = {}

        for mode in ["direct", "verbalized", "latent"]:
            input_ids_key = f"{mode}_input_ids"
            labels_key = f"{mode}_labels"
            attention_mask_key = f"{mode}_attention_mask"

            # Extract mode-specific features
            mode_features = [
                {
                    "input_ids": f[input_ids_key],
                    "labels": f[labels_key],
                    "attention_mask": f[attention_mask_key],
                }
                for f in features
            ]

            # For latent mode, apply KV-cache-friendly padding
            if mode == "latent" and self.latent_id is not None:
                earliest_latent = [
                    mf["input_ids"].index(self.latent_id)
                    for mf in mode_features
                    if self.latent_id in mf["input_ids"]
                ]

                if len(earliest_latent) > 0:
                    latest_earliest_latent = max(earliest_latent)
                    for mf in mode_features:
                        if self.latent_id in mf["input_ids"]:
                            n_tok_pad = latest_earliest_latent - mf["input_ids"].index(
                                self.latent_id
                            )
                        else:
                            n_tok_pad = 0
                        mf["position_ids"] = [0] * n_tok_pad + list(
                            range(len(mf["input_ids"]))
                        )
                        mf["input_ids"] = [
                            self.tokenizer.pad_token_id
                        ] * n_tok_pad + mf["input_ids"]
                        mf["labels"] = [self.label_pad_token_id] * n_tok_pad + mf[
                            "labels"
                        ]
                        mf["attention_mask"] = [0] * n_tok_pad + mf["attention_mask"]

            # Pad input_ids and attention_mask
            non_label_features = [
                {k: v for k, v in mf.items() if k not in ["labels", "position_ids"]}
                for mf in mode_features
            ]

            padded = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                non_label_features,
                padding=True,
                pad_to_multiple_of=None,
                return_tensors="pt",
            )

            batch[input_ids_key] = padded["input_ids"].clone()
            batch[attention_mask_key] = padded["attention_mask"].clone()

            # Pad labels
            labels = [mf["labels"] for mf in mode_features]
            max_label_length = max(len(l) for l in labels)
            batch[labels_key] = torch.tensor(
                [
                    label + [self.label_pad_token_id] * (max_label_length - len(label))
                    for label in labels
                ],
                dtype=torch.int64,
            )

            # Pad position_ids if present (for latent mode)
            if "position_ids" in mode_features[0]:
                position_ids = [mf["position_ids"] for mf in mode_features]
                max_pos_length = max(len(p) for p in position_ids)
                batch[f"{mode}_position_ids"] = torch.tensor(
                    [
                        pos + [0] * (max_pos_length - len(pos))
                        for pos in position_ids
                    ],
                    dtype=torch.int64,
                )
            else:
                # Generate position_ids for non-latent modes
                seq_len = batch[input_ids_key].shape[1]
                batch_size = batch[input_ids_key].shape[0]
                batch[f"{mode}_position_ids"] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).clone()

        # Keep idx
        batch["idx"] = [f["idx"] for f in features]

        return batch


def get_multimode_eval_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    eot_id,
    bocot_id,
    eocot_id,
    latent_id,
    mode="latent",
    use_eot=True,
):
    """
    For inference/evaluation in a specific mode.
    - Direct: {question}[eot]<|eocot|> → generate answer
    - Verbalized: {question}[eot]<|bocot|> → generate CoT + <|eocot|> + answer
    - Latent: {question}[eot]<|bocot|><|latent|>... → run hidden state forward, generate remaining + <|eocot|> + answer

    Args:
        use_eot: If True (Llama), include eot_id in sequences. If False (GPT-2), skip eot_id.
    """

    def process_dataset(sample):
        question_tokens = sample["question_tokenized"]

        # Control tokens after prompt depend on use_eot
        eot_tokens = [eot_id] if use_eot else []

        if mode == "direct":
            # Format: {question}[eot]<|eocot|>
            tokens = question_tokens + eot_tokens + [eocot_id]

        elif mode == "verbalized":
            # Format: {question}[eot]<|bocot|>
            tokens = question_tokens + eot_tokens + [bocot_id]

        elif mode == "latent":
            # Format: {question}[eot]<|bocot|><|latent|>...
            if configs.pad_latent_to_max:
                max_latent_stage = configs.max_latent_stage
            else:
                max_latent_stage = min(
                    configs.max_latent_stage, len(sample["steps_tokenized"])
                )
            k = min(max_latent_stage, scheduled_stage)
            k *= configs.c_thought

            tokens = question_tokens + eot_tokens + [bocot_id] + [latent_id] * k

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset.map(
        process_dataset, remove_columns=list(base_dataset.features), num_proc=32
    )
