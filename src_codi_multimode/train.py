# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import Trainer, TrainerCallback
from math import ceil
from peft import LoraConfig, TaskType
from datasets import load_dataset

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            self.log({
                "loss": loss.item(),
                "ce_loss": outputs["ce_loss"],
                "distill_loss": outputs["distill_loss"],
                "ref_ce_loss": outputs["ref_ce_loss"],
                "direct_ce_loss": outputs.get("direct_ce_loss", 0),
            })
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=1024,#training_args.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_for_eval(sentence: str, data_name: str):
    """Extract answer from generated text for evaluation."""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "prontoqa" in data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                return None  # Could not extract answer
        elif "prosqa" in data_name.lower():
            if "The answer is:" in sentence:
                return sentence.split("The answer is:")[-1].strip()
            return sentence.strip()
        return None
    # For numeric answers, use the last number
    return float(pred[-1])


def run_inference(model, tokenizer, valid_data, inference_mode, data_args, training_args):
    """Run inference on validation data and return accuracy.

    Args:
        model: The CODI model
        tokenizer: The tokenizer
        valid_data: List of validation examples
        inference_mode: "latent", "verbalized", or "direct"
        data_args: Data arguments
        training_args: Training arguments

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    model.eval()
    device = next(model.parameters()).device

    # Prepare questions and answers
    questions = [f"{example['question'].strip().replace('  ', ' ')}" for example in valid_data]
    gold_answers = []
    for example in valid_data:
        ans = example['answer']
        if isinstance(ans, bool):
            gold_answers.append(ans)
        elif ans in ["True", "False"]:
            gold_answers.append(ans == "True")
        elif "prosqa" in data_args.data_name.lower():
            gold_answers.append(str(ans).strip())
        else:
            gold_answers.append(ans)

    # Batch size for inference
    batch_size = 8  # Use smaller batch size for validation
    eval_steps = math.ceil(len(questions) / batch_size)

    # Prepare batches
    question_data = []
    for i in range(eval_steps):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        batch = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding="longest",
        )

        # Construct control tokens based on inference mode
        if inference_mode == "direct":
            if model.use_eot_id:
                tokens = [model.eot_id, model.eocot_id]
            else:
                tokens = [model.eocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens
        elif inference_mode == "verbalized":
            if model.use_eot_id:
                tokens = [model.eot_id, model.bocot_id]
            else:
                tokens = [model.bocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens
        else:  # latent
            if model.use_eot_id:
                tokens = [model.eot_id, model.bocot_id]
            else:
                tokens = [model.bocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens

        control_tokens = torch.tensor(tokens, dtype=torch.long).expand(batch["input_ids"].size(0), len(tokens))
        batch["input_ids"] = torch.cat((batch["input_ids"], control_tokens), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(control_tokens)), dim=1)
        batch['inference_mode'] = inference_mode

        question_data.append({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
    }

    pred_answers = []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in question_data:
            curr_batch_size = batch["input_ids"].size(0)
            inf_mode = batch.get('inference_mode', 'latent')

            # Encode the question
            past_key_values = None
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"]
            )
            past_key_values = outputs.past_key_values

            if inf_mode == "latent":
                # Latent mode: run latent iterations
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

                inf_latent_iterations = training_args.inf_latent_iterations
                for _ in range(inf_latent_iterations):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

                # After latent iterations, add eocot_id
                if training_args.remove_eos:
                    eocot_emb = model.get_embd(model.codi, model.model_name)(
                        torch.tensor([model.eocot_id], dtype=torch.long, device=device)
                    ).unsqueeze(0)
                else:
                    eocot_emb = model.get_embd(model.codi, model.model_name)(
                        torch.tensor([model.eocot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)
                    ).unsqueeze(0)

                eocot_emb = eocot_emb.expand(curr_batch_size, -1, -1)
                outputs = model.codi(
                    inputs_embeds=eocot_emb,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

            # Autoregressive generation
            finished = torch.zeros(curr_batch_size, dtype=torch.bool, device=device)
            pred_tokens = [[] for _ in range(curr_batch_size)]

            logits = outputs.logits[:, -1, :]
            first_iteration = True

            for _ in range(gen_kwargs["max_new_tokens"]):
                if not first_iteration:
                    out = model.codi(
                        inputs_embeds=output_emb,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]

                first_iteration = False

                # Greedy decoding for validation (deterministic)
                next_token_ids = torch.argmax(logits, dim=-1)
                if curr_batch_size > 1 and next_token_ids.dim() > 0:
                    next_token_ids = next_token_ids.squeeze(-1) if next_token_ids.dim() > 1 else next_token_ids

                # Handle EOS for each sequence
                for b in range(curr_batch_size):
                    if not finished[b]:
                        token_id = next_token_ids[b].item() if curr_batch_size > 1 else next_token_ids.item()
                        pred_tokens[b].append(token_id)
                        if token_id == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output_emb = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1)

            # Decode predictions
            for pred_token in pred_tokens:
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                pred_answer = extract_answer_for_eval(decoded_pred, data_args.data_name)
                pred_answers.append(pred_answer)

    # Compute accuracy
    correct = 0
    for pred, gold in zip(pred_answers, gold_answers):
        if pred == gold:
            correct += 1

    accuracy = correct / len(gold_answers) if gold_answers else 0.0
    model.train()
    return accuracy


class ValidationAccuracyCallback(TrainerCallback):
    """Callback to run validation at end of each epoch and save best checkpoint."""

    def __init__(self, model, tokenizer, valid_data, data_args, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.valid_data = valid_data
        self.data_args = data_args
        self.training_args = training_args
        self.best_harmonic_mean = 0.0
        self.best_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        # Run inference for each mode
        print(f"\n{'='*60}")
        print(f"Running validation at epoch {state.epoch:.0f}...")
        print(f"{'='*60}")

        accuracies = {}
        for mode in ["latent", "verbalized", "direct"]:
            acc = run_inference(
                self.model, self.tokenizer, self.valid_data,
                mode, self.data_args, self.training_args
            )
            accuracies[mode] = acc
            print(f"  {mode}: {acc:.2%}")

        # Compute harmonic mean
        if all(a > 0 for a in accuracies.values()):
            harmonic_mean = 3 / (1/accuracies["latent"] + 1/accuracies["verbalized"] + 1/accuracies["direct"])
        else:
            harmonic_mean = 0.0

        print(f"\nEpoch {state.epoch:.0f}: latent={accuracies['latent']:.2%}, "
              f"verbalized={accuracies['verbalized']:.2%}, direct={accuracies['direct']:.2%}, "
              f"harmonic_mean={harmonic_mean:.2%}")

        # Save if best
        if harmonic_mean > self.best_harmonic_mean:
            self.best_harmonic_mean = harmonic_mean
            self.best_epoch = state.epoch
            # Save best model (same format as test.py expects)
            best_checkpoint_dir = f"{args.output_dir}/best_checkpoint"
            os.makedirs(best_checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), f"{best_checkpoint_dir}/pytorch_model.bin")
            print(f"New best model saved to {best_checkpoint_dir}! Harmonic mean: {harmonic_mean:.2%}")

        print(f"{'='*60}\n")

        return control


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )


    model = CODI(model_args, training_args, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token is None:
        # Use eos_token as pad_token (standard practice)
        # The custom pad token added in model.py is unused but harmless
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens to tokenizer so it can decode them during validation
    # Order must match model.py: bocot_id = vocab_size, eocot_id = vocab_size + 1
    tokenizer.add_tokens(["<bocot>", "<eocot>"], special_tokens=True)

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()

    def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        eot_id: int,      # end of prompt (None for GPT-2)
        bocot_id: int,    # begin CoT
        eocot_id: int,    # end CoT
        use_eot_id: bool, # whether to include eot_id (True for Llama, False for GPT-2)
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        # Direct answer: prompt + [eot_id +] eocot_id + answer (no bocot_id, skip CoT)
        if use_eot_id:
            direct_control_tokens = torch.tensor([eot_id, eocot_id])
            num_direct_control = 2
        else:
            direct_control_tokens = torch.tensor([eocot_id])
            num_direct_control = 1
        direct_input_ids = [
            torch.cat([x, direct_control_tokens, z]).to(torch.long)
            for x, z in zip(sources_id, answers_id)
        ]

        # Labels for direct: mask prompt + control tokens
        direct_labels = []
        for direct_input, source in zip(direct_input_ids, sources_id):
            lbl = direct_input.clone()
            lbl[:len(source) + num_direct_control] = -100
            direct_labels.append(lbl)

        # Verbalized CoT (teacher): prompt + [eot_id +] bocot_id + cot + eocot_id + answer
        if use_eot_id:
            begin_cot_tokens = torch.tensor([eot_id, bocot_id])
            num_begin_cot = 2
        else:
            begin_cot_tokens = torch.tensor([bocot_id])
            num_begin_cot = 1
        ref_input_ids = [
            torch.cat([
                x,                                    # prompt
                begin_cot_tokens,                     # begin CoT
                y,                                    # cot
                torch.tensor([eocot_id]),            # end CoT
                z                                     # answer
            ]).to(torch.long)
            for x, y, z in zip(sources_id, cot_id, answers_id)
        ]
        ref_labels = []
        for ref_input, source in zip(ref_input_ids, sources_id):
            lbl = ref_input.clone()
            lbl[:len(source) + num_begin_cot] = -100  # mask prompt + control tokens
            ref_labels.append(lbl)

        # Latent CoT (student encoder): prompt + [eot_id +] bocot_id
        if use_eot_id:
            encoder_input_ids = [torch.tensor(x.numpy().tolist() + [eot_id, bocot_id], dtype=torch.long) for x in sources_id]
        else:
            encoder_input_ids = [torch.tensor(x.numpy().tolist() + [bocot_id], dtype=torch.long) for x in sources_id]

        # Latent CoT (student decoder): eocot_id + answer
        if training_args.remove_eos:
            decoder_input_ids = [torch.tensor([eocot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            decoder_input_ids = [torch.tensor([eocot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]

        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in decoder_input_ids]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in decoder_input_ids]
        return dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=decoder_input_ids,
            ref_answer_position=ref_answer_position,
            model_answer_position=model_answer_position,
            ref_eos_position=ref_eos_position,
            model_eos_position=model_eos_position,
            ref_labels=ref_labels,
            direct_input_ids=direct_input_ids,
            direct_labels=direct_labels,
        )


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, eot_id, bocot_id, eocot_id, use_eot_id):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")

            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            for num_iter, example in enumerate(raw_data):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name: # icot-full (GSM8k-Aug-NL)
                    # bad data
                    if example["answer"] is None: # or example["response"] is None:
                        continue
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]

                    answer = example['answer'].split(' ')[-1]
                    if not answer[0].isdigit():
                        continue
                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    
                    if cot:
                        cot = ". ".join(cot)+".\n"
                    else:
                        cot = ""
                    cots.append(cot)
                    answers.append(answer)
                elif "icot" in self.data_name: # icot (GSM8k-Aug)
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    if not training_args.include_last_cot:
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.num_latent):
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in data_args.data_name or "prosqa" in data_args.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(questions, cots, answers, tokenizer, eot_id, bocot_id, eocot_id, use_eot_id)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels, direct_input_ids, direct_labels = \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels", "direct_input_ids", "direct_labels"))

            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)

            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX)

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            # Pad direct answer sequences
            direct_input_ids = torch.nn.utils.rnn.pad_sequence(direct_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            direct_labels = torch.nn.utils.rnn.pad_sequence(direct_labels, batch_first=True, padding_value=IGNORE_INDEX)

            return dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
                direct_input_ids=direct_input_ids,
                direct_labels=direct_labels,
                direct_attention_mask=direct_input_ids.ne(self.tokenizer.pad_token_id),
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        if "icot" in data_args.data_name:
            if 'full' in data_args.data_name:
                dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
            else:
                dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_args.data_name:
            with open("data/prontoqa_train.json") as f:
                train_data = json.load(f)
            with open("data/prontoqa_valid.json") as f:
                valid_data = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=train_data, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator, valid_data=valid_data)
        elif "prosqa_single_token" in data_args.data_name:
            with open("data/prosqa_train_single_token_name.json") as f:
                train_data = json.load(f)
            with open("data/prosqa_valid.json") as f:
                valid_data = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=train_data, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator, valid_data=valid_data)
        elif "prosqa" in data_args.data_name:
            with open("data/prosqa_train.json") as f:
                train_data = json.load(f)
            with open("data/prosqa_valid.json") as f:
                valid_data = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=train_data, tokenizer=tokenizer, eot_id=model.eot_id, bocot_id=model.bocot_id, eocot_id=model.eocot_id, use_eot_id=model.use_eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator, valid_data=valid_data)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Extract valid_data for the callback (if available)
    valid_data = data_module.pop('valid_data', None)

    # Create callbacks list
    callbacks = []
    if valid_data is not None:
        validation_callback = ValidationAccuracyCallback(
            model=model,
            tokenizer=tokenizer,
            valid_data=valid_data,
            data_args=data_args,
            training_args=training_args,
        )
        callbacks.append(validation_callback)
        print(f"Validation callback registered with {len(valid_data)} validation examples")

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module
    )
    trainer.train()

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
