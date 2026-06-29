#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import hashlib
import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluation(model_args, data_args, training_args):
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
    else:
        raise NotImplementedError
    
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # new_state_dict = { k.replace("coconut", "codi"): v for k, v in state_dict.items() }
    # torch.save(new_state_dict, "/scratch/prj/inf_multimodal_qa/scratch_tmp/transfer/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        # Use eos_token as pad_token (common practice for GPT-2, Llama, etc.)
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens to tokenizer so it can decode them
    # Order must match model.py: bocot_id = vocab_size, eocot_id = vocab_size + 1
    tokenizer.add_tokens(["<bocot>", "<eocot>"], special_tokens=True)

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    print(f"\n=== Evaluation using inference mode: {training_args.inference_mode} ===\n")
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    elif "prosqa" == data_args.data_name:
        with open("data/prosqa_test.json") as f:
            test_set = json.load(f)
    elif "prontoqa" == data_args.data_name:
        with open("data/prontoqa_test.json") as f:
            test_set = json.load(f)
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    # # MODIFIED: Only process first sample for debugging
    # test_set = [test_set[0]]
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = []

    # get answer (format depends on dataset)
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        # prosqa: answers are entity names like "shumpus" or "hilpus"
        if "prosqa" in data_args.data_name.lower():
            answer.append(str(example).strip())
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        # Construct input based on inference mode
        # For Llama: include eot_id (instruction-tuned, has semantic meaning)
        # For GPT-2: skip eot_id (not instruction-tuned)
        inference_mode = training_args.inference_mode
        if inference_mode == "direct":
            # Direct: prompt + [eot_id +] eocot_id -> answer (no bocot_id, skip CoT)
            if model.use_eot_id:
                tokens = [model.eot_id, model.eocot_id]
            else:
                tokens = [model.eocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens
            control_tokens = torch.tensor(tokens, dtype=torch.long).expand(batch["input_ids"].size(0), len(tokens))
        elif inference_mode == "verbalized":
            # Verbalized: prompt + [eot_id +] bocot_id -> generate cot + eocot_id + answer
            if model.use_eot_id:
                tokens = [model.eot_id, model.bocot_id]
            else:
                tokens = [model.bocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens
            control_tokens = torch.tensor(tokens, dtype=torch.long).expand(batch["input_ids"].size(0), len(tokens))
        else:  # latent (default)
            # Latent: prompt + [eot_id +] bocot_id -> [latent iterations] -> eocot_id + answer
            if model.use_eot_id:
                tokens = [model.eot_id, model.bocot_id]
            else:
                tokens = [model.bocot_id]
            if not training_args.remove_eos:
                tokens = [tokenizer.eos_token_id] + tokens
            control_tokens = torch.tensor(tokens, dtype=torch.long).expand(batch["input_ids"].size(0), len(tokens))
        batch["input_ids"] = torch.cat((batch["input_ids"], control_tokens), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(control_tokens)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        batch['inference_mode'] = inference_mode

        # DEBUG: Print first batch info
        if i == 0:
            print(f"\nDEBUG - First batch tokenization:")
            print(f"  Inference mode: {inference_mode}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  First question input IDs: {batch['input_ids'][0].tolist()}")
            print(f"  EOT ID (end of prompt): {model.eot_id}")
            print(f"  BOCOT ID (begin CoT): {model.bocot_id}")
            print(f"  EOCOT ID (end CoT): {model.eocot_id}")
            print(f"  Vocab size: {model.codi.config.vocab_size}")

        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []
    #set_seed(42)
    gating_probs_sums = None
    len_cot = []
    model.eval()
    attn_to_latent_list = []
    
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        inference_mode = batch.get('inference_mode', 'latent')
        with torch.no_grad():
            # encode the question
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values

            if inference_mode == "direct":
                # Direct mode: no latent iterations, eot_id + eocot_id already in input
                # Just start generating answer
                pass  # No latent processing needed
            elif inference_mode == "verbalized":
                # Verbalized mode: generate CoT tokens, then eocot_id, then answer
                pass  # Just start generating, no latent iterations
            else:  # latent mode
                # Latent mode: run latent iterations
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

                inf_latent_iterations = training_args.inf_latent_iterations
                for i in range(inf_latent_iterations):
                    # decode the latent embeddings
                    outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

                # After latent iterations, add eocot_id for latent mode
                if training_args.remove_eos:
                    eocot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eocot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
                else:
                    eocot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eocot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)

                eocot_emb = eocot_emb.expand(batch["input_ids"].size(0), -1, -1)
                outputs = model.codi(inputs_embeds=eocot_emb, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values

            # Start autoregressive generation
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
            pred_tokens = [[] for _ in range(batch_size)]

            # Get the first token from the last output
            logits = outputs.logits[:, -1, :]
            first_iteration = True

            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1

                if not first_iteration:
                    out = model.codi(
                            inputs_embeds=output,
                            output_hidden_states=False,
                            attention_mask=None,
                            use_cache=True,
                            output_attentions=False,
                            past_key_values=past_key_values
                        )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]

                first_iteration = False

                # implement the sampling process
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                    if batch_size > 1:
                        next_token_ids = next_token_ids.squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        token_id = next_token_ids[b].item() if batch_size > 1 else next_token_ids.item()
                        pred_tokens[b].append(token_id)
                        if token_id == tokenizer.eos_token_id:
                            finished[b] = True

                # DEBUG: Print first few generated tokens for first question
                if step == 0 and i < 5 and len(pred_tokens[0]) <= 10:
                    token_id = next_token_ids[0].item() if batch_size > 1 else next_token_ids.item()
                    print(f"  Generated token {len(pred_tokens[0])}: {token_id} = '{tokenizer.decode([token_id])}'")
                    # Print top-5 logits at token 5
                    if len(pred_tokens[0]) == 5:
                        top_logits, top_indices = torch.topk(logits[0], k=5)
                        print(f"  Top-5 logits at token 5:")
                        for j in range(5):
                            print(f"    {top_indices[j].item()} = '{tokenizer.decode([top_indices[j].item()])}': {top_logits[j].item():.4f}")

                # Break if all sequences have finished
                if finished.all():
                    break

                #output = model.codi.get_base_model().transformer.wte(next_token_ids).unsqueeze(1).to(device)
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                sample_idx = step*data_args.batch_size+mini_step
                pred_answer = extract_answer_number(decoded_pred)
                gold_answer = answer[sample_idx]
                is_correct = (pred_answer == gold_answer)
                if do_print:
                    print(f"Question {sample_idx} Starts...")
                    print(f"Q: {question[sample_idx]}")
                    print(decoded_pred)
                    print(f"Question {sample_idx} Ends")
                    correct_str = "CORRECT" if is_correct else "WRONG"
                    print(f"Prediction={pred_answer}; Groundtruth={gold_answer}; {correct_str}")
                    print("")
                ans_pred_list.append(pred_answer)
      
    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"\n{'='*60}")
    print(f"RESULTS: {data_args.data_name} | Mode: {training_args.inference_mode}")
    print(f"{'='*60}")
    print(f"Accuracy: {100*accuracy:.2f}% ({int(accuracy * len(answer))}/{len(answer)} correct)")
    print(f"Average output length: {sum(len_cot)/len(len_cot):.1f} tokens")
    print(f"{'='*60}\n")

    # Hash trainable parameters
    param_hash = hashlib.md5()
    trainable_params = []
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            trainable_params.append(name)
            param_bytes = param.detach().cpu().to(torch.float32).numpy().tobytes()
            param_hash.update(param_bytes)

    # Count total trainable elements
    total_elements = sum(dict(model.named_parameters())[name].numel() for name in trainable_params)
    print(f'Trainable parameters: {len(trainable_params)}')
    print(f'Total trainable elements: {total_elements:,}')
    print(f'Vocab size: {model.codi.config.vocab_size}')
    # Get embedding weights (different paths for GPT-2 vs Llama)
    embed_weight = model.get_embd(model.codi, model.model_name).weight
    print(f'Embedding weight shape: {embed_weight.shape}')
    print(f'LM head weight shape: {model.codi.get_base_model().lm_head.weight.shape}')
    print(f'Parameter hash: {param_hash.hexdigest()}')

    return 100*accuracy

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                return "C"
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        elif "prosqa" in data_args.data_name.lower():
            # Extract answer from "The answer is: <answer>" format
            if "The answer is:" in sentence:
                return sentence.split("The answer is:")[-1].strip()
            return sentence.strip()
        return float('inf')

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"\nFINAL: {data_args.data_name} | Mode: {training_args.inference_mode}")
    print(f"Average accuracy over {training_args.inf_num_iterations} runs: {sum(accu_list)/len(accu_list):.2f}%")
