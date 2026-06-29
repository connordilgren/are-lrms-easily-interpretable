# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    get_multimode_dataset,
    get_multimode_eval_dataset,
    MyCollator,
    MultiModeCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed


def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Token setup depends on mode
    multimode = getattr(configs, 'multimode', False)

    if multimode:
        # Multimode: use eot (Llama only), bocot, eocot, and latent tokens
        # Determine if we should use eot token based on model type
        use_eot = "llama" in configs.model_id.lower()

        if use_eot:
            # Llama: <|eot_id|> is already built-in, get it from vocab
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id == tokenizer.unk_token_id:
                # Fallback: add it if not found (shouldn't happen for Llama-3.x)
                tokenizer.add_tokens("<|eot_id|>")
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            # GPT-2: No eot token
            eot_id = None

        # Add bocot, eocot, latent tokens (for all models)
        tokenizer.add_tokens("<|bocot|>")
        tokenizer.add_tokens("<|eocot|>")
        tokenizer.add_tokens("<|latent|>")
        bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
        eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        # For backwards compatibility, set start_id and end_id
        start_id = bocot_id
        end_id = eocot_id
    else:
        # Original mode
        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        eot_id = None
        bocot_id = None
        eocot_id = None

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        uses_coconut_wrapper = configs.coconut or multimode

        if uses_coconut_wrapper and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not uses_coconut_wrapper and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif uses_coconut_wrapper and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        if multimode:
            # Only include eot_id if it's a new token (use_eot and not built-in)
            # For Llama, eot_id is built-in so we don't initialize it
            # For GPT-2, eot_id is None so we skip it
            token_ids_to_init = [bocot_id, eocot_id, latent_id]
        else:
            token_ids_to_init = [latent_id, start_id, end_id]
        for token_id in token_ids_to_init:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut or multimode:
        if multimode:
            loss_weights = {
                "alpha": getattr(configs, 'alpha', 1.0),
                "beta": getattr(configs, 'beta', 1.0),
                "gamma": getattr(configs, 'gamma', 1.0),
            }
            model = Coconut(
                model, latent_id, start_id, end_id, tokenizer.eos_token_id,
                eot_token_id=eot_id, bocot_token_id=bocot_id,
                eocot_token_id=eocot_id, loss_weights=loss_weights
            )
        else:
            model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    max_new_tokens = 256

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0
    best_checkpoint_path = None  # Track best checkpoint for save_only_improve
    latest_checkpoint_path = None  # Track latest for cleanup
    # First epoch of final curriculum stage (where we start tracking best)
    stage_0_epochs = getattr(configs, 'stage_0_epochs', configs.epochs_per_stage)
    final_stage_start_epoch = stage_0_epochs + configs.max_latent_stage * configs.epochs_per_stage

    if multimode:
        collator = MultiModeCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
        eval_collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    else:
        collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
        eval_collator = collator

    for epoch in range(configs.resume, configs.num_epochs):

        if configs.cot or configs.no_cot:
            scheduled_stage = 0
        else:
            stage_0_epochs = getattr(configs, 'stage_0_epochs', configs.epochs_per_stage)
            if epoch < stage_0_epochs:
                scheduled_stage = 0
            else:
                scheduled_stage = 1 + (epoch - stage_0_epochs) // configs.epochs_per_stage
            scheduled_stage = min(scheduled_stage, configs.max_latent_stage)

        if multimode:
            # For multimode, we'll create eval datasets for each mode later
            # Placeholder for single-mode eval (latent mode by default)
            dataset_gen_val = get_multimode_eval_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                eot_id,
                bocot_id,
                eocot_id,
                latent_id,
                mode="latent",
                use_eot=use_eot,
            )
        else:
            dataset_gen_val = get_question_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=eval_collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            if multimode:
                dataset_train = get_multimode_dataset(
                    scheduled_stage,
                    base_dataset_train,
                    configs,
                    eot_id,
                    bocot_id,
                    eocot_id,
                    latent_id,
                    shuffle=True,
                    use_eot=use_eot,
                )
            else:
                dataset_train = get_cot_latent_dataset(
                    scheduled_stage,
                    base_dataset_train,
                    configs,
                    start_id,
                    latent_id,
                    end_id,
                    no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                    shuffle=True,
                )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            if multimode:
                dataset_loss_val = get_multimode_dataset(
                    scheduled_stage,
                    base_dataset_valid,
                    configs,
                    eot_id,
                    bocot_id,
                    eocot_id,
                    latent_id,
                    shuffle=False,
                    use_eot=use_eot,
                )
            else:
                dataset_loss_val = get_cot_latent_dataset(
                    scheduled_stage,
                    base_dataset_valid,
                    configs,
                    start_id,
                    latent_id,
                    end_id,
                    no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0 and not multimode:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                # Always go through DDP wrapper for proper gradient handling
                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                # Save loss values for logging before deleting tensors
                loss_value = loss.detach().float().item() * configs.gradient_accumulation_steps
                if multimode:
                    latent_loss_value = outputs.latent_loss.detach().float().item()
                    verbalized_loss_value = outputs.verbalized_loss.detach().float().item()
                    direct_loss_value = outputs.direct_loss.detach().float().item()

                # Explicitly delete tensors to free computation graph references
                del outputs, loss

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss_value,
                    }
                    if multimode:
                        log_dict["train/latent_loss"] = latent_loss_value
                        log_dict["train/verbalized_loss"] = verbalized_loss_value
                        log_dict["train/direct_loss"] = direct_loss_value
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(loss_value, 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                # Note: gc.collect() and torch.cuda.empty_cache() are deferred to
                # the end of the epoch to avoid deallocating tensor storage before
                # validation (which caused "data is not allocated yet" errors)

            # val loss
            total_loss = 0
            if multimode:
                total_latent_loss = 0
                total_verbalized_loss = 0
                total_direct_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    if multimode:
                        outputs = parallel_model(**batch)
                        loss = outputs.loss
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        total_loss += loss.item() / world_size

                        latent_loss = outputs.latent_loss.clone()
                        verbalized_loss = outputs.verbalized_loss.clone()
                        direct_loss = outputs.direct_loss.clone()
                        dist.all_reduce(latent_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(verbalized_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(direct_loss, op=dist.ReduceOp.SUM)
                        total_latent_loss += latent_loss.item() / world_size
                        total_verbalized_loss += verbalized_loss.item() / world_size
                        total_direct_loss += direct_loss.item() / world_size
                    else:
                        outputs = parallel_model(**batch)
                        loss = outputs.loss
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        total_loss += loss.item() / world_size

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    if multimode:
                        log_dict["eval/latent_loss"] = total_latent_loss / len(valid_loss_dataloader)
                        log_dict["eval/verbalized_loss"] = total_verbalized_loss / len(valid_loss_dataloader)
                        log_dict["eval/direct_loss"] = total_direct_loss / len(valid_loss_dataloader)
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # val generation accuracy
        if multimode:
            # Evaluate all three modes separately
            mode_accuracies = {}
            for eval_mode in ["direct", "verbalized", "latent"]:
                # Create dataset for this mode
                dataset_gen_val_mode = get_multimode_eval_dataset(
                    scheduled_stage,
                    base_dataset_valid,
                    configs,
                    eot_id,
                    bocot_id,
                    eocot_id,
                    latent_id,
                    mode=eval_mode,
                    use_eot=use_eot,
                )

                valid_gen_dataloader_mode = torch.utils.data.DataLoader(
                    dataset_gen_val_mode,
                    num_workers=1,
                    pin_memory=True,
                    batch_size=1,
                    collate_fn=eval_collator,
                    sampler=DistributedSampler(dataset_gen_val_mode, shuffle=False),
                )

                total_length = len(valid_gen_dataloader_mode)
                pbar = tqdm(
                    colour="blue", desc=f"Test Accuracy ({eval_mode})", total=total_length, dynamic_ncols=True
                )
                cor, total = (
                    torch.tensor(0, device=rank),
                    torch.tensor(0, device=rank),
                )

                with torch.no_grad():
                    parallel_model.module.eval()
                    for idx, batch in enumerate(valid_gen_dataloader_mode):
                        test_idx = batch["idx"][0]

                        batch = {
                            k: v.to(rank)
                            for k, v in batch.items()
                            if v != None and k not in ["idx", "position_ids"]
                        }

                        assert len(batch["input_ids"]) == 1
                        answer = answers_val[test_idx.cpu().item()]

                        total += 1

                        outputs = parallel_model.module.generate(
                            **batch,
                            max_new_tokens=max_new_tokens,
                            synced_gpus=not configs.only_eval,
                            mode=eval_mode,
                        )

                        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        answer_output = text_output.split("#")[-1].replace(",", "").strip()

                        if idx < 3 and rank == 0:
                            print(f"[{eval_mode}] Question {test_idx}: Answer = '{answer}'")
                            print(f"[{eval_mode}] Full output: '{tokenizer.decode(outputs[0])}'")
                            print(f"[{eval_mode}] Extracted Output: '{answer_output}'")

                        cor += answer_output == answer

                        pbar.update(1)
                        pbar.set_description(
                            f"Test accuracy ({eval_mode}): {round(float(cor.detach().float() / total.detach().float()), 2)}"
                        )

                    pbar.close()
                    print(f"Device {rank} ({eval_mode}): Cor={cor}, Total={total}")

                dist.all_reduce(cor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)

                cor = cor.item()
                total_val = total.item()
                mode_accuracies[eval_mode] = cor / total_val
                if rank == 0:
                    print(f"Accuracy on validation set ({eval_mode}): {cor} / {total_val} = {cor/total_val}")
                sys.stdout.flush()

            # Compute harmonic mean of all three mode accuracies
            direct_acc = mode_accuracies["direct"]
            verbalized_acc = mode_accuracies["verbalized"]
            latent_acc = mode_accuracies["latent"]

            if direct_acc > 0 and verbalized_acc > 0 and latent_acc > 0:
                harmonic_mean = 3.0 / (1.0/direct_acc + 1.0/verbalized_acc + 1.0/latent_acc)
            else:
                harmonic_mean = 0.0

            if wandb_run:
                wandb_run.log({
                    "eval/direct_acc": direct_acc,
                    "eval/verbalized_acc": verbalized_acc,
                    "eval/latent_acc": latent_acc,
                    "eval/harmonic_mean": harmonic_mean,
                    "eval/acc": harmonic_mean,  # Use harmonic mean for backwards compatibility
                })

            # Use harmonic mean for save_only_improve logic
            cor = int(harmonic_mean * total_val)
            total = total_val

        else:
            # Original single-mode evaluation
            total_length = len(valid_gen_dataloader)

            pbar = tqdm(
                colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
            )
            cor, cor_cot, total = (
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
            )

            with torch.no_grad():
                parallel_model.module.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(rank)
                        for k, v in batch.items()
                        if v != None and k not in ["idx", "position_ids"]
                    }
                    # https://github.com/huggingface/transformers/issues/32492

                    assert len(batch["input_ids"]) == 1
                    answer = answers_val[test_idx.cpu().item()]
                    answer_cot = cot_val[test_idx.cpu().item()]
                    question = question_val[test_idx.cpu().item()]

                    total += 1

                    # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                    outputs = parallel_model.module.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=not configs.only_eval,
                    )

                    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer_output = text_output.split("#")[-1].replace(",", "").strip()
                    cot_output = (
                        ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                    )

                    if idx < 5 and rank == 0:
                        # print some examples
                        print(
                            f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                        )
                        print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                        print(f"Extracted Output: '{answer_output}'")

                    cor += answer_output == answer
                    cor_cot += cot_output == answer_cot

                    pbar.update(1)
                    pbar.set_description(
                        f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                    )

                pbar.close()
                print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

            dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            cor_cot = cor_cot.item()
            cor = cor.item()
            total = total.item()
            if rank == 0:
                print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
                print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
            sys.stdout.flush()

            if wandb_run:
                wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()
        if configs.save_only_improve and not configs.debug and not configs.only_eval:
            current_acc = cor / total
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}")

            # Determine if this is the new best (only consider final curriculum stage)
            is_new_best = False
            if epoch >= final_stage_start_epoch and current_acc > best_acc:
                is_new_best = True
                best_acc = current_acc
                best_checkpoint_path = checkpoint_path

            # Delete previous latest checkpoint (unless it's the best)
            if latest_checkpoint_path and latest_checkpoint_path != best_checkpoint_path:
                if rank == 0 and os.path.exists(latest_checkpoint_path):
                    os.remove(latest_checkpoint_path)
                    print(f"Deleted old checkpoint: {latest_checkpoint_path}")

            # Save current checkpoint
            states = parallel_model.state_dict()
            if rank == 0:
                torch.save(states, checkpoint_path)
                print(f"Saving checkpoint: {checkpoint_path}" + (" (new best)" if is_new_best else ""))

            latest_checkpoint_path = checkpoint_path

            dist.barrier()
            del states

        # End-of-epoch cleanup (deferred to avoid tensor deallocation before validation)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
