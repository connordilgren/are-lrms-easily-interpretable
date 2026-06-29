SAVE_DIR=models
EXP_NAME=llama32-1b_prontoqa_codi

mkdir -p "$SAVE_DIR"

cp scripts/train_llama1b_prontoqa.sh "$SAVE_DIR"

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXP_NAME" \
	--logging_dir "$SAVE_DIR/logs/$EXPT_NAME" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name prontoqa \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 4 \
	--bf16 \
	--num_train_epochs 10 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_safetensors False \
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
	--num_latent 6 \
	--logging_strategy "steps" \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 200 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True
