#!/bin/bash
# sh scripts/modpo/beavertails/run.sh

# Check for available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs found. Running on CPU..."
    LAUNCH="accelerate launch --config_file scripts/accelerate_configs/cpu_config.yaml"
else
    echo "Found $NUM_GPUS GPUs"
    LAUNCH="accelerate launch --config_file scripts/accelerate_configs/default_config.yaml --num_processes=$NUM_GPUS"
fi

sft_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
sanity_check=False
output_dir="./output"
max_length=512
per_device_train_batch_size=6
per_device_eval_batch_size=6
gradient_accumulation_steps=2
learning_rate=5e-4

# Reward Modeling: Run DPO on safe preferences to train a safe reward model that encourages safe response
rm_run_name="${dataset_name}/modpo/rm/safer"
PYTHONPATH=. $LAUNCH scripts/examples/dpo/dpo.py \
    --sft_model_name ${sft_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-safer" \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${rm_run_name}" \
    --training_args.run_name ${rm_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0

# Language Modeling: Run MODPO on helpful preferences, with the safe reward as margin
# r = (w)r_better + (1-w)r_safer
for w in 0.1 0.5 0.9; do
    lm_run_name="${dataset_name}/modpo/lm/($w)better+(1-$w)safer"
    PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/modpo.py \
        --sft_model_name ${sft_model_name} \
        --margin_reward_model_name "${output_dir}/${rm_run_name}/best_checkpoint" \
        --prompt_template "${prompt_template}" \
        --dataset_name "${dataset_name}-better" \
        --sanity_check ${sanity_check} \
        --w ${w} \
        --max_length ${max_length} \
        --training_args.output_dir "${output_dir}/${lm_run_name}" \
        --training_args.run_name ${lm_run_name} \
        --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
        --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
        --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
        --training_args.learning_rate ${learning_rate} \
        --peft_config.r 64 \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --peft_config.lora_alpha 1 \
        --peft_config.lora_dropout 0
done
