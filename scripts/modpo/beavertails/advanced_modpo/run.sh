#!/bin/bash
# sh scripts/modpo/beavertails/advanced_modpo/run.sh

# Check for available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs found. Running on CPU..."
    LAUNCH="accelerate launch --config_file scripts/accelerate_configs/cpu_config.yaml"
else
    echo "Found $NUM_GPUS GPUs"
    LAUNCH="accelerate launch --config_file scripts/accelerate_configs/default_config.yaml --num_processes=$NUM_GPUS"
fi

sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
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
rm_run_name="${dataset_name}/advanced_modpo/rm/safer"
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

# Language Modeling: Run Advanced MODPO with different scalarization methods
# Test different methods with w=0.5
for method in "linear" "chebyshev" "exponential" "power"; do
    lm_run_name="${dataset_name}/advanced_modpo/lm/${method}_w0.5"
    PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/advanced_modpo/advanced_modpo.py \
        --sft_model_name ${sft_model_name} \
        --margin_reward_model_name "${output_dir}/${rm_run_name}/best_checkpoint" \
        --prompt_template "${prompt_template}" \
        --dataset_name "${dataset_name}-better" \
        --sanity_check ${sanity_check} \
        --scalarization_method ${method} \
        --w 0.5 \
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

# Additional experiments for exponential and power utilities with different parameters
# Test exponential utility with different risk aversion values
for risk_aversion in 0.5 2.0; do
    lm_run_name="${dataset_name}/advanced_modpo/lm/exponential_risk${risk_aversion}"
    PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/advanced_modpo/advanced_modpo.py \
        --sft_model_name ${sft_model_name} \
        --margin_reward_model_name "${output_dir}/${rm_run_name}/best_checkpoint" \
        --prompt_template "${prompt_template}" \
        --dataset_name "${dataset_name}-better" \
        --sanity_check ${sanity_check} \
        --scalarization_method "exponential" \
        --risk_aversion ${risk_aversion} \
        --w 0.5 \
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

# Test power utility with different risk parameters
for risk_parameter in 0.2 0.8; do
    lm_run_name="${dataset_name}/advanced_modpo/lm/power_risk${risk_parameter}"
    PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/advanced_modpo/advanced_modpo.py \
        --sft_model_name ${sft_model_name} \
        --margin_reward_model_name "${output_dir}/${rm_run_name}/best_checkpoint" \
        --prompt_template "${prompt_template}" \
        --dataset_name "${dataset_name}-better" \
        --sanity_check ${sanity_check} \
        --scalarization_method "power" \
        --risk_parameter ${risk_parameter} \
        --w 0.5 \
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