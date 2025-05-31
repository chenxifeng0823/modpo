import os
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate.utils import is_cuda_available, is_xpu_available

from src.trainer.advanced_modpo_trainer import AdvancedMODPOTrainer
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    print_local_main,
    disable_progress_bar_non_local_main,
    set_seeds,
    prepare_model_for_peft,
    param_sharding_enabled,
    PeftAsPreTrained,
)
from src.utils.reward import RewardWrapperList, ImplicitRewardWrapper

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft model name"})
    margin_reward_model_name: str = field(
        metadata={"help": "the margin reward model name"})
    use_flash_attention_2: Optional[bool] = field(
        default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(
        default=DEFAULT_PROMPT_TEMPLATE,
        metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(
        default="PKU-Alignment/PKU-SafeRLHF-10K",
        metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(
        default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "whether to conduct sanity check"})

    # Advanced MODPO specific arguments
    scalarization_method: Literal["linear", "chebyshev", "exponential",
                                  "power"] = field(
                                      default="linear",
                                      metadata={
                                          "help": "scalarization method to use"
                                      })
    risk_aversion: Optional[float] = field(
        default=1.0,
        metadata={"help": "risk aversion parameter for exponential utility"})
    risk_parameter: Optional[float] = field(
        default=0.5, metadata={"help": "risk parameter for power utility"})

    w: Optional[float] = field(default=0.5, metadata={"help": "weight"})
    beta: Optional[float] = field(default=0.1,
                                  metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(
        default=4, metadata={"help": "num_proc for dataset.map"})
    generate_during_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to generate during evaluation"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/advanced_modpo",
            overwrite_output_dir=True,
            seed=42,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=is_cuda_available(),
            remove_unused_columns=False,
            run_name="dev_advanced_modpo",
            report_to="wandb",
            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        ))

    peft: Optional[bool] = field(
        default=True, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(default_factory=lambda: LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ))


def get_device_setup():
    """Get appropriate device setup based on available hardware."""
    if is_cuda_available():
        return {"device_map": {"": Accelerator().local_process_index}}
    elif is_xpu_available():
        return {"device_map": "auto"}
    else:
        return {"device_map": "cpu"}


def main():
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.training_args.seed)
    if not script_args.peft:
        script_args.peft_config = None

    # Get device setup
    device_setup = get_device_setup()

    # base model
    print_local_main("loading model...")
    try:
        sft_model = AutoModelForCausalLM.from_pretrained(
            script_args.sft_model_name,
            use_flash_attention_2=script_args.use_flash_attention_2,
            torch_dtype=torch.bfloat16,
            **device_setup)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("CUDA error encountered. Falling back to CPU...")
            device_setup = {"device_map": "cpu"}
            sft_model = AutoModelForCausalLM.from_pretrained(
                script_args.sft_model_name,
                use_flash_attention_2=False,
                torch_dtype=torch.float32,
                **device_setup)
        else:
            raise e

    sft_model.config.update({
        "use_cache": False,
        "pad_token_id": sft_model.config.eos_token_id
    })
    print_local_main(sft_model)
    print_local_main(script_args.peft_config)

    # peft
    model = prepare_model_for_peft(sft_model,
                                   peft_config=script_args.peft_config,
                                   args=script_args.training_args)
    model.load_adapter(script_args.margin_reward_model_name,
                       adapter_name="margin_reward")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dataset
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check,
    )
    train_dataset = rdp.get_preference_dataset(split="train")
    eval_dataset = rdp.get_preference_dataset(split="validation")

    # get ready for training
    print_local_main("start training...")
    trainer = AdvancedMODPOTrainer(
        model=model,
        beta=script_args.beta,
        args=script_args.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        num_proc=script_args.num_proc,
        generate_during_eval=script_args.generate_during_eval,
        scalarization_method=script_args.scalarization_method,
        risk_aversion=script_args.risk_aversion,
        risk_parameter=script_args.risk_parameter,
    )

    if Accelerator().is_local_main_process:
        trainer.model.print_trainable_parameters()

    trainer.set_wrapped_margin_reward_model_list(
        RewardWrapperList([
            ImplicitRewardWrapper(
                model=PeftAsPreTrained(trainer.model, "margin_reward"),
                ref_model=PeftAsPreTrained(trainer.model),
                tokenizer=tokenizer,
                beta=script_args.beta,
                prompt_template=script_args.prompt_template,
            )
        ]),
        w=(script_args.w, 1 - script_args.w),
        prepare=False,
    )
    trainer.train()

    save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
    trainer.model.save_pretrained(
        os.path.join(script_args.training_args.output_dir, save_name))
    trainer.tokenizer.save_pretrained(
        os.path.join(script_args.training_args.output_dir, save_name))


if __name__ == "__main__":
    main()
