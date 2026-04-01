from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from config import load_config
from reward.grpo_reward import RewardWeights, compute_total_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=cfg["model"].get("use_4bit", True), bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name_or_path"], quantization_config=bnb_cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=cfg["paths"]["grpo_train_jsonl"], split="train")
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM"
    )

    weights = RewardWeights(
        defense_weight=float(cfg["reward"]["defense_weight"]),
        task_weight=float(cfg["reward"]["task_weight"]),
        over_refusal_weight=float(cfg["reward"]["over_refusal_weight"]),
        unsafe_weight=float(cfg["reward"]["unsafe_weight"]),
    )

    def reward_fn(completions, **kwargs):
        metas = kwargs.get("prompts_metadata", kwargs.get("metadata", [{}] * len(completions)))
        return [compute_total_reward(ans, meta, weights) for ans, meta in zip(completions, metas)]

    grpo_cfg = GRPOConfig(
        output_dir=cfg["paths"]["save_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        logging_steps=cfg["train"]["logging_steps"],
        num_generations=int(cfg["grpo"]["num_generations"]),
        max_completion_length=int(cfg["grpo"]["max_new_tokens"]),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["save_dir"])


if __name__ == "__main__":
    main()
