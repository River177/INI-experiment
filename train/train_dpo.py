from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=cfg["model"].get("use_4bit", True), bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name_or_path"], quantization_config=bnb_cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=cfg["paths"]["dpo_train_jsonl"], split="train")
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM"
    )
    args_train = TrainingArguments(
        output_dir=cfg["paths"]["save_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=float(cfg["train"]["num_train_epochs"]),
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args_train,
        beta=float(cfg["dpo"]["beta"]),
        train_dataset=ds,
        tokenizer=tok,
        peft_config=lora_cfg,
        max_prompt_length=int(cfg["dpo"]["max_prompt_length"]),
        max_length=int(cfg["dpo"]["max_length"]),
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["save_dir"])


if __name__ == "__main__":
    main()
