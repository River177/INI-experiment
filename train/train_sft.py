from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=cfg["model"].get("use_4bit", True), bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name_or_path"], quantization_config=bnb_cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    tok.pad_token = tok.eos_token

    sft_train_path = cfg["paths"].get("sft_train_jsonl", cfg["paths"].get("train_jsonl"))
    ds = load_dataset("json", data_files=sft_train_path, split="train")
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        peft_config=lora_cfg,
        dataset_text_field="prompt",
        max_seq_length=cfg["train"]["max_length"],
        args=train_args,
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["save_dir"])


if __name__ == "__main__":
    main()
