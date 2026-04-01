from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            s = json.loads(line)
            t0 = time.time()
            inputs = tok(s["prompt"], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            pred = tok.decode(out[0], skip_special_tokens=True)
            latency = time.time() - t0
            rec = {
                "sample_id": s.get("id", ""),
                "task": s.get("task", "unknown"),
                "prompt": s.get("prompt", ""),
                "prediction": pred,
                "reference": s.get("reference_answer", ""),
                "attack_type": s.get("attack_type", "none"),
                "is_attack": bool(s.get("is_attack", False)),
                "model_name": args.model,
                "latency_sec": latency,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
