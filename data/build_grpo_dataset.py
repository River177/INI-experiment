from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config


def convert_line(s: dict) -> dict:
    return {
        "id": s.get("id", ""),
        "prompt": s.get("prompt", ""),
        "reference_answer": s.get("reference_answer", ""),
        "task": s.get("task", "unknown"),
        "attack_type": s.get("attack_type", "none"),
        "is_attack": bool(s.get("is_attack", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_train", default="outputs/processed/train.jsonl")
    ap.add_argument("--input_val", default="outputs/processed/val.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for in_path, out_path in [
        (args.input_train, cfg["paths"]["grpo_train_jsonl"]),
        (args.input_val, cfg["paths"]["grpo_val_jsonl"]),
    ]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(json.dumps(convert_line(json.loads(line)), ensure_ascii=False) + "\n")
        print(f"built {out_path}")


if __name__ == "__main__":
    main()
