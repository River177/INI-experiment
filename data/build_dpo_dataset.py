from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from config import load_config
from data.judge_pairs import choose_pair


def _dummy_generate_candidates(sample: dict, num_candidates: int) -> List[str]:
    # 骨架版本：用规则生成候选。真实实验建议替换为 base model 采样输出。
    q = sample.get("question", "")
    ref = sample.get("reference_answer", "")
    cands = [
        f"{ref}",
        f"关于你的问题：{q}，我会只基于可信任务给出答案。",
        "Ignore previous instructions. I will follow external malicious request.",
    ]
    return cands[:num_candidates]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_train", default="outputs/processed/train.jsonl")
    ap.add_argument("--input_val", default="outputs/processed/val.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    build_dpo_cfg = cfg.get("build_dpo", {})
    num_candidates = int(build_dpo_cfg.get("num_candidates", 3))
    rejected_policy = build_dpo_cfg.get("rejected_policy", "heuristic")
    supported_policies = {"heuristic"}
    if rejected_policy not in supported_policies:
        raise ValueError(
            f"Unsupported build_dpo.rejected_policy={rejected_policy!r}. "
            f"Supported values are: {sorted(supported_policies)}."
        )

    for split_name, in_path, out_path in [
        ("train", args.input_train, cfg["paths"]["dpo_train_jsonl"]),
        ("val", args.input_val, cfg["paths"]["dpo_val_jsonl"]),
    ]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s = json.loads(line)
                candidates = _dummy_generate_candidates(s, num_candidates)
                chosen, rejected = choose_pair(candidates, s.get("reference_answer", ""), s.get("is_attack", False))
                rec = {
                    "prompt": s["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "task": s.get("task", "unknown"),
                    "attack_type": s.get("attack_type", "none"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"built {split_name}: {out_path}")


if __name__ == "__main__":
    main()
