from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from data.prompt_builder import build_prompt


def _normalize(raw: dict, task: str, max_prompt_chars: int) -> dict:
    sample = {
        "id": str(raw.get("id", raw.get("sample_id", ""))),
        "task": task,
        "question": raw.get("question", raw.get("instruction", "")),
        "external_content": raw.get("external_content", raw.get("context", "")),
        "reference_answer": raw.get("reference_answer", raw.get("answer", "")),
        "attack_type": raw.get("attack_type", "none"),
        "is_attack": bool(raw.get("is_attack", raw.get("attack_type") not in (None, "", "none"))),
    }
    if len(sample["external_content"]) > max_prompt_chars:
        sample["external_content"] = sample["external_content"][:max_prompt_chars]
    sample["prompt"] = build_prompt(sample)
    return sample


def load_task_jsonl(task_path: Path, task: str, max_prompt_chars: int) -> List[dict]:
    records: List[dict] = []
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(_normalize(raw, task, max_prompt_chars))
    return records


def split_dataset(samples: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    return train, val, test


def load_bipia_root(root: str | Path, tasks: List[str], max_prompt_chars: int) -> List[dict]:
    root = Path(root)
    all_records: List[dict] = []
    for task in tasks:
        p = root / f"{task}.jsonl"
        if p.exists():
            all_records.extend(load_task_jsonl(p, task, max_prompt_chars))
    return all_records


def save_jsonl(samples: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_prompt_chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_samples = load_bipia_root(args.root, args.tasks, args.max_prompt_chars)
    tr, va, te = split_dataset(all_samples, args.train_ratio, args.val_ratio, args.seed)
    save_jsonl(tr, Path(args.out_dir) / "train.jsonl")
    save_jsonl(va, Path(args.out_dir) / "val.jsonl")
    save_jsonl(te, Path(args.out_dir) / "test.jsonl")
