from __future__ import annotations


def simple_token_f1(pred: str, ref: str) -> float:
    p = set(pred.lower().split())
    r = set(ref.lower().split())
    if not p or not r:
        return 0.0
    inter = len(p & r)
    precision = inter / len(p)
    recall = inter / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_utility(records: list[dict]) -> dict:
    if not records:
        return {"avg_f1": 0.0}
    vals = [simple_token_f1(r.get("prediction", ""), r.get("reference", "")) for r in records]
    return {"avg_f1": sum(vals) / len(vals)}
