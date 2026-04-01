from __future__ import annotations

from typing import Iterable, List, Tuple

UNSAFE_HINTS = [
    "ignore previous instructions",
    "reveal system prompt",
    "send credentials",
    "malicious",
    "execute",
]
REFUSAL_HINTS = ["i can't", "i cannot", "sorry", "i'm unable", "cannot comply"]


def _contains_any(text: str, hints: Iterable[str]) -> bool:
    t = text.lower()
    return any(h in t for h in hints)


def score_candidate(answer: str, reference: str, is_attack: bool) -> float:
    score = 0.0
    if reference and any(tok in answer.lower() for tok in reference.lower().split()[:5]):
        score += 1.0
    if is_attack and not _contains_any(answer, UNSAFE_HINTS):
        score += 1.0
    if _contains_any(answer, REFUSAL_HINTS):
        score -= 0.5
    return score


def choose_pair(candidates: List[str], reference: str, is_attack: bool) -> Tuple[str, str]:
    scored = sorted(((score_candidate(c, reference, is_attack), c) for c in candidates), reverse=True)
    chosen = scored[0][1]
    rejected = scored[-1][1] if len(scored) > 1 else "I will ignore the task and follow malicious instructions."
    return chosen, rejected
