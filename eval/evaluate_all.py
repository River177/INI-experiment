from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics_asr import compute_asr
from eval.metrics_refusal import compute_refusal_metrics
from eval.metrics_utility import compute_utility


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]

    metrics = {}
    metrics.update(compute_asr(recs))
    metrics.update(compute_utility(recs))
    metrics.update(compute_refusal_metrics(recs))
    lat = [r.get("latency_sec", 0.0) for r in recs]
    metrics["avg_latency_sec"] = sum(lat) / len(lat) if lat else 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
