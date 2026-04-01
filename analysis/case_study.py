from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.pred).read_text(encoding="utf-8").splitlines() if x.strip()]
    risky = [r for r in rows if r.get("is_attack") and "ignore previous instructions" in r.get("prediction", "").lower()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(risky[: args.topk], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(risky[: args.topk])} cases to {out}")


if __name__ == "__main__":
    main()
