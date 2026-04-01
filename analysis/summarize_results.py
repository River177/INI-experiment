from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in Path(args.pred_dir).glob("*.metrics.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"file": p.name, **m})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df)


if __name__ == "__main__":
    main()
