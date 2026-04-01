#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-outputs/checkpoints/dpo}
python eval/generate_answers.py --model "$MODEL_PATH" --input outputs/processed/test.jsonl --output outputs/predictions/pred.jsonl
python eval/evaluate_all.py --pred outputs/predictions/pred.jsonl --out outputs/predictions/pred.metrics.json
