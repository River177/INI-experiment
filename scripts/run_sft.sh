#!/usr/bin/env bash
set -euo pipefail
python train/train_sft.py --config configs/sft.yaml
