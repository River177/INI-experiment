#!/usr/bin/env bash
set -euo pipefail
python train/train_dpo.py --config configs/dpo.yaml
