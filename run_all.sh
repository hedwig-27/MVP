#!/bin/bash
# Ensure the script runs from the MVP root directory.
cd "$(dirname "$0")" || exit 1
# Run full training and evaluation pipeline for primitive discovery on PDEBench-1D data.

# 1) Train primitive discovery model (jointly learned primitives + router)
python train/train_primitives.py --config configs/primitive.yaml

# 2) Evaluate trained model (ID + OOD, one-step + rollout)
python eval/eval_suite.py --config configs/primitive.yaml --model outputs/latest_primitive/primitive_model.pt --steps 20 --sample_idx 0
