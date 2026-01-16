#!/bin/bash
# Ensure the script runs from the MVP root directory.
cd "$(dirname "$0")" || exit 1
# Run full training and evaluation pipeline for primitive discovery on PDEBench-1D data.

# 1. Train primitive discovery model (jointly learned primitives + router)
python train/train_primitives.py --config configs/primitive.yaml

# 2. Train baseline FNO model on the same multi-dataset mix
python train/train_fno.py --config configs/fno.yaml

# 3. Evaluate one-step prediction error for baseline FNO and primitive model
python eval/eval_nextstep.py --config configs/fno.yaml  # Baseline FNO
python eval/eval_nextstep.py --config configs/primitive.yaml     # Primitive discovery model

# 4. Evaluate multi-step rollout errors for primitive model (and optionally baseline)
python eval/eval_rollout.py --config configs/primitive.yaml --model outputs/latest_primitive/primitive_model.pt --steps 20 --sample_idx 0

# 5. Evaluate baseline FNO rollout for comparison
python eval/eval_rollout.py --config configs/fno.yaml --model outputs/latest_fno/fno_model.pt --steps 20 --sample_idx 0

# 6. Generate comparison plots (includes rollout comparisons)
python scripts/make_plots.py
