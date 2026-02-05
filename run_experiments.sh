#!/bin/bash
cd "$(dirname "$0")" || exit 1
python scripts/run_experiment_suite.py --suite configs/experiment_suite.yaml
