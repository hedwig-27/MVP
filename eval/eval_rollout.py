import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.multi_loader import build_eval_loaders
from utils import setup_logger


def _nrmse(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target)
    denom = torch.sqrt(torch.mean(target ** 2)) + eps
    return torch.sqrt(mse) / denom


def _resolve_run_dir(model_path):
    if model_path:
        try:
            path = Path(model_path).resolve()
            if path.exists():
                return path.parent
        except OSError:
            pass
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _call_model(model, u_t, pde_params=None, dataset_id=None, equation=None, sparse_primitives=False):
    if pde_params is None and dataset_id is None and equation is None and not sparse_primitives:
        return model(u_t)
    try:
        return model(
            u_t,
            pde_params=pde_params,
            dataset_id=dataset_id,
            equation=equation,
            sparse_primitives=sparse_primitives,
        )
    except TypeError:
        return model(u_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-step rollout error growth")
    parser.add_argument('--config', type=str, help="Path to YAML config for data")
    parser.add_argument('--model', type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument('--steps', type=int, default=20, help="Number of rollout steps to evaluate")
    parser.add_argument('--sample_idx', type=int, default=0, help="Index of sample sequence to use for rollout")
    args = parser.parse_args()

    # Load data config and model
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    data_conf = cfg.get("data")
    if not data_conf:
        print("Data path must be provided via --config.")
        sys.exit(1)
    model_path = args.model or (cfg.get("training", {}).get("output_model") if "training" in cfg else None)
    if not model_path:
        print("Model checkpoint path must be provided via --model or in config.")
        sys.exit(1)
    run_dir = _resolve_run_dir(model_path)
    logger, log_path = setup_logger("eval_rollout", run_dir)
    logger.info("Log file: %s", log_path)
    model = torch.load(model_path)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    loaders = build_eval_loaders(data_conf, split="test")
    if not loaders:
        logger.info("Test split not available in config.")
        sys.exit(1)

    plot_root = run_dir / "plots"
    plot_dir = plot_root / "curves" / "rollout"
    plot_dir.mkdir(parents=True, exist_ok=True)
    legacy_dirs = [
        run_dir / "rollout_plots",
        plot_root / "rollout_plots",
        plot_root / "rollout_curves",
        plot_root / "curves" / "rollout_curves",
    ]
    for legacy_dir in legacy_dirs:
        if legacy_dir.exists():
            for path in legacy_dir.glob("rollout_error_*.png"):
                safe = path.name.replace("rollout_error_", "rollout_curve_")
                target = plot_dir / safe
                if not target.exists():
                    try:
                        path.rename(target)
                    except OSError:
                        pass
    for path in run_dir.glob("rollout_error_*.png"):
        safe = path.name.replace("rollout_error_", "rollout_curve_")
        target = plot_dir / safe
        if not target.exists():
            try:
                path.rename(target)
            except OSError:
                pass

    for name, loader, _, _ in loaders:
        dataset = loader.dataset
        seq = dataset.get_sequence(args.sample_idx)  # (T, X, C)
        T = seq.shape[0]
        C = seq.shape[2]
        init_state = seq[0]
        init_tensor = torch.from_numpy(init_state.astype("float32")).unsqueeze(0)
        init_tensor = init_tensor.to(next(model.parameters()).device)
        if dataset.include_grid:
            grid = torch.from_numpy(dataset.xcoord.astype("float32")).unsqueeze(0).unsqueeze(-1)
            grid = grid.to(init_tensor.device)
            init_tensor = torch.cat([init_tensor, grid], dim=-1)

        dataset_id = None
        pde_params = None
        equation = None
        if hasattr(dataset, "dataset_id"):
            dataset_id = torch.tensor([dataset.dataset_id], device=init_tensor.device, dtype=torch.long)
        if hasattr(dataset, "params") and dataset.params is not None:
            pde_params = dataset.params.to(init_tensor.device).float().unsqueeze(0)
        if hasattr(dataset, "equation") and dataset.equation is not None:
            equation = dataset.equation.to(init_tensor.device).float().unsqueeze(0)

        steps = args.steps
        max_steps = min(steps, T - 1)
        errors = []
        pred = init_tensor
        for step in tqdm(range(1, max_steps + 1), desc=f"Rollout {name}", leave=False):
            with torch.no_grad():
                pred = _call_model(
                    model,
                    pred,
                    pde_params=pde_params,
                    dataset_id=dataset_id,
                    equation=equation,
                    sparse_primitives=True,
                )
                if dataset.include_grid:
                    pred = torch.cat([pred, grid], dim=-1)
            gt = torch.from_numpy(seq[step].astype("float32")).unsqueeze(0).to(pred.device)
            pred_field = pred[..., :C]
            nrmse = _nrmse(pred_field, gt).item()
            errors.append(nrmse)

        if max_steps >= 5:
            logger.info("NRMSE after 5 steps (%s): %.6f", name, errors[4])
        if max_steps >= 10:
            logger.info("NRMSE after 10 steps (%s): %.6f", name, errors[9])
        if max_steps >= 20:
            logger.info("NRMSE after 20 steps (%s): %.6f", name, errors[19])

        plt.figure()
        plt.plot(range(1, max_steps + 1), errors, marker="o")
        plt.title(f"Rollout NRMSE vs Step ({name})")
        plt.xlabel("Prediction Step")
        plt.ylabel("NRMSE")
        plt.grid(True)
        plt.tight_layout()
        safe_name = name.replace("/", "_")
        output_path = plot_dir / f"rollout_curve_{safe_name}.png"
        plt.savefig(output_path)
        logger.info("Rollout curve plot saved to %s", output_path)
