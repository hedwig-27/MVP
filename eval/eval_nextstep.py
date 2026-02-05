import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import inspect
import json

import torch
import yaml
from utils import get_progress

from dataloader.multi_loader import build_eval_loaders
from utils import setup_logger


def _nrmse(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    denom = torch.sqrt(torch.mean(target ** 2)) + eps
    return torch.sqrt(mse) / denom


def _unpack_batch(batch, device):
    if len(batch) >= 4 and isinstance(batch[2], dict):
        x, y, cond, _ = batch[0], batch[1], batch[2], batch[3]
        dataset_id = cond.get("dataset_id")
        pde_params = cond.get("params")
        equation = cond.get("equation")
    else:
        x, y = batch[0], batch[1]
        dataset_id = None
        pde_params = None
        equation = None
    x = x.float().to(device)
    y = y.float().to(device)
    if pde_params is not None:
        pde_params = pde_params.to(device).float()
    if dataset_id is not None:
        dataset_id = dataset_id.to(device).long()
    if equation is not None:
        equation = equation.to(device).float()
    return x, y, pde_params, dataset_id, equation


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
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        accepts_pde = "pde_params" in params or accepts_kwargs
        accepts_dataset = "dataset_id" in params or accepts_kwargs
        accepts_equation = "equation" in params or accepts_kwargs
        if (
            (accepts_pde or accepts_dataset or accepts_equation)
            and (
                pde_params is not None
                or dataset_id is not None
                or equation is not None
                or sparse_primitives
            )
        ):
            return model(
                u_t,
                pde_params=pde_params,
                dataset_id=dataset_id,
                equation=equation,
                sparse_primitives=sparse_primitives,
            )
    except (ValueError, TypeError):
        pass
    return model(u_t)


def evaluate_model(model_path, data_conf, split, logger, run_dir):
    model = torch.load(model_path)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    loaders = build_eval_loaders(data_conf, split=split)
    criterion = _nrmse
    all_usage = []
    total_loss = 0.0
    total_batches = 0

    for name, loader, _, _ in loaders:
        usage_counts = None
        if hasattr(model, "route") and (hasattr(model, "total_primitives") or hasattr(model, "num_primitives")):
            total = getattr(model, "total_primitives", model.num_primitives)
            usage_counts = torch.zeros(total, dtype=torch.long)
        ds_loss = 0.0
        with torch.no_grad():
            pbar = get_progress(loader, desc=f"Eval {name}", leave=False)
            for batch in pbar:
                u_t, u_tp1, pde_params, dataset_id, equation = _unpack_batch(
                    batch, device=next(model.parameters()).device
                )
                pred = _call_model(
                    model,
                    u_t,
                    pde_params=pde_params,
                    dataset_id=dataset_id,
                    equation=equation,
                    sparse_primitives=True,
                )
                loss = criterion(pred, u_tp1)
                ds_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                if usage_counts is not None:
                    _, topk_idx = model.route(
                        u_t, pde_params=pde_params, dataset_id=dataset_id, equation=equation
                    )
                    flat = topk_idx.reshape(-1).cpu()
                    usage_counts += torch.bincount(flat, minlength=usage_counts.numel())
        ds_loss /= max(len(loader), 1)
        logger.info("One-step NRMSE (%s): %.6f", name, ds_loss)
        total_loss += ds_loss * len(loader)
        total_batches += len(loader)
        if usage_counts is not None:
            all_usage.append(
                {"dataset": name, "counts": usage_counts.tolist(), "total": int(usage_counts.sum().item())}
            )

    avg_loss = total_loss / max(total_batches, 1)
    logger.info("One-step NRMSE (avg): %.6f", avg_loss)
    if all_usage:
        with open(run_dir / "primitive_usage.json", "w") as f:
            json.dump(all_usage, f, indent=2)
        logger.info("Primitive usage saved to %s", run_dir / "primitive_usage.json")
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one-step prediction error")
    parser.add_argument('--config', type=str, help="Path to YAML config used for model training")
    parser.add_argument('--model', type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to evaluate on (default: test)")
    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    # Determine model path and data path
    model_path = args.model
    data_path = None
    split = args.split
    if not model_path and "training" in cfg:
        model_path = cfg["training"].get("output_model")
    if "data" in cfg:
        data_conf = cfg["data"]
    else:
        data_conf = None
    if not data_conf or not model_path:
        print("Error: data path or model path not provided.")
        sys.exit(1)

    run_dir = _resolve_run_dir(model_path)
    logger, log_path = setup_logger("eval_next", run_dir)
    logger.info("Log file: %s", log_path)

    # Evaluate
    evaluate_model(model_path, data_conf, split, logger, run_dir)
