import argparse
import json
import sys
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _get_eval_conf(data_conf, split):
    if split == "ood_param" and data_conf.get("eval_datasets"):
        conf = dict(data_conf)
        conf["datasets"] = data_conf["eval_datasets"]
        conf.pop("eval_datasets", None)
        conf.pop("eval_equation_datasets", None)
        return conf
    if split == "ood_equation" and data_conf.get("eval_equation_datasets"):
        conf = dict(data_conf)
        conf["datasets"] = data_conf["eval_equation_datasets"]
        conf.pop("eval_datasets", None)
        conf.pop("eval_equation_datasets", None)
        return conf
    return data_conf


def _eval_onestep(model, loaders, device, use_dataset_id=True):
    results = {}
    total_loss = 0.0
    total_batches = 0
    usage_counts = None
    if hasattr(model, "route") and hasattr(model, "num_primitives"):
        usage_counts = torch.zeros(model.num_primitives, dtype=torch.long)

    with torch.no_grad():
        for name, loader, _, _ in loaders:
            ds_loss = 0.0
            pbar = tqdm(loader, desc=f"One-step {name}", leave=False)
            for batch in pbar:
                u_t, u_tp1, pde_params, dataset_id, equation = _unpack_batch(batch, device=device)
                if not use_dataset_id:
                    dataset_id = None
                pred = _call_model(
                    model,
                    u_t,
                    pde_params=pde_params,
                    dataset_id=dataset_id,
                    equation=equation,
                    sparse_primitives=True,
                )
                loss = _nrmse(pred, u_tp1)
                ds_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                if usage_counts is not None:
                    _, topk_idx = model.route(
                        u_t, pde_params=pde_params, dataset_id=dataset_id, equation=equation
                    )
                    flat = topk_idx.reshape(-1).cpu()
                    usage_counts += torch.bincount(flat, minlength=usage_counts.numel())
            ds_loss /= max(len(loader), 1)
            results[name] = ds_loss
            total_loss += ds_loss * len(loader)
            total_batches += len(loader)
    avg_loss = total_loss / max(total_batches, 1)
    return results, avg_loss, usage_counts


def _eval_rollout(model, loaders, device, steps, sample_idx, use_dataset_id=True):
    curves = {}
    max_steps_all = 0
    for name, loader, _, _ in loaders:
        dataset = loader.dataset
        dataset_id = None
        pde_params = None
        equation = None
        if use_dataset_id and hasattr(dataset, "dataset_id"):
            dataset_id = torch.tensor([dataset.dataset_id], device=device, dtype=torch.long)
        if hasattr(dataset, "params") and dataset.params is not None:
            pde_params = dataset.params.to(device).float().unsqueeze(0)
        if hasattr(dataset, "equation") and dataset.equation is not None:
            equation = dataset.equation.to(device).float().unsqueeze(0)
        seq = dataset.get_sequence(sample_idx)
        T = seq.shape[0]
        C = seq.shape[2]
        init_state = seq[0]
        init_tensor = torch.from_numpy(init_state.astype("float32")).unsqueeze(0).to(device)
        if dataset.include_grid:
            grid = torch.from_numpy(dataset.xcoord.astype("float32")).unsqueeze(0).unsqueeze(-1)
            grid = grid.to(init_tensor.device)
            init_tensor = torch.cat([init_tensor, grid], dim=-1)

        max_steps = min(steps, T - 1)
        max_steps_all = max(max_steps_all, max_steps)
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
            errors.append(_nrmse(pred_field, gt).item())
        curves[name] = errors

    avg_curve = []
    for step_idx in range(max_steps_all):
        vals = []
        for series in curves.values():
            if step_idx < len(series):
                vals.append(series[step_idx])
        avg_curve.append(sum(vals) / max(len(vals), 1))
    return curves, avg_curve


def _plot_onestep(results, avg, out_path, title):
    if not results:
        return
    names = list(results.keys())
    vals = [results[n] for n in names]
    x = range(len(names))
    plt.figure(figsize=(max(7, len(names) * 0.6), 4))
    plt.bar(x, vals)
    plt.axhline(avg, color="red", linestyle="--", linewidth=1, label=f"avg={avg:.4f}")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_rollout(curves, avg_curve, out_path, title):
    if not curves:
        return
    steps = list(range(1, len(avg_curve) + 1))
    # Average curve + bar@20
    plt.figure(figsize=(7, 6))
    plt.subplot(2, 1, 1)
    plt.plot(steps, avg_curve, marker="o", label="avg")
    plt.xlabel("Step")
    plt.ylabel("NRMSE")
    plt.title(f"{title} (avg curve)")
    plt.legend()

    plt.subplot(2, 1, 2)
    names = list(curves.keys())
    vals = []
    for name in names:
        series = curves[name]
        vals.append(series[min(19, len(series) - 1)] if series else 0.0)
    x = range(len(names))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("NRMSE@20")
    plt.title("Rollout@20 by dataset")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_usage(counts, out_path, title):
    if counts is None:
        return
    x = list(range(1, len(counts) + 1))
    plt.figure(figsize=(6, 4))
    plt.bar(x, counts.tolist())
    plt.xticks(x, [f"P{i}" for i in x])
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate ID/OOD one-step and rollout")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--steps", type=int, default=20, help="Rollout steps")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sequence index for rollout")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_conf = cfg.get("data")
    if not data_conf:
        raise ValueError("Config missing data section.")

    model_path = args.model or cfg.get("training", {}).get("output_model")
    if not model_path:
        raise ValueError("Model checkpoint path must be provided.")

    run_dir = _resolve_run_dir(model_path)
    logger, log_path = setup_logger("eval", run_dir)
    logger.info("Log file: %s", log_path)

    model = torch.load(model_path)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    device = next(model.parameters()).device
    router_conf = cfg.get("model", {}).get("router", {})
    if hasattr(model, "router") and "top_k" in router_conf:
        model.router.top_k = int(router_conf["top_k"])

    plot_root = run_dir / "plots" / "eval"
    plot_root.mkdir(parents=True, exist_ok=True)

    splits = ["id"]
    if data_conf.get("eval_datasets"):
        splits.append("ood_param")
    if data_conf.get("eval_equation_datasets"):
        splits.append("ood_equation")

    all_usage = None

    for split in splits:
        eval_conf = _get_eval_conf(data_conf, split)
        loaders = build_eval_loaders(eval_conf, split="test")
        if not loaders:
            continue

        use_dataset_id = split == "id"
        one_step, one_avg, usage_counts = _eval_onestep(model, loaders, device, use_dataset_id=use_dataset_id)
        curves, avg_curve = _eval_rollout(
            model, loaders, device, args.steps, args.sample_idx, use_dataset_id=use_dataset_id
        )

        if usage_counts is not None:
            all_usage = usage_counts if all_usage is None else all_usage + usage_counts

        metrics = {
            "one_step": one_step,
            "one_step_avg": one_avg,
            "rollout_curves": curves,
            "rollout_avg_curve": avg_curve,
        }
        with open(plot_root / f"eval_metrics_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        _plot_onestep(
            one_step, one_avg, plot_root / f"eval_onestep_{split}.png", f"One-step NRMSE ({split})"
        )
        _plot_rollout(
            curves, avg_curve, plot_root / f"eval_rollout_{split}.png", f"Rollout NRMSE ({split})"
        )

        logger.info("[%s] One-step NRMSE avg: %.6f", split.upper(), one_avg)
        for name, val in one_step.items():
            logger.info("[%s] One-step NRMSE %s: %.6f", split.upper(), name, val)

        rollout_20 = {}
        for name, series in curves.items():
            if series:
                rollout_20[name] = series[min(19, len(series) - 1)]
        if avg_curve:
            logger.info("[%s] Rollout NRMSE@20 avg: %.6f", split.upper(), avg_curve[min(19, len(avg_curve) - 1)])
        for name, val in rollout_20.items():
            logger.info("[%s] Rollout NRMSE@20 %s: %.6f", split.upper(), name, val)


    if all_usage is not None:
        with open(plot_root / "router_usage.json", "w") as f:
            json.dump({"counts": all_usage.tolist()}, f, indent=2)
        _plot_usage(all_usage, plot_root / "router_usage.png", "Router usage (all eval)")

    logger.info("Evaluation complete. Plots in %s", plot_root)


if __name__ == "__main__":
    main()
