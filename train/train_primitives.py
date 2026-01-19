import json
import sys
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.multi_loader import build_loaders
from models.primitive_composer import PrimitiveAggregator, PrimitiveCNO, Router
from models.primitive_operator import PrimitiveOperator
from utils import create_run_dir, set_latest_link, setup_logger


def _nrmse(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    denom = torch.sqrt(torch.mean(target ** 2)) + eps
    return torch.sqrt(mse) / denom


def _diversity_penalty(delta_stack):
    # delta_stack: (batch, x, channels, k)
    bsz, xsz, csz, ksz = delta_stack.shape
    flat = delta_stack.reshape(bsz, xsz * csz, ksz).permute(0, 2, 1)
    flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(flat, flat.transpose(1, 2))
    off_diag = sim - torch.eye(ksz, device=sim.device).unsqueeze(0)
    return off_diag.abs().mean()


def _save_training_plots(run_dir, train_history, val_history, val_by_dataset):
    plot_dir = run_dir / "plots" / "training"
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_history) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_history, marker="o", label="train")
    plt.plot(epochs, val_history, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NRMSE")
    plt.title("Train/Val NRMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "train_val_nrmse.png")
    plt.close()

    if val_by_dataset:
        plt.figure(figsize=(7, 4))
        for name, series in val_by_dataset.items():
            if not series:
                continue
            plt.plot(epochs, series, marker="o", label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Val NRMSE")
        plt.title("Val NRMSE by Dataset")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_dir / "val_nrmse_by_dataset.png")
        plt.close()


def _unpack_batch(batch, device, return_meta=False):
    if len(batch) >= 4 and isinstance(batch[2], dict):
        x, y, cond, meta = batch[0], batch[1], batch[2], batch[3]
        dataset_id = cond.get("dataset_id")
        pde_params = cond.get("params")
        equation = cond.get("equation")
    else:
        x, y = batch[0], batch[1]
        dataset_id = None
        pde_params = None
        equation = None
        meta = None
    x = x.float().to(device)
    y = y.float().to(device)
    if pde_params is not None:
        pde_params = pde_params.to(device).float()
    if equation is not None:
        equation = equation.to(device).float()
    if dataset_id is not None:
        dataset_id = dataset_id.to(device).long()
    if return_meta:
        return x, y, pde_params, dataset_id, equation, meta
    return x, y, pde_params, dataset_id, equation


def _load_balance_loss(weights):
    if weights is None:
        return 0.0
    num = weights.size(-1)
    target = torch.full((num,), 1.0 / num, device=weights.device, dtype=weights.dtype)
    mean_w = weights.mean(dim=0)
    return torch.sum((mean_w - target) ** 2)


def _build_dataset_map(train_loader):
    if hasattr(train_loader, "streams"):
        return {s.dataset_id: s.loader.dataset.dataset for s in train_loader.streams}
    dataset = train_loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return {0: dataset}


def _read_future_state(dataset, sample_id, t_idx, device):
    f = dataset._get_file()
    abs_t = dataset.t_indices[t_idx]
    arr = dataset._read_frame(f, sample_id, abs_t)
    m, s = dataset.stats["mean"], dataset.stats["std"]
    arr = (arr - m) / s
    return torch.from_numpy(arr).to(device).float()


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_dir = create_run_dir("primitive")
    logger, log_path = setup_logger("primitive", run_dir)
    linked = set_latest_link(run_dir, "latest_primitive")
    if not linked:
        logger.info("Could not update outputs/latest_primitive link.")
    logger.info("Log file: %s", log_path)

    if "seed" in config:
        torch.manual_seed(config["seed"])

    data_conf = config["data"]
    train_loader, val_loaders, test_loaders, dataset_specs = build_loaders(data_conf)
    dataset_map = _build_dataset_map(train_loader)

    for spec in dataset_specs:
        logger.info(
            "Dataset[%d] %s params=%s train_pairs=%d val_pairs=%d test_pairs=%d weight=%s",
            spec["id"],
            spec["name"],
            spec.get("params"),
            spec.get("train_pairs", 0),
            spec.get("val_pairs", 0),
            spec.get("test_pairs", 0),
            spec.get("weight", "n/a"),
        )

    model_conf = config["model"]
    num_primitives = model_conf["num_primitives"]
    primitive_conf = model_conf["primitive"]
    router_conf = model_conf["router"]

    primitives = []
    if hasattr(train_loader, "streams"):
        sample_ds = train_loader.streams[0].loader.dataset
    else:
        sample_ds = train_loader.dataset
    input_channels = sample_ds.input_channels
    output_channels = sample_ds.solution_channels

    for _ in range(num_primitives):
        primitives.append(
            PrimitiveOperator(
                modes=primitive_conf["modes"],
                width=primitive_conf["width"],
                depth=primitive_conf["depth"],
                input_channels=input_channels,
                output_channels=output_channels,
                fc_dim=primitive_conf.get("fc_dim", 128),
                primitive_type=primitive_conf.get("type", "fno"),
                kernel_size=primitive_conf.get("kernel_size", 5),
            )
        )

    num_datasets = len(dataset_specs)
    equation_terms = data_conf.get("equation_terms", [])
    equation_text_dim = int(data_conf.get("equation_text_dim", 0))
    equation_dim = len(equation_terms) + equation_text_dim
    if equation_dim == 0:
        equation_dim = router_conf.get("equation_dim", 0)
    router = Router(
        num_primitives=num_primitives,
        state_channels=output_channels,
        hidden_dim=router_conf.get("hidden_dim", 64),
        top_k=router_conf.get("top_k", 2),
        stats=router_conf.get("stats", ["mean", "std", "min", "max"]),
        local_segments=router_conf.get("local_segments", 0),
        local_stats=router_conf.get("local_stats", ["mean", "std"]),
        fft_bins=router_conf.get("fft_bins", 0),
        equation_dim=equation_dim,
        pde_param_dim=router_conf.get("pde_param_dim", 0),
        num_datasets=num_datasets,
        dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
    )

    agg_conf = config.get("model", {}).get("aggregator", {})
    agg_type = agg_conf.get("type", "linear")
    agg_hidden = int(agg_conf.get("hidden_dim", 32))
    aggregator = PrimitiveAggregator(num_primitives, agg_type=agg_type, hidden_dim=agg_hidden)

    model = PrimitiveCNO(primitives, router, output_channels, aggregator=aggregator)
    model = model.cuda() if torch.cuda.is_available() else model

    opt = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = _nrmse
    epochs = config["training"]["epochs"]
    entropy_weight = config["training"].get("entropy_weight", 0.0)
    diversity_weight = config["training"].get("diversity_weight", 0.0)

    output_model = run_dir / Path(config["training"]["output_model"]).name
    best_val_loss = float("inf")
    train_history = []
    val_history = []
    val_by_dataset = {}
    for name, loader, _, _ in val_loaders:
        if loader is None:
            continue
        val_by_dataset[name] = []

    rollout_steps = max(1, int(config["training"].get("rollout_steps", 1)))
    rollout_gamma = float(config["training"].get("rollout_gamma", 1.0))
    ss_conf = config["training"].get("scheduled_sampling", {})
    ss_start = float(ss_conf.get("start", 0.0))
    ss_end = float(ss_conf.get("end", 0.0))
    load_balance_weight = float(config["training"].get("load_balance_weight", 0.0))
    sparse_primitives = bool(config["training"].get("sparse_primitives", True))
    warmup_epochs = int(config["training"].get("topk_warmup_epochs", 0))
    base_top_k = router.top_k

    include_grid = getattr(sample_ds, "include_grid", False)
    device = next(model.parameters()).device

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0:
            router.top_k = num_primitives if epoch <= warmup_epochs else base_top_k
        model.train()
        train_loss = 0.0
        if epochs > 1:
            ss_prob = ss_start + (ss_end - ss_start) * (epoch - 1) / (epochs - 1)
        else:
            ss_prob = ss_end
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for batch in train_pbar:
            u_t, u_tp1, pde_params, dataset_id, equation, meta = _unpack_batch(
                batch, device=device, return_meta=True
            )

            opt.zero_grad()
            grid = u_t[..., output_channels:] if include_grid else None
            current_input = u_t
            total_loss = 0.0
            weight_sum = 0.0

            sample_ids = None
            t_idx = None
            if meta and isinstance(meta, dict):
                sample_ids = meta.get("sample_id")
                t_idx = meta.get("t_idx")
            if sample_ids is not None:
                sample_ids = sample_ids.to("cpu")
            if t_idx is not None:
                t_idx = t_idx.to("cpu")

            for step in range(1, rollout_steps + 1):
                use_sparse = sparse_primitives and router.top_k < num_primitives
                if diversity_weight > 0:
                    pred, weights, _, delta_stack = model(
                        current_input,
                        pde_params=pde_params,
                        dataset_id=dataset_id,
                        equation=equation,
                        return_weights=True,
                        return_deltas=True,
                        sparse_primitives=use_sparse,
                    )
                else:
                    pred, weights, _ = model(
                        current_input,
                        pde_params=pde_params,
                        dataset_id=dataset_id,
                        equation=equation,
                        return_weights=True,
                        return_deltas=False,
                        sparse_primitives=use_sparse,
                    )
                    delta_stack = None

                if step == 1:
                    gt = u_tp1
                    valid = torch.ones(gt.size(0), dtype=torch.bool, device=device)
                else:
                    gt = pred.detach().clone()
                    valid = torch.ones(gt.size(0), dtype=torch.bool, device=device)
                    if sample_ids is not None and t_idx is not None and dataset_id is not None:
                        for i in range(gt.size(0)):
                            ds_id = int(dataset_id[i].item())
                            ds = dataset_map.get(ds_id)
                            if ds is None:
                                valid[i] = False
                                continue
                            next_idx = int(t_idx[i].item()) + step
                            if next_idx >= ds.timesteps:
                                valid[i] = False
                                continue
                            gt[i] = _read_future_state(ds, int(sample_ids[i].item()), next_idx, device)

                if valid.any():
                    step_loss = criterion(pred[valid], gt[valid])
                    step_weight = rollout_gamma ** (step - 1)
                    total_loss = total_loss + step_weight * step_loss
                    weight_sum = weight_sum + step_weight

                if entropy_weight > 0:
                    entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                    total_loss = total_loss + entropy_weight * entropy
                if diversity_weight > 0 and step == 1:
                    diversity = _diversity_penalty(delta_stack)
                    total_loss = total_loss + diversity_weight * diversity
                if load_balance_weight > 0:
                    total_loss = total_loss + load_balance_weight * _load_balance_loss(weights)

                if step < rollout_steps:
                    if ss_prob > 0:
                        use_pred = torch.rand(pred.size(0), device=pred.device) < ss_prob
                        mask = use_pred & valid
                        mask = mask.view(-1, 1, 1)
                        next_state = torch.where(mask, pred, gt.detach())
                    else:
                        next_state = gt.detach()
                    if include_grid:
                        current_input = torch.cat([next_state, grid], dim=-1)
                    else:
                        current_input = next_state

            if weight_sum > 0:
                loss = total_loss / weight_sum
            else:
                loss = total_loss

            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        val_loss = 0.0
        if val_loaders:
            model.eval()
            with torch.no_grad():
                total_batches = 0
                for name, loader, _, _ in val_loaders:
                    ds_loss = 0.0
                    val_pbar = tqdm(loader, desc=f"Val {name} {epoch}/{epochs}", leave=False)
                    for batch in val_pbar:
                        u_t, u_tp1, pde_params, dataset_id, equation = _unpack_batch(
                            batch, device=next(model.parameters()).device
                        )
                        pred = model(
                            u_t,
                            pde_params=pde_params,
                            dataset_id=dataset_id,
                            equation=equation,
                            sparse_primitives=sparse_primitives,
                        )
                        loss = criterion(pred, u_tp1)
                        ds_loss += loss.item()
                        val_pbar.set_postfix(loss=loss.item())
                    ds_loss /= max(len(loader), 1)
                    logger.info("Val NRMSE (%s): %.6f", name, ds_loss)
                    if name in val_by_dataset:
                        val_by_dataset[name].append(ds_loss)
                    val_loss += ds_loss * len(loader)
                    total_batches += len(loader)
                if total_batches > 0:
                    val_loss /= total_batches
        else:
            val_loss = train_loss

        logger.info(
            "[Epoch %d] Train NRMSE: %.6f, Val NRMSE: %.6f",
            epoch,
            train_loss,
            val_loss,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved_top_k = router.top_k
            router.top_k = base_top_k
            torch.save(model, output_model)
            router.top_k = saved_top_k

    _save_training_plots(run_dir, train_history, val_history, val_by_dataset)

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": None,
        "num_primitives": num_primitives,
    }
    with open(run_dir / "primitive_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Primitive training complete. Model saved to: %s", str(output_model))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_primitives.py --config configs/primitive.yaml")
    else:
        cfg_path = sys.argv[-1]
        main(cfg_path)
