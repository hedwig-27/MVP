import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.multi_loader import build_loaders
from models.primitive_composer import PrimitiveCNO, Router
from models.primitive_operator import PrimitiveOperator
from utils import create_run_dir, set_latest_link, setup_logger


def _diversity_penalty(delta_stack):
    # delta_stack: (batch, x, channels, k)
    bsz, xsz, csz, ksz = delta_stack.shape
    flat = delta_stack.reshape(bsz, xsz * csz, ksz).permute(0, 2, 1)
    flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(flat, flat.transpose(1, 2))
    off_diag = sim - torch.eye(ksz, device=sim.device).unsqueeze(0)
    return off_diag.abs().mean()


def _unpack_batch(batch, device):
    if len(batch) >= 4 and isinstance(batch[2], dict):
        x, y, cond, _ = batch[0], batch[1], batch[2], batch[3]
        dataset_id = cond.get("dataset_id")
        pde_params = cond.get("params")
    else:
        x, y = batch[0], batch[1]
        dataset_id = None
        pde_params = None
    x = x.float().to(device)
    y = y.float().to(device)
    if pde_params is not None:
        pde_params = pde_params.to(device).float()
    if dataset_id is not None:
        dataset_id = dataset_id.to(device).long()
    return x, y, pde_params, dataset_id


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
    router = Router(
        num_primitives=num_primitives,
        state_channels=output_channels,
        hidden_dim=router_conf.get("hidden_dim", 64),
        top_k=router_conf.get("top_k", 2),
        stats=router_conf.get("stats", ["mean", "std", "min", "max"]),
        fft_bins=router_conf.get("fft_bins", 0),
        pde_param_dim=router_conf.get("pde_param_dim", 0),
        num_datasets=num_datasets,
        dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
    )

    model = PrimitiveCNO(primitives, router, output_channels)
    model = model.cuda() if torch.cuda.is_available() else model

    opt = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()
    epochs = config["training"]["epochs"]
    entropy_weight = config["training"].get("entropy_weight", 0.0)
    diversity_weight = config["training"].get("diversity_weight", 0.0)

    output_model = run_dir / Path(config["training"]["output_model"]).name
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for batch in train_pbar:
            u_t, u_tp1, pde_params, dataset_id = _unpack_batch(batch, device=next(model.parameters()).device)
            u_t = u_t.float()
            u_tp1 = u_tp1.float()
            u_t = u_t.to(next(model.parameters()).device)
            u_tp1 = u_tp1.to(next(model.parameters()).device)
            if pde_params is not None:
                pde_params = pde_params.to(next(model.parameters()).device)

            opt.zero_grad()
            pred, weights, _, delta_stack = model(
                u_t,
                pde_params=pde_params,
                dataset_id=dataset_id,
                return_weights=True,
                return_deltas=True,
            )
            loss = criterion(pred, u_tp1)

            if entropy_weight > 0:
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                loss = loss + entropy_weight * entropy
            if diversity_weight > 0:
                diversity = _diversity_penalty(delta_stack)
                loss = loss + diversity_weight * diversity

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
                        u_t, u_tp1, pde_params, dataset_id = _unpack_batch(
                            batch, device=next(model.parameters()).device
                        )
                        pred = model(u_t, pde_params=pde_params, dataset_id=dataset_id)
                        loss = criterion(pred, u_tp1)
                        ds_loss += loss.item()
                        val_pbar.set_postfix(loss=loss.item())
                    ds_loss /= max(len(loader), 1)
                    logger.info("Val Loss (%s): %.6f", name, ds_loss)
                    val_loss += ds_loss * len(loader)
                    total_batches += len(loader)
                if total_batches > 0:
                    val_loss /= total_batches
        else:
            val_loss = train_loss

        logger.info(
            "[Epoch %d] Train Loss: %.6f, Val Loss: %.6f",
            epoch,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, output_model)

    test_loss = 0.0
    if test_loaders:
        model.eval()
        with torch.no_grad():
            total_batches = 0
            for name, loader, _, _ in test_loaders:
                ds_loss = 0.0
                test_pbar = tqdm(loader, desc=f"Test {name}", leave=False)
                for batch in test_pbar:
                    u_t, u_tp1, pde_params, dataset_id = _unpack_batch(
                        batch, device=next(model.parameters()).device
                    )
                    pred = model(u_t, pde_params=pde_params, dataset_id=dataset_id)
                    loss = criterion(pred, u_tp1)
                    ds_loss += loss.item()
                    test_pbar.set_postfix(loss=loss.item())
                ds_loss /= max(len(loader), 1)
                logger.info("Test Loss (%s): %.6f", name, ds_loss)
                test_loss += ds_loss * len(loader)
                total_batches += len(loader)
            if total_batches > 0:
                test_loss /= total_batches

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss if test_loaders else None,
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
