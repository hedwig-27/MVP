import json
import sys
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from utils import get_progress

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.multi_loader import build_loaders
from models.fno1d import FNO1D
from utils import create_run_dir, log_config, set_latest_link, setup_logger


def _nrmse(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    denom = torch.sqrt(torch.mean(target ** 2)) + eps
    return torch.sqrt(mse) / denom


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


def _unpack_batch(batch, device):
    if len(batch) >= 4 and isinstance(batch[2], dict):
        x, y, _, _ = batch[0], batch[1], batch[2], batch[3]
    else:
        x, y = batch[0], batch[1]
    x = x.float().to(device)
    y = y.float().to(device)
    return x, y, None, None


def main(config_path):
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    run_dir = create_run_dir("fno")
    logger, log_path = setup_logger("fno", run_dir)
    linked = set_latest_link(run_dir, "latest_fno")
    if not linked:
        logger.info("Could not update outputs/latest_fno link.")
    logger.info("Log file: %s", log_path)
    log_config(logger, config, title="Experiment config")
    # Set random seed for reproducibility (if needed)
    if 'seed' in config:
        torch.manual_seed(config['seed'])
    # Prepare datasets
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

    # Initialize FNO model
    model_conf = config["model"]
    if hasattr(train_loader, "streams"):
        sample_ds = train_loader.streams[0].loader.dataset
    else:
        sample_ds = train_loader.dataset
    input_channels = model_conf.get("input_channels", sample_ds.input_channels)
    output_channels = model_conf.get("output_channels", sample_ds.solution_channels)
    model = FNO1D(
        modes=model_conf["modes"],
        width=model_conf["width"],
        depth=model_conf["depth"],
        input_channels=input_channels,
        output_channels=output_channels,
        fc_dim=model_conf.get("fc_dim", 128),
    )
    model = model.cuda() if torch.cuda.is_available() else model
    # Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = _nrmse
    epochs = config["training"]["epochs"]

    output_model = run_dir / Path(config["training"]["output_model"]).name

    best_val_loss = float("inf")
    train_history = []
    val_history = []
    val_by_dataset = {}
    for name, loader, _, _ in val_loaders:
        if loader is None:
            continue
        val_by_dataset[name] = []
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_pbar = get_progress(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for batch in train_pbar:
            u_t, u_tp1, _, _ = _unpack_batch(batch, device=next(model.parameters()).device)
            opt.zero_grad()
            pred = model(u_t)
            loss = criterion(pred, u_tp1)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        # Validate
        val_loss = 0.0
        if val_loaders:
            model.eval()
            with torch.no_grad():
                total_batches = 0
                for name, loader, _, _ in val_loaders:
                    ds_loss = 0.0
                    val_pbar = get_progress(loader, desc=f"Val {name} {epoch}/{epochs}", leave=False)
                    for batch in val_pbar:
                        u_t, u_tp1, _, _ = _unpack_batch(
                            batch, device=next(model.parameters()).device
                        )
                        pred = model(u_t)
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

        # Logging
        logger.info("[Epoch %d] Train NRMSE: %.6f, Val NRMSE: %.6f", epoch, train_loss, val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, output_model)
    _save_training_plots(run_dir, train_history, val_history, val_by_dataset)
    # Save metrics
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": None
    }
    with open(run_dir / "fno_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Baseline FNO training complete. Model saved to: %s", str(output_model))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_fno.py --config path/to/fno_burgers.yaml")
    else:
        # Simple arg parsing
        cfg_path = sys.argv[-1]
        main(cfg_path)
