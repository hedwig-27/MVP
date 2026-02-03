import json
import sys
import time
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
from models.primitive_composer import (
    PrimitiveAggregator,
    PrimitiveCNO,
    Router,
    TransformerRouter,
    CompositePDEModel,
    TermLibrary,
)
from models.primitive_operator import PrimitiveOperator
from utils import create_run_dir, set_latest_link, setup_logger

TQDM_KWARGS = dict(ncols=80, dynamic_ncols=True, bar_format="{l_bar}{bar:10}{r_bar}")


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


def _set_requires_grad(module, flag):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def _term_align_loss(term_outputs, term_features, output_channels):
    if not term_outputs or not term_features:
        return 0.0
    total = 0.0
    count = 0
    for term, out in term_outputs.items():
        feats = term_features.get(term)
        if feats is None:
            continue
        if feats.size(-1) < output_channels:
            continue
        # select target feature per term
        if term == "reaction" and feats.size(-1) >= output_channels * 2:
            u = feats[..., :output_channels]
            u2 = feats[..., output_channels : 2 * output_channels]
            target = u - u2
        elif term == "sorption" and feats.size(-1) >= output_channels * 2:
            target = feats[..., output_channels : 2 * output_channels]
        else:
            target = feats[..., :output_channels]

        out_flat = out.reshape(out.size(0), -1)
        tgt_flat = target.reshape(target.size(0), -1)
        out_norm = out_flat / (out_flat.norm(dim=1, keepdim=True) + 1e-8)
        tgt_norm = tgt_flat / (tgt_flat.norm(dim=1, keepdim=True) + 1e-8)
        cos = (out_norm * tgt_norm).sum(dim=1).abs().mean()
        total += 1.0 - cos
        count += 1
    return total / max(count, 1)


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


def _load_balance_loss(weights, num_primitives, ignore_index=None):
    if weights is None:
        return 0.0
    if weights.dim() == 3 and weights.size(1) == num_primitives:
        # vector weights: (B, P, C)
        mean_w = weights.mean(dim=(0, 2))
    elif weights.dim() == 3:
        # spatial weights: (B, X, P)
        mean_w = weights.mean(dim=(0, 1))
    else:
        mean_w = weights.mean(dim=0)
    if ignore_index is not None and mean_w.numel() > 1:
        mask = torch.ones_like(mean_w, dtype=torch.bool)
        if 0 <= ignore_index < mean_w.numel():
            mask[ignore_index] = False
            mean_w = mean_w[mask]
            num_primitives = mean_w.numel()
    if num_primitives <= 0:
        return 0.0
    target = torch.full((num_primitives,), 1.0 / num_primitives, device=weights.device, dtype=weights.dtype)
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
    model_type = str(model_conf.get("type", "primitive")).lower()
    router_conf = model_conf.get("router", {})

    if hasattr(train_loader, "streams"):
        sample_ds = train_loader.streams[0].loader.dataset
    else:
        sample_ds = train_loader.dataset
    input_channels = sample_ds.input_channels
    output_channels = sample_ds.solution_channels

    num_datasets = len(dataset_specs)
    equation_terms = data_conf.get("equation_terms", [])
    equation_text_dim = int(data_conf.get("equation_text_dim", 0))
    equation_dim = len(equation_terms) + equation_text_dim
    if equation_dim == 0:
        equation_dim = router_conf.get("equation_dim", 0)

    if model_type == "composite":
        cond_conf = model_conf.get("cond", {})
        use_dataset_embed = bool(cond_conf.get("use_dataset_embed", router_conf.get("use_dataset_embed", False)))
        pde_param_dim = int(router_conf.get("pde_param_dim", 0))
        cond_dim = pde_param_dim + equation_dim
        if use_dataset_embed and num_datasets > 1:
            cond_dim += int(router_conf.get("dataset_embed_dim", 8))

        base_conf = model_conf.get("base_operator", {})
        base_operator = None
        if base_conf and base_conf.get("enabled", True):
            base_operator = PrimitiveOperator(
                modes=base_conf.get("modes", 16),
                width=base_conf.get("width", 64),
                depth=base_conf.get("depth", 4),
                input_channels=input_channels,
                output_channels=output_channels,
                fc_dim=base_conf.get("fc_dim", 128),
                primitive_type=base_conf.get("type", "fno"),
                kernel_size=base_conf.get("kernel_size", 5),
                cond_dim=cond_dim,
            )

        term_conf = model_conf.get("term_library", {})
        term_library = None
        if term_conf.get("enabled", True):
            term_library = TermLibrary(
                term_names=equation_terms,
                output_channels=output_channels,
                hidden_dim=term_conf.get("hidden_dim", 32),
                cond_dim=cond_dim,
                use_scale_mlp=term_conf.get("use_scale_mlp", True),
            )

        residual_conf = model_conf.get("residual", model_conf.get("primitive", {}))
        num_residuals = int(residual_conf.get("num_experts", model_conf.get("num_primitives", 4)))
        residual_experts = []
        for _ in range(num_residuals):
            residual_experts.append(
                PrimitiveOperator(
                    modes=residual_conf.get("modes", 8),
                    width=residual_conf.get("width", 32),
                    depth=residual_conf.get("depth", 3),
                    input_channels=input_channels,
                    output_channels=output_channels,
                    fc_dim=residual_conf.get("fc_dim", 64),
                    primitive_type=residual_conf.get("type", "fno"),
                    kernel_size=residual_conf.get("kernel_size", 5),
                    cond_dim=cond_dim,
                )
            )

        router_type = str(router_conf.get("type", "transformer")).lower()
        top_k = min(int(router_conf.get("top_k", num_residuals)), num_residuals)
        if router_type in ("transformer", "attn", "attention"):
            router = TransformerRouter(
                num_primitives=num_residuals,
                state_channels=output_channels,
                hidden_dim=router_conf.get("d_model", router_conf.get("hidden_dim", 64)),
                top_k=top_k,
                stats=router_conf.get("stats", ["mean", "std", "min", "max"]),
                local_segments=router_conf.get("local_segments", 0),
                local_stats=router_conf.get("local_stats", ["mean", "std"]),
                fft_bins=router_conf.get("fft_bins", 0),
                equation_dim=equation_dim,
                pde_param_dim=pde_param_dim,
                num_datasets=num_datasets,
                dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
                use_dataset_embed=router_conf.get("use_dataset_embed", use_dataset_embed),
                code_dim=router_conf.get("code_dim", 0),
                code_as_weight=router_conf.get("code_as_weight", True),
                use_state_tokens=router_conf.get("use_state_tokens", True),
                state_downsample=router_conf.get("state_downsample", 4),
                use_stats_token=router_conf.get("use_stats_token", True),
                n_layers=router_conf.get("n_layers", 2),
                n_heads=router_conf.get("n_heads", 4),
                ff_dim=router_conf.get("ff_dim"),
                dropout=router_conf.get("dropout", 0.0),
                use_primitive_queries=router_conf.get("use_primitive_queries", True),
            )
        else:
            router = Router(
                num_primitives=num_residuals,
                state_channels=output_channels,
                hidden_dim=router_conf.get("hidden_dim", 64),
                top_k=top_k,
                stats=router_conf.get("stats", ["mean", "std", "min", "max"]),
                local_segments=router_conf.get("local_segments", 0),
                local_stats=router_conf.get("local_stats", ["mean", "std"]),
                fft_bins=router_conf.get("fft_bins", 0),
                equation_dim=equation_dim,
                pde_param_dim=pde_param_dim,
                num_datasets=num_datasets,
                dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
                use_dataset_embed=router_conf.get("use_dataset_embed", use_dataset_embed),
                code_dim=router_conf.get("code_dim", 0),
                code_as_weight=router_conf.get("code_as_weight", True),
                spatial=router_conf.get("spatial", False),
                spatial_kernel=router_conf.get("spatial_kernel", 3),
                spatial_downsample=router_conf.get("spatial_downsample", 1),
                spatial_use_state=router_conf.get("spatial_use_state", True),
                spatial_hidden_dim=router_conf.get("spatial_hidden_dim"),
            )

        delta_clip = model_conf.get("delta_clip", None)
        model = CompositePDEModel(
            base_operator=base_operator,
            term_library=term_library,
            residual_experts=residual_experts,
            router=router,
            output_channels=output_channels,
            term_count=len(equation_terms),
            cond_dim=cond_dim,
            use_dataset_embed=use_dataset_embed,
            dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
            num_datasets=num_datasets,
            delta_clip=delta_clip,
        )
        always_on_index = None
        total_primitives = num_residuals
    else:
        num_primitives = model_conf["num_primitives"]
        primitive_conf = model_conf["primitive"]

        primitives = []
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

        base_operator = None
        base_conf = model_conf.get("base_operator") or model_conf.get("shared_base")
        if base_conf:
            enabled = True
            if isinstance(base_conf, dict):
                enabled = base_conf.get("enabled", True)
            if enabled:
                base_type = base_conf.get("type", primitive_conf.get("type", "fno")) if isinstance(base_conf, dict) else primitive_conf.get("type", "fno")
                base_modes = base_conf.get("modes", max(1, primitive_conf["modes"] // 2)) if isinstance(base_conf, dict) else max(1, primitive_conf["modes"] // 2)
                base_width = base_conf.get("width", max(8, primitive_conf["width"] // 2)) if isinstance(base_conf, dict) else max(8, primitive_conf["width"] // 2)
                base_depth = base_conf.get("depth", max(1, primitive_conf["depth"] // 2)) if isinstance(base_conf, dict) else max(1, primitive_conf["depth"] // 2)
                base_fc_dim = base_conf.get("fc_dim", primitive_conf.get("fc_dim", 128)) if isinstance(base_conf, dict) else primitive_conf.get("fc_dim", 128)
                base_kernel = base_conf.get("kernel_size", primitive_conf.get("kernel_size", 5)) if isinstance(base_conf, dict) else primitive_conf.get("kernel_size", 5)
                base_operator = PrimitiveOperator(
                    modes=base_modes,
                    width=base_width,
                    depth=base_depth,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    fc_dim=base_fc_dim,
                    primitive_type=base_type,
                    kernel_size=base_kernel,
                )

        total_primitives = num_primitives + (1 if base_operator is not None else 0)
        always_on_index = total_primitives - 1 if base_operator is not None else None

        router_type = str(router_conf.get("type", "mlp")).lower()
        if router_type in ("transformer", "attn", "attention"):
            router = TransformerRouter(
                num_primitives=total_primitives,
                state_channels=output_channels,
                hidden_dim=router_conf.get("d_model", router_conf.get("hidden_dim", 64)),
                top_k=router_conf.get("top_k", 2),
                stats=router_conf.get("stats", ["mean", "std", "min", "max"]),
                local_segments=router_conf.get("local_segments", 0),
                local_stats=router_conf.get("local_stats", ["mean", "std"]),
                fft_bins=router_conf.get("fft_bins", 0),
                equation_dim=equation_dim,
                pde_param_dim=router_conf.get("pde_param_dim", 0),
                num_datasets=num_datasets,
                dataset_embed_dim=router_conf.get("dataset_embed_dim", 8),
                use_dataset_embed=router_conf.get("use_dataset_embed", True),
                code_dim=router_conf.get("code_dim", 0),
                code_as_weight=router_conf.get("code_as_weight", True),
                use_state_tokens=router_conf.get("use_state_tokens", True),
                state_downsample=router_conf.get("state_downsample", 4),
                use_stats_token=router_conf.get("use_stats_token", True),
                n_layers=router_conf.get("n_layers", 2),
                n_heads=router_conf.get("n_heads", 4),
                ff_dim=router_conf.get("ff_dim"),
                dropout=router_conf.get("dropout", 0.0),
                use_primitive_queries=router_conf.get("use_primitive_queries", True),
                always_on_index=always_on_index,
            )
        else:
            router = Router(
                num_primitives=total_primitives,
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
                use_dataset_embed=router_conf.get("use_dataset_embed", True),
                code_dim=router_conf.get("code_dim", 0),
                code_as_weight=router_conf.get("code_as_weight", True),
                spatial=router_conf.get("spatial", False),
                spatial_kernel=router_conf.get("spatial_kernel", 3),
                spatial_downsample=router_conf.get("spatial_downsample", 1),
                spatial_use_state=router_conf.get("spatial_use_state", True),
                spatial_hidden_dim=router_conf.get("spatial_hidden_dim"),
                always_on_index=always_on_index,
            )

        agg_conf = config.get("model", {}).get("aggregator", {})
        agg_type = agg_conf.get("type", "linear")
        agg_hidden = int(agg_conf.get("hidden_dim", 32))
        router_code_dim = int(router_conf.get("code_dim", 0))
        agg_code_dim = int(agg_conf.get("code_dim", router_code_dim))
        aggregator = PrimitiveAggregator(
            total_primitives,
            agg_type=agg_type,
            hidden_dim=agg_hidden,
            output_channels=output_channels,
            code_dim=agg_code_dim,
        )

        model = PrimitiveCNO(
            primitives,
            router,
            output_channels,
            aggregator=aggregator,
            base_operator=base_operator,
        )

    model = model.cuda() if torch.cuda.is_available() else model

    include_grid = getattr(sample_ds, "include_grid", False)
    device = next(model.parameters()).device

    if model_type == "composite":
        output_model = run_dir / Path(config["training"]["output_model"]).name
        best_val_loss = float("inf")
        train_history = []
        val_history = []
        val_by_dataset = {}
        for name, loader, _, _ in val_loaders:
            if loader is None:
                continue
            val_by_dataset[name] = []

        train_conf = config["training"]
        lr = float(train_conf.get("learning_rate", 1e-3))
        finetune_lr = float(train_conf.get("finetune_lr", lr * 0.3))
        base_epochs = int(train_conf.get("base_pretrain_epochs", 0))
        residual_epochs = int(train_conf.get("residual_pretrain_epochs", 0))
        joint_epochs = int(train_conf.get("joint_finetune_epochs", 0))
        if base_epochs + residual_epochs + joint_epochs == 0:
            base_epochs = int(train_conf.get("epochs", 50))

        dataset_id_dropout = float(train_conf.get("dataset_id_dropout", 0.0))
        load_balance_weight = float(train_conf.get("load_balance_weight", 0.0))
        term_align_weight = float(train_conf.get("term_align_weight", 0.0))
        rollout_steps = max(1, int(train_conf.get("rollout_steps", 1)))
        rollout_gamma = float(train_conf.get("rollout_gamma", 1.0))
        ss_conf = train_conf.get("scheduled_sampling", {})
        ss_start = float(ss_conf.get("start", 0.0))
        ss_end = float(ss_conf.get("end", 0.0))
        grad_clip = float(train_conf.get("grad_clip", 0.0))

        stages = []
        if base_epochs > 0 and (residual_epochs > 0 or joint_epochs > 0):
            stages.append(("base", base_epochs, True, False, False, True, False, False, lr))
        if residual_epochs > 0:
            # keep base active but frozen so residual learns correction
            stages.append(("residual", residual_epochs, False, True, True, True, True, True, lr))
        if joint_epochs > 0:
            stages.append(("joint", joint_epochs, True, True, True, True, True, True, finetune_lr))
        if not stages:
            stages.append(("joint", base_epochs, True, True, True, True, True, True, lr))

        total_epochs = sum(s[1] for s in stages)
        global_epoch = 0
        for stage_name, stage_epochs, train_base, train_terms, train_residual, use_base, use_terms, use_residual, stage_lr in stages:
            logger.info("Stage %s: epochs=%d lr=%.6f", stage_name, stage_epochs, stage_lr)
            _set_requires_grad(model.base_operator, train_base)
            _set_requires_grad(model.term_library, train_terms)
            _set_requires_grad(model.residual_experts, train_residual)
            _set_requires_grad(model.router, train_residual)
            use_base = bool(use_base and model.base_operator is not None)
            use_terms = bool(use_terms and model.term_library is not None)
            use_residual = bool(use_residual and model.residual_experts and model.router is not None)

            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=stage_lr)
            for epoch in range(1, stage_epochs + 1):
                global_epoch += 1
                epoch_start = time.perf_counter()
                model.train()
                train_loss = 0.0
                stage_rollout_steps = rollout_steps if (use_terms or use_residual) else 1
                if stage_rollout_steps > 1:
                    if total_epochs > 1:
                        ss_prob = ss_start + (ss_end - ss_start) * (global_epoch - 1) / (total_epochs - 1)
                    else:
                        ss_prob = ss_end
                else:
                    ss_prob = 0.0
                train_pbar = tqdm(train_loader, desc=f"{stage_name} {global_epoch}", leave=False, **TQDM_KWARGS)
                for batch in train_pbar:
                    u_t, u_tp1, pde_params, dataset_id, equation, meta = _unpack_batch(
                        batch, device=device, return_meta=True
                    )
                    if dataset_id is not None and dataset_id_dropout > 0.0:
                        if torch.rand(1).item() < dataset_id_dropout:
                            dataset_id = None

                    opt.zero_grad()
                    weights = None
                    align_loss = 0.0
                    total_loss = 0.0
                    weight_sum = 0.0

                    grid = u_t[..., output_channels:] if include_grid else None
                    current_input = u_t

                    sample_ids = None
                    t_idx = None
                    if meta and isinstance(meta, dict):
                        sample_ids = meta.get("sample_id")
                        t_idx = meta.get("t_idx")
                    if sample_ids is not None:
                        sample_ids = sample_ids.to("cpu")
                    if t_idx is not None:
                        t_idx = t_idx.to("cpu")

                    for step in range(1, stage_rollout_steps + 1):
                        if step == 1 and term_align_weight > 0 and use_terms:
                            pred, parts = model(
                                current_input,
                                pde_params=pde_params,
                                dataset_id=dataset_id,
                                equation=equation,
                                return_parts=True,
                                return_terms=True,
                                use_base=use_base,
                                use_terms=use_terms,
                                use_residual=use_residual,
                            )
                            weights = parts.get("residual_weights")
                            align_loss = _term_align_loss(
                                parts.get("term_outputs", {}),
                                parts.get("term_features", {}),
                                output_channels,
                            )
                        else:
                            pred, weights, _ = model(
                                current_input,
                                pde_params=pde_params,
                                dataset_id=dataset_id,
                                equation=equation,
                                return_weights=True,
                                use_base=use_base,
                                use_terms=use_terms,
                                use_residual=use_residual,
                            )

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
                            step_loss = _nrmse(pred[valid], gt[valid])
                            step_weight = rollout_gamma ** (step - 1)
                            total_loss = total_loss + step_weight * step_loss
                            weight_sum = weight_sum + step_weight

                        if load_balance_weight > 0 and weights is not None:
                            lb = _load_balance_loss(
                                weights, getattr(model.router, "num_primitives", weights.size(-1))
                            )
                            total_loss = total_loss + load_balance_weight * lb

                        if step < stage_rollout_steps:
                            if stage_rollout_steps > 1 and ss_prob > 0:
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
                    if term_align_weight > 0 and use_terms:
                        loss = loss + term_align_weight * align_loss

                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    train_loss += loss.item()
                    train_pbar.set_postfix(loss=loss.item())

                train_loss /= max(len(train_loader), 1)

                # validation
                val_loss = 0.0
                if val_loaders:
                    model.eval()
                    with torch.no_grad():
                        total_batches = 0
                        for name, loader, _, _ in val_loaders:
                            ds_loss = 0.0
                            val_pbar = tqdm(loader, desc=f"Val {name} {global_epoch}", leave=False, **TQDM_KWARGS)
                            for batch in val_pbar:
                                u_t, u_tp1, pde_params, dataset_id, equation = _unpack_batch(
                                    batch, device=device
                                )
                                pred = model(
                                    u_t,
                                    pde_params=pde_params,
                                    dataset_id=dataset_id,
                                    equation=equation,
                                    use_base=use_base,
                                    use_terms=use_terms,
                                    use_residual=use_residual,
                                )
                                loss = _nrmse(pred, u_tp1)
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
                    "[Stage %s | Epoch %d] Train NRMSE: %.6f, Val NRMSE: %.6f, time: %.1fs",
                    stage_name,
                    global_epoch,
                    train_loss,
                    val_loss,
                    time.perf_counter() - epoch_start,
                )
                train_history.append(train_loss)
                val_history.append(val_loss)

                has_aux = bool(model.term_library is not None or (model.residual_experts and model.router))
                track_best = (not has_aux) or use_terms or use_residual
                if track_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model, output_model)

        _save_training_plots(run_dir, train_history, val_history, val_by_dataset)
        metrics = {
            "train_loss": train_history[-1] if train_history else None,
            "val_loss": val_history[-1] if val_history else None,
            "test_loss": None,
        }
        with open(run_dir / "primitive_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Composite training complete. Model saved to: %s", str(output_model))
        return

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
        epoch_start = time.perf_counter()
        if warmup_epochs > 0:
            router.top_k = router.num_primitives if epoch <= warmup_epochs else base_top_k
        model.train()
        train_loss = 0.0
        if epochs > 1:
            ss_prob = ss_start + (ss_end - ss_start) * (epoch - 1) / (epochs - 1)
        else:
            ss_prob = ss_end
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, **TQDM_KWARGS)
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
                use_sparse = sparse_primitives and router.top_k < router.num_primitives
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
                    if weights.dim() == 3 and weights.size(1) == router.num_primitives:
                        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
                    else:
                        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                    total_loss = total_loss + entropy_weight * entropy
                if diversity_weight > 0 and step == 1:
                    diversity = _diversity_penalty(delta_stack)
                    total_loss = total_loss + diversity_weight * diversity
                if load_balance_weight > 0:
                    total_loss = total_loss + load_balance_weight * _load_balance_loss(
                        weights, router.num_primitives, ignore_index=always_on_index
                    )

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
                    val_pbar = tqdm(loader, desc=f"Val {name} {epoch}/{epochs}", leave=False, **TQDM_KWARGS)
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
            "[Epoch %d] Train NRMSE: %.6f, Val NRMSE: %.6f, time: %.1fs",
            epoch,
            train_loss,
            val_loss,
            time.perf_counter() - epoch_start,
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
