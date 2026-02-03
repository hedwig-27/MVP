from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataloader.pdebench_loader import build_datasets


def _merge_conf(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _dataset_name(conf: Dict[str, Any]) -> str:
    if "name" in conf:
        return str(conf["name"])
    path = Path(conf["path"])
    return path.stem


def _parse_params(params: Any) -> Tuple[Optional[torch.Tensor], List[str]]:
    if params is None:
        return None, []
    if isinstance(params, dict):
        keys = sorted(params.keys())
        vals = [float(params[k]) for k in keys]
        return torch.tensor(vals, dtype=torch.float32), keys
    if isinstance(params, (list, tuple)):
        vals = [float(v) for v in params]
        keys = [f"p{i}" for i in range(len(vals))]
        return torch.tensor(vals, dtype=torch.float32), keys
    return torch.tensor([float(params)], dtype=torch.float32), ["p0"]


def _encode_equation_text(text: str, dim: int) -> Optional[torch.Tensor]:
    if not text or dim <= 0:
        return None
    cleaned = re.sub(r"\s+", "", str(text).lower())
    if not cleaned:
        return None
    vec = np.zeros(dim, dtype=np.float32)
    n = 3
    if len(cleaned) < n:
        n = 1
    for i in range(0, len(cleaned) - n + 1):
        ngram = cleaned[i : i + n]
        digest = hashlib.md5(ngram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec) + 1e-8
    vec = vec / norm
    return torch.tensor(vec, dtype=torch.float32)


def _build_equation_vector(
    terms: Optional[Sequence[str]],
    conf: Dict[str, Any],
    *,
    text: Optional[str] = None,
    text_dim: int = 0,
) -> Tuple[Optional[torch.Tensor], List[str]]:
    term_list = [str(t) for t in terms] if terms else []
    coeffs = {t: 0.0 for t in term_list}
    eq_conf = conf.get("equation_coeffs") or conf.get("equation")
    if isinstance(eq_conf, dict):
        for key, val in eq_conf.items():
            if key in coeffs:
                coeffs[key] = float(val)
    else:
        params_conf = conf.get("params")
        if isinstance(params_conf, dict):
            if "beta" in params_conf and "advection" in coeffs:
                coeffs["advection"] = float(params_conf["beta"])
            if "nu" in params_conf and "diffusion" in coeffs:
                coeffs["diffusion"] = float(params_conf["nu"])
            if "rho" in params_conf and "reaction" in coeffs:
                coeffs["reaction"] = float(params_conf["rho"])
        name = _dataset_name(conf).lower()
        if "burgers" in name and "nonlinear_advection" in coeffs and coeffs["nonlinear_advection"] == 0.0:
            coeffs["nonlinear_advection"] = 1.0
        if "sorp" in name and "sorption" in coeffs and coeffs["sorption"] == 0.0:
            coeffs["sorption"] = 1.0
        if ("cns" in name or "cfd" in name or "navier" in name) and "cns" in coeffs and coeffs["cns"] == 0.0:
            coeffs["cns"] = 1.0
    parts: List[torch.Tensor] = []
    if term_list:
        parts.append(torch.tensor([coeffs[t] for t in term_list], dtype=torch.float32))
    text_vec = None
    if text_dim > 0:
        text_vec = _encode_equation_text(text, text_dim)
        if text_vec is None:
            text_vec = torch.zeros(text_dim, dtype=torch.float32)
        parts.append(text_vec)
    if not parts:
        return None, term_list
    vec = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    return vec, term_list


class DatasetWithMeta(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_id: int,
        dataset_name: str,
        params: Optional[torch.Tensor] = None,
        param_names: Optional[List[str]] = None,
        equation: Optional[torch.Tensor] = None,
        equation_terms: Optional[List[str]] = None,
        dt: Optional[float] = None,
        append_dt: bool = False,
    ):
        self.dataset = dataset
        self.dataset_id = int(dataset_id)
        self.dataset_name = dataset_name
        self.params = params
        self.param_names = param_names or []
        self.dt = float(dt) if dt is not None else None
        if append_dt:
            dt_val = self.dt if self.dt is not None else 1.0
            dt_tensor = torch.tensor([dt_val], dtype=torch.float32)
            if self.params is None:
                self.params = dt_tensor
                self.param_names = ["dt"]
            else:
                self.params = torch.cat([self.params, dt_tensor], dim=0)
                self.param_names = list(self.param_names) + ["dt"]
        self.equation = equation
        self.equation_terms = equation_terms or []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int):
        x, y, meta = self.dataset[idx]
        cond: Dict[str, Any] = {"dataset_id": self.dataset_id}
        if self.params is not None:
            cond["params"] = self.params
            cond["param_names"] = self.param_names
        if self.equation is not None:
            cond["equation"] = self.equation
        meta = dict(meta)
        meta["dataset_id"] = self.dataset_id
        meta["dataset_name"] = self.dataset_name
        return x, y, cond, meta


@dataclass
class DatasetStream:
    name: str
    loader: DataLoader
    weight: float
    dataset_id: int


def _normalize_weights(streams: Sequence[DatasetStream]) -> torch.Tensor:
    w = torch.tensor([max(0.0, float(s.weight)) for s in streams], dtype=torch.float32)
    if torch.all(w == 0):
        w = torch.ones_like(w)
    return w / w.sum()


def _combine_stats(datasets: Sequence[Dataset]) -> Optional[Dict[str, float]]:
    total_count = 0.0
    total_sum = 0.0
    total_sumsq = 0.0
    for ds in datasets:
        if not hasattr(ds, "stats") or ds.stats is None:
            continue
        if not hasattr(ds, "stats_count"):
            continue
        count = float(ds.stats_count())
        if count <= 0:
            continue
        mean = float(ds.stats.get("mean", 0.0))
        std = float(ds.stats.get("std", 1.0))
        var = std * std
        total_count += count
        total_sum += mean * count
        total_sumsq += (var + mean * mean) * count
    if total_count <= 0:
        return None
    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean
    std = float(np.sqrt(max(var, 0.0)) + 1e-8)
    return {"mean": float(mean), "std": std}


class MixLoader:
    def __init__(
        self,
        streams: Sequence[DatasetStream],
        *,
        steps_per_epoch: int,
        seed: int = 0,
    ):
        if len(streams) == 0:
            raise ValueError("MixLoader requires at least one DatasetStream")
        self.streams = list(streams)
        self.steps_per_epoch = int(steps_per_epoch)
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        self.weights = _normalize_weights(self.streams)
        self.gen = torch.Generator().manual_seed(int(seed))
        self._iters = [iter(s.loader) for s in self.streams]

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _next_from_stream(self, i: int):
        try:
            return next(self._iters[i])
        except StopIteration:
            self._iters[i] = iter(self.streams[i].loader)
            return next(self._iters[i])

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            i = int(torch.multinomial(self.weights, num_samples=1, replacement=True, generator=self.gen).item())
            yield self._next_from_stream(i)


def build_loaders(data_conf: Dict[str, Any], split: str = "train"):
    """
    Returns loaders for train/val/test (train may be MixLoader).
    For multi-dataset configs, val/test are lists of (name, loader, dataset_id, params).
    """
    if "datasets" not in data_conf:
        train_ds, val_ds, test_ds, _ = build_datasets(data_conf)
        name = _dataset_name(data_conf)
        params, param_names = _parse_params(data_conf.get("params"))
        append_dt = bool(data_conf.get("append_dt_to_params", False))
        wrapped_train = DatasetWithMeta(train_ds, 0, name, params, param_names, dt=getattr(train_ds, "dt", None), append_dt=append_dt)
        wrapped_val = DatasetWithMeta(val_ds, 0, name, params, param_names, dt=getattr(val_ds, "dt", None), append_dt=append_dt) if val_ds else None
        wrapped_test = DatasetWithMeta(test_ds, 0, name, params, param_names, dt=getattr(test_ds, "dt", None), append_dt=append_dt) if test_ds else None

        batch_size = data_conf.get("batch_size", 32)
        num_workers = data_conf.get("num_workers", 0)
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            wrapped_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = None
        test_loader = None
        if wrapped_val:
            val_loader = DataLoader(
                wrapped_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        if wrapped_test:
            test_loader = DataLoader(
                wrapped_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

        specs = [
            {
                "id": 0,
                "name": name,
                "params": params.tolist() if params is not None else None,
                "param_names": param_names,
                "train_pairs": len(train_ds),
                "val_pairs": len(val_ds) if val_ds else 0,
                "test_pairs": len(test_ds) if test_ds else 0,
            }
        ]
        return train_loader, [("single", val_loader)], [("single", test_loader)], specs

    datasets_conf = data_conf["datasets"]
    if split == "test" and data_conf.get("eval_datasets"):
        datasets_conf = data_conf["eval_datasets"]
    default_conf = dict(data_conf)
    default_conf.pop("datasets", None)
    default_conf.pop("eval_datasets", None)
    default_conf.pop("eval_equation_datasets", None)
    global_normalize = bool(default_conf.get("global_normalize", False))
    append_dt = bool(default_conf.get("append_dt_to_params", False))
    if global_normalize:
        default_conf["normalize"] = True
    eval_datasets_conf = data_conf.get("eval_datasets")
    equation_terms = data_conf.get("equation_terms")
    equation_text_dim = int(data_conf.get("equation_text_dim", 0))

    batch_size = data_conf.get("batch_size", 32)
    num_workers = data_conf.get("num_workers", 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    steps_per_epoch = data_conf.get("steps_per_epoch")
    val_steps_per_epoch = data_conf.get("val_steps_per_epoch")
    seed = data_conf.get("seed", 0)

    streams: List[DatasetStream] = []
    val_loaders: List[Tuple[str, DataLoader, int, Optional[torch.Tensor]]] = []
    test_loaders: List[Tuple[str, DataLoader, int, Optional[torch.Tensor]]] = []
    specs: List[Dict[str, Any]] = []
    train_datasets_for_stats: List[Dataset] = []
    all_datasets: List[Dataset] = []

    for idx, ds_conf in enumerate(datasets_conf):
        merged = _merge_conf(default_conf, ds_conf)
        train_ds, val_ds, test_ds, _ = build_datasets(merged)
        if global_normalize and merged.get("normalize", False):
            train_datasets_for_stats.append(train_ds)
        all_datasets.extend([d for d in (train_ds, val_ds, test_ds) if d is not None])
        name = _dataset_name(merged)
        params, param_names = _parse_params(merged.get("params"))
        eq_terms = equation_terms or merged.get("equation_terms")
        equation_text = merged.get("equation_text")
        equation, eq_terms = _build_equation_vector(
            eq_terms,
            merged,
            text=equation_text,
            text_dim=equation_text_dim,
        )
        weight = float(merged.get("weight", 1.0))

        wrapped_train = DatasetWithMeta(
            train_ds, idx, name, params, param_names, equation, eq_terms,
            dt=getattr(train_ds, "dt", None), append_dt=append_dt
        )
        wrapped_val = DatasetWithMeta(
            val_ds, idx, name, params, param_names, equation, eq_terms,
            dt=getattr(val_ds, "dt", None), append_dt=append_dt
        ) if val_ds else None
        wrapped_test = DatasetWithMeta(
            test_ds, idx, name, params, param_names, equation, eq_terms,
            dt=getattr(test_ds, "dt", None), append_dt=append_dt
        ) if test_ds else None

        train_loader = DataLoader(
            wrapped_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        streams.append(DatasetStream(name=name, loader=train_loader, weight=weight, dataset_id=idx))

        if wrapped_val:
            val_loader = DataLoader(
                wrapped_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            val_loaders.append((name, val_loader, idx, params))
        if wrapped_test:
            test_loader = DataLoader(
                wrapped_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            test_loaders.append((name, test_loader, idx, params))

        specs.append(
            {
                "id": idx,
                "name": name,
                "params": params.tolist() if params is not None else None,
                "param_names": param_names,
                "train_pairs": len(train_ds),
                "val_pairs": len(val_ds) if val_ds else 0,
                "test_pairs": len(test_ds) if test_ds else 0,
                "weight": weight,
            }
        )

    if global_normalize:
        stats = _combine_stats(train_datasets_for_stats)
        if stats is not None:
            for ds in all_datasets:
                ds.stats = stats

    if steps_per_epoch is None:
        steps_per_epoch = sum(len(s.loader) for s in streams)
    train_loader = MixLoader(streams, steps_per_epoch=steps_per_epoch, seed=seed)

    if val_steps_per_epoch is not None and val_loaders:
        val_streams = [
            DatasetStream(name=n, loader=ldr, weight=1.0, dataset_id=idx)
            for n, ldr, idx, _ in val_loaders
        ]
        val_loader = MixLoader(val_streams, steps_per_epoch=val_steps_per_epoch, seed=seed + 1)
        val_loaders = [("mix", val_loader, -1, None)]

    if eval_datasets_conf:
        test_loaders = build_eval_loaders(_merge_conf(default_conf, {"datasets": eval_datasets_conf}), split="test")

    return train_loader, val_loaders, test_loaders, specs


def build_eval_loaders(data_conf: Dict[str, Any], split: str = "test"):
    if "datasets" not in data_conf:
        ds, _, _, _ = build_datasets(data_conf)
        name = _dataset_name(data_conf)
        params, _ = _parse_params(data_conf.get("params"))
        equation_terms = data_conf.get("equation_terms")
        equation_text_dim = int(data_conf.get("equation_text_dim", 0))
        equation_text = data_conf.get("equation_text")
        equation, eq_terms = _build_equation_vector(
            equation_terms,
            data_conf,
            text=equation_text,
            text_dim=equation_text_dim,
        )
        append_dt = bool(data_conf.get("append_dt_to_params", False))
        wrapped = DatasetWithMeta(
            ds, 0, name, params, equation=equation, equation_terms=eq_terms,
            dt=getattr(ds, "dt", None), append_dt=append_dt
        )
        batch_size = data_conf.get("batch_size", 32)
        num_workers = data_conf.get("num_workers", 0)
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0
        loader = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return [(name, loader, 0, params)]

    datasets_conf = data_conf["datasets"]
    default_conf = dict(data_conf)
    default_conf.pop("datasets", None)
    default_conf.pop("eval_datasets", None)
    default_conf.pop("eval_equation_datasets", None)
    global_normalize = bool(default_conf.get("global_normalize", False))
    append_dt = bool(default_conf.get("append_dt_to_params", False))
    if global_normalize:
        default_conf["normalize"] = True
    equation_terms = data_conf.get("equation_terms")
    equation_text_dim = int(data_conf.get("equation_text_dim", 0))
    loaders: List[Tuple[str, DataLoader, int, Optional[torch.Tensor]]] = []
    batch_size = data_conf.get("batch_size", 32)
    num_workers = data_conf.get("num_workers", 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_datasets_for_stats: List[Dataset] = []
    all_datasets: List[Dataset] = []

    for idx, ds_conf in enumerate(datasets_conf):
        merged = _merge_conf(default_conf, ds_conf)
        train_ds, val_ds, test_ds, _ = build_datasets(merged)
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
        if dataset is None:
            continue
        if global_normalize and merged.get("normalize", False):
            train_datasets_for_stats.append(train_ds)
        all_datasets.extend([d for d in (train_ds, val_ds, test_ds) if d is not None])
        name = _dataset_name(merged)
        params, param_names = _parse_params(merged.get("params"))
        eq_terms = equation_terms or merged.get("equation_terms")
        equation_text = merged.get("equation_text")
        equation, eq_terms = _build_equation_vector(
            eq_terms,
            merged,
            text=equation_text,
            text_dim=equation_text_dim,
        )
        wrapped = DatasetWithMeta(
            dataset, idx, name, params, param_names, equation, eq_terms,
            dt=getattr(dataset, "dt", None), append_dt=append_dt
        )
        loader = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        loaders.append((name, loader, idx, params))

    if global_normalize:
        stats = None
        stats_sources = data_conf.get("global_stats_datasets")
        if stats_sources:
            # compute stats from explicitly provided dataset configs
            tmp = []
            for ds_conf in stats_sources:
                merged = _merge_conf(default_conf, ds_conf)
                train_ds, _, _, _ = build_datasets(merged)
                tmp.append(train_ds)
            stats = _combine_stats(tmp)
        else:
            stats = _combine_stats(train_datasets_for_stats)
        if stats is not None:
            for ds in all_datasets:
                ds.stats = stats

    return loaders
