from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


class DatasetWithMeta(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_id: int,
        dataset_name: str,
        params: Optional[torch.Tensor] = None,
        param_names: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.dataset_id = int(dataset_id)
        self.dataset_name = dataset_name
        self.params = params
        self.param_names = param_names or []

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
        wrapped_train = DatasetWithMeta(train_ds, 0, name, params, param_names)
        wrapped_val = DatasetWithMeta(val_ds, 0, name, params, param_names) if val_ds else None
        wrapped_test = DatasetWithMeta(test_ds, 0, name, params, param_names) if test_ds else None

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
    eval_datasets_conf = data_conf.get("eval_datasets")

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

    for idx, ds_conf in enumerate(datasets_conf):
        merged = _merge_conf(default_conf, ds_conf)
        train_ds, val_ds, test_ds, _ = build_datasets(merged)
        name = _dataset_name(merged)
        params, param_names = _parse_params(merged.get("params"))
        weight = float(merged.get("weight", 1.0))

        wrapped_train = DatasetWithMeta(train_ds, idx, name, params, param_names)
        wrapped_val = DatasetWithMeta(val_ds, idx, name, params, param_names) if val_ds else None
        wrapped_test = DatasetWithMeta(test_ds, idx, name, params, param_names) if test_ds else None

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
        wrapped = DatasetWithMeta(ds, 0, name, params)
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
    loaders: List[Tuple[str, DataLoader, int, Optional[torch.Tensor]]] = []
    batch_size = data_conf.get("batch_size", 32)
    num_workers = data_conf.get("num_workers", 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    for idx, ds_conf in enumerate(datasets_conf):
        merged = _merge_conf(default_conf, ds_conf)
        train_ds, val_ds, test_ds, _ = build_datasets(merged)
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
        if dataset is None:
            continue
        name = _dataset_name(merged)
        params, param_names = _parse_params(merged.get("params"))
        wrapped = DatasetWithMeta(dataset, idx, name, params, param_names)
        loader = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        loaders.append((name, loader, idx, params))

    return loaders
