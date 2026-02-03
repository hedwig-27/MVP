import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _is_coord_key(key):
    return "coordinate" in key


def _detect_h5(path, data_keys=None):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        if "tensor" in keys:
            d = f["tensor"]
            mode = "tensor"
            n_samples = d.shape[0]
            t_full = d.shape[1]
            x_full = d.shape[2]
            channels = d.shape[3] if d.ndim == 4 else 1
            xcoord = f["x-coordinate"][()] if "x-coordinate" in f else np.arange(x_full)
            tcoord = None
            if "t-coordinate" in f:
                tcoord = f["t-coordinate"][()]
            elif "t_coordinate" in f:
                tcoord = f["t_coordinate"][()]
            return {
                "mode": mode,
                "data_keys": ["tensor"],
                "n_samples": n_samples,
                "t_full": t_full,
                "x_full": x_full,
                "channels": channels,
                "xcoord": np.asarray(xcoord, dtype=np.float32),
                "tcoord": np.asarray(tcoord, dtype=np.float32) if tcoord is not None else None,
            }

        data_keys = data_keys or [k for k in keys if not _is_coord_key(k)]
        if data_keys and isinstance(f[data_keys[0]], h5py.Dataset):
            d0 = f[data_keys[0]]
            mode = "multi"
            n_samples = d0.shape[0]
            t_full = d0.shape[1]
            x_full = d0.shape[2]
            xcoord = f["x-coordinate"][()] if "x-coordinate" in f else np.arange(x_full)
            tcoord = None
            if "t-coordinate" in f:
                tcoord = f["t-coordinate"][()]
            elif "t_coordinate" in f:
                tcoord = f["t_coordinate"][()]
            return {
                "mode": mode,
                "data_keys": data_keys,
                "n_samples": n_samples,
                "t_full": t_full,
                "x_full": x_full,
                "channels": len(data_keys),
                "xcoord": np.asarray(xcoord, dtype=np.float32),
                "tcoord": np.asarray(tcoord, dtype=np.float32) if tcoord is not None else None,
            }

        group_names = sorted(keys)
        if len(group_names) == 0:
            raise ValueError(f"No groups found in {path}")
        g0 = f[group_names[0]]
        if "data" not in g0:
            raise ValueError(f"Group {group_names[0]} has no 'data' dataset")
        d0 = g0["data"]
        t_full = d0.shape[0]
        x_full = d0.shape[1]
        channels = d0.shape[2] if d0.ndim == 3 else 1
        xcoord = None
        if "grid" in g0 and "x" in g0["grid"]:
            xcoord = np.asarray(g0["grid"]["x"][()], dtype=np.float32)
        else:
            xcoord = np.arange(x_full, dtype=np.float32)
        tcoord = None
        if "grid" in g0 and "t" in g0["grid"]:
            tcoord = np.asarray(g0["grid"]["t"][()], dtype=np.float32)
        return {
            "mode": "group",
            "data_keys": None,
            "group_names": group_names,
            "n_samples": len(group_names),
            "t_full": t_full,
            "x_full": x_full,
            "channels": channels,
            "xcoord": xcoord,
            "tcoord": tcoord,
        }


def _split_indices(
    n_samples,
    seed,
    n_train=None,
    n_val=None,
    n_test=None,
    train_frac=0.8,
    val_frac=0.1,
    sample_ratio=None,
):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    if sample_ratio is not None:
        if isinstance(sample_ratio, int):
            ratio = float(sample_ratio) / 100.0
        else:
            ratio = float(sample_ratio)
            if ratio > 1.0:
                ratio = ratio / 100.0
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError("sample_ratio must be in (0, 1] or a percent in (0, 100].")
        n_pool = max(1, int(n_samples * ratio))
        perm = perm[:n_pool]
        n_samples = n_pool

    if n_train is None:
        n_train = int(n_samples * train_frac)
    if n_val is None:
        n_val = int(n_samples * val_frac)
    if n_test is None:
        n_test = n_samples - n_train - n_val

    total = n_train + n_val + n_test
    if total > n_samples:
        scale = n_samples / max(total, 1)
        n_train = int(n_train * scale)
        n_val = int(n_val * scale)
        n_test = n_samples - n_train - n_val

    total = n_train + n_val + n_test
    if total < n_samples:
        n_train += n_samples - total

    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val : n_train + n_val + n_test]
    return {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
    }


class PDEBenchDataset(Dataset):
    """
    Unified 1D dataset loader for PDEBench-like HDF5 and group-per-sample H5 files.
    Returns (x, y, meta), where x includes optional grid channel and y is the next-step field.
    """
    def __init__(
        self,
        path,
        split,
        seed=0,
        n_res=128,
        timesteps=41,
        time_downsample=5,
        include_grid=True,
        normalize=False,
        stats=None,
        data_keys=None,
        split_ids=None,
        stats_samples=None,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.include_grid = include_grid
        self.time_downsample = int(time_downsample)
        self.timesteps = int(timesteps)
        self.stats_samples = stats_samples

        info = _detect_h5(path, data_keys=data_keys)
        self.mode = info["mode"]
        self.data_keys = info.get("data_keys", None)
        self.group_names_all = info.get("group_names", None)
        self.n_samples = info["n_samples"]
        self.t_full = info["t_full"]
        self.x_full = info["x_full"]
        self.solution_channels = info["channels"]
        self.tcoord = info.get("tcoord", None)

        # spatial sampling
        self.x_idx = np.linspace(0, self.x_full - 1, n_res, dtype=np.int64)
        self.xcoord = np.asarray(info["xcoord"], dtype=np.float32)[self.x_idx]

        # time indices after downsampling
        t_idx = np.arange(0, self.t_full, self.time_downsample, dtype=np.int64)
        if self.timesteps is not None:
            t_idx = t_idx[: self.timesteps]
        self.t_indices = t_idx
        self.timesteps = len(self.t_indices)

        # dt (time step size)
        dt = None
        if self.tcoord is not None and len(self.tcoord) > 1:
            try:
                t_vals = np.asarray(self.tcoord, dtype=np.float32)[self.t_indices]
                if len(t_vals) > 1:
                    dt = float(np.mean(np.diff(t_vals)))
            except Exception:
                dt = None
        if dt is None:
            dt = float(self.time_downsample)
        self.dt = float(dt)

        # split ids
        if split_ids is None:
            split_ids = _split_indices(self.n_samples, seed)
        self.sample_ids = split_ids[split]

        self.pairs_per_traj = max(self.timesteps - 1, 0)
        self.length = len(self.sample_ids) * self.pairs_per_traj

        self.input_channels = self.solution_channels + (1 if include_grid else 0)
        self._file = None

        if normalize:
            if stats is None:
                cached = self._load_stats_cache()
                if cached is not None:
                    self.stats = cached
                else:
                    self.stats = self._compute_stats(self.sample_ids)
                    self._save_stats_cache(self.stats)
            else:
                self.stats = stats
        else:
            self.stats = {"mean": 0.0, "std": 1.0}

    def stats_count(self):
        use_ids = self.sample_ids
        if self.stats_samples is not None:
            use_ids = use_ids[: int(self.stats_samples)]
        return (
            len(use_ids)
            * len(self.t_indices)
            * len(self.x_idx)
            * int(self.solution_channels)
        )

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __del__(self):
        f = getattr(self, "_file", None)
        if f is None:
            return
        try:
            file_id = getattr(f, "id", None)
            if file_id is None or getattr(file_id, "valid", False):
                f.close()
        except Exception:
            pass
        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _read_frame(self, f, sample_id, t_idx):
        if self.mode == "tensor":
            dset = f["tensor"]
            if dset.ndim == 3:
                u = dset[sample_id, t_idx, self.x_idx]
                u = u[:, None]
            else:
                u = dset[sample_id, t_idx, self.x_idx, :]
            return np.asarray(u, dtype=np.float32)

        if self.mode == "multi":
            arrs = []
            for key in self.data_keys:
                dset = f[key]
                u = dset[sample_id, t_idx, self.x_idx]
                arrs.append(u[:, None])
            u = np.concatenate(arrs, axis=-1)
            return np.asarray(u, dtype=np.float32)

        gname = self.group_names_all[sample_id]
        g = f[gname]
        dset = g["data"]
        if dset.ndim == 2:
            u = dset[t_idx, self.x_idx]
            u = u[:, None]
        else:
            u = dset[t_idx, self.x_idx, :]
        return np.asarray(u, dtype=np.float32)

    def _read_pair(self, f, sample_id, t0, t1):
        if self.mode == "tensor":
            dset = f["tensor"]
            if dset.ndim == 3:
                u = dset[sample_id, [t0, t1]]
                u = u[:, self.x_idx]
                u = u[..., None]
            else:
                u = dset[sample_id, [t0, t1]]
                u = u[:, self.x_idx, :]
            return np.asarray(u[0], dtype=np.float32), np.asarray(u[1], dtype=np.float32)

        if self.mode == "multi":
            arrs = []
            for key in self.data_keys:
                dset = f[key]
                u = dset[sample_id, [t0, t1]]
                u = u[:, self.x_idx]
                arrs.append(u[..., None])
            u = np.concatenate(arrs, axis=-1)
            return np.asarray(u[0], dtype=np.float32), np.asarray(u[1], dtype=np.float32)

        gname = self.group_names_all[sample_id]
        g = f[gname]
        dset = g["data"]
        if dset.ndim == 2:
            u = dset[[t0, t1]]
            u = u[:, self.x_idx]
            u = u[..., None]
        else:
            u = dset[[t0, t1]]
            u = u[:, self.x_idx, :]
        return np.asarray(u[0], dtype=np.float32), np.asarray(u[1], dtype=np.float32)

    def _stats_cache_path(self):
        cache_dir = Path(self.path).resolve().parent / ".stats_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sample_bytes = np.asarray(self.sample_ids, dtype=np.int64).tobytes()
        sample_hash = hashlib.md5(sample_bytes).hexdigest()
        key = {
            "path": str(Path(self.path).resolve()),
            "mode": self.mode,
            "data_keys": self.data_keys,
            "n_res": int(len(self.x_idx)),
            "timesteps": int(self.timesteps),
            "time_downsample": int(self.time_downsample),
            "stats_samples": int(self.stats_samples) if self.stats_samples is not None else None,
            "sample_ids": sample_hash,
        }
        digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
        name = f"{Path(self.path).stem}_stats_{digest}.json"
        return cache_dir / name

    def _load_stats_cache(self):
        path = self._stats_cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if "mean" in data and "std" in data:
            return {"mean": float(data["mean"]), "std": float(data["std"])}
        return None

    def _save_stats_cache(self, stats):
        path = self._stats_cache_path()
        try:
            path.write_text(json.dumps({"mean": float(stats["mean"]), "std": float(stats["std"])}))
        except OSError:
            pass

    def _compute_stats(self, sample_ids):
        use_ids = sample_ids
        if self.stats_samples is not None:
            use_ids = use_ids[: int(self.stats_samples)]

        count = 0
        sum_ = 0.0
        sumsq = 0.0
        f = self._get_file()
        for sid in use_ids:
            for t_idx in self.t_indices:
                u = self._read_frame(f, sid, t_idx)
                arr = u.reshape(-1)
                count += arr.size
                sum_ += float(arr.sum())
                sumsq += float((arr * arr).sum())

        mean = sum_ / max(count, 1)
        var = sumsq / max(count, 1) - mean * mean
        std = float(np.sqrt(max(var, 0.0)) + 1e-8)
        return {"mean": float(mean), "std": std}

    def __getitem__(self, idx):
        traj_id = idx // self.pairs_per_traj
        t = idx % self.pairs_per_traj
        sample_id = self.sample_ids[traj_id]

        t0 = self.t_indices[t]
        t1 = self.t_indices[t + 1]

        f = self._get_file()
        u_t, u_tp1 = self._read_pair(f, sample_id, t0, t1)

        m, s = self.stats["mean"], self.stats["std"]
        u_t = (u_t - m) / s
        u_tp1 = (u_tp1 - m) / s

        x = torch.from_numpy(u_t).float()
        y = torch.from_numpy(u_tp1).float()

        if self.include_grid:
            g = torch.from_numpy(self.xcoord).float().unsqueeze(-1)
            x = torch.cat([x, g], dim=-1)

        meta = {"sample_id": int(sample_id), "t_idx": int(t)}
        return x, y, meta

    def get_sequence(self, sample_idx):
        sample_id = self.sample_ids[sample_idx]
        f = self._get_file()
        if self.mode == "tensor":
            dset = f["tensor"]
            if dset.ndim == 3:
                seq = dset[sample_id]
                seq = seq[self.t_indices]
                seq = seq[:, self.x_idx]
                seq = seq[..., None]
            else:
                seq = dset[sample_id]
                seq = seq[self.t_indices]
                seq = seq[:, self.x_idx, :]
        elif self.mode == "multi":
            arrs = []
            for key in self.data_keys:
                dset = f[key]
                u = dset[sample_id]
                u = u[self.t_indices]
                u = u[:, self.x_idx]
                arrs.append(u[..., None])
            seq = np.concatenate(arrs, axis=-1)
        else:
            gname = self.group_names_all[sample_id]
            g = f[gname]
            dset = g["data"]
            if dset.ndim == 2:
                seq = dset[self.t_indices]
                seq = seq[:, self.x_idx]
                seq = seq[..., None]
            else:
                seq = dset[self.t_indices]
                seq = seq[:, self.x_idx, :]
        seq = np.asarray(seq, dtype=np.float32)

        m, s = self.stats["mean"], self.stats["std"]
        seq = (seq - m) / s
        return seq


def build_datasets(data_conf):
    path = data_conf["path"]
    seed = data_conf.get("seed", 0)
    n_res = data_conf.get("n_res", 128)
    timesteps = data_conf.get("timesteps", 41)
    time_downsample = data_conf.get("time_downsample", 5)
    include_grid = data_conf.get("include_grid", True)
    normalize = data_conf.get("normalize", False)
    data_keys = data_conf.get("data_keys")
    stats_samples = data_conf.get("stats_samples")

    info = _detect_h5(path, data_keys=data_keys)
    split_ids = _split_indices(
        info["n_samples"],
        seed=seed,
        n_train=data_conf.get("n_train"),
        n_val=data_conf.get("n_val"),
        n_test=data_conf.get("n_test"),
        train_frac=data_conf.get("train_frac", 0.8),
        val_frac=data_conf.get("val_frac", 0.1),
        sample_ratio=data_conf.get("sample_ratio"),
    )

    train_ds = PDEBenchDataset(
        path=path,
        split="train",
        seed=seed,
        n_res=n_res,
        timesteps=timesteps,
        time_downsample=time_downsample,
        include_grid=include_grid,
        normalize=normalize,
        stats=None,
        data_keys=data_keys,
        split_ids=split_ids,
        stats_samples=stats_samples,
    )

    stats = train_ds.stats
    val_ds = None
    test_ds = None
    if len(split_ids["val"]) > 0:
        val_ds = PDEBenchDataset(
            path=path,
            split="val",
            seed=seed,
            n_res=n_res,
            timesteps=timesteps,
            time_downsample=time_downsample,
            include_grid=include_grid,
            normalize=normalize,
            stats=stats,
            data_keys=data_keys,
            split_ids=split_ids,
            stats_samples=stats_samples,
        )
    if len(split_ids["test"]) > 0:
        test_ds = PDEBenchDataset(
            path=path,
            split="test",
            seed=seed,
            n_res=n_res,
            timesteps=timesteps,
            time_downsample=time_downsample,
            include_grid=include_grid,
            normalize=normalize,
            stats=stats,
            data_keys=data_keys,
            split_ids=split_ids,
            stats_samples=stats_samples,
        )

    return train_ds, val_ds, test_ds, stats
