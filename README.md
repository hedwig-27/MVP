# MVP

This repository is an MVP for automatic primitive discovery on PDEBench-1D datasets. We compare a baseline **Fourier Neural Operator (FNO)** model with a primitive discovery model that learns K anonymous operator primitives and a sparse router (top-k gating). The router uses low-dimensional statistics of `u_t` (and optional PDE parameters) to select primitives, so the primitives are discovered automatically during end-to-end training. The goal is to evaluate whether composing learned primitives improves generalization, particularly on **out-of-distribution (OOD)** scenarios (e.g. Burgers' equation with unseen parameters).

## Project Structure

- `models/`: FNO backbone, primitive operators, router, and primitive discovery wrapper.
- `dataloader/`: Unified 1D dataset loader for PDEBench-like HDF5 and group-per-sample H5 files.
- `train/`: Training scripts for primitive discovery and baseline FNO.
- `eval/`: One-step and rollout evaluation scripts.
- `configs/`: Example configs pointing to `../MySolver/datasets`.

## Data

This project is configured to read data from `datasets/`. The loader (`dataloader/pdebench_loader.py`) supports:
- PDEBench-like HDF5: top-level `tensor` or multiple variable datasets + `x-coordinate`.
- Group-per-sample H5: many groups (`0000`, `0001`, ...) each with `data` and `grid/x`.

The loader builds one-step pairs `(u_t, u_{t+1})`, with optional grid concatenation. `n_res` subsamples the spatial grid and `time_downsample` controls the temporal stride.

### Dataset inventory (current contents of `datasets/`)
- `1D_Advection_Sols_beta{0.1,0.4,1.0}.hdf5`: tensor mode, shape `(10000, 201, 1024)`, scalar field, parameter `beta`.
- `1D_Burgers_Sols_Nu{0.001,0.01,0.1,1.0}.hdf5`: tensor mode, shape `(10000, 201, 1024)`, scalar field, parameter `nu`.
- `ReacDiff_Nu0.5_Rho{1.0,5.0,10.0}.hdf5`: tensor mode, shape `(10000, 101, 1024)`, scalar field, parameters `nu=0.5` and `rho`.
- `1D_diff-sorp_NA_NA.h5`: group-per-sample mode, 10000 groups, each `data` is `(101, 1024, 1)`.
- `1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5`: multi-variable mode with datasets `Vx`, `density`, `pressure` (3 channels), shape `(10000, 101, 1024)`.

Note: The default configs mix only **single-channel** datasets. The CFD dataset is multi-channel and should be trained in a separate run (or after extending the model to support mixed channel counts).

### Multi-dataset config

Use `data.datasets` to list multiple datasets and `data.eval_datasets` for OOD evaluation. Each entry can add a `weight` (sampling probability) and `params` (PDE parameters). To keep the router input dimension fixed, all datasets should use the **same parameter keys**. In the default configs we use `{beta, nu, rho}` and fill missing values with `0.0`.
Evaluation scripts use `eval_datasets` automatically when `split=test`.

Example snippet (see `configs/primitive.yaml` / `configs/fno.yaml`):

```yaml
data:
  sample_ratio: 20
  steps_per_epoch: 2000
  datasets:
    - name: burgers_nu0.001
      path: datasets/1D_Burgers_Sols_Nu0.001.hdf5
      params: {beta: 0.0, nu: 0.001, rho: 0.0}
      weight: 1.0
  eval_datasets:
    - name: burgers_nu1
      path: datasets/1D_Burgers_Sols_Nu1.0.hdf5
      params: {beta: 0.0, nu: 1.0, rho: 0.0}
```

## Quickstart

```bash
bash run_all.sh
```

This runs:
1) primitive discovery training (joint primitives + router)
2) baseline FNO training
3) one-step evaluation + rollout plot

Outputs are saved under `outputs/<exp>_<timestamp>/` for each run. Latest checkpoints are
also symlinked under `outputs/latest_primitive` and `outputs/latest_fno`.

If training is slow, increase `num_workers` in the configs to speed up data loading.

Primitive usage frequency is logged during evaluation in `primitive_usage.json`.

For quick experiments, you can set `data.sample_ratio` to use a fraction of the dataset.
Integers are treated as percentages (e.g. `1` = 1%, `20` = 20%, `100` = 100%), and
floats in (0, 1] are treated as fractions (e.g. `0.2` = 20%). If the ratio makes the
dataset smaller than `n_train + n_val + n_test`, the counts are scaled down automatically.
The default configs now use `sample_ratio: 50` for a stronger training signal.

If you observe router collapse (some primitives unused), enable small regularizers:
`training.entropy_weight` and `training.diversity_weight` in `configs/primitive.yaml`.

## Example Commands

```bash
# baseline
python train/train_fno.py --config configs/fno.yaml

# primitive discovery model
python train/train_primitives.py --config configs/primitive.yaml
```

## Plotting

Generate comparison plots after a run:

```bash
python scripts/make_plots.py
```

This writes plots into `outputs/<primitive_run>/plots/`, including:
- `compare_val_loss.png`: per-dataset validation loss (primitive vs FNO).
- `compare_ood_test_loss.png`: OOD test loss (primitive vs FNO).
- `compare_onestep_mse.png` or `primitive_onestep_mse.png`: one-step evaluation MSE.
- `primitive_usage.png`: total primitive usage histogram.
- `compare_train_val_curve.png`: train/val loss curves vs epoch (primitive vs FNO).
- `compare_val_curve_<dataset>.png`: per-dataset val curves vs epoch (primitive vs FNO).
- `compare_rollout_<step>.png`: rollout MSE at fixed steps (primitive vs FNO).

If you want to save plots into a specific directory, pass `--out <path>`.

## Latest Results (2026-01-15)

Run setup:
- Configs: `configs/primitive.yaml`, `configs/fno.yaml`
- Data mix: advection (beta 0.1/0.4/1.0), burgers (nu 0.001/0.01/0.1), reaction-diffusion (rho 1/5), diff-sorp
- OOD eval: burgers (nu 1.0), reaction-diffusion (rho 10.0)
- Settings: `sample_ratio=20`, `steps_per_epoch=2000`, `epochs=5`, `batch_size=32`

Primitive discovery model (`outputs/primitive_20260115_184626/`):
- Train/val/test loss: `0.002506 / 0.006064 / 0.004146`
- OOD test loss: burgers_nu1 `0.005663`, reacdiff_rho10 `0.001230`
- One-step MSE (in-distribution avg): `0.003224` (range `0.000904`–`0.007646`)
- Rollout MSE at 20 steps: min `0.304` (diff_sorp), max `2.054` (adv_beta1.0)
- Primitive usage (top-k): P4 ~35.7%, P6 ~23.0%, P3 ~21.9%, P1 ~19.5%, P2/P5 unused

Baseline FNO (`outputs/fno_20260115_185822/`):
- Train/val/test loss: `0.013792 / 0.017252 / 0.003591`
- OOD test loss: burgers_nu1 `0.004513`, reacdiff_rho10 `0.001820`
