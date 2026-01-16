# MVP

## 项目简介

本项目是一个用于 **PDEBench-1D** 的自动原语发现（Primitive Discovery）MVP。核心思想是用一组匿名原语算子 + 稀疏路由（top-k）来组合动力学更新，并与 **FNO** 基线进行对比，重点观察在 **OOD（参数外推）** 情况下的泛化能力。

模型形式：

```
u_{t+1} = u_t + sum_i alpha_i * PrimitiveOperator_i(u_t)
```

路由器根据 `u_t` 的低维统计量（可选 PDE 参数）选择原语权重，原语在端到端训练中自动“分工”。

## 项目结构

- `models/`：FNO、原语算子、路由器、Primitive 组合模型
- `dataloader/`：统一 PDEBench-1D HDF5/H5 加载器
- `train/`：Primitive 与 FNO 训练脚本
- `eval/`：一步预测与 rollout 评估脚本
- `configs/`：多数据集训练/评估配置

## 数据与读取

### 支持格式

- PDEBench 风格 HDF5：顶层 `tensor` 或多变量数据集 + `x-coordinate`
- Group-per-sample H5：多个组（`0000`/`0001`…），每组含 `data` 与 `grid/x`

加载器会构造一步预测对 `(u_t, u_{t+1})`，可拼接网格坐标；`n_res` 为空间下采样，`time_downsample` 为时间步长抽样。

### 数据集清单（`datasets/`）

- `1D_Advection_Sols_beta{0.1,0.4,1.0}.hdf5`：tensor 模式，`(10000, 201, 1024)`，标量场，参数 `beta`
- `1D_Burgers_Sols_Nu{0.001,0.01,0.1,1.0}.hdf5`：tensor 模式，`(10000, 201, 1024)`，标量场，参数 `nu`
- `ReacDiff_Nu0.5_Rho{1.0,5.0,10.0}.hdf5`：tensor 模式，`(10000, 101, 1024)`，标量场，参数 `nu=0.5`、`rho`
- `1D_diff-sorp_NA_NA.h5`：group-per-sample 模式，10000 组，每组 `data` 为 `(101, 1024, 1)`
- `1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5`：多变量（`Vx/density/pressure`，3 通道）

注意：默认配置只混合 **单通道** 数据集。CFD 多通道数据建议单独训练或扩展模型后再混合。

### 多数据集配置说明

`data.datasets` 用于训练集混合，`data.eval_datasets` 用于 OOD 测试。每个条目可设置：

- `weight`：采样概率权重
- `params`：PDE 参数（需所有数据集使用相同键名，例如 `{beta, nu, rho}`）

示例（见 `configs/primitive.yaml` / `configs/fno.yaml`）：

```yaml
data:
  sample_ratio: 50
  steps_per_epoch: 2000
  datasets:
    - name: burgers_nu0.001
      path: datasets/1D_Burgers_Sols_Nu0.001.hdf5
      params: {beta: 0.0, nu: 0.001, rho: 0.0}
  eval_datasets:
    - name: burgers_nu1
      path: datasets/1D_Burgers_Sols_Nu1.0.hdf5
      params: {beta: 0.0, nu: 1.0, rho: 0.0}
```

## 快速开始

```bash
bash run_all.sh
```

流程包含：
1) Primitive 训练
2) FNO 训练
3) 一步预测评估
4) rollout 评估
5) 绘图对比

输出会写入 `outputs/<exp>_<timestamp>/`，并维护 `outputs/latest_primitive` 与 `outputs/latest_fno` 软链接。

## 训练参数建议

- `data.sample_ratio`：用更高比例（如 50/100）可显著提升稳定性，但会更耗时。
- `data.num_workers`：加速数据加载。
- Primitive 若出现路由塌缩，可调小正则权重：  
  `training.entropy_weight` / `training.diversity_weight`
- Loss/metrics 统一采用 **NRMSE**（归一化 RMSE）。

## 可视化

```bash
python scripts/make_plots.py
```

图表输出到 `outputs/<primitive_run>/plots/`，其中：
- `plots/overview/`：整体对比图（train/val、一阶预测、OOD、rollout 等）
- `plots/usage/`：原语使用分布
- `plots/curves/val/`：各数据集的 val 曲线（val vs epoch）
- `plots/curves/rollout/`：各数据集的 rollout 曲线

命名规则（示例）：
- overview：`val_last_compare.png`, `ood_compare.png`, `onestep_compare.png`,
  `train_val_compare.png`, `rollout_step_20_compare.png`
- usage：`primitive_usage.png`
- curves/val：`val_epoch_<dataset>.png`
- curves/rollout：`rollout_curve_<dataset>.png`

## 结果记录（历史）
### 指标说明
- 2026-01-16 之前的历史结果使用 **MSE**（当时 loss/评估均为 MSE）。
- 代码已切换为 **NRMSE**，后续新结果将以 NRMSE 记录。
- 表格中的 avg：一步预测使用日志中的 avg，其余为简单平均。

### 2026-01-15（MSE，sample_ratio=20，epochs=5）

#### 训练/验证/测试

| 模型 | Train | Val | Test |
| --- | --- | --- | --- |
| Primitive | 0.002506 | 0.006064 | 0.004146 |
| FNO | 0.013792 | 0.017252 | 0.003591 |

#### OOD

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| burgers_nu1 | 0.005663 | 0.004513 | -25.5% |
| reacdiff_rho10 | 0.001230 | 0.001820 | +32.4% |
| avg | 0.003446 | 0.003167 | -8.8% |

#### 一步预测（ID）

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| adv_beta0.1 | 0.001994 | 0.034495 | +94.2% |
| adv_beta0.4 | 0.007646 | 0.007550 | -1.3% |
| adv_beta1.0 | 0.004824 | 0.064795 | +92.6% |
| burgers_nu0.001 | 0.004635 | 0.007596 | +39.0% |
| burgers_nu0.01 | 0.001937 | 0.005378 | +64.0% |
| burgers_nu0.1 | 0.000904 | 0.002725 | +66.8% |
| reacdiff_rho1 | 0.001020 | 0.002399 | +57.5% |
| reacdiff_rho5 | 0.001670 | 0.004363 | +61.7% |
| diff_sorp | 0.001988 | 0.002001 | +0.6% |
| avg | 0.003224 | 0.016812 | +80.8% |

#### Rollout@20

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| adv_beta0.1 | 0.811092 | - | - |
| adv_beta0.4 | 0.402949 | - | - |
| adv_beta1.0 | 2.054479 | - | - |
| burgers_nu0.001 | 1.040186 | - | - |
| burgers_nu0.01 | 1.003086 | - | - |
| burgers_nu0.1 | 1.000074 | - | - |
| reacdiff_rho1 | 1.047472 | - | - |
| reacdiff_rho5 | 0.861433 | - | - |
| diff_sorp | 0.304495 | - | - |
| avg | 0.947252 | - | - |

#### 路由使用（Primitive）

| 原语 | 选择次数 | 占比 |
| --- | --- | --- |
| P1 | 2339 | 19.5% |
| P2 | 0 | 0.0% |
| P3 | 2626 | 21.9% |
| P4 | 4280 | 35.7% |
| P5 | 0 | 0.0% |
| P6 | 2755 | 23.0% |

#### 客观结论

- 一步预测：Primitive 在 8/9 个数据集更低；FNO 更低的数据集为 adv_beta0.4。
- OOD：burgers_nu1（FNO 更低）; reacdiff_rho10（Primitive 更低）。
- Rollout@20：仅有 Primitive 结果，FNO 未评估。

### 2026-01-16（MSE，sample_ratio=50，epochs=50）

#### 训练/验证/测试

| 模型 | Train | Val | Test |
| --- | --- | --- | --- |
| Primitive | 0.001824 | 0.002038 | 0.004104 |
| FNO | 0.013471 | 0.015448 | 0.004382 |

#### OOD

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| burgers_nu1 | 0.005141 | 0.005549 | +7.4% |
| reacdiff_rho10 | 0.002032 | 0.002053 | +1.0% |
| avg | 0.003586 | 0.003801 | +5.6% |

#### 一步预测（ID）

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| adv_beta0.1 | 0.000640 | 0.036309 | +98.2% |
| adv_beta0.4 | 0.001490 | 0.006185 | +75.9% |
| adv_beta1.0 | 0.000866 | 0.069984 | +98.8% |
| burgers_nu0.001 | 0.001995 | 0.004487 | +55.5% |
| burgers_nu0.01 | 0.000774 | 0.003436 | +77.5% |
| burgers_nu0.1 | 0.001164 | 0.002839 | +59.0% |
| reacdiff_rho1 | 0.000436 | 0.000242 | -80.2% |
| reacdiff_rho5 | 0.000708 | 0.000706 | -0.3% |
| diff_sorp | 0.000046 | 0.001012 | +95.5% |
| avg | 0.001003 | 0.016558 | +93.9% |

#### Rollout@20

| 数据集 | Primitive | FNO | 提升 |
| --- | --- | --- | --- |
| adv_beta0.1 | 0.013207 | 0.026354 | +49.9% |
| adv_beta0.4 | 0.040428 | 0.011148 | -262.6% |
| adv_beta1.0 | 0.010939 | 0.032553 | +66.4% |
| burgers_nu0.001 | 0.036102 | 0.009122 | -295.8% |
| burgers_nu0.01 | 0.034790 | 0.008646 | -302.4% |
| burgers_nu0.1 | 0.021211 | 0.009233 | -129.7% |
| reacdiff_rho1 | 0.106780 | 0.044700 | -138.9% |
| reacdiff_rho5 | 0.289292 | 0.291132 | +0.6% |
| diff_sorp | 0.015012 | 0.135529 | +88.9% |
| avg | 0.063085 | 0.063157 | +0.1% |

#### 路由使用（Primitive）

| 原语 | 选择次数 | 占比 |
| --- | --- | --- |
| P1 | 0 | 0.0% |
| P2 | 57482 | 19.2% |
| P3 | 20649 | 6.9% |
| P4 | 9497 | 3.2% |
| P5 | 113552 | 37.9% |
| P6 | 98820 | 32.9% |

#### 客观结论

- 一步预测：Primitive 在 7/9 个数据集更低；FNO 更低的数据集为 reacdiff_rho1, reacdiff_rho5。
- OOD：burgers_nu1（Primitive 更低）; reacdiff_rho10（Primitive 更低）。
- Rollout@20：Primitive 在 4/9 个数据集更低；FNO 更低的数据集为 adv_beta0.4, burgers_nu0.001, burgers_nu0.01, burgers_nu0.1, reacdiff_rho1。
