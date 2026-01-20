# MVP

## 项目简介

本项目是一个用于 **PDEBench-1D** 的自动原语发现（Primitive Discovery）MVP。核心思想是用一组匿名原语算子 + 稀疏路由（top-k）来组合动力学更新，重点观察在 **OOD（参数外推 / 方程外推）** 情况下的泛化能力，并与 UPS 论文结果做外部对照。

模型形式（支持线性/MLP 聚合器）：

```
delta_i = Primitive_i(u_t)
w = Router(stats(u_t), pde_params, equation, dataset_id)
delta = Aggregate(w, {delta_i})  # 线性加权或 MLP 聚合
u_{t+1} = u_t + delta
```

路由器使用 `u_t` 的全局/局部统计与频域特征，并可拼接 PDE 参数、数据集 embedding、结构化方程系数与 LaTeX 文本编码；训练时可启用 top-k 稀疏路由。

## 项目结构

- `models/`：FNO、原语算子、路由器、Primitive 组合模型
- `dataloader/`：统一 PDEBench-1D HDF5/H5 加载器
- `train/`：Primitive 训练脚本（FNO 脚本保留但默认不使用）
- `eval/`：统一评估脚本（ID/OOD，一步预测 + rollout）
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
- `1D_diff-sorp_NA_NA.h5`：group-per-sample 模式，10000 组，每组 `data` 为 `(101, 1024, 1)`，含 `grid/x` 与 `grid/t`
- `1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5`：multi 模式，`Vx/density/pressure` 3 通道（每个变量 `(10000, 101, 1024)`）

注意：默认配置只混合 **单通道** 数据集。CFD 多通道数据建议单独训练或扩展模型后再混合。

### 多数据集配置说明

`data.datasets` 用于训练集混合，`data.eval_datasets` 用于 **OOD-参数** 测试，`data.eval_equation_datasets` 用于 **OOD-方程** 测试。每个条目可设置：

- `weight`：采样概率权重
- `params`：PDE 参数（需所有数据集使用相同键名，例如 `{beta, nu, rho}`）
- `data_keys`：多通道数据集选择单通道（例如 CFD 只取 `density`）
- `equation_coeffs`：结构化公式向量（配合 `data.equation_terms` 的顺序）
- `equation_text`：LaTeX 公式文本（会编码成固定维度向量）

示例（见 `configs/primitive.yaml`）：

```yaml
data:
  sample_ratio: 100
  steps_per_epoch: 1000
  equation_terms: [advection, nonlinear_advection, diffusion, reaction, sorption, cns]
  equation_text_dim: 128
  datasets:
    - name: burgers_nu0.001
      path: datasets/1D_Burgers_Sols_Nu0.001.hdf5
      params: {beta: 0.0, nu: 0.001, rho: 0.0}
      equation_coeffs: {nonlinear_advection: 1.0, diffusion: 0.00031831}
      equation_text: '\partial_t u(t,x) + \partial_x \frac{u(t,x)^2}{2} = \frac{\nu}{\pi} \partial_{xx} u(t,x), \nu=0.001'
  eval_datasets:
    - name: burgers_nu1
      path: datasets/1D_Burgers_Sols_Nu1.0.hdf5
      params: {beta: 0.0, nu: 1.0, rho: 0.0}
      equation_coeffs: {nonlinear_advection: 1.0, diffusion: 0.31831}
  eval_equation_datasets:
    - name: diff_sorp
      path: datasets/1D_diff-sorp_NA_NA.h5
      params: {beta: 0.0, nu: 0.0, rho: 0.0}
      equation_coeffs: {diffusion: 0.0005, sorption: 1.0}
```

注意：
- `equation_terms` 的顺序决定结构化系数向量的顺序。
- `equation_text` 会用字符 n-gram 哈希编码到 `equation_text_dim` 维；缺失时自动补零向量。
- `params` 会按 key 排序拼成向量，`router.pde_param_dim` 需与其长度一致。

### 数据集用途（当前默认配置）

| 用途 | 数据集 |
| --- | --- |
| 训练/ID | adv_beta0.4, burgers_nu0.001, diff_sorp |
| OOD-参数 | adv_beta1.0, burgers_nu1 |
| OOD-方程 | reacdiff_rho1, reacdiff_rho5, reacdiff_rho10 |

## 快速开始

```bash
bash run_all.sh
```

流程包含两个阶段：
1) 训练阶段：训练 MVP（Primitive），记录训练/验证 NRMSE，并保存最优模型，同时生成训练曲线。
2) 评估阶段：读取最优模型，对 ID/OOD 测试集做一步预测与 rollout 评估，并生成评估图。

输出会写入 `outputs/<exp>_<timestamp>/`，并维护 `outputs/latest_primitive` 软链接。

## 训练/评估流程说明

- 训练阶段：以 **NRMSE** 为主损失，可叠加 entropy/diversity/load-balance 正则；支持 top-k warmup、rollout_steps 与 scheduled sampling；保存 Val 最优模型。
- 评估阶段：基于最优模型，在 **test split** 上评估：
  - ID：`data.datasets`
  - OOD-参数：`data.eval_datasets`（若配置）
  - OOD-方程：`data.eval_equation_datasets`（若配置）
- OOD 评估默认不使用 `dataset_id`（避免数据集 embedding 带来泄露）。
- 评估结果写入同一 `run.log`，并输出少量关键图表。
- 评估入口统一使用 `eval/eval_suite.py`（其他评估脚本为历史保留，不在默认流程中使用）。


## TODO


## 已实现功能
- 数据加载：支持 PDEBench tensor/multi 与 group-per-sample H5；空间/时间下采样；可拼接网格坐标；`sample_ratio` 采样；训练集统计归一化并缓存；提供 `get_sequence` 供 rollout。
- 多数据集混训：MixLoader 按权重采样并固定 `steps_per_epoch`；每数据集携带 `dataset_id`、`params`、`equation`；支持 `data_keys` 选取多通道子集。
- 方程条件：`equation_terms` 系数向量 + `equation_text` 哈希向量；缺失自动补零；可从数据集名/参数推断常见项。
- 路由器：全局统计（mean/std/min/max/grad_norm）+ 局部统计 + FFT 频域特征；可拼接 PDE 参数、数据集 embedding、方程向量；top-k 稀疏路由。
- 原语算子：FNO1D 或 CNN 作为匿名原语；输出 delta；聚合器支持 linear 或 MLP。
- 训练流程：NRMSE + entropy/diversity/load-balance 正则；rollout 训练与 scheduled sampling；top-k warmup；可选 sparse_primitives 加速；保存最佳模型与训练曲线。
- 评估流程：ID/OOD 一步预测 + rollout 曲线与 NRMSE@20；输出 JSON/图表；统计路由使用频次。
- 输出与复现：`outputs/<exp>_<timestamp>` + `outputs/latest_primitive` 软链；统一 `run.log`；`run_all.sh` 一键训练+评估。
- 基线：提供 FNO 训练脚本与配置（非默认流程）。


## 公式文本输入（LaTeX）

当前配置会把每个数据集的 `equation_text` 编码成向量（与 `equation_coeffs` 拼接后输入路由器）。编码方式为字符 n-gram 哈希。示例公式如下：

**Advection (1D)**
```latex
\partial_t u(t,x) + \beta \partial_x u(t,x) = 0
```

**Burgers (1D)**
```latex
\partial_t u(t,x) + \partial_x \frac{u(t,x)^2}{2} = \frac{\nu}{\pi} \partial_{xx} u(t,x)
```

**Diffusion-Sorption (1D)**
```latex
\partial_t u(t,x) = \frac{D}{R(u)} \partial_{xx} u(t,x),\quad D=5\times10^{-4}
```

**Reaction-Diffusion (1D)**
```latex
\partial_t u(t,x) - \nu \partial_{xx} u(t,x) = \rho u(t,x)(1 - u(t,x))
```

**Compressible Navier-Stokes (1D/2D)**
```latex
\partial_t \rho + \nabla \cdot (\rho u) = 0;\;
\rho(\partial_t u + u \cdot \nabla u) = -\nabla p + \eta \Delta u + (\zeta + \eta/3)\nabla(\nabla \cdot u);\;
\partial_t(\epsilon + \rho ||u||_2^2/2) + \nabla \cdot ((p + \epsilon + \rho u^2/2)u - u \cdot \sigma) = 0
```


## 版本记录

### 20260119

#### 关键配置
- 相比 20260118：训练集权重从 1/1/1 调整为 adv_beta0.4=5.0、burgers_nu0.001=1.0、diff_sorp=25.0。
- epochs: 100, steps_per_epoch: 2000。

### 20260118

#### 关键配置
- 训练集：adv_beta0.4、burgers_nu0.001、diff_sorp（多数据集混训）。
- OOD-参数：adv_beta1.0、burgers_nu1；OOD-方程：reacdiff_rho1/5/10。
- Primitive：FNO（modes=8, width=32, depth=3, fc_dim=64），`num_primitives=6`。
- 组合器：MLP（hidden_dim=32）。
- 路由器：hidden_dim=64，top-k=3，stats=mean/std/min/max/grad_norm，local_segments=4，local_stats=mean/std，fft_bins=8，pde_param_dim=3，dataset_embed_dim=8，equation_dim=6+128。
- 采样与步数：sample_ratio=100，steps_per_epoch=1000，batch_size=32。
- 训练设置：epochs=50，rollout_steps=1，topk_warmup_epochs=5，scheduled_sampling=0。
- time_downsample=5。

## 结果记录
**记录说明**
- 所有数值均为 **NRMSE**。
- 当前记录 20260118/20260119；后续新版本在现有表格中新增一列（或一行）。

**训练/验证**

| 版本 | Train | Val |
| --- | --- | --- |
| 20260119 | 0.006706 | 0.031957 |
| 20260118 | 0.024300 | 0.025395 |

**ID 一步预测**

| 数据集 | MVP（20260118） | MVP（20260119） | FNO（Single-Family） | FNO（Unified） | UPS-B | UPS-L |
| --- | --- | --- | --- | --- | --- | --- |
| adv_beta0.4 | 0.010871 | 0.010781 | 0.011 | 0.0130 | 0.0027 | 0.0022 |
| burgers_nu0.001 | 0.044805 | 0.057242 | 0.042 | 0.0501 | 0.0399 | 0.0373 |
| diff_sorp | 0.002099 | 0.002115 | 0.0017 | 0.0041 | 0.0009 | 0.0009 |
| avg | 0.022690 | 0.027632 | — | — | — | — |

**ID Rollout@20**

| 数据集 | MVP（20260118） | MVP（20260119） |
| --- | --- | --- |
| adv_beta0.4 | 0.102977 | 0.102891 |
| burgers_nu0.001 | 0.256642 | 0.206329 |
| diff_sorp | 0.003666 | 0.016400 |
| avg | 0.121095 | 0.108540 |

**OOD-参数**

| 数据集 | MVP（20260118）一步预测 | MVP（20260119）一步预测 | MVP（20260118）Rollout@20 | MVP（20260119）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- | --- | --- |
| adv_beta1.0 | 0.433994 | 0.445080 | 1.201563 | 1.220268 | — | — |
| burgers_nu1 | 0.256437 | 0.290047 | 0.314546 | 0.301903 | 0.0566 | 1.0342 |
| avg | 0.345216 | 0.367564 | 0.758054 | 0.761085 | — | — |

**OOD-方程（reacdiff）**

| 数据集 | MVP（20260118）一步预测 | MVP（20260119）一步预测 | MVP（20260118）Rollout@20 | MVP（20260119）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- | --- | --- |
| reacdiff_rho1 | 1.396651 | 1.852961 | 0.812408 | 1.150930 | 0.0557 | 0.1839 |
| reacdiff_rho5 | 0.544398 | 0.591105 | 5.235736 | 5.316033 | — | — |
| reacdiff_rho10 | 0.728883 | 0.765395 | 8.439720 | 8.108418 | — | — |
| avg | 0.889977 | 1.069820 | 4.829288 | 4.858460 | — | — |

**路由使用（Primitive）**

| 原语 | MVP（20260118）选择次数 | MVP（20260118）占比 | MVP（20260119）选择次数 | MVP（20260119）占比 |
| --- | --- | --- | --- | --- |
| P1 | 122710 | 17.0% | 81923 | 11.4% |
| P2 | 167632 | 23.3% | 175881 | 24.4% |
| P3 | 164587 | 22.9% | 68708 | 9.5% |
| P4 | 97473 | 13.5% | 125554 | 17.4% |
| P5 | 36123 | 5.0% | 86183 | 12.0% |
| P6 | 131475 | 18.3% | 181751 | 25.2% |
