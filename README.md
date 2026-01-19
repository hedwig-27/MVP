# MVP

## 项目简介

本项目是一个用于 **PDEBench-1D** 的自动原语发现（Primitive Discovery）MVP。核心思想是用一组匿名原语算子 + 稀疏路由（top-k）来组合动力学更新，重点观察在 **OOD（参数外推 / 方程外推）** 情况下的泛化能力，并与 UPS 论文结果做外部对照。

模型形式：

```
u_{t+1} = u_t + sum_i alpha_i * PrimitiveOperator_i(u_t)
```

路由器根据 `u_t` 的低维统计量（可选 PDE 参数）选择原语权重，原语在端到端训练中自动“分工”。

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
- `1D_diff-sorp_NA_NA.h5`：group-per-sample 模式，10000 组，每组 `data` 为 `(101, 1024, 1)`
- `1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5`：多变量（`Vx/density/pressure`，3 通道）

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
  sample_ratio: 50
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

- 训练阶段：仅计算 **Train/Val NRMSE**，保存 Val 最优模型。
- 评估阶段：基于最优模型，在 **test split** 上评估：
  - ID：`data.datasets`
  - OOD-参数：`data.eval_datasets`（若配置）
  - OOD-方程：`data.eval_equation_datasets`（若配置）
- 评估结果写入同一 `run.log`，并输出少量关键图表。
- 评估入口统一使用 `eval/eval_suite.py`（其他评估脚本为历史保留，不在默认流程中使用）。

## 训练参数建议

- `data.sample_ratio`：当前默认 50（加速），需要更稳定可升到 100。
- `data.normalize`：多数据集混训建议启用（按数据集各自统计）。
- `data.num_workers`：加速数据加载。
- `data.steps_per_epoch`：当前默认 1000（加速），稳定后可上调。
- Primitive 若出现路由塌缩，可调小正则权重：  
  `training.entropy_weight` / `training.diversity_weight`
- Rollout 优化相关：`training.rollout_steps` / `training.rollout_gamma` / `training.scheduled_sampling`
- 单步优化：`training.rollout_steps=1` 且 `training.scheduled_sampling.start/end=0`
- 路由负载均衡：`training.load_balance_weight`
- Top-k 预热：`training.topk_warmup_epochs`
- 稀疏原语计算：`training.sparse_primitives`
- Loss/metrics 统一采用 **NRMSE**（归一化 RMSE）。

## TODO（关键改进项）

- 方程公式的 LLM 编码器（当前仅使用结构化公式向量）。
- 路由塌缩缓解：更强的负载均衡损失、温度/噪声探索、dataset_id dropout。
- 归一化策略完善：多通道按通道统计；OOD 评估时可选使用训练统计以避免“统计泄露”。
- 训练效率优化：多步未来帧读取的 HDF5 访问可批量化或缓存。
- 原语数量与 top-k 的消融：验证容量/稀疏性对性能的影响。
- OOD-方程留出实验扩展：基于更多方程数据集做“留一类方程”测试。

## 已实现功能

- 多数据集混训与加权采样（`MixLoader`），支持 `sample_ratio` 与 `steps_per_epoch`。
- 数据集级标准化（按训练集统计 mean/std），可选网格坐标拼接。
- 归一化统计缓存（避免每次训练重复统计）。
- Primitive 端到端训练：匿名原语算子（FNO/CNN）+ 显式增量更新。
- 原语组合器：支持线性加权与非线性 MLP 组合（可配置）。
- 路由器条件输入：全局统计、梯度范数、局部统计、FFT 频域特征、PDE 参数、数据集 embedding、结构化公式向量、LaTeX 公式文本编码。
- 稀疏路由（top-k）+ 预热阶段 + 负载均衡正则。
- 多步训练目标（rollout-aware）与 scheduled sampling。
- 统一评估流程（ID/OOD 一步预测 + rollout），并统计 router usage。

## 公式文本输入（LaTeX）

当前配置会把每个数据集的 `equation_text` 编码成向量（与 `equation_coeffs` 拼接后输入路由器）。示例公式如下：

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

### 20260118_202629（当前）

#### 关键配置
- 训练集：adv_beta0.4、burgers_nu0.001、diff_sorp（多数据集混训）。
- OOD-参数：adv_beta1.0、burgers_nu1；OOD-方程：reacdiff_rho1/5/10。
- Primitive：FNO（modes=8, width=32, depth=3）。
- 组合器：MLP（hidden_dim=32），top-k=3。
- 路由特征：全局统计 + 局部统计 + FFT + PDE 参数 + 数据集 embedding + 结构化公式 + LaTeX 公式编码。
- 采样与步数：sample_ratio=50，steps_per_epoch=1000，batch_size=32。
- 单步优化：rollout_steps=1，scheduled_sampling=0。
- time_downsample=5（与 UPS 论文的 41/21 timesteps 对齐）。

## 结果记录
**记录说明**
- 所有数值均为 **NRMSE**。
- 后续新版本直接在现有表格中新增一列（或一行）“MVP(时间戳)”。

**训练/验证**

| 版本 | Train | Val |
| --- | --- | --- |
| 20260118_202629 | 0.024300 | 0.025395 |

**ID 一步预测**

| 数据集 | MVP（20260118_202629） | FNO（Single-Family） | FNO（Unified） | UPS-B | UPS-L |
| --- | --- | --- | --- | --- | --- |
| adv_beta0.4 | 0.010871 | 0.011 | 0.0130 | 0.0027 | 0.0022 |
| burgers_nu0.001 | 0.044805 | 0.042 | 0.0501 | 0.0399 | 0.0373 |
| diff_sorp | 0.002099 | 0.0017 | 0.0041 | 0.0009 | 0.0009 |
| avg | 0.022690 | — | — | — | — |

**ID Rollout@20**

| 数据集 | MVP（20260118_202629） |
| --- | --- |
| adv_beta0.4 | 0.102977 |
| burgers_nu0.001 | 0.256642 |
| diff_sorp | 0.003666 |
| avg | 0.121095 |

**OOD-参数**

| 数据集 | MVP（20260118_202629）一步预测 | MVP（20260118_202629）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- |
| adv_beta1.0 | 0.433994 | 1.201563 | — | — |
| burgers_nu1 | 0.256437 | 0.314546 | 0.0566 | 1.0342 |
| avg | 0.345216 | 0.758054 | — | — |

**OOD-方程（reacdiff）**

| 数据集 | MVP（20260118_202629）一步预测 | MVP（20260118_202629）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- |
| reacdiff_rho1 | 1.396651 | 0.812408 | 0.0557 | 0.1839 |
| reacdiff_rho5 | 0.544398 | 5.235736 | — | — |
| reacdiff_rho10 | 0.728883 | 8.439720 | — | — |
| avg | 0.889977 | 4.829288 | — | — |

**路由使用（Primitive）**

| 原语 | MVP（20260118_202629）选择次数 | MVP（20260118_202629）占比 |
| --- | --- | --- |
| P1 | 122710 | 17.0% |
| P2 | 167632 | 23.3% |
| P3 | 164587 | 22.9% |
| P4 | 97473 | 13.5% |
| P5 | 36123 | 5.0% |
| P6 | 131475 | 18.3% |
