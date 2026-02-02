# MVP

## TODO
- 加强原语分工信号：在现有正则基础上加入/强化输出正交或差异性约束 + load-balance，防止专家塌缩
- 非线性组合器升级：从“权重×delta 后 MLP”升级为对所有原语输出的联合融合（concat/attention）
- 多尺度/分域路由：基于已有 FFT/坐标特征加入频段/空间区域 gating，提升多尺度拟合与泛化
- 中长期：探索超网络/动态模块生成（低频更新或按 task 生成），验证对 OOD 的收益

## 项目简介

本项目是一个用于 **PDEBench-1D** 的自动原语发现（Primitive Discovery）MVP。核心思想是用一组匿名原语算子 + 稀疏路由（top-k）来组合动力学更新，重点观察在 **OOD（参数外推 / 方程外推）** 情况下的泛化能力，并与 UPS 论文结果做外部对照。

整体流程（高层概览）：
- 数据读取：从 PDEBench/H5 读取序列，构造一步预测对 `(u_t, u_{t+1})`，按配置做时空下采样与归一化，可拼接网格坐标。
- 条件构造：为每个数据集准备 PDE 参数向量、结构化方程系数 + 公式文本哈希向量、数据集 embedding。
- 路由选择：用 `u_t` 生成函数空间 token，经 Transformer Router 编码；条件向量作为侧面调制（Scale/Shift）影响注意力与前馈，输出对“残差原语 + 共享基础算子”的权重，并做带 always-on 的 top-k 选择（base 恒被选中）。
- 原语预测：每个原语（FNO/CNN）预测 `delta_i = Primitive_i(u_t)`。
- 组合更新：用权重对原语输出聚合（线性或 MLP），得到 `delta` 并更新 `u_{t+1} = u_t + delta`。
- 训练评估：训练阶段优化 NRMSE + 正则；评估阶段在 ID/OOD 上做一步预测与 rollout，并输出图表与路由使用统计。

模型形式（支持线性/MLP 聚合器）：

```
delta_i = Primitive_i(u_t)
delta_base = Base(u_t)
w = Router(u_t; stats(u_t) | cond(pde_params, equation, dataset_id))  # 对 [delta_i, delta_base] 给权重，base always-on
delta = Aggregate(w, {delta_i} ∪ {delta_base})  # 权重先与 delta 相乘，再经 MLP 聚合
u_{t+1} = u_t + delta
```

路由器使用 `u_t` 的函数空间 token（下采样序列 + 统计/FFT token）作为内容输入；PDE 参数、数据集 embedding、结构化方程系数与 LaTeX 文本编码组成条件向量，作为 Transformer 的侧面调制（Scale/Shift），不直接作为输入 token。输出对“残差原语 + 共享基础算子”的权重；当 base 启用时，base 作为额外原语参与组合器并在 top-k 中恒被选中（always-on）。

模型架构示意（ASCII）：

```
u_t --> [State/Stats/FFT Tokens] --> Transformer Router --> vector weights (residual + base)
                 ^                      ^
                 |                      |
     Params/Equation/Embed --------- (Scale & Shift)

Primitive_1(u_t) -> delta_1 ----+
Primitive_2(u_t) -> delta_2 ----+--> Aggregator MLP (weighted delta) -> delta
              ...               |
Primitive_K(u_t) -> delta_K ----+
 Base(u_t)       -> delta_base -+

u_{t+1} = u_t + delta
```




## 项目结构

- `models/`：FNO、原语算子、路由器、Primitive 组合模型
- `dataloader/`：统一 PDEBench-1D HDF5/H5 加载器
- `train/`：Primitive 训练脚本（FNO 脚本保留但默认不使用）
- `eval/`：统一评估脚本（ID/OOD，一步预测 + rollout）
- `configs/`：多数据集训练/评估配置
- `outputs/`：训练/评估输出目录（含 `latest_primitive` 软链接）

## 环境依赖

```bash
pip install -r requirements.txt
```

如需 GPU，请按你的 CUDA 版本安装匹配的 PyTorch。

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
| 训练/ID | adv_beta0.1（weight=1.0）, adv_beta0.4（weight=1.0）, burgers_nu0.001（weight=1.0）, burgers_nu0.01（weight=1.0）, burgers_nu0.1（weight=1.0）, diff_sorp（weight=1.0，timesteps=21, time_downsample=5） |
| OOD-参数 | adv_beta1.0, burgers_nu1 |
| OOD-方程 | reacdiff_rho1, reacdiff_rho5, reacdiff_rho10（均 timesteps=21, time_downsample=5） |

## 快速开始

```bash
bash run_all.sh
```

流程包含两个阶段：
1) 训练阶段：训练 MVP（Primitive），记录训练/验证 NRMSE，并保存最优模型，同时生成训练曲线。
2) 评估阶段：读取最优模型，对 ID/OOD 测试集做一步预测与 rollout 评估，并生成评估图。

输出会写入 `outputs/<exp>_<timestamp>/`，并维护 `outputs/latest_primitive` 软链接。

若需单独运行：

```bash
python train/train_primitives.py --config configs/primitive.yaml
python eval/eval_suite.py --config configs/primitive.yaml --model outputs/latest_primitive/primitive_model.pt --steps 20 --sample_idx 0
```

## 训练/评估流程说明

- 训练阶段：以 **NRMSE** 为主损失，可叠加 entropy/diversity/load-balance 正则；支持 top-k warmup、rollout_steps 与 scheduled sampling；保存 Val 最优模型。
- 当 `model.base_operator.enabled=true` 时，路由与组合器会使用 `num_primitives + 1`（base 作为最后一个原语且 always-on）；`router.top_k` 会把 base 算在内（例如 `top_k=3` 表示 base + 2 个残差原语），load-balance 默认忽略 base。
- 评估阶段：基于最优模型，在 **test split** 上评估：
  - ID：`data.datasets`
  - OOD-参数：`data.eval_datasets`（若配置）
  - OOD-方程：`data.eval_equation_datasets`（若配置）
- OOD 评估默认不使用 `dataset_id`（避免数据集 embedding 带来泄露）。
- 评估结果写入同一 `run.log`，并输出少量关键图表。
- 评估入口统一使用 `eval/eval_suite.py`（其他评估脚本为历史保留，不在默认流程中使用）。





## 已实现功能
- 数据加载：支持 PDEBench tensor/multi 与 group-per-sample H5；空间/时间下采样；可拼接网格坐标；`sample_ratio` 采样；训练集统计归一化并缓存；提供 `get_sequence` 供 rollout。
- 多数据集混训：MixLoader 按权重采样并固定 `steps_per_epoch`；每数据集携带 `dataset_id`、`params`、`equation`；支持 `data_keys` 选取多通道子集。
- 方程条件：`equation_terms` 系数向量 + `equation_text` 哈希向量；缺失自动补零；可从数据集名/参数推断常见项。
- 路由器：Transformer 条件调制（FiLM/AdaLN）；输入为 `u_t` 下采样序列 + 统计/FFT token，条件向量（PDE 参数、数据集 embedding、方程向量）作为侧面 Scale/Shift 调制；输出每原语向量权重用于 top-k 选择与通道加权。
- 原语算子：FNO1D 或 CNN 作为匿名原语；输出 delta；聚合器支持 linear 与 MLP（权重+delta）；共享基础算子（小型 FNO）作为“额外原语”进入组合器并赋予权重，且在 top-k 中恒被选中（always-on，load-balance 默认忽略 base）。
- 训练流程：NRMSE + entropy/diversity/load-balance 正则；rollout 训练与 scheduled sampling；top-k warmup；可选 sparse_primitives 加速；保存最佳模型与训练曲线。
- 评估流程：ID/OOD 一步预测 + rollout 曲线与 NRMSE@20；输出 JSON/图表；统计路由使用频次。
- 输出与复现：`outputs/<exp>_<timestamp>` + `outputs/latest_primitive` 软链；统一 `run.log`；`run_all.sh` 一键训练+评估。
- 基线：提供 FNO 训练脚本与配置（非默认流程）。


## 公式文本输入（LaTeX）

当前配置会把每个数据集的 `equation_text` 编码成向量（与 `equation_coeffs` 拼接为条件向量，作为路由器的侧面调制输入）。编码方式为字符 n-gram 哈希。示例公式如下：

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

### 20260202

- 训练集扩展为 adv_beta0.1、adv_beta0.4、burgers_nu0.001、burgers_nu0.01、burgers_nu0.1、diff_sorp（见 `configs/primitive.yaml`）。
- OOD 评估数据集保持不变。

### 20260127

- 训练集权重统一为 adv_beta0.4=1.0、burgers_nu0.001=1.0、diff_sorp=1.0（见 `configs/primitive.yaml`）。
- 当前默认配置 `sample_ratio=100`（见 `configs/primitive.yaml`）。
- diff_sorp 与 reacdiff 系列默认使用 `timesteps=21`、`time_downsample=5` 的数据子采样设置。
- 路由器升级为 Transformer 条件调制（Scale/Shift），条件向量不再作为输入 token。
- 新增共享基础算子路径（小型 FNO），并作为“额外原语”参与组合器且赋予权重；base 在 top-k 中恒被选中（always-on），load-balance 默认忽略 base。


### 20260118

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
- 当前记录 20260118/20260127；其中 20260118 对应 `outputs/primitive_20260118_202629`，20260127 对应 `outputs/primitive_20260127_150601`。

**训练/验证**

| 版本 | Train | Val |
| --- | --- | --- |
| 20260127 | 0.026071 | 0.027718 |
| 20260118 | 0.024300 | 0.025395 |

**ID 一步预测**

| 数据集 | MVP（20260118） | MVP（20260127） | FNO（Single-Family） | FNO（Unified） | UPS-B | UPS-L |
| --- | --- | --- | --- | --- | --- | --- |
| adv_beta0.4 | 0.010871 | 0.011057 | 0.011 | 0.0130 | 0.0027 | 0.0022 |
| burgers_nu0.001 | 0.044805 | 0.049449 | 0.042 | 0.0501 | 0.0399 | 0.0373 |
| diff_sorp | 0.002099 | 0.002506 | 0.0017 | 0.0041 | 0.0009 | 0.0009 |

**ID Rollout@20**

| 数据集 | MVP（20260118） | MVP（20260127） |
| --- | --- | --- |
| adv_beta0.4 | 0.102977 | 0.062999 |
| burgers_nu0.001 | 0.256642 | 0.168464 |
| diff_sorp | 0.003666 | 0.014009 |

**OOD-参数**

| 数据集 | MVP（20260118）一步预测 | MVP（20260127）一步预测 | MVP（20260118）Rollout@20 | MVP（20260127）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- | --- | --- |
| adv_beta1.0 | 0.433994 | 0.406106 | 1.201563 | 0.440921 | — | — |
| burgers_nu1 | 0.256437 | 0.229572 | 0.314546 | 0.186229 | 0.0566 | 1.0342 |

**OOD-方程（reacdiff）**

| 数据集 | MVP（20260118）一步预测 | MVP（20260127）一步预测 | MVP（20260118）Rollout@20 | MVP（20260127）Rollout@20 | UPS-B（0 samples） | FNO（0 samples） |
| --- | --- | --- | --- | --- | --- | --- |
| reacdiff_rho1 | 1.396651 | 1.909669 | 0.812408 | 9.873865 | 0.0557 | 0.1839 |
| reacdiff_rho5 | 0.544398 | 0.466353 | 5.235736 | 2.690305 | — | — |
| reacdiff_rho10 | 0.728883 | 0.640842 | 8.439720 | 4.357972 | — | — |

**路由使用（Primitive）**

注：以下统计来自 20260118 的旧版路由；20260127 启用 base 后会多一个 always-on 的 base 原语（索引为最后一个），未在此表中展示。

| 原语 | MVP（20260118）选择次数 | MVP（20260118）占比 |
| --- | --- | --- |
| P1 | 122710 | 17.0% |
| P2 | 167632 | 23.3% |
| P3 | 164587 | 22.9% |
| P4 | 97473 | 13.5% |
| P5 | 36123 | 5.0% |
| P6 | 131475 | 18.3% |
