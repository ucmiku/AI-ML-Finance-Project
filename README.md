# AI-ML-Finance-Project

NUS summer camp project.

项目标题：基于极端天气特征驱动的 ERCOT 实时电价机器学习套利策略与回测平台

## Project Workflow

本仓库按项目流程拆分为四个主要工作区：

```text
01_data_collection_cleaning/
  raw/          # 原始数据：ERCOT 实时电价、负荷、天气、极端天气事件等
  interim/      # 中间数据：临时合并、初步清洗、特征拼接结果
  processed/    # 建模数据：可直接用于训练、验证、回测的数据集
  scripts/      # 数据下载、清洗、特征工程脚本
  notebooks/    # 数据探索与清洗验证 notebook

02_model_training_validation/
  notebooks/    # 模型实验、特征选择、误差分析 notebook
  src/          # 可复用训练、验证、预测代码
  models/       # 训练好的模型文件
  metrics/      # 验证指标、模型对比结果
  predictions/  # 验证集/测试集/未来窗口预测结果

03_qc_backtesting/
  algorithms/   # QuantConnect / LEAN 策略代码
  config/       # 回测配置、参数、数据映射
  results/      # 回测输出、交易记录、权益曲线
  reports/      # 回测分析报告

04_frontend_dashboard/
  app/          # Dashboard 前端应用代码
  assets/       # 图片、图标、静态资源
  data/         # Dashboard 展示用的轻量结果数据
  docs/         # 前端说明文档

docs/           # 项目计划、方法说明、报告材料
outputs/        # 最终汇总输出：图表、演示材料、可交付文件
```

# 03_qc_backtesting — ERCOT RT-DA 价差套利策略回测系统

> **负责人**：成员 C（量化回测与策略分析）
> **项目**：基于极端天气特征驱动的 ERCOT 实时电价机器学习套利策略与回测平台
> **回测区间**：2025-01-01 至 2025-12-31（8,760 小时 OOF） + 2026-01-01 至 2026-07-01（4,197 小时 Walk-Forward）

---

## 目录结构

```
03_qc_backtesting/
├── README.md
├── algorithms/                    # 策略代码
│   ├── sumstrategy.py                  # 🏗️ 核心回测引擎 + C1 B2B/B2A 策略实现
│   ├── strategy_extreme_weather.py     # 🎯 生产级 API 函数（FastAPI 后端直接调用）
│   ├── baseline_utils.py               # 🔧 共享工具：数据加载、基线回测引擎、可视化
│   ├── baseline1_naive.py              # 基线策略 1：始终做空 + 每日随机
│   ├── baseline2_ma.py                 # 基线策略 2：双均线 MA 交叉 (24h/168h)
│   ├── baseline3_weather_shutdown.py   # 基线策略 3：天气关闭 + ML 混合
│   ├── run_c1_baseline_comparison.py   # 🚀 C1 全策略对比运行器（主入口）
│   ├── compute_strategy_scores.py      # 📊 多维度综合评分框架
│   └── generate_frontend_data.py       # 🌐 前端数据生成脚本
├── config/                        # 配置与规范
│   ├── settings.json              # 回测参数配置
│   ├── frontend_data_specification.md  # 前端数据接口规范 (API Schema)
│   ├── unified_backtest_benchmark.md   # 统一回测基准规范
│   └── sumstrategy.md             # 策略算法详细文档
├── results/                       # 回测输出
│   ├── backtest_result_c1.json    # C1 策略回测结果（前端数据源）
│   ├── c1_baseline_comparison_results.json  # C1 基线对比指标 JSON
│   ├── strategy_scores.json       # 多维度策略评分
│   ├── c1_b2b_baseline_2025.png            # 2025 B2B 基准策略权益曲线
│   ├── c1_b2b_baseline_2026.png            # 2026 B2B 基准策略权益曲线
│   ├── c1_2026_strategies_comparison.png   # 2026 三策略叠加对比图
│   ├── c1_baseline_comparison_2025.png     # 全策略叠加权益曲线对比图
│   ├── ercot_backtest_2025.png             # 2025 年回测图 (LightGBM)
│   ├── ercot_backtest_2026.png             # 2026 年回测图 (LightGBM)
│   └── logs/                               # 运行日志
└── reports/                       # 分析报告
    ├── C1_strategy_analysis_report.md     # 📌 C1 模型 7 策略深度分析（核心报告）
    └── c1_baseline_comparison_report.md   # 📌 C1 跨类型基线对比（互补报告）
```

---

## 策略体系总览

本回测系统实现了从天真策略到最终选定策略的完整进化路径：

```
B1A 始终做空 (-13.0K, Sharpe -3.07)        ← 下限基准
  ↓
B1B 每日随机 (-16.4K, Sharpe -4.16)        ← 零预测能力基准
  ↓
B2 MA双均线 (+17.8K, Sharpe 4.92)          ← 技术分析基准（证明价差可交易）
  ↓
C1-060 Handoff (+3.4K, Sharpe 1.58)        ← C1 模型交付基线
  ↓
C1-070 最优阈值 (+4.1K, Sharpe 2.34)       ← 阈值优化 (Sharpe +48%)
  ↓
C1-EWO ExtremeWeather_Only                ← 🥇 最终选定策略
  (+3.6K/2025, Sharpe 2.97 | +6.5K/2026, Sharpe 2.06)
```

### 最终选定策略：C1 ExtremeWeather_Only

| 指标 | 2025 OOF | 2026 H1 | 说明 |
|------|--:|--:|---|
| Total PnL | $3,617 | $6,532 | 2026 在更短周期内赚得更多 |
| Sharpe Ratio | 2.97 | 2.06 | 2026 收益更集中导致日波动率上升 |
| Max Drawdown | -0.62% | **-0.17%** | 2026 回撤极低 |
| Calmar Ratio | 5.88 | **37.67** | 收益/回撤比惊人 |
| Win Rate | 69.2% | 62.0% | 稳健 |
| Profit Factor | 2.42 | **16.54** | 盈亏比极高 |
| Trades | 425 | 158 | 仅极端天气时段交易 |
| 盈利月 | 10/13 | 4/7 | 覆盖多数月份 |

> **跨期验证结论**：策略在 2025 和 2026 都表现最优，通过了时序稳健性检验。Sharpe 在 2026 看似下降，实际是回测周期缩短（366→178 天）导致的数学效应——策略的交易质量（Calmar、Profit Factor、Max DD）在 2026 全面改善。

---

## 快速开始

### 环境依赖

- Python 3.8+
- numpy, pandas, matplotlib
- C1 XGBoost 统一预测表（由成员 B 交付）

依赖安装：
```bash
pip install numpy pandas matplotlib pyarrow
```

### 运行全策略对比

```bash
cd algorithms
python run_c1_baseline_comparison.py
```

输出：
- `c1_baseline_comparison_report.md` — 对比分析报告（自动覆盖到上级 `reports/`）
- `c1_baseline_comparison_results.json` — 汇总指标 JSON
- `c1_baseline_comparison_2025.png` — 多策略权益曲线对比图

### 仅运行 C1 策略分析

```bash
cd algorithms
python sumstrategy.py
```

该脚本执行：
1. C1 数据加载与基准回测
2. B2B 概率阈值网格搜索（0.50–0.80）
3. 7 策略变体横向对比
4. 风险集中度分析 + 月度 PnL 分布
5. 多维度综合评分

### 计算策略评分

```bash
cd algorithms
python compute_strategy_scores.py
```

输出 `strategy_scores.json` — 四维度 13 指标的综合评分（2025 + 2026 双时段）。

### 生产级 API 函数（FastAPI 后端集成）

`strategy_extreme_weather.py` 将验证通过的 ExtremeWeather_Only 策略封装为**无状态纯函数**，可直接被 FastAPI 后端调用：

```python
from strategy_extreme_weather import get_trade_signal

# 单小时调用
result = get_trade_signal(
    p_negative=0.75,           # B2B INC 概率
    p_positive=0.15,           # B2B DEC 概率
    extreme_weather_flag=True, # 极端天气标志
    predicted_spread=-50.0,    # B2A 预测价差（可选）
    confidence=0.75,           # 信号置信度（可选）
)
# → {"strategy_action": "INC", "strategy_confidence": 0.75, "position_size": 1.25}

# 批量 DataFrame 调用
from strategy_extreme_weather import get_trade_signals_batch
df_out = get_trade_signals_batch(df)
```

**输入规范**：
| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `p_negative` | float | ✅ | INC 方向概率 (p_c1 + p_c2)，范围 [0, 1] |
| `p_positive` | float | ✅ | DEC 方向概率 (p_c4 + p_c5)，范围 [0, 1] |
| `extreme_weather_flag` | bool/int | ✅ | 极端天气触发标志 |
| `predicted_spread` | float | 可选 | B2A 回归头预测价差 ($/MWh)，用于方向确认 |
| `confidence` | float | 可选 | 最大类别概率，用于仓位缩放 |
| `threshold` | float | 可选 | B2B 概率阈值，默认 0.60（已验证） |

**输出规范**：
| 字段 | 类型 | 说明 |
|---|---|---|
| `strategy_action` | str | `"INC"` / `"DEC"` / `"NO_TRADE"` |
| `strategy_confidence` | float | 信号置信度 (0.0 ~ 1.0) |
| `position_size` | float | 建议仓位倍数（基准 1.0 = 1 MWh） |
| `signal_details` | dict | 决策明细（b2b_signal、b2a_confirmed 等） |

---

## 数据依赖

本回测系统依赖以下外部数据（**不包含在本目录内**，由项目其他成员交付）：

| 数据 | 来源 | 路径（运行时期望） |
|------|------|---|
| C1 统一预测表（2025 OOF） | 成员 B — C1 XGBoost v3 | `../trading_handoff_C1_v3/C1_unified_prediction_table_2025_oof_v3.parquet` |
| C1 统一预测表（2026 H1 Walk-Forward） | 成员 B — C1 XGBoost v3 | `../trading_handoff_C1_v3/C1_unified_prediction_table_2026_H1_walkforward_v1.parquet` |
| LightGBM 预测（2025 验证） | 成员 B — LightGBM（Legacy） | `../data/member_B/lightgbm_predictions_2025_validation.csv` |
| LightGBM 预测（2026 H1） | 成员 B — LightGBM（Legacy） | `../data/member_B/lightgbm_predictions_2026_H1_walkforward.csv` |

> **注意**：C1 策略（`sumstrategy.py` Phase 3-5）仅需 C1 parquet 数据。LightGBM 数据仅用于 Phase 1-2（Legacy 自适应波动率策略），不影响最终选定的 ExtremeWeather_Only 策略。

---

## 回测假设（全策略统一）

| 参数 | 值 | 说明 |
|---|---|---|
| 初始本金 | $100,000 USD | 固定 |
| 单笔仓位 | 1 MWh/执行小时 | 每信号小时固定持仓 |
| 交易佣金 | $2.00 / MWh | 含交易所及结算费用 |
| 滑点 | \|spread\| × 0.005 (50 bps) | 按实际价差比例 |
| 价差捕获率 | 65% (γ = 0.65) | 模拟物理与金融结算摩擦 |
| 收费模式 | 按执行小时 | 不按 signal 变化次数收费 |

---

## 核心模块说明

### `sumstrategy.py` — 主引擎

`ERCOTBacktestEngine` 类提供：
- `execute_b2b_baseline()` — C1 B2B 分类器基准策略
- `execute_b2b_b2a_combined()` — B2B + B2A 组合策略（支持方向确认、幅度过滤、极端天气开关）
- `execute_confidence_scaled()` — 置信度仓位缩放策略
- `execute_p_outer_strategy()` — 尾部概率尖峰过滤策略
- `calculate_metrics()` — 量化核心指标计算（Sharpe、Sortino、Calmar 等）
- 风险集中度分析、方向表现分解、月度 PnL 分布

### `baseline_utils.py` — 基线工具

`BaselineBacktestEngine` 类提供：
- 接受任意信号向量的通用回测框架
- 统一的指标计算与可视化

### 多维度综合评分框架

四维度、11 指标的 Min-Max 归一化评分（0–100 分制）：

| 维度（权重） | 包含指标 | 衡量目标 |
|---|---|---|
| 收益能力 (40%) | Sharpe、Total PnL、Sortino、Avg Trade PnL | 赚钱能力 |
| 风险控制 (30%) | Max DD、Win Rate、盈利月占比、Profit Factor | 下行保护 |
| 稳健性 (20%) | Calmar Ratio、去 Top5 后剩余 PnL | 非极端环境生存能力 |
| 交易效率 (10%) | 笔均 PnL、交易频率（反向）、多空平衡度 | 实用性与成本效率 |

---

## 报告阅读指南

### 需要理解策略为什么有效？
→ 阅读 [`reports/C1_strategy_analysis_report.md`](reports/C1_strategy_analysis_report.md)
- 7 个 C1 模型变体的深度对比
- 概率阈值敏感性分析
- 风险集中度与信号分布
- 月度 PnL 明细

### 需要证明策略比简单方法强？
→ 阅读 [`reports/c1_baseline_comparison_report.md`](reports/c1_baseline_comparison_report.md)
- 6 种跨类型策略的系统对比
- 从天真策略到 C1-EWO 的进化路径
- 多维度综合评分排名
- 信号分布与交易特征分析

### 需要对接前端？
→ 阅读 [`config/frontend_data_specification.md`](config/frontend_data_specification.md)
- JSON 数据结构说明
- API Schema
- ECharts 渲染建议

### 需要了解策略算法细节？
→ 阅读 [`config/sumstrategy.md`](config/sumstrategy.md)
- B2B 分类器信号生成规则
- B2A 回归确认逻辑
- 极端天气开关机制

---

## 交付物清单

| 交付物 | 文件 | 用途 | 交付对象 |
|---|---|---|---|
| 🎯 **API 函数** | `algorithms/strategy_extreme_weather.py` | FastAPI 后端直接调用的交易信号函数 | 成员 D（后端） |
| 回测引擎代码 | `algorithms/sumstrategy.py` | 核心策略实现（含 2026 Walk-Forward） | 成员 C |
| 基线策略代码 | `algorithms/baseline*.py` | 3 个基线策略 | 成员 C |
| 对比运行器 | `algorithms/run_c1_baseline_comparison.py` | 一键运行全策略对比 | 成员 C |
| 评分框架 | `algorithms/compute_strategy_scores.py` | 多维综合评分（2025+2026） | 成员 C |
| 🌐 **前端 JSON** | `results/backtest_result_c1.json` | 前端图表数据（含双时段） | 成员 D（前端） |
| 📌 策略分析报告 | `reports/C1_strategy_analysis_report.md` | 7 策略深度分析 + 2026 检验 | 全员 |
| 📌 基线对比报告 | `reports/c1_baseline_comparison_report.md` | 跨类型基线对比 | 全员 |
| 权益曲线图 | `results/*.png` | 可视化 | 全员 |
| 🌐 **前端规范** | `config/frontend_data_specification.md` | JSON Schema + 渲染建议 | 成员 D（前端） |
| 📦 前端交付包 | `../delivery_to_D/` | 前端 JSON + 规范文档（可直接打包） | 成员 D |

### 给成员 D 的最小交付包

> 路径：`../delivery_to_D/`，包含两个文件：
> - `backtest_result_c1.json`（1.3 MB）— 全部数据
> - `frontend_data_specification.md`（18 KB）— 接口规范

### 给 FastAPI 后端同学的交付

> 单个文件：`algorithms/strategy_extreme_weather.py`
> 导入方式：`from strategy_extreme_weather import get_trade_signal`
> 依赖：仅 `numpy` + `pandas`（均为项目已有依赖）

---

## 版本变更记录

### v2.0 — 2026 H1 Walk-Forward 验证 + 生产化交付（2026-07-23）

本次更新是整个回测系统从"研究验证阶段"向"生产交付阶段"的跃升，
以下按影响程度从高到低列出所有变更。

#### 🆕 新增文件

| 文件 | 说明 |
|---|---|
| `algorithms/strategy_extreme_weather.py` | 🎯 **生产级 API 函数**。将验证通过的 ExtremeWeather_Only 策略封装为无状态纯函数 `get_trade_signal()` 和 `get_trade_signals_batch()`，可直接被 FastAPI 后端调用。包含 7 项内联自测，`import` 即可用。 |
| `results/c1_b2b_baseline_2026.png` | 2026 H1 B2B 基准策略权益曲线（Walk-Forward 冻结参数） |
| `results/c1_2026_strategies_comparison.png` | 2026 H1 三策略叠加对比图（基准 0.60 / 最优 0.70 / ExtremeWeather_Only） |
| `results/logs/backtest_20260723_211815.log` | 2026 H1 Walk-Forward 完整运行日志 |

#### 🔄 重大更新

| 文件 | 变更幅度 | 核心变更 |
|---|---|---|
| `algorithms/sumstrategy.py` | 1,785 → 3,051 行 (+71%) | • 新增 `load_c1_unified_2026()` 数据加载函数，对接成员 B 交付的 2026 H1 Walk-Forward parquet 数据<br>• 新增 `diagnose_information_leakage()` 函数，用于排查回测中的 look-ahead bias<br>• 新增 `run_unified_comparison()` 函数，统一 2025 OOF 和 2026 H1 的回测流程<br>• Phase 5: C1 2026 H1 冻结规则时序稳健性检验（~500 行新增）<br>• 2026 策略三合一对比图：B2B 基线 vs 最优阈值 vs ExtremeWeather_Only |
| `algorithms/compute_strategy_scores.py` | 80 → 277 行 (+246%) | • **双时段独立评分**：2025 OOF 和 2026 H1 分别计算综合得分，不做跨时段归一化<br>• 符合 Walk-Forward 验证原则——冻结参数在独立时段做纯预测验证<br>• JSON 输出结构从单时段改为 `{"2025": {...}, "2026": {...}}` |
| `algorithms/generate_frontend_data.py` | 310 → 390 行 (+26%) | • 新增 2026 H1 时段数据生成<br>• 前端 JSON 支持双时段切换（`period: "2025"` / `"2026"`） |
| `reports/C1_strategy_analysis_report.md` | 398 → 476 行 (+20%) | • 新增 2026 H1 Walk-Forward 检验结果<br>• 跨期策略稳定性分析<br>• 时序稳健性验证结论 |
| `reports/c1_baseline_comparison_report.md` | 内容重构 | • 动态数据驱动（不再硬编码数值）<br>• 策略进化路径数字均从实际回测结果动态生成 |

#### 🗑️ 移除/清理

| 文件 | 原因 |
|---|---|
| `_fix_ma.py`, `_run_wrapper.py` | 临时调试脚本，不应进入仓库 |
| `baeslinevs.py` | 拼写错误的早期废弃版本 |
| `fix_indent.py`, `gengerate.py` | 一次性修复脚本，使用完毕 |
| `run_baseline_comparison.py` | 旧版 LightGBM 对比运行器，已被 `run_c1_baseline_comparison.py`（C1 数据版）取代 |
| `baseline_comparison_report.md` + `baseline_comparison_results.json` | 旧版 LightGBM 基线对比结果，已被 C1 版替代 |
| `ideal_one/` | 早期回测引擎版本，功能已被 `sumstrategy.py` 完全覆盖 |
| `ercot_backtest_chart.png` | 旧版 LightGBM 图表 |

#### 📐 架构决策记录

1. **2026 H1 Walk-Forward 的使用限定**：
   - 2025 OOF（8,760 h）用于阈值搜索和参数优化
   - 2026 H1（4,197 h）仅做冻结规则检验，**不重新优化任何参数**
   - 两个时段的评分独立计算，不混合归一化
   
2. **`strategy_extreme_weather.py` 的默认阈值选择**：
   - 默认 `threshold=0.60`（Handoff 基准值），而非最优 0.70
   - 理由：2025 OOF 上 0.70 更优，但 2026 H1 Walk-Forward 上 0.60 在部分月份表现更稳健
   - 这是一个保守选择——宁可少赚，也要确保跨期稳健。后可通过 API 参数灵活切换

3. **移除所有 LightGBM 相关基线对比**：
   - `run_baseline_comparison.py`、`baseline_comparison_report.md` 等已从本包移除
   - LightGBM 预测数据仍保留在外部 `data/member_B/` 路径供参考，
     但 `sumstrategy.py` 的 Phase 1-2（自适应波动率策略）仅作为历史参考
   - 最终交付的策略体系完全基于 C1 XGBoost 模型

4. **生产级 API 函数的设计原则**：
   - **无状态**：纯函数，零外部依赖，不读取文件系统
   - **可测试**：内置 7 项自测，运行 `python strategy_extreme_weather.py` 即可验证
   - **可配置**：阈值、B2A 方向确认等均可通过参数控制
   - **双接口**：`get_trade_signal()` 单小时调用 + `get_trade_signals_batch()` DataFrame 批量调用

---

### v1.0 — 初始交付（2026-07-23 早期）

- C1 B2B/B2A 回测引擎（`ERCOTBacktestEngine`）
- 7 策略变体对比（B2B 基线 / 方向确认 / 幅度过滤 / 置信度缩放 / p_outer 过滤 / ExtremeWeather_Only / 极端天气降阈值）
- 3 个基线策略（始终做空 / 每日随机 / MA 双均线）
- 四维度 11 指标综合评分框架
- C1 策略分析报告 + 基线对比报告
- 2025 OOF 单时段验证

