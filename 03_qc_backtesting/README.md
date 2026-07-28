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
│   ├── sumstrategy.py                  # 核心回测引擎 + C1 B2B/B2A 策略实现
│   ├── strategy_extreme_weather.py     # 生产级 API 函数（FastAPI 后端直接调用）
│   ├── baseline_utils.py               # 共享工具：数据加载、基线回测引擎、可视化
│   ├── baseline1_naive.py              # 基线策略 1：始终做空 + 每日随机
│   ├── baseline2_ma.py                 # 基线策略 2：双均线 MA 交叉 (24h/168h)
│   ├── baseline3_weather_shutdown.py   # 基线策略 3：天气关闭 + ML 混合
│   ├── run_c1_baseline_comparison.py   # C1 全策略对比运行器（主入口）
│   ├── compute_strategy_scores.py      # 多维度综合评分框架
│   └── generate_frontend_data.py       # 前端数据生成脚本
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
    ├── C1_strategy_analysis_report.md     # C1 模型 7 策略深度分析（核心报告）
    └── c1_baseline_comparison_report.md   # C1 跨类型基线对比（互补报告）
```

---

## 策略体系总览

本回测系统实现了从天真策略到最终选定策略的完整进化路径：

```
B1A 始终做空 (-13.0K, Sharpe -3.07)        ← 下限基准
  ↓
B1B 每日随机 (-16.4K, Sharpe -4.16)        ← 零预测能力基准
  ↓
B2 MA双均线 (+17.8K, Sharpe 4.92)          ← 技术分析基准（证明价差趋势可交易）
  ↓
C1-060 Handoff (+3.4K, Sharpe 1.58)        ← C1 模型交付基线
  ↓
C1-070 最优阈值 (+4.1K, Sharpe 2.34)       ← 阈值优化 (Sharpe +48%)
  ↓
C1-EWO ExtremeWeather_Only                ← 最终选定策略
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

> **跨期验证结论**：策略在 2025 和 2026 都表现最优，通过了时序稳健性检验。

---

## 快速开始

### 环境依赖

- Python 3.8+
- numpy, pandas, matplotlib
- C1 XGBoost 统一预测表（由成员 B 交付）

```bash
pip install numpy pandas matplotlib pyarrow
```

### 运行全策略对比

```bash
cd algorithms
python run_c1_baseline_comparison.py
```

### 运行 C1 完整策略分析

```bash
cd algorithms
python sumstrategy.py
```

### 计算策略评分

```bash
cd algorithms
python compute_strategy_scores.py
```

### 生产级 API 函数

`strategy_extreme_weather.py` 封装为无状态纯函数，可直接被 FastAPI 后端调用：

```python
from strategy_extreme_weather import get_trade_signal

result = get_trade_signal(
    p_negative=0.75, p_positive=0.15,
    extreme_weather_flag=True,
    predicted_spread=-50.0,
    confidence=0.75,
)
# → {"strategy_action": "INC", "strategy_confidence": 0.75, "position_size": 1.25}
```

---

## 数据依赖

本回测系统依赖以下外部数据（**不包含在本目录内**，由项目其他成员交付）：

| 数据 | 来源 | 路径（运行时期望） |
|------|------|---|
| C1 统一预测表（2025 OOF） | 成员 B | `../trading_handoff_C1_v3/C1_unified_prediction_table_2025_oof_v3.parquet` |
| C1 统一预测表（2026 H1） | 成员 B | `../trading_handoff_C1_v3/C1_unified_prediction_table_2026_H1_walkforward_v1.parquet` |
| LightGBM 预测（2025/2026） | 成员 B（Legacy） | `../data/member_B/` |

> C1 策略仅需 C1 parquet 数据。LightGBM 数据仅用于 Phase 1-2（Legacy 参考），不影响最终选定的 ExtremeWeather_Only 策略。

---

## 回测假设（全策略统一）

| 参数 | 值 |
|---|---|
| 初始本金 | $100,000 USD |
| 单笔仓位 | 1 MWh/执行小时 |
| 交易佣金 | $2.00 / MWh |
| 滑点 | \|spread\| × 0.005 (50 bps) |
| 价差捕获率 | 65% (γ = 0.65) |
| 收费模式 | 按执行小时 |

---

## 核心模块说明

### `sumstrategy.py` — 主引擎

`ERCOTBacktestEngine` 类提供：
- `execute_b2b_baseline()` — C1 B2B 分类器基准策略
- `execute_b2b_b2a_combined()` — B2B + B2A 组合策略（方向确认、幅度过滤、极端天气开关）
- `execute_confidence_scaled()` — 置信度仓位缩放策略
- `execute_p_outer_strategy()` — 尾部概率尖峰过滤策略
- `calculate_metrics()` — 量化核心指标（Sharpe、Sortino、Calmar 等）
- 风险集中度分析、方向表现分解、月度 PnL 分布

### `baseline_utils.py` — 基线工具

`BaselineBacktestEngine` 类：接受任意信号向量的通用回测框架，统一指标计算与可视化。

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

- **理解策略为什么有效？** → [`reports/C1_strategy_analysis_report.md`](reports/C1_strategy_analysis_report.md)
- **证明策略比简单方法强？** → [`reports/c1_baseline_comparison_report.md`](reports/c1_baseline_comparison_report.md)
- **对接前端？** → [`config/frontend_data_specification.md`](config/frontend_data_specification.md)
- **了解算法细节？** → [`config/sumstrategy.md`](config/sumstrategy.md)

---

## 交付物清单

| 交付物 | 文件 | 交付对象 |
|---|---|---|
| API 函数 | `algorithms/strategy_extreme_weather.py` | 成员 D（后端） |
| 回测引擎 | `algorithms/sumstrategy.py` | 成员 C |
| 基线策略 | `algorithms/baseline*.py` | 成员 C |
| 对比运行器 | `algorithms/run_c1_baseline_comparison.py` | 成员 C |
| 评分框架 | `algorithms/compute_strategy_scores.py` | 成员 C |
| 前端 JSON | `results/backtest_result_c1.json` | 成员 D（前端） |
| 策略分析报告 | `reports/C1_strategy_analysis_report.md` | 全员 |
| 基线对比报告 | `reports/c1_baseline_comparison_report.md` | 全员 |
| 前端规范 | `config/frontend_data_specification.md` | 成员 D（前端） |

---

## 版本变更记录

### v2.0 — 2026 H1 Walk-Forward 验证 + 生产化交付（2026-07-23）

#### 新增文件

| 文件 | 说明 |
|---|---|
| `algorithms/strategy_extreme_weather.py` | 生产级 API 函数，无状态纯函数，FastAPI 可直接调用。含 7 项内联自测 |
| `results/c1_b2b_baseline_2026.png` | 2026 H1 B2B 基准策略权益曲线 |
| `results/c1_2026_strategies_comparison.png` | 2026 H1 三策略叠加对比图 |
| `results/logs/backtest_20260723_211815.log` | 2026 H1 Walk-Forward 运行日志 |

#### 重大更新

| 文件 | 变更 | 核心变更 |
|---|---|---|
| `sumstrategy.py` | 1,785→3,051 行 (+71%) | 新增 `load_c1_unified_2026()`、`diagnose_information_leakage()`、`run_unified_comparison()`、Phase 5 2026 Walk-Forward 检验 |
| `compute_strategy_scores.py` | 80→277 行 (+246%) | 双时段独立评分，JSON 输出改为 `{"2025": {...}, "2026": {...}}` |
| `generate_frontend_data.py` | 310→390 行 (+26%) | 新增 2026 H1 时段数据生成 |
| `C1_strategy_analysis_report.md` | 398→476 行 (+20%) | 新增 2026 Walk-Forward 检验结果与跨期稳健性分析 |
| `c1_baseline_comparison_report.md` | 内容重构 | 全动态数据驱动 |

#### 移除/清理

`_fix_ma.py`, `_run_wrapper.py`, `baeslinevs.py`, `fix_indent.py`, `gengerate.py`, `run_baseline_comparison.py`, `baseline_comparison_report.md`, `baseline_comparison_results.json`, `ideal_one/`, `ercot_backtest_chart.png`

#### 架构决策

1. **Walk-Forward 原则**：2025 OOF 用于参数优化，2026 H1 仅做冻结检验，两时段评分独立
2. **API 默认阈值选 0.60**：保守选择，确保跨期稳健性，可通过参数灵活切换
3. **完全移除 LightGBM 基线对比**：最终交付完全基于 C1 XGBoost
4. **生产级 API 四原则**：无状态、可测试、可配置、双接口

### v1.0 — 初始交付（2026-07-23 早期）

- C1 B2B/B2A 回测引擎
- 7 策略变体对比 + 3 基线策略
- 四维度 11 指标综合评分框架
- C1 策略分析报告 + 基线对比报告
- 2025 OOF 单时段验证
