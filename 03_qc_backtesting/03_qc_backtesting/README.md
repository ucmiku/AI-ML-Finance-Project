# 03_qc_backtesting — ERCOT RT-DA 价差套利策略回测系统

> **负责人**：成员 C（量化回测与策略分析）
> **项目**：基于极端天气特征驱动的 ERCOT 实时电价机器学习套利策略与回测平台
> **回测区间**：2025-01-01 至 2025-12-31（8,760 小时）

---

## 目录结构

```
03_qc_backtesting/
├── README.md
├── algorithms/                    # 策略代码
│   ├── sumstrategy.py             # 🏗️ 核心回测引擎 + C1 B2B/B2A 策略实现
│   ├── baseline_utils.py          # 🔧 共享工具：数据加载、基线回测引擎、可视化
│   ├── baseline1_naive.py         # 基线策略 1：始终做空 + 每日随机
│   ├── baseline2_ma.py            # 基线策略 2：双均线 MA 交叉 (24h/168h)
│   ├── baseline3_weather_shutdown.py  # 基线策略 3：天气关闭 + ML 混合
│   ├── run_c1_baseline_comparison.py  # 🚀 C1 全策略对比运行器（主入口）
│   ├── compute_strategy_scores.py     # 📊 多维度综合评分框架
│   └── generate_frontend_data.py      # 🌐 前端数据生成脚本
├── config/                        # 配置与规范
│   ├── settings.json              # 回测参数配置
│   ├── frontend_data_specification.md  # 前端数据接口规范 (API Schema)
│   ├── unified_backtest_benchmark.md   # 统一回测基准规范
│   └── sumstrategy.md             # 策略算法详细文档
├── results/                       # 回测输出
│   ├── backtest_result_c1.json    # C1 策略回测结果（前端数据源）
│   ├── c1_baseline_comparison_results.json  # C1 基线对比指标 JSON
│   ├── strategy_scores.json       # 多维度策略评分
│   ├── c1_b2b_baseline_2025.png   # B2B 基准策略权益曲线
│   ├── c1_baseline_comparison_2025.png     # 全策略叠加权益曲线对比图
│   ├── ercot_backtest_2025.png    # 2025 年回测图
│   ├── ercot_backtest_2026.png    # 2026 H1 回测图
│   └── logs/                      # 运行日志
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
  (+3.7K, Sharpe 3.01, Max DD -0.35%)
```

### 最终选定策略：C1 ExtremeWeather_Only

| 指标 | 数值 | 排名 |
|------|------|:---:|
| Sharpe Ratio | **3.01** | 🥇 最高 |
| Max Drawdown | **-0.35%** | 🥇 最低 |
| Win Rate | **75.0%** | 🥇 最高 |
| 交易效率（综合） | **100.0/100** | 🥇 满分 |
| 风险控制（综合） | **100.0/100** | 🥇 满分 |
| 交易笔数 | 232 | 仅 2.6% 持仓时间 |

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

输出 `strategy_scores.json` — 四维度 13 指标的综合评分。

---

## 数据依赖

本回测系统依赖以下外部数据（**不包含在本目录内**，由项目其他成员交付）：

| 数据 | 来源 | 路径（运行时期望） |
|------|------|---|
| C1 统一预测表（2025 OOF） | 成员 B — C1 XGBoost v3 | `../trading_handoff_C1_v3/C1_unified_prediction_table_2025_oof_v3.parquet` |
| LightGBM 预测（2025 验证） | 成员 B — LightGBM | `../data/member_B/lightgbm_predictions_2025_validation.csv` |
| LightGBM 预测（2026 H1） | 成员 B — LightGBM | `../data/member_B/lightgbm_predictions_2026_H1_walkforward.csv` |

> **注意**：C1 基线对比（`run_c1_baseline_comparison.py`）仅需 C1 parquet 数据，不再依赖 LightGBM CSV。

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

| 交付物 | 文件 | 用途 |
|---|---|---|
| 回测引擎代码 | `algorithms/sumstrategy.py` | 核心策略实现 |
| 基线策略代码 | `algorithms/baseline*.py` | 3 个基线策略 |
| 对比运行器 | `algorithms/run_c1_baseline_comparison.py` | 一键运行全策略对比 |
| 评分框架 | `algorithms/compute_strategy_scores.py` | 多维综合评分 |
| 前端 JSON | `results/backtest_result_c1.json` | 前端图表数据 |
| 策略分析报告 | `reports/C1_strategy_analysis_report.md` | 📌 核心交付 |
| 基线对比报告 | `reports/c1_baseline_comparison_report.md` | 📌 互补交付 |
| 权益曲线图 | `results/*.png` | 可视化 |
| 前端规范 | `config/frontend_data_specification.md` | 成员 D 对接 |
