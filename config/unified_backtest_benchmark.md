# ERCOT 价差套利策略 —— 统一回测基准假设与核心策略代码

> 本文档面向课程项目中不同模型（如 LightGBM、LSTM、Transformer 等）的经济评价对标分析，提供统一的回测基准假设场景与极简可复用的固定阈值交易策略代码。

---

## 第一部分：统一回测基准假设场景

本节定义了一个标准化的回测环境，所有成员模型的预测结果均需在此环境下进行经济评价，以确保对比的公平性与可复现性。

### 1.1 交易标的与市场机制

| 项目 | 说明 |
|---|---|
| **交易市场** | ERCOT（德州电力可靠性委员会）实时市场 |
| **交易标的** | RT-DA 价差（Real-Time 价格 − Day-Ahead 价格），单位为 USD/MWh |
| **价差含义** | 正价差表示实时价格高于日前价格（多头盈利方向），负价差表示实时价格低于日前价格（空头盈利方向） |
| **交易机制** | 每小时为一个交易时段，策略在每小时初根据预测信号决定持仓方向，在该小时末按实际价差结算 |
| **价差裁剪** | 实际价差被截断至 $[-1000, 5000]$ 区间，防止极端异常值扭曲回测结果 |

### 1.2 初始状态与资金参数

| 参数 | 取值 | 说明 |
|---|---|---|
| **初始本金** | $100,000 USD | 回测起始权益 |
| **初始仓位** | 0（空仓） | 回测开始时无任何持仓 |
| **最大持仓** | 1 个单位（每小时最多持有一个方向的仓位） | 每小时要么做多、要么做空、要么空仓，不可同时持有多空双向仓位 |
| **仓位规模** | 固定 1 MW 名义功率 | 每笔交易的名义功率为 1 MW，即 1 小时对应 1 MWh 电量 |

### 1.3 信号生成与开平仓规则（固定阈值策略）

策略采用**双向套利**模式，既可做多也可做空，信号生成规则如下：

#### 1.3.1 动态阈值设定

系统根据当前小时是否被标记为极端天气时段（`extreme_hour_flag`），动态选取不同的开仓阈值：

$$
\text{Threshold}_t = \begin{cases}
\theta_{\text{extreme}}, & \text{if } \text{extreme\_hour\_flag}_t = 1 \\[4pt]
\theta_{\text{normal}}, & \text{otherwise}
\end{cases}
$$

其中 $\theta_{\text{normal}}$ 和 $\theta_{\text{extreme}}$ 经网格搜索优化确定（见 §1.3.4）。

#### 1.3.2 原始信号生成

$$
\text{Raw\_Signal}_t = \begin{cases}
+1 \quad (\text{做多}), & \text{if } \hat{S}_t > \text{Threshold}_t \\[4pt]
-1 \quad (\text{做空}), & \text{if } \hat{S}_t < -\text{Threshold}_t \\[4pt]
0 \quad (\text{空仓}), & \text{otherwise}
\end{cases}
$$

其中 $\hat{S}_t$ 为模型在时刻 $t$ 对 RT-DA 价差的预测值。

#### 1.3.3 方向过滤（Direction Filter）

若启用方向一致性过滤（`direction_filter=True`），则进一步约束信号方向必须与预测方向标记一致：

$$
\text{Signal}_t = \begin{cases}
+1, & \text{if } \text{Raw\_Signal}_t = +1 \;\land\; \text{pred\_direction}_t = +1 \\[4pt]
-1, & \text{if } \text{Raw\_Signal}_t = -1 \;\land\; \text{pred\_direction}_t = -1 \\[4pt]
0, & \text{otherwise}
\end{cases}
$$

此过滤确保仅在模型对价差符号和幅度判断一致时才入场，降低因预测方向与幅度矛盾导致的噪声交易。

#### 1.3.4 阈值搜索空间与优化目标

| 项目 | 取值 |
|---|---|
| **常规阈值搜索空间** ($\theta_{\text{normal}}$) | [5, 10, 20, 30, 50, 75, 100, 150, 200] $/MWh |
| **极端阈值搜索空间** ($\theta_{\text{extreme}}$) | [100, 150, 200, 300, 500] $/MWh |
| **约束条件** | $\theta_{\text{extreme}} \geq \theta_{\text{normal}}$（极端天气阈值不低于常规阈值） |
| **优化目标** | 最大化年化夏普比率（Sharpe Ratio） |
| **方向过滤** | 默认启用（`direction_filter=True`） |

### 1.4 交易成本与摩擦

| 参数 | 取值 | 计算方式 |
|---|---|---|
| **佣金 (Commission)** | $2.00 / MWh | 按每笔交易动作（开仓或平仓方向切换）收取，即每当 `Signal` 发生变化时计费一次 |
| **滑点 (Slippage)** | 50 bps（0.50%） | 按实际价差的绝对值乘以滑点基点数计算：$\text{Slippage}_t = \mathbb{1}[\text{Trade}] \times |S_t^{\text{real}}| \times \frac{50}{10000}$ |
| **价差捕获率 (Capture Rate)** | 65% | 现实中无法 100% 捕获 RT-DA 价差，策略仅能捕获理论价差的 65%。捕获收益计算为：$\text{Captured}_t = \text{Signal}_t \times S_t^{\text{real}} \times 0.65$ |

### 1.5 风控约束

| 约束项 | 取值 | 说明 |
|---|---|---|
| **最大连续持仓小时数** | 48 小时 | 若同一方向的持仓连续超过 48 小时，第 49 小时起强制平仓（`Signal` 置 0），防止单边风险暴露过久 |
| **价差截断** | $[-1000, 5000]$ | 极端价差被截断以控制尾部风险 |
| **方向一致性过滤** | 启用 | 预测方向与预测幅度符号必须一致（见 §1.3.3） |
| **每日结算** | 按小时逐笔结算 | 每小时的盈亏独立计算并累加 |

### 1.6 回测时间范围

| 阶段 | 数据集 | 时间范围 | 用途 |
|---|---|---|---|
| **Phase 1: 策略开发** | 2025 年 LightGBM 独立验证集 | 2025 全年 | 阈值参数网格搜索与策略调优 |
| **Phase 2: 独立测试** | 2026 H1 周度滚动预测集 | 2026 年上半年 | 冻结策略参数后的最终独立评估（仅含 `target_available=1` 的小时） |

### 1.7 评价指标体系

所有策略模型统一使用以下指标进行经济评价对比：

| 指标类别 | 指标名称 | 计算方式 |
|---|---|---|
| **收益类** | 总收益 (Total Return) | $\frac{\sum \text{Hourly\_Pnl}}{\text{Initial\_Capital}}$ |
| | 总盈亏 (Total PnL) | $\sum \text{Hourly\_Pnl}$ |
| | 平均每笔交易盈亏 (Avg Trade PnL) | $\frac{\sum \text{Trade\_Pnl}}{\text{交易笔数}}$ |
| **风险调整收益** | 年化夏普比率 (Sharpe Ratio) | $\frac{\mu_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{365}$ |
| | 年化索提诺比率 (Sortino Ratio) | $\frac{\mu_{\text{daily}}}{\sigma_{\text{downside}}} \times \sqrt{365}$ |
| | 卡玛比率 (Calmar Ratio) | $\frac{\text{Total\_Return}}{|\text{Max\_Drawdown}|}$ |
| **风险类** | 最大回撤 (Max Drawdown) | $\min\left(\frac{\text{Equity}_t - \text{Peak}_t}{\text{Peak}_t}\right)$ |
| **交易质量** | 胜率 (Win Rate) | $\frac{\text{盈利交易笔数}}{\text{总交易笔数}}$ |
| | 盈亏因子 (Profit Factor) | $\frac{\sum \text{盈利金额}}{\sum |\text{亏损金额}|}$ |
| | 交易次数 (Total Trades) | 持仓不为 0 的小时总数 |
| | 多/空交易次数 | 分别统计做多/做空小时数 |

---

## 第二部分：固定阈值简易交易策略 —— 核心代码片段

以下代码为剥离了数据加载、模型训练、可视化等冗余逻辑后的**纯策略核心**，可直接复用于不同模型的回测评价。代码仅包含交易判断、仓位计算与风控约束，不依赖任何特定预测模型。

```python
import numpy as np
import pandas as pd


def fixed_threshold_strategy(
    market_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    spread_threshold: float = 50.0,
    extreme_spread_threshold: float = 200.0,
    direction_filter: bool = True,
    max_consecutive_hours: int = 48,
    fee_per_mwh: float = 2.0,
    slippage_bps: float = 50.0,
    capture_rate: float = 0.65,
    spread_clip: tuple = (-1000, 5000),
) -> pd.DataFrame:
    """
    固定阈值双向套利策略 —— 统一回测核心。

    Parameters
    ----------
    market_df : pd.DataFrame
        市场真实数据，必须包含列:
        - delivery_hour_utc : 时间戳
        - spread_usd_per_mwh : 实际 RT-DA 价差 ($/MWh)
    pred_df : pd.DataFrame
        模型预测数据，必须包含列:
        - delivery_hour_utc : 时间戳
        - predicted_spread : 预测价差
        - predicted_direction : 预测方向 (+1 / -1)，若 direction_filter=False 可省略
        - extreme_hour_flag : 极端天气标记 (0/1)，可选，若无则全部使用常规阈值
    initial_capital : float
        初始本金 ($)
    spread_threshold : float
        常规时段开仓阈值 ($/MWh)
    extreme_spread_threshold : float
        极端天气时段开仓阈值 ($/MWh)
    direction_filter : bool
        是否要求预测方向与预测价差符号一致
    max_consecutive_hours : int
        最大连续持仓小时数（风控平仓）
    fee_per_mwh : float
        每 MWh 交易佣金 ($)
    slippage_bps : float
        滑点基点 (1 bp = 0.01%)
    capture_rate : float
        价差捕获率 (0~1)
    spread_clip : tuple
        实际价差截断区间 (min, max)

    Returns
    -------
    pd.DataFrame
        包含 Signal, Hourly_Pnl, Cumulative_Pnl, Equity 等完整回测列
    """

    # ---- 1. 数据合并与预处理 ----
    data = pd.merge(
        market_df, pred_df, on='delivery_hour_utc', how='inner'
    ).sort_values('delivery_hour_utc').reset_index(drop=True)

    # 价差截断（风控）
    data['spread_usd_per_mwh'] = np.clip(
        data['spread_usd_per_mwh'], spread_clip[0], spread_clip[1]
    )

    # 极端小时标记补全
    if 'extreme_hour_flag' not in data.columns:
        data['extreme_hour_flag'] = False

    # ---- 2. 动态阈值计算 ----
    data['dynamic_threshold'] = np.where(
        data['extreme_hour_flag'] == 1,
        extreme_spread_threshold,
        spread_threshold,
    )

    # ---- 3. 信号生成（核心交易判断） ----
    # 原始多空信号：预测价差超过动态阈值即开仓
    long_signal = data['predicted_spread'] > data['dynamic_threshold']
    short_signal = data['predicted_spread'] < -data['dynamic_threshold']

    # 方向一致性过滤
    if direction_filter and 'predicted_direction' in data.columns:
        long_signal = long_signal & (data['predicted_direction'] == 1)
        short_signal = short_signal & (data['predicted_direction'] == -1)

    data['Raw_Signal'] = 0
    data.loc[long_signal, 'Raw_Signal'] = 1
    data.loc[short_signal, 'Raw_Signal'] = -1

    # ---- 4. 连续持仓风控 ----
    data['Signal'] = data['Raw_Signal'].copy()
    consecutive = 0
    for i in range(len(data)):
        if data.loc[i, 'Raw_Signal'] != 0:
            consecutive += 1
            if consecutive > max_consecutive_hours:
                data.loc[i, 'Signal'] = 0   # 强制平仓
        else:
            consecutive = 0

    # ---- 5. 交易成本与盈亏计算 ----
    # 交易动作检测（信号变化 = 发生交易）
    data['Trade_Action'] = data['Signal'].diff().fillna(0).abs()

    # 佣金：仅在发生交易动作时收取
    data['Commission'] = data['Trade_Action'] * fee_per_mwh

    # 滑点：按实际价差绝对值 × 滑点基点数计算
    data['Slippage'] = (
        data['Trade_Action']
        * np.abs(data['spread_usd_per_mwh'])
        * (slippage_bps / 10000.0)
    )

    # 捕获收益：方向 × 实际价差 × 捕获率
    data['Captured_Spread'] = (
        data['Signal'] * data['spread_usd_per_mwh'] * capture_rate
    )

    # 小时级净盈亏
    data['Hourly_Pnl'] = (
        data['Captured_Spread'] - data['Commission'] - data['Slippage']
    )

    # 累计盈亏与权益曲线
    data['Cumulative_Pnl'] = data['Hourly_Pnl'].cumsum()
    data['Equity'] = initial_capital + data['Cumulative_Pnl']

    return data
```

### 使用示例

```python
# 假设 market_df 和 pred_df 已由各模型成员准备好
result = fixed_threshold_strategy(
    market_df,
    pred_df,
    initial_capital=100_000,
    spread_threshold=50.0,           # 最佳常规阈值（经网格搜索确定）
    extreme_spread_threshold=200.0,  # 最佳极端阈值（经网格搜索确定）
    direction_filter=True,
    max_consecutive_hours=48,
    fee_per_mwh=2.0,
    slippage_bps=50.0,
    capture_rate=0.65,
)
```

---

## 附录：关键参数速查表

| 参数 | 基准值 | 作用域 |
|---|---|---|
| `initial_capital` | 100,000 | 初始权益 |
| `spread_threshold` | 50.0（网格搜索最优） | 常规开仓阈值 |
| `extreme_spread_threshold` | 200.0（网格搜索最优） | 极端天气开仓阈值 |
| `direction_filter` | True | 方向一致性过滤 |
| `max_consecutive_hours` | 48 | 最大连续持仓 |
| `fee_per_mwh` | 2.0 | 佣金 |
| `slippage_bps` | 50.0 | 滑点 |
| `capture_rate` | 0.65 | 价差捕获率 |
| `spread_clip` | (-1000, 5000) | 价差截断区间 |
