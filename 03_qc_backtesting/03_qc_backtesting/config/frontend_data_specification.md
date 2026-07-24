# 前端数据接口规范

> **交付对象**：成员 D（全栈/前端开发）
> **数据文件**：`bk_testing/backtest_result_c1.json`（930 KB）
> **生成脚本**：`bk_testing/generate_frontend_data.py`
> **更新频率**：策略参数变更后重新运行 `generate_frontend_data.py` 即可

---

## 1. JSON 顶层结构

```json
{
  "meta":                  { ... },  // 项目元信息与回测假设
  "strategy_comparison":   [ ... ],  // 7 策略核心指标对比表
  "equity_curves":         { ... },  // 各策略每日净值曲线
  "trades":                { ... },  // 各策略交易明细（近 300 笔）
  "monthly_pnl":           { ... },  // 各策略月度 PnL 分布
  "risk_reports":          { ... },  // 各策略风险集中度分析
  "market_data":           { ... },  // 市场背景数据（日度均价差）
  "threshold_sensitivity": [ ... ]   // B2B 阈值敏感性曲线
}
```

---

## 2. `meta` — 回测元信息

```json
{
  "meta": {
    "project": "ERCOT Extreme-Weather-Driven ML Arbitrage Strategy & Backtest Platform",
    "data_source": "C1_unified_prediction_table_2025_oof_v3.parquet",
    "model": "C1_XGBoost_Prediction_Agent (B2A Regression + B2B 5/20 Classifier)",
    "model_version": "v3",
    "backtest_period": "2025-01-01 to 2025-12-31 (8760 hours)",
    "assumptions": {
      "initial_capital_usd": 100000,
      "position_size_mwh": 1,
      "commission_per_mwh_usd": 2.0,
      "slippage_formula": "abs(spread) * 0.005",
      "capture_rate": 0.65,
      "cost_model": "per_execution_hour",
      "handoff_baseline_threshold": 0.60,
      "handoff_baseline_rule": "p_positive>=0.60 => DEC(+1); p_negative>=0.60 => INC(-1); else NO_TRADE(0)"
    },
    "generated_at": "2026-07-23 ..."
  }
}
```

**前端用途**：页面顶部显示回测参数摘要，或作为 tooltip 提供上下文。

---

## 3. `strategy_comparison` — 多策略对比表

```json
{
  "strategy_comparison": [
    {
      "name": "B2B_Baseline_060",
      "label": "B2B Baseline 060",
      "total_pnl": 3439.00,
      "total_return": 0.0344,
      "sharpe_ratio": 1.5828,
      "sortino_ratio": 1.3339,
      "max_drawdown": -0.0176,
      "calmar_ratio": 89.94,
      "win_rate": 0.6014,
      "profit_factor": 1.4089,
      "total_trades": 1174,
      "long_trades": 200,
      "short_trades": 974,
      "avg_trade_pnl": 2.93,
      "direction_precision": 0.5681,
      "profitable_months": "8/13",
      "pnl_per_trade": 2.93
    }
    // ... 其余 6 个策略
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 | 前端渲染建议 |
|---|---|---|---|
| `name` | string | 策略内部标识 | — |
| `label` | string | 策略显示名称 | 表格/图例标签 |
| `total_pnl` | float | 总盈亏 ($) | ⚠️ 标注货币单位 |
| `total_return` | float | 总收益率（小数） | ×100 显示为百分比 |
| `sharpe_ratio` | float | 年化夏普比率 | >1.5 绿色，<0.5 红色 |
| `sortino_ratio` | float | 年化索提诺比率 | 下行风险调整 |
| `max_drawdown` | float | 最大回撤（小数） | ×100 显示为百分比 |
| `calmar_ratio` | float | 卡尔玛比率 | 收益/回撤比 |
| `win_rate` | float | 胜率（小数） | ×100 显示为百分比 |
| `profit_factor` | float | 盈亏比 | >1.5 绿色，<1.0 红色 |
| `total_trades` | int | 总交易笔数 | 纯数字 |
| `long_trades` | int | 多头交易笔数 | 🟢 绿色 |
| `short_trades` | int | 空头交易笔数 | 🔴 红色 |
| `avg_trade_pnl` | float | 平均每笔盈亏 ($) | — |
| `direction_precision` | float | 方向精度（小数） | 信号与实际方向一致率 |
| `profitable_months` | string | 盈利月数/总月数 | "8/13" |
| `pnl_per_trade` | float | PnL/交易 ($) | = total_pnl / total_trades |

**前端渲染建议**：使用**可排序的表格**，默认按 `sharpe_ratio` 降序排列。对关键列（Sharpe、PnL、Win Rate）使用条件着色。

---

## 4. `equity_curves` — 净值曲线

```json
{
  "equity_curves": {
    "B2B_Baseline_060": [
      { "date": "2025-01-01", "equity": 99999.98 },
      { "date": "2025-01-02", "equity": 100012.04 }
      // ... 每日一条（约 365 条/策略）
    ],
    "B2B_Optimal_070": [ ... ],
    "ExtremeWeather_Only": [ ... ]
    // ... 共 7 条曲线
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `date` | string | 日期 `YYYY-MM-DD` |
| `equity` | float | 当日末净值 ($) |

**前端渲染建议**：
- 使用 **ECharts/Chart.js 多线折线图**，横轴日期，纵轴净值。
- 默认显示 3 条：B2B_Baseline_060（灰色虚线）、B2B_Optimal_070（蓝色实线）、ExtremeWeather_Only（红色实线）。
- 其他 4 条通过图例切换显示/隐藏。
- 添加 `$100,000` 基准水平线。
- 考虑提供 Y 轴对数刻度选项（等比例观察回撤）。

---

## 5. `trades` — 交易明细

```json
{
  "trades": {
    "B2B_Baseline_060": [
      {
        "timestamp": "2025-11-15 17:00:00+00:00",
        "action": "SHORT",
        "signal": -1,
        "actual_spread": -45.32,
        "predicted_spread": -0.42,
        "p_positive": 0.0833,
        "p_negative": 0.6129,
        "confidence": 0.6129,
        "hourly_pnl": 24.55,
        "extreme_weather": false
      }
      // ... 策略最近 300 笔交易
    ]
    // ... 其余 6 个策略
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | string | 交易时间 UTC `YYYY-MM-DD HH:MM:SS+00:00` |
| `action` | string | `"LONG"` 或 `"SHORT"` |
| `signal` | int | +1 (DEC) / -1 (INC) |
| `actual_spread` | float | 实际 RT-DA 价差 ($/MWh) |
| `predicted_spread` | float | B2A 模型预测价差 ($/MWh) |
| `p_positive` | float | DEC 信号概率 (p_c4 + p_c5) |
| `p_negative` | float | INC 信号概率 (p_c1 + p_c2) |
| `confidence` | float | 最大类别概率 = max(p_c1..p_c5) |
| `hourly_pnl` | float | 该小时净盈亏 ($) |
| `extreme_weather` | bool | 是否为极端天气小时 |

**前端渲染建议**：
- 使用**可翻页的表格**展示最近交易。
- 对 `hourly_pnl` > 0 的行标绿，< 0 的行标红。
- 添加策略切换下拉框，用户选择不同策略查看其交易明细。
- 可选：交易时间轴（Timeline），标注极端天气交易。

---

## 6. `monthly_pnl` — 月度 PnL 柱状图

```json
{
  "monthly_pnl": {
    "B2B_Baseline_060": [
      { "month": "2025-01", "pnl": 447.32 },
      { "month": "2025-02", "pnl": 283.80 }
      // ... 12 个月
    ]
    // ... 其余策略
  }
}
```

**前端渲染建议**：使用**分组柱状图**，横轴月份，纵轴 PnL。同时展示 2-3 条策略便于对比。正值绿色，负值红色。

---

## 7. `risk_reports` — 风险集中度仪表盘

```json
{
  "risk_reports": {
    "B2B_Baseline_060": {
      "total_pnl": 3439.00,
      "top5_days_pnl": 2517.50,
      "top5_concentration": 0.7320,
      "pnl_ex_top5": 921.50,
      "top10_days_pnl": 3747.41,
      "top10_concentration": 1.0897,
      "pnl_ex_top10": -308.41,
      "profitable_months": "8/13",
      "january_pnl": 447.32,
      "non_january_pnl": 2991.67,
      "january_concentration": 0.13,
      "extreme_weather_pnl": 3876.12,
      "normal_weather_pnl": -437.12,
      "extreme_trades": 496,
      "normal_trades": 678
    }
    // ... 其余策略
  }
}
```

| 字段 | 类型 | 说明 | 前端渲染 |
|---|---|---|---|
| `total_pnl` | float | 总盈亏 | Stat tile |
| `top5_days_pnl` | float | Top 5 天 PnL | — |
| `top5_concentration` | float | Top 5 集中度 | ⚠️ >70% 红色警报 |
| `pnl_ex_top5` | float | 去掉 Top 5 后 PnL | <0 红色警报 |
| `extreme_weather_pnl` | float | 极端天气总 PnL | 🟢 Stat tile |
| `normal_weather_pnl` | float | 正常天气总 PnL | <0 红色 Stile |
| `extreme_trades` | int | 极端天气交易数 | — |
| `normal_trades` | int | 正常天气交易数 | — |

**前端渲染建议**：
- 使用 **Stat Tiles（指标卡片）** 展示核心风险数字。
- 使用 **饼图/环形图** 展示极端 vs 正常天气 PnL 占比。
- 使用 **水平条形图** 展示 Top 5/Top 10 集中度。
- 对 PnL ex-Top 5 为负的情况用红色高亮 + 警告图标。

---

## 8. `threshold_sensitivity` — 阈值敏感性曲线

```json
{
  "threshold_sensitivity": [
    { "threshold": 0.50, "total_pnl": -1161.76, "sharpe": -0.3832, "trades": 2598, "win_rate": 0.5239, "max_dd": -0.0344 },
    { "threshold": 0.55, "total_pnl": 700.20,  "sharpe": 0.2558,  "trades": 1789, "win_rate": 0.5567, "max_dd": -0.0222 },
    { "threshold": 0.60, "total_pnl": 3439.00, "sharpe": 1.5828,  "trades": 1174, "win_rate": 0.6014, "max_dd": -0.0176 },
    { "threshold": 0.65, "total_pnl": 3457.08, "sharpe": 1.6739,  "trades": 762,  "win_rate": 0.6535, "max_dd": -0.0171 },
    { "threshold": 0.70, "total_pnl": 4129.86, "sharpe": 2.3362,  "trades": 474,  "win_rate": 0.6920, "max_dd": -0.0143 },
    { "threshold": 0.75, "total_pnl": 3919.47, "sharpe": 2.1551,  "trades": 260,  "win_rate": 0.7500, "max_dd": -0.0142 },
    { "threshold": 0.80, "total_pnl": 2693.80, "sharpe": 1.6323,  "trades": 149,  "win_rate": 0.7651, "max_dd": -0.0122 }
  ]
}
```

**前端渲染建议**：使用**双 Y 轴折线图**：
- 左轴（蓝色）：Sharpe Ratio、Win Rate
- 右轴（绿色）：Total PnL ($)
- 横轴：阈值 0.50 → 0.80
- 标注最优阈值点（0.70）的位置

---

## 9. `market_data` — 市场背景

```json
{
  "market_data": {
    "daily_avg_spread": [
      { "date": "2025-01-01", "avg_spread": -3.45 },
      { "date": "2025-01-02", "avg_spread": 2.18 }
      // ... 365 天
    ],
    "total_hours": 8760,
    "extreme_weather_hours": 1871
  }
}
```

**前端渲染建议**：可选的背景参考线/子图，展示每日平均实际价差，帮助用户理解市场环境。

---

## 10. 推荐前端页面布局

```
┌─────────────────────────────────────────────────────────┐
│  🏠 ERCOT Extreme-Weather ML Arbitrage Dashboard         │
│  回测区间: 2025 | 模型: C1 XGBoost v3 | 初始本金: $100K  │
├────────────┬────────────┬────────────┬──────────────────┤
│ Total PnL  │ Sharpe     │ Max DD     │ Profitable       │
│ $4,130     │ 2.34       │ -1.43%     │ Months: 8/13     │
│ (Stat Tile)│ (Stat Tile)│ (Stat Tile)│ (Stat Tile)      │
├────────────┴────────────┴────────────┴──────────────────┤
│                                                          │
│  📈 多策略净值曲线对比 (Equity Curves)                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  $104K │                    ╭── Optimal 0.70       │  │
│  │  $103K │               ╭──╯                        │  │
│  │  $102K │          ╭──╯                             │  │
│  │  $101K │     ╭──╯    Baseline 0.60                 │  │
│  │  $100K │────╯                                      │  │
│  │        ├────┼────┼────┼────┼────┼────┼────┼────┤  │  │
│  │        Jan  Mar  May  Jul  Sep  Nov                │  │
│  └────────────────────────────────────────────────────┘  │
│  [图例切换: ☑Baseline ☑Optimal ☑ExtremeWx ☐Others]      │
├──────────────────────────┬───────────────────────────────┤
│                          │                               │
│  📊 策略指标对比表        │  ⚠️ 风险集中度分析             │
│  (可排序/可筛选)         │  🥧 极端 vs 正常天气 PnL 饼图  │
│                          │  📊 Top 5 集中度条形图         │
│                          │                               │
├──────────────────────────┴───────────────────────────────┤
│                                                          │
│  📉 阈值敏感性曲线 (Threshold Sensitivity)                │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Sharpe ▲                               ● PnL       │  │
│  │  2.0 │                          ●────── $4K        │  │
│  │  1.0 │               ●──────●                      │  │
│  │  0.0 │    ●────●                                   │  │
│  │      ├────┼────┼────┼────┼────┼────┤              │  │
│  │     0.50 0.55 0.60 0.65 0.70 0.75 0.80            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
├──────────────────────────┬───────────────────────────────┤
│  📋 最近交易明细          │  📊 月度 PnL 柱状图            │
│  (策略选择器 + 翻页表格)  │  (分组柱状图，正负着色)       │
│                          │                               │
└──────────────────────────┴───────────────────────────────┘
```

---

## 11. 快速接入指南

### Step 1: 获取数据
```javascript
// 直接导入 JSON（或通过 API fetch）
const response = await fetch('/api/backtest_result_c1.json');
const data = await response.json();
```

### Step 2: 渲染策略对比表
```javascript
// 按 Sharpe 降序排列
const sorted = data.strategy_comparison
  .sort((a, b) => b.sharpe_ratio - a.sharpe_ratio);

// 渲染表格，对关键指标着色
sorted.forEach(s => {
  const sharpeColor = s.sharpe_ratio > 1.5 ? 'green' :
                      s.sharpe_ratio > 0.5 ? 'orange' : 'red';
  // ... 渲染到 DOM
});
```

### Step 3: 渲染净值曲线（ECharts 示例）
```javascript
const series = Object.entries(data.equity_curves).map(([name, curve]) => ({
  name: name,
  type: 'line',
  data: curve.map(d => [d.date, d.equity]),
  smooth: true,
}));

const option = {
  title: { text: 'Multi-Strategy Equity Curves' },
  xAxis: { type: 'time' },
  yAxis: { type: 'value', axisLabel: { formatter: '$ {value}' } },
  series: series,
};
```

### Step 4: 渲染阈值敏感性
```javascript
// 双 Y 轴：Sharpe（左）、PnL（右）
const sharpeSeries = data.threshold_sensitivity.map(d => [d.threshold, d.sharpe]);
const pnlSeries = data.threshold_sensitivity.map(d => [d.threshold, d.total_pnl]);
```

---

## 12. 数据更新流程

当策略参数变更时，运行以下命令重新生成前端数据：

```bash
cd bk_testing
python generate_frontend_data.py
```

输出文件：`backtest_result_c1.json`（覆盖更新）。

---

> 📎 **战略分析报告**：[`C1_strategy_analysis_report.md`](C1_strategy_analysis_report.md) — 包含完整策略分析结论、风险集中度解读和后续研究方向。
