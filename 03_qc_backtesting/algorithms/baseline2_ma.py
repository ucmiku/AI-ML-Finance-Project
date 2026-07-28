"""
Baseline 2: 经典双均线 MA 策略
===============================
基于真实 RT-DA 价差的双均线交叉策略（纯技术分析，不依赖 ML 预测）。

策略逻辑：
  - 短期均线 (MA_24): 过去 24 小时真实价差 (actual_spread) 的滚动均值
  - 长期均线 (MA_168): 过去 168 小时（一周）真实价差的滚动均值
  - 金叉 (MA_24 > MA_168): 价差趋势向上 → 做多 (Signal = +1)
  - 死叉 (MA_24 < MA_168): 价差趋势向下 → 做空 (Signal = -1)

额外风控：
  - 最小价差阈值：价差幅度过小时不交易（避免噪音）
  - 波动率过滤器：市场过于平稳时不开仓

回测假设（与 ML 策略完全一致）：
  - 初始本金: $100,000
  - 单笔仓位: 1 MWh/执行小时
  - 交易佣金: $2.00/MWh
  - 滑点: abs(spread) * 0.005 (50 bps)
  - 价差捕获率: 65%
"""
import numpy as np
import pandas as pd
import os
import sys
from baseline_utils import (
    load_validation_2025, load_test_2026,
    BaselineBacktestEngine, print_metrics, save_metrics_json,
    plot_backtest_result, OUTPUT_DIR
)


def strategy_ma_crossover(market_df, pred_df=None,
                           short_window=24, long_window=168,
                           min_spread_threshold=5.0,
                           vol_filter_window=168,
                           vol_filter_threshold=3.0):
    """
    双均线交叉策略

    Parameters
    ----------
    market_df : pd.DataFrame
        市场数据，含 delivery_hour_utc, spread_usd_per_mwh
    pred_df : pd.DataFrame, optional
        预测数据，用于对齐时间
    short_window : int
        短期均线窗口（小时），默认 24h
    long_window : int
        长期均线窗口（小时），默认 168h（一周）
    min_spread_threshold : float
        最小价差阈值，|spread| < 此值时不开仓
    vol_filter_window : int
        波动率过滤器窗口
    vol_filter_threshold : float
        波动率低于此值时不开仓

    Returns
    -------
    signals : pd.Series
    """
    if pred_df is not None:
        data = pd.merge(
            market_df, pred_df[['delivery_hour_utc']],
            on='delivery_hour_utc', how='inner'
        ).sort_values('delivery_hour_utc').reset_index(drop=True)
    else:
        data = market_df.sort_values('delivery_hour_utc').reset_index(drop=True)

    spread = data['spread_usd_per_mwh'].values

    spread_lagged = pd.Series(spread).shift(1).values

    short_ma = pd.Series(spread_lagged).rolling(
        window=short_window, min_periods=short_window
    ).mean().values
    long_ma = pd.Series(spread_lagged).rolling(
        window=long_window, min_periods=long_window
    ).mean().values

    volatility = pd.Series(spread_lagged).rolling(
        window=vol_filter_window, min_periods=vol_filter_window
    ).std().values

    signals = np.zeros(len(data), dtype=int)

    for i in range(max(short_window, long_window, vol_filter_window) + 1, len(data)):
        if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
            continue

        if volatility[i] < vol_filter_threshold:
            continue

        if abs(spread_lagged[i]) < min_spread_threshold:
            continue

        if short_ma[i] > long_ma[i]:
            signals[i] = 1  # 金叉 → 做多
        elif short_ma[i] < long_ma[i]:
            signals[i] = -1  # 死叉 → 做空

    return pd.Series(signals, index=data.index, dtype=int)


def run_baseline2():
    """运行 Baseline 2 双均线策略"""
    results = {}

    print("=" * 60)
    print("Baseline 2: 经典双均线 MA 策略")
    print("=" * 60)

    # ---- 2025 数据 ----
    print("\n>>> 加载 2025 验证集数据...")
    market_2025, pred_2025 = load_validation_2025()
    print(f"  2025 数据: {len(market_2025)} 小时")

    engine_2025 = BaselineBacktestEngine(
        market_2025, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    print("\n--- MA 策略参数 ---")
    print(f"  短期均线: 24h, 长期均线: 168h (一周)")
    print(f"  最小价差阈值: $5.0/MWh, 波动率最低阈值: $3.0/MWh")

    sig_ma_2025 = strategy_ma_crossover(
        market_2025, pred_2025,
        short_window=24, long_window=168,
        min_spread_threshold=5.0,
        vol_filter_threshold=3.0,
    )
    res_ma_2025 = engine_2025.run_backtest(sig_ma_2025)
    metrics_ma_2025 = BaselineBacktestEngine.calculate_metrics(
        res_ma_2025, engine_2025.initial_capital
    )

    print("\n--- 2025 双均线策略指标 ---")
    print_metrics(metrics_ma_2025)

    n_long = (sig_ma_2025 == 1).sum()
    n_short = (sig_ma_2025 == -1).sum()
    n_none = (sig_ma_2025 == 0).sum()
    print(f"\n  信号分布: LONG={n_long}, SHORT={n_short}, NO_TRADE={n_none}")

    results['B2_MA_Crossover_2025'] = metrics_ma_2025

    plot_backtest_result(
        res_ma_2025,
        title="Baseline 2: MA Crossover (24h/168h) — 2025 Validation",
        save_path=os.path.join(OUTPUT_DIR, 'baseline2_ma_2025.png')
    )

    # ---- 2026 数据 ----
    print("\n>>> 加载 2026 H1 测试集数据...")
    market_2026, pred_2026 = load_test_2026()
    print(f"  2026 数据: {len(market_2026)} 小时")

    engine_2026 = BaselineBacktestEngine(
        market_2026, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    sig_ma_2026 = strategy_ma_crossover(
        market_2026, pred_2026,
        short_window=24, long_window=168,
        min_spread_threshold=5.0,
        vol_filter_threshold=3.0,
    )
    res_ma_2026 = engine_2026.run_backtest(sig_ma_2026)
    metrics_ma_2026 = BaselineBacktestEngine.calculate_metrics(
        res_ma_2026, engine_2026.initial_capital
    )

    print("\n--- 2026 双均线策略指标 ---")
    print_metrics(metrics_ma_2026)

    n_long = (sig_ma_2026 == 1).sum()
    n_short = (sig_ma_2026 == -1).sum()
    n_none = (sig_ma_2026 == 0).sum()
    print(f"\n  信号分布: LONG={n_long}, SHORT={n_short}, NO_TRADE={n_none}")

    results['B2_MA_Crossover_2026'] = metrics_ma_2026

    plot_backtest_result(
        res_ma_2026,
        title="Baseline 2: MA Crossover (24h/168h) — 2026 H1 (Frozen Test)",
        save_path=os.path.join(OUTPUT_DIR, 'baseline2_ma_2026.png')
    )

    # 保存结果
    save_metrics_json(results, os.path.join(OUTPUT_DIR, 'baseline2_results.json'))
    print(f"\n✅ Baseline 2 结果已保存至 baseline2_results.json")

    return results


if __name__ == "__main__":
    run_baseline2()