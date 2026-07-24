"""
Baseline 1: 天真策略 / 随机策略
=================================
包含两个子策略：
  A) 盲目跟日前市场 (Always Short): 始终做空价差 (RT < DA)
  B) 每日随机策略 (Random Daily): 每天随机选择做多/做空/不交易

用途：作为策略性能的下限基准，任何有效策略应显著优于这两个天真策略。

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


def strategy_always_short(market_df, pred_df):
    """
    策略 A: 盲目跟日前市场 — 始终做空价差

    逻辑：在 ERCOT 市场中，实时电价 (RT) 历史平均低于日前电价 (DA)，
    因此盲目做空价差（即押注 RT < DA）是一个简单的基准策略。
    这相当于"闭眼跟市场"的天真策略。

    信号: Signal = -1 (全部小时做空)
    """
    if pred_df is not None:
        data = pd.merge(
            market_df, pred_df[['delivery_hour_utc']],
            on='delivery_hour_utc', how='inner'
        )
    else:
        data = market_df
    return pd.Series(-1, index=data.index, dtype=int)


def strategy_random_daily(market_df, pred_df, seed=42):
    """
    策略 B: 每日随机策略

    逻辑：每天随机选择做多、做空或不交易。
    模拟一个没有预测能力的随机交易者。

    概率分配：40% 做空, 40% 做多, 20% 不交易
    """
    if pred_df is not None:
        data = pd.merge(
            market_df, pred_df[['delivery_hour_utc']],
            on='delivery_hour_utc', how='inner'
        )
    else:
        data = market_df

    data = data.sort_values('delivery_hour_utc').reset_index(drop=True)
    data['date'] = data['delivery_hour_utc'].dt.date

    np.random.seed(seed)
    daily_choices = {}
    for d in data['date'].unique():
        daily_choices[d] = np.random.choice([-1, 1, 0], p=[0.40, 0.40, 0.20])

    signals = data['date'].map(daily_choices).astype(int)
    return signals


def run_baseline1():
    """运行 Baseline 1 的所有子策略"""
    results = {}

    print("=" * 60)
    print("Baseline 1: 天真策略 / 随机策略")
    print("=" * 60)

    # ---- 2025 数据 ----
    print("\n>>> 加载 2025 验证集数据...")
    market_2025, pred_2025 = load_validation_2025()
    print(f"  2025 数据: {len(market_2025)} 小时")

    engine_2025 = BaselineBacktestEngine(
        market_2025, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    # 策略 A: Always Short
    print("\n--- 策略 A: Always Short (始终做空) ---")
    sig_a_2025 = strategy_always_short(market_2025, pred_2025)
    res_a_2025 = engine_2025.run_backtest(sig_a_2025)
    metrics_a_2025 = BaselineBacktestEngine.calculate_metrics(
        res_a_2025, engine_2025.initial_capital
    )
    print_metrics(metrics_a_2025)
    results['B1A_AlwaysShort_2025'] = metrics_a_2025

    plot_backtest_result(
        res_a_2025,
        title="Baseline 1A: Always Short — 2025 Validation",
        save_path=os.path.join(OUTPUT_DIR, 'baseline1a_2025.png')
    )

    # 策略 B: Random Daily
    print("\n--- 策略 B: Random Daily (每日随机) ---")
    sig_b_2025 = strategy_random_daily(market_2025, pred_2025, seed=42)
    res_b_2025 = engine_2025.run_backtest(sig_b_2025)
    metrics_b_2025 = BaselineBacktestEngine.calculate_metrics(
        res_b_2025, engine_2025.initial_capital
    )
    print_metrics(metrics_b_2025)
    results['B1B_RandomDaily_2025'] = metrics_b_2025

    plot_backtest_result(
        res_b_2025,
        title="Baseline 1B: Random Daily — 2025 Validation",
        save_path=os.path.join(OUTPUT_DIR, 'baseline1b_2025.png')
    )

    # ---- 2026 数据 ----
    print("\n>>> 加载 2026 H1 测试集数据...")
    market_2026, pred_2026 = load_test_2026()
    print(f"  2026 数据: {len(market_2026)} 小时")

    engine_2026 = BaselineBacktestEngine(
        market_2026, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    # 策略 A: Always Short (2026)
    print("\n--- 策略 A: Always Short (2026) ---")
    sig_a_2026 = strategy_always_short(market_2026, pred_2026)
    res_a_2026 = engine_2026.run_backtest(sig_a_2026)
    metrics_a_2026 = BaselineBacktestEngine.calculate_metrics(
        res_a_2026, engine_2026.initial_capital
    )
    print_metrics(metrics_a_2026)
    results['B1A_AlwaysShort_2026'] = metrics_a_2026

    plot_backtest_result(
        res_a_2026,
        title="Baseline 1A: Always Short — 2026 H1 (Frozen Test)",
        save_path=os.path.join(OUTPUT_DIR, 'baseline1a_2026.png')
    )

    # 策略 B: Random Daily (2026)
    print("\n--- 策略 B: Random Daily (2026) ---")
    sig_b_2026 = strategy_random_daily(market_2026, pred_2026, seed=123)
    res_b_2026 = engine_2026.run_backtest(sig_b_2026)
    metrics_b_2026 = BaselineBacktestEngine.calculate_metrics(
        res_b_2026, engine_2026.initial_capital
    )
    print_metrics(metrics_b_2026)
    results['B1B_RandomDaily_2026'] = metrics_b_2026

    plot_backtest_result(
        res_b_2026,
        title="Baseline 1B: Random Daily — 2026 H1 (Frozen Test)",
        save_path=os.path.join(OUTPUT_DIR, 'baseline1b_2026.png')
    )

    # 保存结果
    save_metrics_json(results, os.path.join(OUTPUT_DIR, 'baseline1_results.json'))
    print(f"\n✅ Baseline 1 结果已保存至 baseline1_results.json")

    return results


if __name__ == "__main__":
    run_baseline1()