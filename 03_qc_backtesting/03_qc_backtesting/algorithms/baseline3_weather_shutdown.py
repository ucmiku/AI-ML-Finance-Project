"""
Baseline 3: 真实天气关闭策略 + 极端天气策略
=============================================
结合真实天气条件（extreme_hour_flag）与 ML 预测信号的混合策略。

策略逻辑：
  1. 使用 ML 预测信号 (predicted_spread) 生成原始交易信号
  2. 叠加真实天气关闭过滤器：
     - 当真实天气条件为极端 (extreme_hour_flag=True) → 允许交易
     - 当真实天气条件为正常 (extreme_hour_flag=False) → 关闭交易
  3. 对于 2026 数据（无 extreme_hour_flag），使用实际价差幅度作为
     极端天气的代理指标 (|actual_spread| > 100 视为极端)

与现有 ML 策略的区别：
  - 现有 ML 策略：使用 predicted_spread 的波动率自适应阈值，在极端天气时
    切换为固定阈值
  - Baseline 3：使用真实天气条件作为开关，仅在极端天气时允许 ML 信号交易，
    正常天气时完全关闭（风险管理的终极保守形式）

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


def strategy_weather_shutdown_ml(market_df, pred_df,
                                  rolling_window=168,
                                  std_multiplier=1.5,
                                  short_risk_multiplier=1.0,
                                  vol_regime_threshold=3.0,
                                  min_profit_ratio=1.0,
                                  extreme_spread_threshold=200.0,
                                  min_base_threshold=10.0,
                                  direction_filter=True,
                                  extreme_proxy_threshold=100.0):
    """
    真实天气关闭 + ML 极端天气策略

    核心流程：
      1. 生成 ML 原始信号（基于 predicted_spread 的自适应阈值）
      2. 检测真实天气条件：
         - 有 extreme_hour_flag → 直接使用
         - 无 extreme_hour_flag → 使用 |actual_spread| > extreme_proxy_threshold
      3. 仅在真实天气极端时保留 ML 信号，否则关闭

    Parameters
    ----------
    extreme_proxy_threshold : float
        当 extreme_hour_flag 不可用时，|actual_spread| 超过此值视为极端天气
    """
    data = pd.merge(
        market_df, pred_df[['delivery_hour_utc', 'predicted_spread',
                            'predicted_direction']],
        on='delivery_hour_utc', how='inner'
    ).sort_values('delivery_hour_utc').reset_index(drop=True)

    data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

    # Step 1: 生成 ML 原始信号（自适应波动率阈值）
    rolling_std = data['predicted_spread'].rolling(
        window=rolling_window, min_periods=24
    ).std()

    data['adaptive_threshold'] = np.maximum(
        min_base_threshold,
        rolling_std.fillna(min_base_threshold) * std_multiplier
    )

    # 极端天气覆盖
    if 'extreme_hour_flag' in data.columns:
        data['dynamic_threshold'] = np.where(
            data['extreme_hour_flag'] == True,
            extreme_spread_threshold,
            data['adaptive_threshold']
        )
    else:
        data['dynamic_threshold'] = data['adaptive_threshold']

    # 波动率 Regime 过滤器
    data['market_volatility'] = (
        data['spread_usd_per_mwh']
        .rolling(window=rolling_window, min_periods=24)
        .std()
        .fillna(vol_regime_threshold + 1)
    )
    data['low_vol_regime'] = data['market_volatility'] < vol_regime_threshold

    # 生成 ML 原始信号
    long_signal = data['predicted_spread'] > data['dynamic_threshold']
    short_signal = data['predicted_spread'] < -(
        data['dynamic_threshold'] * short_risk_multiplier
    )

    if direction_filter and 'predicted_direction' in data.columns:
        long_signal = long_signal & (data['predicted_direction'] == 1)
        short_signal = short_signal & (data['predicted_direction'] == -1)

    # 预期利润过滤器
    expected_gross_profit = np.abs(data['predicted_spread']) * 0.65
    estimated_cost = 2.0 + np.abs(data['predicted_spread']) * (50.0 / 10000.0)
    trade_worthwhile = expected_gross_profit > estimated_cost * min_profit_ratio

    ml_raw_signal = pd.Series(0, index=data.index, dtype=int)
    ml_raw_signal[long_signal & trade_worthwhile & ~data['low_vol_regime']] = 1
    ml_raw_signal[short_signal & trade_worthwhile & ~data['low_vol_regime']] = -1

    # Step 2: 检测真实天气条件
    if 'extreme_hour_flag' in data.columns:
        is_real_extreme = data['extreme_hour_flag'].astype(bool)
        weather_source = "extreme_hour_flag (真实天气特征)"
    else:
        is_real_extreme = np.abs(data['spread_usd_per_mwh']) > extreme_proxy_threshold
        weather_source = f"|actual_spread| > ${extreme_proxy_threshold} (代理指标)"

    # Step 3: 天气关闭过滤器
    final_signal = ml_raw_signal.copy()
    final_signal[~is_real_extreme] = 0

    print(f"\n  天气数据来源: {weather_source}")
    print(f"  极端天气小时数: {is_real_extreme.sum()} / {len(data)} "
          f"({is_real_extreme.sum()/len(data)*100:.1f}%)")
    print(f"  ML 原始信号数: {(ml_raw_signal != 0).sum()}")
    print(f"  天气关闭后信号数: {(final_signal != 0).sum()}")
    print(f"  被关闭信号数: {(ml_raw_signal != 0).sum() - (final_signal != 0).sum()}")

    return final_signal


def run_baseline3():
    """运行 Baseline 3: 真实天气关闭 + 极端天气策略"""
    results = {}

    print("=" * 60)
    print("Baseline 3: 真实天气关闭 + 极端天气策略")
    print("=" * 60)

    # ---- 2025 数据 (有 extreme_hour_flag) ----
    print("\n>>> 加载 2025 验证集数据...")
    market_2025, pred_2025 = load_validation_2025()
    print(f"  2025 数据: {len(market_2025)} 小时")

    engine_2025 = BaselineBacktestEngine(
        market_2025, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    print("\n--- 策略参数 ---")
    print(f"  滚动窗口: 168h, 波动率倍数: 1.5x")
    print(f"  波动率最低阈值: $3.0/MWh, 极端价差阈值: $200/MWh")
    print(f"  空头风险系数: 1.0, 方向过滤: 开启")

    sig_ws_2025 = strategy_weather_shutdown_ml(
        market_2025, pred_2025,
        rolling_window=168, std_multiplier=1.5,
        short_risk_multiplier=1.0,
        vol_regime_threshold=3.0,
        min_profit_ratio=1.0,
        direction_filter=True,
    )
    res_ws_2025 = engine_2025.run_backtest(sig_ws_2025)
    metrics_ws_2025 = BaselineBacktestEngine.calculate_metrics(
        res_ws_2025, engine_2025.initial_capital
    )

    print("\n--- 2025 天气关闭策略指标 ---")
    print_metrics(metrics_ws_2025)

    n_long = (sig_ws_2025 == 1).sum()
    n_short = (sig_ws_2025 == -1).sum()
    print(f"\n  信号分布: LONG={n_long}, SHORT={n_short}")

    # 极端 vs 正常天气表现分解
    if 'extreme_hour_flag' in pred_2025.columns:
        merged = res_ws_2025.merge(
            pred_2025[['delivery_hour_utc', 'extreme_hour_flag']],
            on='delivery_hour_utc', how='left'
        )
        ext_mask = merged['extreme_hour_flag'] == True
        extreme_pnl = merged.loc[ext_mask, 'Hourly_Pnl'].sum()
        normal_pnl = merged.loc[~ext_mask, 'Hourly_Pnl'].sum()
        print(f"\n  极端天气 PnL: ${extreme_pnl:,.2f}")
        print(f"  正常天气 PnL: ${normal_pnl:,.2f} (应接近 $0，因为关闭交易)")

    results['B3_WeatherShutdown_2025'] = metrics_ws_2025

    plot_backtest_result(
        res_ws_2025,
        title="Baseline 3: Weather Shutdown + ML — 2025 Validation",
        save_path=os.path.join(OUTPUT_DIR, 'baseline3_weather_2025.png')
    )

    # ---- 2026 数据 (无 extreme_hour_flag，使用代理) ----
    print("\n>>> 加载 2026 H1 测试集数据...")
    market_2026, pred_2026 = load_test_2026()
    print(f"  2026 数据: {len(market_2026)} 小时")

    engine_2026 = BaselineBacktestEngine(
        market_2026, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65
    )

    sig_ws_2026 = strategy_weather_shutdown_ml(
        market_2026, pred_2026,
        rolling_window=168, std_multiplier=1.5,
        short_risk_multiplier=1.0,
        vol_regime_threshold=3.0,
        min_profit_ratio=1.0,
        direction_filter=True,
        extreme_proxy_threshold=100.0,
    )
    res_ws_2026 = engine_2026.run_backtest(sig_ws_2026)
    metrics_ws_2026 = BaselineBacktestEngine.calculate_metrics(
        res_ws_2026, engine_2026.initial_capital
    )

    print("\n--- 2026 天气关闭策略指标 ---")
    print_metrics(metrics_ws_2026)

    n_long = (sig_ws_2026 == 1).sum()
    n_short = (sig_ws_2026 == -1).sum()
    print(f"\n  信号分布: LONG={n_long}, SHORT={n_short}")

    results['B3_WeatherShutdown_2026'] = metrics_ws_2026

    plot_backtest_result(
        res_ws_2026,
        title="Baseline 3: Weather Shutdown + ML — 2026 H1 (Frozen Test)",
        save_path=os.path.join(OUTPUT_DIR, 'baseline3_weather_2026.png')
    )

    # 保存结果
    save_metrics_json(results, os.path.join(OUTPUT_DIR, 'baseline3_results.json'))
    print(f"\n✅ Baseline 3 结果已保存至 baseline3_results.json")

    return results


if __name__ == "__main__":
    run_baseline3()