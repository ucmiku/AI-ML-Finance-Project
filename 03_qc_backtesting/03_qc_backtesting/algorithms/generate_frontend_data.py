"""
Generate comprehensive frontend JSON data from C1 strategy backtest results.
Output: backtest_result_c1.json — consumed by member D (frontend developer).
"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sumstrategy import (
    load_c1_unified_2025, ERCOTBacktestEngine,
    analyze_risk_concentration, analyze_strategy_comparison
)


def main():
    start_time = time.time()
    market_c1, pred_c1 = load_c1_unified_2025()
    engine = ERCOTBacktestEngine(
        market_c1,
        initial_capital=100000,
        fee_per_mwh=2.0,
        slippage_bps=50.0,
        capture_rate=0.65,
    )

    # ==========================================
    # Run all 7 strategies
    # ==========================================
    strategies = {}

    res_s1 = engine.execute_b2b_baseline(pred_c1, threshold=0.60)
    strategies['B2B_Baseline_060'] = res_s1

    res_s2 = engine.execute_b2b_b2a_combined(
        pred_c1, threshold=0.60,
        use_b2a_direction=True, use_b2a_magnitude=False,
    )
    strategies['B2B_B2A_Direction'] = res_s2

    res_s3 = engine.execute_b2b_b2a_combined(
        pred_c1, threshold=0.60,
        use_b2a_direction=True, use_b2a_magnitude=True,
        min_magnitude=5.0,
    )
    strategies['B2B_B2A_DirMag'] = res_s3

    res_s4 = engine.execute_confidence_scaled(
        pred_c1, threshold=0.60, use_confidence_sizing=True,
    )
    strategies['ConfidenceScaled_060'] = res_s4

    res_s5 = engine.execute_p_outer_strategy(
        pred_c1, threshold=0.60, p_outer_threshold=0.15,
    )
    strategies['pOuter_Filter_015'] = res_s5

    res_s6 = engine.execute_b2b_b2a_combined(
        pred_c1, threshold=0.60,
        use_b2a_direction=True, extreme_weather_filter=True,
    )
    strategies['ExtremeWeather_Only'] = res_s6

    res_s7 = engine.execute_b2b_baseline(pred_c1, threshold=0.70)
    strategies['B2B_Optimal_070'] = res_s7

    # ==========================================
    # Build comprehensive JSON for frontend
    # ==========================================
    output = {
        'meta': {
            'project': 'ERCOT Extreme-Weather-Driven ML Arbitrage Strategy & Backtest Platform',
            'data_source': 'C1_unified_prediction_table_2025_oof_v3.parquet',
            'model': 'C1_XGBoost_Prediction_Agent (B2A Regression + B2B 5/20 Classifier)',
            'model_version': 'v3',
            'backtest_period': '2025-01-01 to 2025-12-31 (8760 hours)',
            'assumptions': {
                'initial_capital_usd': 100000,
                'position_size_mwh': 1,
                'commission_per_mwh_usd': 2.0,
                'slippage_formula': 'abs(spread) * 0.005',
                'capture_rate': 0.65,
                'cost_model': 'per_execution_hour',
                'handoff_baseline_threshold': 0.60,
                'handoff_baseline_rule': 'p_positive>=0.60 => DEC(+1); p_negative>=0.60 => INC(-1); else NO_TRADE(0)',
            },
            'generated_at': str(pd.Timestamp.now()),
        },

        'strategy_comparison': [],
        'equity_curves': {},
        'trades': {},
        'monthly_pnl': {},
        'risk_reports': {},
        'market_data': {},
    }

    # Build daily aggregated equity curves & strategy metrics
    for name, res in strategies.items():
        m = ERCOTBacktestEngine.calculate_metrics(res, 100000)
        res_copy = res.copy()
        res_copy['date'] = res_copy['delivery_hour_utc'].dt.date

        # Daily aggregation
        daily = res_copy.groupby('date').agg(
            daily_pnl=('Hourly_Pnl', 'sum'),
            trades=('Signal', lambda x: (x != 0).sum()),
            end_equity=('Equity', 'last'),
        ).reset_index()
        daily['cumulative_equity'] = 100000 + daily['daily_pnl'].cumsum()

        # Trade statistics
        trades_df = res[res['Signal'] != 0]
        n_pos = int((res['Signal'] == 1).sum())
        n_neg = int((res['Signal'] == -1).sum())

        # Direction precision: Signal direction matches actual class
        if 'actual_class' in res.columns:
            tr = trades_df
            correct_inc = ((tr['Signal'] == -1) & (tr['actual_class'].isin([1, 2]))).sum()
            correct_dec = ((tr['Signal'] == 1) & (tr['actual_class'].isin([4, 5]))).sum()
            dir_precision = float((correct_inc + correct_dec) / max(len(tr), 1))
        else:
            dir_precision = float((trades_df['Hourly_Pnl'] > 0).mean())

        # Monthly PnL
        res_copy['month_label'] = res_copy['delivery_hour_utc'].dt.strftime('%Y-%m')
        monthly = res_copy.groupby('month_label')['Hourly_Pnl'].sum()
        profitable = int((monthly > 0).sum())

        # Risk concentration
        risk, _, _, _ = analyze_risk_concentration(res)

        # Strategy entry
        output['strategy_comparison'].append({
            'name': name,
            'label': name.replace('_', ' '),
            'total_pnl': round(float(m['Total_Pnl']), 2),
            'total_return': round(float(m['Total_Return']), 4),
            'sharpe_ratio': round(float(m['Sharpe_Ratio']), 4),
            'sortino_ratio': round(float(m['Sortino_Ratio']), 4),
            'max_drawdown': round(float(m['Max_Drawdown']), 4),
            'calmar_ratio': round(float(m['Calmar_Ratio']), 4),
            'win_rate': round(float(m['Win_Rate']), 4),
            'profit_factor': (
                round(float(m['Profit_Factor']), 4)
                if m['Profit_Factor'] != float('inf') else 999.0
            ),
            'total_trades': int(m['Total_Trades']),
            'long_trades': n_pos,
            'short_trades': n_neg,
            'avg_trade_pnl': round(float(m['Avg_Trade_Pnl']), 2),
            'direction_precision': round(float(dir_precision), 4),
            'profitable_months': f'{profitable}/{len(monthly)}',
            'pnl_per_trade': round(float(m['Total_Pnl'] / max(m['Total_Trades'], 1)), 2),
        })

        # Equity curves (daily resolution, for multi-line chart)
        output['equity_curves'][name] = [
            {'date': str(row['date']), 'equity': round(float(row['end_equity']), 2)}
            for _, row in daily.iterrows()
        ]

        # Trade records (latest 300 for frontend table)
        trades_list = []
        for _, row in trades_df.tail(300).iterrows():
            trades_list.append({
                'timestamp': str(row['delivery_hour_utc']),
                'action': 'LONG' if row['Signal'] == 1 else 'SHORT',
                'signal': int(row['Signal']),
                'actual_spread': round(float(row['spread_usd_per_mwh']), 2),
                'predicted_spread': round(float(row.get('predicted_spread', 0)), 2),
                'p_positive': round(float(row.get('p_positive', 0)), 4),
                'p_negative': round(float(row.get('p_negative', 0)), 4),
                'confidence': round(float(row.get('confidence', 0)), 4),
                'hourly_pnl': round(float(row['Hourly_Pnl']), 2),
                'extreme_weather': bool(row.get('fixed_extreme_weather_flag', 0) == 1),
            })
        output['trades'][name] = trades_list

        # Monthly PnL
        output['monthly_pnl'][name] = [
            {'month': str(k), 'pnl': round(float(v), 2)}
            for k, v in monthly.items()
        ]

        # Risk report
        risk_entry = {
            'total_pnl': round(float(risk['Total_PnL']), 2),
            'top5_days_pnl': round(float(risk['Top5_Days_PnL']), 2),
            'top5_concentration': round(float(risk['Top5_Concentration']), 4),
            'pnl_ex_top5': round(float(risk['PnL_ex_Top5']), 2),
            'top10_days_pnl': round(float(risk['Top10_Days_PnL']), 2),
            'top10_concentration': round(float(risk['Top10_Concentration']), 4),
            'pnl_ex_top10': round(float(risk['PnL_ex_Top10']), 2),
            'profitable_months': risk['Profitable_Months'],
            'january_pnl': round(float(risk['January_PnL']), 2),
            'non_january_pnl': round(float(risk['Non_January_PnL']), 2),
            'january_concentration': round(float(risk['January_Concentration']), 4),
        }
        if 'Extreme_Weather_PnL' in risk:
            risk_entry['extreme_weather_pnl'] = round(float(risk['Extreme_Weather_PnL']), 2)
            risk_entry['normal_weather_pnl'] = round(float(risk['Normal_Weather_PnL']), 2)
            risk_entry['extreme_trades'] = int(risk['Extreme_Trades'])
            risk_entry['normal_trades'] = int(risk['Normal_Trades'])
        output['risk_reports'][name] = risk_entry

    # Market data: daily average spread (for context)
    daily_spread = market_c1.copy()
    daily_spread['date'] = daily_spread['delivery_hour_utc'].dt.date
    daily_avg = daily_spread.groupby('date')['spread_usd_per_mwh'].mean().reset_index()
    output['market_data'] = {
        'daily_avg_spread': [
            {'date': str(row['date']), 'avg_spread': round(float(row['spread_usd_per_mwh']), 2)}
            for _, row in daily_avg.iterrows()
        ],
        'total_hours': len(market_c1),
        'extreme_weather_hours': int(pred_c1['fixed_extreme_weather_flag'].sum()),
    }

    # B2B threshold search results (for threshold sensitivity chart)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []
    for th in thresholds:
        r = engine.execute_b2b_baseline(pred_c1, threshold=th)
        m = engine.calculate_metrics(r, 100000)
        threshold_results.append({
            'threshold': th,
            'total_pnl': round(float(m['Total_Pnl']), 2),
            'sharpe': round(float(m['Sharpe_Ratio']), 4),
            'trades': int(m['Total_Trades']),
            'win_rate': round(float(m['Win_Rate']), 4),
            'max_dd': round(float(m['Max_Drawdown']), 4),
        })
    output['threshold_sensitivity'] = threshold_results

    # ---- Load & embed multi-dimensional strategy scores ----
    scores_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_scores.json')
    if os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        output['strategy_scoring'] = {
            'methodology': scores_data.get('scoring_methodology', {}),
            'ranking': scores_data.get('ranking', []),
            'scores': scores_data.get('scores', {}),
        }
        # Also inject composite score into each strategy_comparison entry
        score_map = {name: s['composite'] for name, s in scores_data.get('scores', {}).items()}
        for entry in output['strategy_comparison']:
            entry['composite_score'] = score_map.get(entry['name'], None)
        print('Strategy scores embedded into output.')
    else:
        print('Warning: strategy_scores.json not found. Run compute_strategy_scores.py first.')

    # Write JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'backtest_result_c1.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    print(f'JSON written: {out_path} ({size_kb:.1f} KB)')
    print(f'Strategies: {len(output["strategy_comparison"])}')
    for s in output['strategy_comparison']:
        print(f'  {s["name"]:<30s} PnL=${s["total_pnl"]:>10,.2f}  '
              f'Sharpe={s["sharpe_ratio"]:>7.4f}  Trades={s["total_trades"]:>5d}')

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    if minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"
    print(f"\n程序总执行时间: {time_str}")


if __name__ == '__main__':
    main()