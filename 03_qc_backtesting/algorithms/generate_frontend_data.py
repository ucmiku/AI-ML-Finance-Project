"""
Generate comprehensive frontend JSON data from C1 strategy backtest results.
Covers BOTH 2025 OOF (development) and 2026 H1 Walk-Forward (frozen-rule test).
Output: backtest_result_c1.json — consumed by member D (frontend developer).
"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sumstrategy import (
    load_c1_unified_2025, load_c1_unified_2026,
    ERCOTBacktestEngine,
    analyze_risk_concentration, analyze_strategy_comparison,
)


def build_strategy_data(name, res, initial_capital=100000):
    """Build standardised per-strategy data block for one result DataFrame."""
    m = ERCOTBacktestEngine.calculate_metrics(res, initial_capital)
    res_copy = res.copy()
    res_copy['date'] = res_copy['delivery_hour_utc'].dt.date

    # Daily aggregation
    daily = res_copy.groupby('date').agg(
        daily_pnl=('Hourly_Pnl', 'sum'),
        trades=('Signal', lambda x: (x != 0).sum()),
        end_equity=('Equity', 'last'),
    ).reset_index()
    daily['cumulative_equity'] = initial_capital + daily['daily_pnl'].cumsum()

    # Trade statistics
    trades_df = res[res['Signal'] != 0]
    n_pos = int((res['Signal'] == 1).sum())
    n_neg = int((res['Signal'] == -1).sum())

    # Direction precision
    if 'actual_class' in res.columns:
        tr = trades_df
        correct_inc = int(((tr['Signal'] == -1) & (tr['actual_class'].isin([1, 2]))).sum())
        correct_dec = int(((tr['Signal'] == 1) & (tr['actual_class'].isin([4, 5]))).sum())
        dir_precision = float((correct_inc + correct_dec) / max(len(tr), 1))
    else:
        dir_precision = float((trades_df['Hourly_Pnl'] > 0).mean())

    # Monthly PnL
    res_copy['month_label'] = res_copy['delivery_hour_utc'].dt.strftime('%Y-%m')
    monthly = res_copy.groupby('month_label')['Hourly_Pnl'].sum()
    profitable = int((monthly > 0).sum())

    # Risk concentration
    risk, _, _, _ = analyze_risk_concentration(res)

    # Strategy metrics entry
    metrics_entry = {
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
    }

    # Equity curve (daily resolution)
    equity_curve = [
        {'date': str(row['date']), 'equity': round(float(row['end_equity']), 2)}
        for _, row in daily.iterrows()
    ]

    # Trade records (latest 300)
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

    # Monthly PnL
    monthly_pnl = [
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

    return {
        'metrics': metrics_entry,
        'equity_curve': equity_curve,
        'trades': trades_list,
        'monthly_pnl': monthly_pnl,
        'risk_report': risk_entry,
    }


def main():
    start_time = time.time()

    # ==========================================
    # PART A: 2025 OOF — Strategy Development
    # ==========================================
    print('Loading 2025 C1 OOF data...')
    market_25, pred_25 = load_c1_unified_2025()
    engine_25 = ERCOTBacktestEngine(market_25, initial_capital=100000,
                                     fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65)

    strategies_2025 = {}
    strategies_2025['B2B_Baseline_060']      = engine_25.execute_b2b_baseline(pred_25, threshold=0.60)
    strategies_2025['B2B_B2A_Direction']     = engine_25.execute_b2b_b2a_combined(pred_25, threshold=0.60, use_b2a_direction=True, use_b2a_magnitude=False)
    strategies_2025['B2B_B2A_DirMag']        = engine_25.execute_b2b_b2a_combined(pred_25, threshold=0.60, use_b2a_direction=True, use_b2a_magnitude=True, min_magnitude=5.0)
    strategies_2025['ConfidenceScaled_060']  = engine_25.execute_confidence_scaled(pred_25, threshold=0.60, use_confidence_sizing=True)
    strategies_2025['pOuter_Filter_015']     = engine_25.execute_p_outer_strategy(pred_25, threshold=0.60, p_outer_threshold=0.15)
    strategies_2025['ExtremeWeather_Only']   = engine_25.execute_b2b_b2a_combined(pred_25, threshold=0.60, use_b2a_direction=True, extreme_weather_filter=True)
    strategies_2025['B2B_Optimal_070']       = engine_25.execute_b2b_baseline(pred_25, threshold=0.70)

    # B2B threshold sensitivity (2025)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_sensitivity = []
    for th in thresholds:
        r = engine_25.execute_b2b_baseline(pred_25, threshold=th)
        m = ERCOTBacktestEngine.calculate_metrics(r, 100000)
        threshold_sensitivity.append({
            'threshold': th,
            'total_pnl': round(float(m['Total_Pnl']), 2),
            'sharpe': round(float(m['Sharpe_Ratio']), 4),
            'trades': int(m['Total_Trades']),
            'win_rate': round(float(m['Win_Rate']), 4),
            'max_dd': round(float(m['Max_Drawdown']), 4),
        })

    # ==========================================
    # PART B: 2026 H1 — Frozen-Rule Test
    # ==========================================
    print('Loading 2026 C1 H1 Walk-Forward data...')
    market_26, pred_26 = load_c1_unified_2026()
    engine_26 = ERCOTBacktestEngine(market_26, initial_capital=100000,
                                     fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65)

    strategies_2026 = {}
    # Frozen from 2025: only the top 3 strategies, no re-optimisation
    strategies_2026['B2B_Baseline_060']    = engine_26.execute_b2b_baseline(pred_26, threshold=0.60)
    strategies_2026['B2B_Optimal_070']     = engine_26.execute_b2b_baseline(pred_26, threshold=0.70)
    strategies_2026['ExtremeWeather_Only'] = engine_26.execute_b2b_b2a_combined(pred_26, threshold=0.60, use_b2a_direction=True, extreme_weather_filter=True)

    # ==========================================
    # BUILD OUTPUT JSON
    # ==========================================
    output = {
        'meta': {
            'project': 'ERCOT Extreme-Weather-Driven ML Arbitrage Strategy & Backtest Platform',
            'model': 'C1_XGBoost_Prediction_Agent (B2A Regression + B2B 5/20 Classifier)',
            'model_version': 'v3',
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
            'periods': {
                '2025': {
                    'type': 'OOF development',
                    'source': 'C1_unified_prediction_table_2025_oof_v3.parquet',
                    'range': '2025-01-01 to 2025-12-31',
                    'total_hours': 8760,
                    'usage': 'Strategy development, threshold optimisation, model selection',
                },
                '2026': {
                    'type': 'Walk-Forward frozen-rule test',
                    'source': 'C1_unified_prediction_table_2026_H1_walkforward_v1.parquet',
                    'range': '2026-01-01 to 2026-07-01',
                    'total_hours': 4197,
                    'total_weeks': 27,
                    'usage': 'Frozen-rule temporal robustness check ONLY — no re-optimisation',
                },
            },
            'generated_at': str(pd.Timestamp.now()),
        },

        # Per-period strategy data
        '2025': {
            'strategy_comparison': [],
            'equity_curves': {},
            'trades': {},
            'monthly_pnl': {},
            'risk_reports': {},
            'market_data': {},
            'threshold_sensitivity': threshold_sensitivity,
        },
        '2026': {
            'strategy_comparison': [],
            'equity_curves': {},
            'trades': {},
            'monthly_pnl': {},
            'risk_reports': {},
            'market_data': {},
        },

        # Cross-period comparison
        'cross_period_comparison': [],
    }

    # --- Build 2025 data ---
    for name, res in strategies_2025.items():
        sd = build_strategy_data(name, res)
        output['2025']['strategy_comparison'].append(sd['metrics'])
        output['2025']['equity_curves'][name] = sd['equity_curve']
        output['2025']['trades'][name] = sd['trades']
        output['2025']['monthly_pnl'][name] = sd['monthly_pnl']
        output['2025']['risk_reports'][name] = sd['risk_report']

    # 2025 market background
    daily_25 = market_25.copy()
    daily_25['date'] = daily_25['delivery_hour_utc'].dt.date
    daily_avg_25 = daily_25.groupby('date')['spread_usd_per_mwh'].mean().reset_index()
    output['2025']['market_data'] = {
        'daily_avg_spread': [
            {'date': str(row['date']), 'avg_spread': round(float(row['spread_usd_per_mwh']), 2)}
            for _, row in daily_avg_25.iterrows()
        ],
        'total_hours': len(market_25),
        'extreme_weather_hours': int(pred_25['fixed_extreme_weather_flag'].sum()),
    }

    # --- Build 2026 data ---
    for name, res in strategies_2026.items():
        sd = build_strategy_data(name, res)
        output['2026']['strategy_comparison'].append(sd['metrics'])
        output['2026']['equity_curves'][name] = sd['equity_curve']
        output['2026']['trades'][name] = sd['trades']
        output['2026']['monthly_pnl'][name] = sd['monthly_pnl']
        output['2026']['risk_reports'][name] = sd['risk_report']

    # 2026 market background
    daily_26 = market_26.copy()
    daily_26['date'] = daily_26['delivery_hour_utc'].dt.date
    daily_avg_26 = daily_26.groupby('date')['spread_usd_per_mwh'].mean().reset_index()
    output['2026']['market_data'] = {
        'daily_avg_spread': [
            {'date': str(row['date']), 'avg_spread': round(float(row['spread_usd_per_mwh']), 2)}
            for _, row in daily_avg_26.iterrows()
        ],
        'total_hours': len(market_26),
        'extreme_weather_hours': int(pred_26['fixed_extreme_weather_flag'].sum()),
        'total_weeks': int(pred_26['week_id'].nunique()) if 'week_id' in pred_26.columns else 0,
    }

    # --- Cross-period comparison: same 3 strategies on both years ---
    print('Building cross-period comparison...')
    cross_strategies = [
        ('B2B_Baseline_060',      'baseline',   0.60),
        ('B2B_Optimal_070',       'baseline',   0.70),
        ('ExtremeWeather_Only',   'extreme',    0.60),
    ]
    for label, stype, th in cross_strategies:
        # 2025
        if stype == 'baseline':
            r25 = engine_25.execute_b2b_baseline(pred_25, threshold=th)
        else:
            r25 = engine_25.execute_b2b_b2a_combined(pred_25, threshold=th, use_b2a_direction=True, extreme_weather_filter=True)
        m25 = ERCOTBacktestEngine.calculate_metrics(r25, 100000)
        risk25, _, _, _ = analyze_risk_concentration(r25)

        # 2026
        if stype == 'baseline':
            r26 = engine_26.execute_b2b_baseline(pred_26, threshold=th)
        else:
            r26 = engine_26.execute_b2b_b2a_combined(pred_26, threshold=th, use_b2a_direction=True, extreme_weather_filter=True)
        m26 = ERCOTBacktestEngine.calculate_metrics(r26, 100000)
        risk26, _, _, _ = analyze_risk_concentration(r26)

        output['cross_period_comparison'].append({
            'strategy': label,
            '2025': {
                'total_pnl':     round(float(m25['Total_Pnl']), 2),
                'sharpe':        round(float(m25['Sharpe_Ratio']), 4),
                'max_dd':        round(float(m25['Max_Drawdown']), 4),
                'win_rate':      round(float(m25['Win_Rate']), 4),
                'total_trades':  int(m25['Total_Trades']),
                'avg_trade_pnl': round(float(m25['Avg_Trade_Pnl']), 2),
                'calmar':        round(float(m25['Calmar_Ratio']), 4),
                'profit_factor': round(float(m25['Profit_Factor']), 4) if m25['Profit_Factor'] != float('inf') else 999.0,
                'extreme_pnl':   round(float(risk25.get('Extreme_Weather_PnL', 0)), 2),
                'normal_pnl':    round(float(risk25.get('Normal_Weather_PnL', 0)), 2),
                'total_hours':   8760,
            },
            '2026': {
                'total_pnl':     round(float(m26['Total_Pnl']), 2),
                'sharpe':        round(float(m26['Sharpe_Ratio']), 4),
                'max_dd':        round(float(m26['Max_Drawdown']), 4),
                'win_rate':      round(float(m26['Win_Rate']), 4),
                'total_trades':  int(m26['Total_Trades']),
                'avg_trade_pnl': round(float(m26['Avg_Trade_Pnl']), 2),
                'calmar':        round(float(m26['Calmar_Ratio']), 4),
                'profit_factor': round(float(m26['Profit_Factor']), 4) if m26['Profit_Factor'] != float('inf') else 999.0,
                'extreme_pnl':   round(float(risk26.get('Extreme_Weather_PnL', 0)), 2),
                'normal_pnl':    round(float(risk26.get('Normal_Weather_PnL', 0)), 2),
                'total_hours':   4197,
            },
            'change': {
                'pnl_direction': 'UP' if float(m26['Total_Pnl']) > float(m25['Total_Pnl']) else 'DOWN',
                'sharpe_direction': 'UP' if float(m26['Sharpe_Ratio']) > float(m25['Sharpe_Ratio']) else 'DOWN',
                'max_dd_improved': bool(float(m26['Max_Drawdown']) > float(m25['Max_Drawdown'])),
            },
        })

    # ---- Embed strategy scores ----
    scores_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_scores.json')
    if os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        output['strategy_scoring'] = {
            'methodology': scores_data.get('scoring_methodology', {}),
            '2025_ranking': scores_data.get('2025', {}).get('ranking', scores_data.get('ranking', [])),
            '2026_ranking': scores_data.get('2026', {}).get('ranking', []),
            'scores_2025': scores_data.get('2025', {}).get('scores', scores_data.get('scores', {})),
            'scores_2026': scores_data.get('2026', {}).get('scores', {}),
        }
        # Inject composite scores
        score_map_25 = {n: s['composite'] for n, s in output['strategy_scoring']['scores_2025'].items()} if output['strategy_scoring']['scores_2025'] else {}
        score_map_26 = {n: s['composite'] for n, s in output['strategy_scoring']['scores_2026'].items()} if output['strategy_scoring']['scores_2026'] else {}
        for entry in output['2025']['strategy_comparison']:
            entry['composite_score'] = score_map_25.get(entry['name'], None)
        for entry in output['2026']['strategy_comparison']:
            entry['composite_score'] = score_map_26.get(entry['name'], None)
        print('Strategy scores embedded.')
    else:
        print('Warning: strategy_scores.json not found.')

    # ---- Write JSON ----
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'backtest_result_c1.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    print(f'\nJSON written: {out_path} ({size_kb:.1f} KB)')
    print(f'2025 strategies: {len(output["2025"]["strategy_comparison"])}  |  2026 strategies: {len(output["2026"]["strategy_comparison"])}')
    for s in output['2025']['strategy_comparison']:
        print(f'  2025 {s["name"]:<30s} PnL=${s["total_pnl"]:>10,.2f}  Sharpe={s["sharpe_ratio"]:.2f}')
    for s in output['2026']['strategy_comparison']:
        print(f'  2026 {s["name"]:<30s} PnL=${s["total_pnl"]:>10,.2f}  Sharpe={s["sharpe_ratio"]:.2f}')

    elapsed = time.time() - start_time
    print(f'\nTotal time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
