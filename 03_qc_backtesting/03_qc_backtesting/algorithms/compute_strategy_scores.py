"""
Multi-dimensional strategy scoring framework.
Computes composite scores across Return, Risk, Robustness, and Efficiency dimensions.
Output: strategy_scores.json
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sumstrategy import (
    load_c1_unified_2025, ERCOTBacktestEngine,
    analyze_risk_concentration,
)


# ============================================================
# Scoring dimension definitions
# ============================================================
# Each dimension: (metric_key, direction, weight, display_label)
# direction: 'higher' = higher values are better; 'lower' = lower values are better
# Weights are relative and will be normalized per category

RETURN_DIMS = [
    ('Sharpe_Ratio',              'higher', 0.40, '夏普比率 (Sharpe Ratio)'),
    ('Total_PnL',                 'higher', 0.25, '总盈亏 (Total PnL)'),
    ('Sortino_Ratio',             'higher', 0.20, '索提诺比率 (Sortino Ratio)'),
    ('Avg_Trade_Pnl',             'higher', 0.15, '笔均盈亏 (Avg Trade PnL)'),
]

RISK_DIMS = [
    ('Max_DD',                    'lower',  0.30, '最大回撤 (Max DD)'),
    ('Win_Rate',                  'higher', 0.25, '胜率 (Win Rate)'),
    ('Profitable_Months_Ratio',   'higher', 0.25, '盈利月占比'),
    ('Profit_Factor',             'higher', 0.20, '盈亏比 (Profit Factor)'),
]

ROBUST_DIMS = [
    ('Calmar_Ratio',              'higher', 0.30, '卡尔玛比率 (Calmar Ratio)'),
    ('Normal_Weather_PnL',        'higher', 0.35, '正常天气稳健性'),
    ('PnL_ex_Top5',               'higher', 0.35, '去Top5后剩余PnL'),
]

EFFIC_DIMS = [
    ('Total_Trades',              'lower',  0.35, '交易频率（反向）'),
    ('Avg_Trade_Pnl',             'higher', 0.35, '每笔交易效率'),
    ('Long_Trades',               'higher', 0.30, '多空平衡度'),
]

CATEGORIES = [
    ('收益能力 (Return)',     RETURN_DIMS, 0.40),
    ('风险控制 (Risk)',       RISK_DIMS,   0.30),
    ('稳健性 (Robustness)',   ROBUST_DIMS, 0.20),
    ('交易效率 (Efficiency)', EFFIC_DIMS,  0.10),
]


def minmax_normalize(values, direction):
    """Normalize to 0-100 scale. For 'lower' direction, invert."""
    arr = np.array(values, dtype=float)
    if direction == 'higher':
        lo, hi = arr.min(), arr.max()
    else:
        lo, hi = arr.max(), arr.min()
    if hi == lo:
        return [50.0] * len(values)
    if direction == 'higher':
        return [max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0)) for v in values]
    else:
        return [max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0)) for v in values]


def compute_scores(strategies_results):
    """Compute multi-dimensional scores for all strategies."""
    names = list(strategies_results.keys())

    # --- Collect raw metrics ---
    raw = {}
    for name, res in strategies_results.items():
        m = ERCOTBacktestEngine.calculate_metrics(res, 100000)
        risk, monthly_df, _, _ = analyze_risk_concentration(res)

        n_profitable = int(risk['Profitable_Months'].split('/')[0])
        n_total_months = int(risk['Profitable_Months'].split('/')[1])
        normal_pnl = risk.get('Normal_Weather_PnL', 0)
        n_long = int((res['Signal'] == 1).sum())
        n_short = int((res['Signal'] == -1).sum())

        raw[name] = {
            'Sharpe_Ratio':              float(m['Sharpe_Ratio']),
            'Sortino_Ratio':             float(m['Sortino_Ratio']),
            'Total_PnL':                 float(m['Total_Pnl']),
            'Avg_Trade_Pnl':             float(m['Avg_Trade_Pnl']),
            'Max_DD':                    float(m['Max_Drawdown']),
            'Win_Rate':                  float(m['Win_Rate']),
            'Profit_Factor':             float(m['Profit_Factor']) if m['Profit_Factor'] != float('inf') else 10.0,
            'Profitable_Months_Ratio':   n_profitable / max(n_total_months, 1),
            'Calmar_Ratio':              float(m['Calmar_Ratio']) if m['Calmar_Ratio'] != float('inf') else 200.0,
            'Normal_Weather_PnL':        float(normal_pnl),
            'PnL_ex_Top5':               float(risk['PnL_ex_Top5']),
            'Total_Trades':              int(m['Total_Trades']),
            'Long_Trades':               n_long,
        }

    # --- Normalize & score each dimension ---
    all_dimensions = []
    for cat_name, dims, cat_weight in CATEGORIES:
        for metric_key, direction, dim_weight, label in dims:
            all_dimensions.append((cat_name, cat_weight, metric_key, direction, dim_weight, label))

    dimension_scores = {name: {} for name in names}
    for cat_name, cat_weight, metric_key, direction, dim_weight, label in all_dimensions:
        values = [raw[name][metric_key] for name in names]
        normalized = minmax_normalize(values, direction)
        for i, name in enumerate(names):
            dimension_scores[name][metric_key] = {
                'raw_value':     round(raw[name][metric_key], 4),
                'score_0_100':   round(normalized[i], 1),
                'weight_in_dim': round(dim_weight, 2),
                'weight_in_total': round(cat_weight * dim_weight, 4),
                'weighted_contrib': round(normalized[i] * cat_weight * dim_weight, 2),
                'label':         label,
                'category':      cat_name,
                'direction':     direction,
            }

    # --- Aggregate: dimension → category → composite ---
    final_scores = {}
    for name in names:
        ds = dimension_scores[name]
        cat_scores = {}
        for cat_name, dims, cat_weight in CATEGORIES:
            dim_keys = [d[0] for d in dims]
            cat_raw = sum(ds[k]['score_0_100'] * ds[k]['weight_in_dim'] for k in dim_keys)
            cat_scores[cat_name] = {
                'score_0_100': round(cat_raw, 1),
                'weight': cat_weight,
                'weighted': round(cat_raw * cat_weight, 2),
            }

        composite = sum(cat_scores[c]['weighted'] for c in cat_scores)
        final_scores[name] = {
            'composite': round(composite, 1),
            'categories': cat_scores,
            'dimensions': dimension_scores[name],
        }

    # --- Ranking ---
    ranking = sorted(names, key=lambda n: final_scores[n]['composite'], reverse=True)

    return final_scores, ranking, raw


def main():
    market_c1, pred_c1 = load_c1_unified_2025()
    engine = ERCOTBacktestEngine(
        market_c1, initial_capital=100000,
        fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65,
    )

    # Run all 7 strategies
    results = {}
    results['B2B_Baseline_060']      = engine.execute_b2b_baseline(pred_c1, threshold=0.60)
    results['B2B_B2A_Direction']     = engine.execute_b2b_b2a_combined(pred_c1, threshold=0.60, use_b2a_direction=True, use_b2a_magnitude=False)
    results['B2B_B2A_DirMag']        = engine.execute_b2b_b2a_combined(pred_c1, threshold=0.60, use_b2a_direction=True, use_b2a_magnitude=True, min_magnitude=5.0)
    results['ConfidenceScaled_060']  = engine.execute_confidence_scaled(pred_c1, threshold=0.60, use_confidence_sizing=True)
    results['pOuter_Filter_015']     = engine.execute_p_outer_strategy(pred_c1, threshold=0.60, p_outer_threshold=0.15)
    results['ExtremeWeather_Only']   = engine.execute_b2b_b2a_combined(pred_c1, threshold=0.60, use_b2a_direction=True, extreme_weather_filter=True)
    results['B2B_Optimal_070']       = engine.execute_b2b_baseline(pred_c1, threshold=0.70)

    final_scores, ranking, raw = compute_scores(results)

    # --- Print report ---
    def bar(val, width=20):
        n = int(round(val / 100 * width))
        return '#' * n + '-' * (width - n)

    print('\n' + '=' * 110)
    print('  Multi-Dimensional Strategy Scoring Report')
    print('=' * 110)
    print()
    print(f'  {"Rank":<4s} {"Strategy":<28s} {"Score":>6s}  {"Return(40%)":>10s}  {"Risk(30%)":>10s}  {"Robust(20%)":>10s}  {"Effic(10%)":>10s}')
    print(f'  {"":-<4s} {"":-<28s} {"":>6s}  {"":>10s}  {"":>10s}  {"":>10s}  {"":>10s}')

    for rank, name in enumerate(ranking, 1):
        fs = final_scores[name]
        medal = f'#{rank}'
        cats = fs['categories']
        print(f'  {medal:<4s} {name:<28s} {fs["composite"]:>5.1f}  '
              f'{cats["收益能力 (Return)"]   ["score_0_100"]:>7.1f}  '
              f'{cats["风险控制 (Risk)"]     ["score_0_100"]:>7.1f}  '
              f'{cats["稳健性 (Robustness)"] ["score_0_100"]:>7.1f}  '
              f'{cats["交易效率 (Efficiency)"]["score_0_100"]:>7.1f}')

    # Detailed dimension breakdown per strategy
    print('\n' + '=' * 110)
    print('  各策略分维度详细得分')
    print('=' * 110)

    all_dim_keys = []
    for cat_name, dims, _ in CATEGORIES:
        all_dim_keys.extend([d[0] for d in dims])

    for rank, name in enumerate(ranking, 1):
        fs = final_scores[name]
        print(f'\n  #{rank} {name}  --  Composite Score: {fs["composite"]:.1f}/100  {bar(fs["composite"])}')
        print(f'  {"维度":<32s} {"原始值":>12s}  {"得分":>6s}  {"分布"}')
        print(f'  {"":-<32s} {"":>12s}  {"":>6s}  {"":-<22s}')
        for key in all_dim_keys:
            d = fs['dimensions'][key]
            print(f'  [{d["category"]}] {d["label"]:<20s}  {d["raw_value"]:>12.4f}  {d["score_0_100"]:>5.1f}  {bar(d["score_0_100"], 20)}')

    # Export JSON
    export = {
        'scoring_methodology': {
            'description': 'Multi-dimensional min-max normalization scoring (0-100 scale)',
            'categories': [
                {'name': cn, 'weight': cw,
                 'dimensions': [{'key': d[0], 'direction': d[1], 'label': d[3]} for d in dims]}
                for cn, dims, cw in CATEGORIES
            ],
        },
        'ranking': ranking,
        'scores': {},
        'raw_metrics': {},
    }
    for name in results:
        export['scores'][name] = {
            'composite': final_scores[name]['composite'],
            'categories': final_scores[name]['categories'],
            'dimensions': {
                k: {'raw': v['raw_value'], 'score': v['score_0_100'],
                    'label': v['label'], 'category': v['category'],
                    'direction': v['direction']}
                for k, v in final_scores[name]['dimensions'].items()
            },
        }
        export['raw_metrics'][name] = raw[name]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_scores.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f'\n\n[Scores exported to: {out_path}]')


if __name__ == '__main__':
    main()
