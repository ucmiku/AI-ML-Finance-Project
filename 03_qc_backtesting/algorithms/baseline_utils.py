"""
共享工具模块：Baseline 策略的数据加载、回测引擎、指标计算与可视化
用于 Baseline 1/2/3 和 ML 策略的统一对比
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from matplotlib.markers import MarkerStyle

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ==========================================
# 0. 数据路径
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBER_B_DATA_DIR = os.path.join(BASE_DIR, 'data', 'member_B')
C1_DATA_DIR = os.path.join(BASE_DIR, 'trading_handoff_C1_v3')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 1. 数据加载
# ==========================================
def load_validation_2025(data_dir=None):
    """加载 2025 年 LightGBM 独立验证集预测数据"""
    if data_dir is None:
        data_dir = MEMBER_B_DATA_DIR
    path = os.path.join(data_dir, 'lightgbm_predictions_2025_validation.csv')
    df = pd.read_csv(path)
    df['delivery_hour_utc'] = pd.to_datetime(df['datetime'])

    market_df = df[['delivery_hour_utc', 'actual_spread']].copy()
    market_df.rename(columns={'actual_spread': 'spread_usd_per_mwh'}, inplace=True)

    pred_df = df[[
        'delivery_hour_utc', 'predicted_spread', 'predicted_direction',
        'extreme_hour_flag'
    ]].copy()

    return market_df, pred_df


def load_test_2026(data_dir=None):
    """加载 2026 H1 周度滚动预测数据"""
    if data_dir is None:
        data_dir = MEMBER_B_DATA_DIR
    path = os.path.join(data_dir, 'lightgbm_predictions_2026_H1_walkforward.csv')
    df = pd.read_csv(path)
    df['delivery_hour_utc'] = pd.to_datetime(df['datetime'])

    available = df['target_available'] == 1
    df = df[available].copy()

    market_df = df[['delivery_hour_utc', 'actual_spread']].copy()
    market_df.rename(columns={'actual_spread': 'spread_usd_per_mwh'}, inplace=True)

    pred_df = df[[
        'delivery_hour_utc', 'predicted_spread', 'predicted_direction',
        'week_id', 'train_end'
    ]].copy()

    return market_df, pred_df


# ==========================================
# 2. 通用回测引擎
# ==========================================
class BaselineBacktestEngine:
    """通用回测引擎，接受任意信号向量进行回测"""

    def __init__(self, market_df, initial_capital=100000.0, fee_per_mwh=2.0,
                 slippage_bps=50.0, capture_rate=0.65):
        self.df = market_df.sort_values('delivery_hour_utc').reset_index(drop=True)
        self.df['delivery_hour_utc'] = pd.to_datetime(self.df['delivery_hour_utc'])
        self.initial_capital = initial_capital
        self.fee_per_mwh = fee_per_mwh
        self.slippage_bps = slippage_bps
        self.capture_rate = capture_rate

    def run_backtest(self, signals, pred_df=None):
        """
        执行回测

        Parameters
        ----------
        signals : np.ndarray or pd.Series
            交易信号，1=做多, -1=做空, 0=不交易
        pred_df : pd.DataFrame, optional
            预测数据，用于对齐时间

        Returns
        -------
        result_df : pd.DataFrame
        """
        data = self.df.copy()

        if pred_df is not None:
            data = pd.merge(
                data, pred_df[['delivery_hour_utc']],
                on='delivery_hour_utc', how='inner'
            ).sort_values('delivery_hour_utc').reset_index(drop=True)

        if isinstance(signals, pd.Series):
            signals = signals.values

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)
        data['Signal'] = signals[:len(data)].astype(int)

        has_position = data['Signal'] != 0
        data['Commission'] = has_position.astype(float) * self.fee_per_mwh
        data['Slippage'] = (
            has_position.astype(float)
            * np.abs(data['spread_usd_per_mwh'])
            * (self.slippage_bps / 10000.0)
        )
        data['Captured_Spread'] = (
            data['Signal'] * data['spread_usd_per_mwh'] * self.capture_rate
        )
        data['Hourly_Pnl'] = (
            data['Captured_Spread'] - data['Commission'] - data['Slippage']
        )
        data['Cumulative_Pnl'] = data['Hourly_Pnl'].cumsum()
        data['Equity'] = self.initial_capital + data['Cumulative_Pnl']

        return data

    @staticmethod
    def calculate_metrics(result_df, initial_capital):
        """计算量化核心指标"""
        total_pnl = result_df['Hourly_Pnl'].sum()
        total_return = total_pnl / initial_capital

        dates = result_df['delivery_hour_utc'].dt.date
        daily_pnl = result_df.groupby(dates)['Hourly_Pnl'].sum()
        daily_returns = daily_pnl / initial_capital

        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()

        sharpe = (mean_daily / std_daily) * np.sqrt(365) if std_daily > 1e-10 else 0.0

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0.0
        sortino = (
            (mean_daily / downside_std) * np.sqrt(365)
            if downside_std > 1e-10 else 0.0
        )

        equity = result_df['Equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak.replace(0, np.nan)
        max_dd = drawdown.min()

        calmar = total_return / abs(max_dd) if max_dd != 0 else float('inf')

        trades = result_df[result_df['Signal'] != 0]
        if len(trades) > 0:
            win_rate = (trades['Hourly_Pnl'] > 0).mean()
            avg_trade_pnl = trades['Hourly_Pnl'].mean()
            long_trades = (trades['Signal'] == 1).sum()
            short_trades = (trades['Signal'] == -1).sum()
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            long_trades = 0
            short_trades = 0

        winning = trades[trades['Hourly_Pnl'] > 0]['Hourly_Pnl']
        losing = trades[trades['Hourly_Pnl'] < 0]['Hourly_Pnl']
        if len(losing) > 0 and len(winning) > 0:
            profit_factor = winning.sum() / abs(losing.sum())
        elif len(winning) > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0

        return {
            "Total_Return": total_return,
            "Total_Pnl": total_pnl,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Max_Drawdown": max_dd,
            "Calmar_Ratio": calmar,
            "Win_Rate": win_rate,
            "Profit_Factor": profit_factor,
            "Total_Trades": len(trades),
            "Long_Trades": long_trades,
            "Short_Trades": short_trades,
            "Avg_Trade_Pnl": avg_trade_pnl,
        }


# ==========================================
# 3. 可视化
# ==========================================
def plot_backtest_result(result_df, title="Strategy Equity Curve", save_path=None):
    """标准回测可视化"""
    result_df = result_df.copy()
    time_col = 'delivery_hour_utc'

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 0.5, 1.0, 1.0],
                          hspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax_pos = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3])

    ax1.plot(result_df[time_col], result_df['Equity'],
             label='Strategy Equity', color='#1f77b4', linewidth=1.5, zorder=2)

    initial_eq = result_df['Equity'].iloc[0]
    ax1.axhline(y=initial_eq, color='gray', linestyle=':', alpha=0.5)

    trade_rows = result_df[result_df['Signal'] != 0]
    if not trade_rows.empty:
        long_trades = trade_rows[trade_rows['Signal'] == 1]
        short_trades = trade_rows[trade_rows['Signal'] == -1]
        if not long_trades.empty:
            ax1.scatter(long_trades[time_col], long_trades['Equity'],
                        color='#2ca02c', s=12, marker=MarkerStyle('^'), zorder=5,
                        alpha=0.7, label=f'LONG (n={len(long_trades)})')
        if not short_trades.empty:
            ax1.scatter(short_trades[time_col], short_trades['Equity'],
                        color='#d62728', s=12, marker=MarkerStyle('v'), zorder=5,
                        alpha=0.7, label=f'SHORT (n={len(short_trades)})')

    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_ylabel("Equity ($)", fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.25)

    long_mask = result_df['Signal'] == 1
    short_mask = result_df['Signal'] == -1
    ax_pos.fill_between(result_df[time_col], 0, 1, where=long_mask,
                        color='#2ca02c', alpha=0.7, step='post')
    ax_pos.fill_between(result_df[time_col], 0, 1, where=short_mask,
                        color='#d62728', alpha=0.7, step='post')
    ax_pos.set_yticks([])
    ax_pos.set_ylabel('Pos', fontsize=8, rotation=0, labelpad=15)
    plt.setp(ax_pos.get_xticklabels(), visible=False)

    drawdown = (result_df['Equity'] - result_df['Equity'].cummax()) / \
        result_df['Equity'].cummax().replace(0, np.nan)
    ax2.fill_between(result_df[time_col], 0, drawdown * 100, color='#d62728', alpha=0.3)
    ax2.plot(result_df[time_col], drawdown * 100, color='#d62728', linewidth=0.8)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.grid(True, alpha=0.25)

    trade_pnls = result_df[result_df['Signal'] != 0]['Hourly_Pnl'].values
    if len(trade_pnls) > 0:
        if len(trade_pnls) > 500:
            ax3.plot(trade_pnls, color='#1f77b4', linewidth=0.5, alpha=0.7)
        else:
            colors = ['#2ca02c' if p >= 0 else '#d62728' for p in trade_pnls]
            ax3.bar(range(len(trade_pnls)), trade_pnls, color=colors, width=0.8, alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        win_count = (trade_pnls > 0).sum()
        ax3.set_title(
            f'Trade PnL Distribution  |  Win Rate: {win_count}/{len(trade_pnls)} '
            f'({win_count/len(trade_pnls)*100:.1f}%)', fontsize=9
        )
    ax3.set_xlabel("Trade Sequence", fontsize=10)
    ax3.set_ylabel("PnL ($)", fontsize=10)
    ax3.grid(True, alpha=0.25)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_metrics(metrics, prefix=""):
    """格式化打印指标"""
    for k, v in metrics.items():
        label = f"{prefix}{k}"
        if k in ("Total_Return", "Max_Drawdown", "Win_Rate"):
            print(f"  {label}: {v*100:.2f}%")
        elif k == "Total_Pnl":
            print(f"  {label}: ${v:,.2f}")
        elif k == "Avg_Trade_Pnl":
            print(f"  {label}: ${v:,.2f}")
        elif k == "Profit_Factor":
            print(f"  {label}: {v:.2f}" if v != float('inf') else f"  {label}: inf")
        elif k in ("Long_Trades", "Short_Trades", "Total_Trades"):
            print(f"  {label}: {v}")
        else:
            print(f"  {label}: {v:.4f}")


def save_metrics_json(metrics_dict, filepath):
    """保存指标为 JSON"""
    serializable = {}
    for name, m in metrics_dict.items():
        serializable[name] = {}
        for k, v in m.items():
            if isinstance(v, (np.integer,)):
                serializable[name][k] = int(v)
            elif isinstance(v, (np.floating,)):
                serializable[name][k] = float(v)
            elif v == float('inf'):
                serializable[name][k] = "inf"
            else:
                serializable[name][k] = v
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)