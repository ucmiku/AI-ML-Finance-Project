import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.markers import MarkerStyle
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ==========================================
# 0. 数据加载模块 — 对接成员B的真实预测CSV
# ==========================================
MEMBER_B_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'member_B'
)

C1_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'trading_handoff_C1_v3'
)


def load_validation_2025(data_dir=None):
    """
    加载 2025 年 LightGBM 独立验证集预测数据，用于策略开发与阈值搜索

    Returns
    -------
    market_df : pd.DataFrame
        标准化后的市场数据，列: delivery_hour_utc, actual_spread
    pred_df : pd.DataFrame
        标准化后的预测数据，列: delivery_hour_utc, predicted_spread,
        predicted_direction, extreme_hour_flag
    """
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
    """
    加载 2026 H1 周度滚动预测数据，用于最终独立测试（冻结策略后）

    Returns
    -------
    market_df : pd.DataFrame
    pred_df : pd.DataFrame
    """
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


def load_c1_unified_2025(data_dir=None):
    """
    加载 C1 统一预测表（2025 OOF），对接交易团队交付的 B2A/B2B 双头预测数据。

    C1 统一预测表包含两个模型组件的原始输出：
      - B2A XGBoost Regression Head  → predicted_spread
      - B2B XGBoost 5/20 Classifier Head → p_c1~p_c5, p_negative/neutral/positive/outer

    Returns
    -------
    market_df : pd.DataFrame
        标准化市场数据，列: delivery_hour_utc, spread_usd_per_mwh
    pred_df : pd.DataFrame
        完整预测数据，包含所有 C1 字段:
        predicted_spread, p_c1~p_c5, p_negative, p_neutral, p_positive, p_outer,
        predicted_class, confidence, signal_base, recommended_action_base,
        actual_class, fixed_extreme_weather_flag, target_extreme20, net_pnl
    """
    if data_dir is None:
        data_dir = C1_DATA_DIR
    path = os.path.join(data_dir, 'C1_unified_prediction_table_2025_oof_v3.parquet')
    if not os.path.exists(path):
        # 尝试 CSV 回退
        csv_path = os.path.join(data_dir, 'C1_unified_prediction_table_2025_oof_v3.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"C1 预测表未找到: {path} 或 {csv_path}")
    else:
        df = pd.read_parquet(path)

    df['delivery_hour_utc'] = pd.to_datetime(df['delivery_hour_utc'])

    # 标准化市场数据
    market_df = df[['delivery_hour_utc', 'actual_spread']].copy()
    market_df.rename(columns={'actual_spread': 'spread_usd_per_mwh'}, inplace=True)

    # 完整预测字段
    pred_cols = [
        'delivery_hour_utc', 'predicted_spread',
        'p_c1', 'p_c2', 'p_c3', 'p_c4', 'p_c5',
        'p_negative', 'p_neutral', 'p_positive', 'p_outer',
        'predicted_class', 'confidence',
        'signal_base', 'recommended_action_base',
        'actual_class', 'fixed_extreme_weather_flag',
        'target_extreme20', 'net_pnl',
    ]
    pred_df = df[pred_cols].copy()

    return market_df, pred_df


# ==========================================
# 1. 核心回测引擎类 (BacktestEngine)
# ==========================================
class ERCOTBacktestEngine:
    def __init__(self, market_df, initial_capital=100000.0, fee_per_mwh=2.0,
                 slippage_bps=10.0, capture_rate=0.75):
        """
        严格基于测试集进行回测

        Parameters
        ----------
        market_df : pd.DataFrame
            必须包含 delivery_hour_utc, spread_usd_per_mwh (真实 RT-DA 价差)
        fee_per_mwh : float
            每 MWh 交易佣金（含交易所费用），默认 $2.0/MWh
        slippage_bps : float
            滑点基点（1 bp = 0.01%），按实际价差百分比模拟市场冲击成本
        capture_rate : float
            价差捕获率（0~1），现实中无法 100% 捕获 RT-DA 价差
        """
        self.df = market_df.sort_values('delivery_hour_utc').reset_index(drop=True)
        self.df['delivery_hour_utc'] = pd.to_datetime(self.df['delivery_hour_utc'])
        self.initial_capital = initial_capital
        self.fee_per_mwh = fee_per_mwh
        self.slippage_bps = slippage_bps
        self.capture_rate = capture_rate
        self._merged_cache = {}  # 缓存 pred_df -> merged DataFrame，避免重复 merge

    def _get_merged_data(self, pred_df):
        """获取合并后的数据，自动缓存以避免重复 pd.merge 操作"""
        key = id(pred_df)
        if key not in self._merged_cache:
            data = pd.merge(
                self.df, pred_df, on='delivery_hour_utc', how='inner'
            ).sort_values('delivery_hour_utc').reset_index(drop=True)
            self._merged_cache[key] = data
        return self._merged_cache[key].copy()

    # ==========================================
# 优化后的 execute_strategy 核心代码片段
# （直接替换 ERCOTBacktestEngine 类中的对应方法）
# ==========================================

    def execute_strategy(self, pred_df, 
                     spread_threshold: float = 50.0,
                     rolling_window: int = 168,      # 过去7天(168h)滚动窗口
                     std_multiplier: float = 1.5,    # 波动率倍数阈值
                     short_risk_multiplier: float = 1.3, # 空头风险惩罚系数（空头开仓门槛提高30%）
                     min_base_threshold: float = 10.0, # 最小保底阈值($/MWh)
                     extreme_spread_threshold: float = 200.0,
                     direction_filter: bool = True, 
                     max_consecutive_hours: int = 48,
                     vol_regime_threshold: float = 5.0,  # 市场波动率最低阈值($/MWh)，低于此值不开仓
                     min_profit_ratio: float = 1.5):     # 预期利润/成本最低倍数，低于此值不开仓
        """
        自适应波动率双向套利策略：利用过去 N 小时的预测价差标准差动态计算开仓阈值

        Parameters (新增)
        -----------------
        vol_regime_threshold : float
            市场实际价差的滚动标准差最低阈值。当市场过于平稳时完全不开仓，
            避免在低波动环境中被手续费和滑点侵蚀利润。
        min_profit_ratio : float
            预期净利润与交易成本的最低比值。例如 1.5 表示预期利润至少是
            手续费+滑点的 1.5 倍才开仓，过滤掉"赚1块花2块"的劣质交易。
        """
        data = self._get_merged_data(pred_df)

        # 1. 裁剪异常极值
        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # 2. 计算动态波动率阈值 (Rolling Volatility-based Threshold)
        # 计算过去 N 小时 predicted_spread 的滚动标准差
        rolling_std = data['predicted_spread'].rolling(window=rolling_window, min_periods=24).std()
        
        # 动态阈值 = max( 保底阈值, 滚动标准差 * 倍数 )
        data['adaptive_threshold'] = np.maximum(
            min_base_threshold, 
            rolling_std.fillna(min_base_threshold) * std_multiplier 
        )

        # 3. 极端天气覆盖
        if 'extreme_hour_flag' not in data.columns:
            data['extreme_hour_flag'] = False

        data['dynamic_threshold'] = np.where(
            data['extreme_hour_flag'] == 1,
            extreme_spread_threshold,
            data['adaptive_threshold'] # 使用自适应阈值取代原有的静态阈值
        )

        # 4. 市场波动率 Regime 过滤器
        # 用实际价差（而非预测价差）的滚动标准差判断市场是否过于平稳
        data['market_volatility'] = (
            data['spread_usd_per_mwh']
            .rolling(window=rolling_window, min_periods=24)
            .std()
            .fillna(vol_regime_threshold + 1)  # 数据不足时默认允许交易
        )
        data['low_vol_regime'] = data['market_volatility'] < vol_regime_threshold

        # 5. 开仓信号生成（空头使用非对称更高门槛）
        long_signal = data['predicted_spread'] > data['dynamic_threshold']
        short_signal = data['predicted_spread'] < -(data['dynamic_threshold'] * short_risk_multiplier)

        if direction_filter and 'predicted_direction' in data.columns:
            long_signal = long_signal & (data['predicted_direction'] == 1)
            short_signal = short_signal & (data['predicted_direction'] == -1)

        # 6. 预期净利润过滤器：砍掉"赚得不够手续费"的劣质交易
        expected_gross_profit = np.abs(data['predicted_spread']) * self.capture_rate
        estimated_cost = (
            self.fee_per_mwh
            + np.abs(data['predicted_spread']) * (self.slippage_bps / 10000.0)
        )
        trade_worthwhile = expected_gross_profit > estimated_cost * min_profit_ratio

        data['Raw_Signal'] = 0
        data.loc[long_signal & trade_worthwhile & ~data['low_vol_regime'], 'Raw_Signal'] = 1
        data.loc[short_signal & trade_worthwhile & ~data['low_vol_regime'], 'Raw_Signal'] = -1

        # 7. 风控与持仓拦截（向量化：替代逐行循环，大幅提速）
        data['Signal'] = data['Raw_Signal'].copy()
        is_nonzero = data['Raw_Signal'] != 0
        block_id = (is_nonzero != is_nonzero.shift()).cumsum()
        data['_pos_in_block'] = data.groupby(block_id).cumcount() + 1
        data.loc[is_nonzero & (data['_pos_in_block'] > max_consecutive_hours), 'Signal'] = 0
        data.drop(columns=['_pos_in_block'], inplace=True)

        # 8. PnL 结算逻辑 (保持原样)
        data['Trade_Action'] = data['Signal'].diff().fillna(0).abs()
        data['Commission'] = data['Trade_Action'] * self.fee_per_mwh
        data['Slippage'] = (
            data['Trade_Action'] * np.abs(data['spread_usd_per_mwh']) * (self.slippage_bps / 10000.0)
        )
        data['Captured_Spread'] = data['Signal'] * data['spread_usd_per_mwh'] * self.capture_rate
        data['Hourly_Pnl'] = data['Captured_Spread'] - data['Commission'] - data['Slippage']
        data['Cumulative_Pnl'] = data['Hourly_Pnl'].cumsum()
        data['Equity'] = self.initial_capital + data['Cumulative_Pnl']

        return data

    @staticmethod
    def calculate_metrics(result_df, initial_capital):
        """
        计算常规量化核心指标（含风险调整后收益），支持双向交易
        """
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
    # C1 B2B/B2A 策略方法（对接交易团队交付的统一预测表）
    # ==========================================

    def _settle_pnl_per_execution_hour(self, data):
        """
        PnL 结算 — 按执行小时收费（匹配 C1 handoff 假设）:
        - 每个执行小时 1 MWh
        - commission = 2 USD/MWh（每小时持仓收取）
        - slippage = abs(spread) * 0.005（每小时持仓收取）
        - capture_rate = 65%
        - 不按 signal 变化次数收费；不使用连续持仓限制
        """
        has_position = data['Signal'] != 0
        data['Trade_Count_Per_Hour'] = has_position.astype(int)
        data['Commission'] = data['Trade_Count_Per_Hour'] * self.fee_per_mwh
        data['Slippage'] = (
            data['Trade_Count_Per_Hour']
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

    def execute_b2b_baseline(self, pred_df, threshold=0.60,
                              per_execution_hour_costs=True):
        """
        执行 C1 基准 B2B 分类策略（完全复现 handoff 规则）。

        规则:
          if p_positive >= threshold AND p_positive > p_negative:
              signal = +1  # DEC
          elif p_negative >= threshold AND p_negative > p_positive:
              signal = -1  # INC
          else:
              signal = 0   # No Trade

        Parameters
        ----------
        threshold : float
            B2B 概率阈值，默认 0.60（handoff 基准值）
        per_execution_hour_costs : bool
            True = 按执行小时收费（handoff 假设）；
            False = 按信号变化收费（与原引擎一致）
        """
        data = self._get_merged_data(pred_df)

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # B2B 基准信号生成
        pos_cond = (
            (data['p_positive'] >= threshold)
            & (data['p_positive'] > data['p_negative'])
        )
        neg_cond = (
            (data['p_negative'] >= threshold)
            & (data['p_negative'] > data['p_positive'])
        )

        data['Signal'] = 0
        data.loc[pos_cond, 'Signal'] = 1
        data.loc[neg_cond, 'Signal'] = -1

        # 记录策略元信息
        data['Strategy'] = 'B2B_Baseline'
        data['Active_Threshold'] = threshold

        if per_execution_hour_costs:
            return self._settle_pnl_per_execution_hour(data)
        else:
            # 回退到原引擎的按信号变化收费方式
            data['Trade_Action'] = data['Signal'].diff().fillna(0).abs()
            data['Commission'] = data['Trade_Action'] * self.fee_per_mwh
            data['Slippage'] = (
                data['Trade_Action'] * np.abs(data['spread_usd_per_mwh'])
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

    def execute_b2b_b2a_combined(self, pred_df, threshold=0.60,
                                  use_b2a_direction=True,
                                  use_b2a_magnitude=False,
                                  min_magnitude=10.0,
                                  extreme_weather_filter=False,
                                  per_execution_hour_costs=True):
        """
        B2B 信号 + B2A 回归确认的组合策略。

        支持多种组合方式（对应 handoff 第5节建议）:
          - B2B-only（关闭所有 B2A 选项）
          - B2B 信号 + B2A 方向确认（use_b2a_direction=True）
          - B2B 信号 + B2A 幅度过滤（use_b2a_magnitude=True）
          - 极端天气风险开关（extreme_weather_filter=True）

        Parameters
        ----------
        use_b2a_direction : bool
            True 时 B2A predicted_spread 符号必须与 B2B 信号方向一致
        use_b2a_magnitude : bool
            True 时 |B2A predicted_spread| 必须超过 min_magnitude
        min_magnitude : float
            B2A 最低预测价差幅度 ($/MWh)
        extreme_weather_filter : bool
            True 时仅在 fixed_extreme_weather_flag=1 的小时交易（风险开关）
        """
        data = self._get_merged_data(pred_df)

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # Step 1: B2B 基准信号
        pos_cond = (
            (data['p_positive'] >= threshold)
            & (data['p_positive'] > data['p_negative'])
        )
        neg_cond = (
            (data['p_negative'] >= threshold)
            & (data['p_negative'] > data['p_positive'])
        )

        b2b_signal = pd.Series(0, index=data.index, dtype=int)
        b2b_signal[pos_cond] = 1
        b2b_signal[neg_cond] = -1

        # 记录过滤统计
        data['Filter_B2B_Raw'] = b2b_signal
        filter_stats = {'B2B_Raw': (b2b_signal != 0).sum()}

        # Step 2: B2A 方向确认
        b2a_direction = np.sign(data['predicted_spread'])
        if use_b2a_direction:
            direction_mismatch = (
                (b2b_signal != 0) & (b2b_signal != b2a_direction)
            )
            b2b_signal[direction_mismatch] = 0
            filter_stats['After_B2A_Direction'] = (b2b_signal != 0).sum()

        # Step 3: B2A 幅度过滤
        if use_b2a_magnitude:
            insufficient_mag = (
                (b2b_signal != 0)
                & (np.abs(data['predicted_spread']) < min_magnitude)
            )
            b2b_signal[insufficient_mag] = 0
            filter_stats['After_B2A_Magnitude'] = (b2b_signal != 0).sum()

        # Step 4: 极端天气风险开关
        if extreme_weather_filter:
            if 'fixed_extreme_weather_flag' in data.columns:
                non_extreme = (
                    (b2b_signal != 0)
                    & (data['fixed_extreme_weather_flag'] != 1)
                )
                b2b_signal[non_extreme] = 0
                filter_stats['After_ExtremeWeather_Only'] = (b2b_signal != 0).sum()

        data['Signal'] = b2b_signal.astype(int)
        data['Strategy'] = 'B2B_B2A_Combined'
        data['Active_Threshold'] = threshold
        data['_filter_stats'] = str(filter_stats)

        if per_execution_hour_costs:
            return self._settle_pnl_per_execution_hour(data)
        else:
            data['Trade_Action'] = data['Signal'].diff().fillna(0).abs()
            data['Commission'] = data['Trade_Action'] * self.fee_per_mwh
            data['Slippage'] = (
                data['Trade_Action'] * np.abs(data['spread_usd_per_mwh'])
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

    def execute_confidence_scaled(self, pred_df, threshold=0.60,
                                   use_confidence_sizing=True,
                                   size_cap=2.0, size_floor=0.5,
                                   per_execution_hour_costs=True):
        """
        B2B 策略 + confidence 仓位缩放。

        用 confidence（最大类别概率）缩放仓位大小，高置信度加仓，低置信度减仓。
        对应 handoff 第5节: "用 confidence 做仓位缩放"。

        Parameters
        ----------
        use_confidence_sizing : bool
            True 时按 confidence/threshold 缩放仓位
        size_cap : float
            最大仓位倍数
        size_floor : float
            最小仓位倍数
        """
        data = self._get_merged_data(pred_df)

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # B2B 基准信号
        pos_cond = (
            (data['p_positive'] >= threshold)
            & (data['p_positive'] > data['p_negative'])
        )
        neg_cond = (
            (data['p_negative'] >= threshold)
            & (data['p_negative'] > data['p_positive'])
        )

        data['Signal'] = 0
        data.loc[pos_cond, 'Signal'] = 1
        data.loc[neg_cond, 'Signal'] = -1

        # Confidence 仓位缩放
        if use_confidence_sizing and 'confidence' in data.columns:
            base_conf = threshold
            data['Position_Size'] = np.where(
                data['Signal'] != 0,
                np.clip(data['confidence'] / base_conf, size_floor, size_cap),
                0.0
            )
        else:
            data['Position_Size'] = (data['Signal'] != 0).astype(float)

        data['Strategy'] = 'B2B_ConfidenceScaled'
        data['Active_Threshold'] = threshold

        # PnL with position sizing
        has_position = data['Signal'] != 0
        data['Trade_Count_Per_Hour'] = has_position.astype(float) * data['Position_Size']
        data['Commission'] = data['Trade_Count_Per_Hour'] * self.fee_per_mwh
        data['Slippage'] = (
            data['Trade_Count_Per_Hour']
            * np.abs(data['spread_usd_per_mwh'])
            * (self.slippage_bps / 10000.0)
        )
        data['Captured_Spread'] = (
            data['Signal'] * data['spread_usd_per_mwh']
            * self.capture_rate * data['Position_Size']
        )
        data['Hourly_Pnl'] = (
            data['Captured_Spread'] - data['Commission'] - data['Slippage']
        )
        data['Cumulative_Pnl'] = data['Hourly_Pnl'].cumsum()
        data['Equity'] = self.initial_capital + data['Cumulative_Pnl']
        return data

    def execute_p_outer_strategy(self, pred_df, threshold=0.60,
                                  p_outer_threshold=0.20,
                                  per_execution_hour_costs=True):
        """
        使用 p_outer（两端极端概率 p_c1+p_c5）识别尾部尖峰机会的策略。

        对应 handoff 第5节: "用 p_outer 识别两端尖峰机会"。

        Parameters
        ----------
        threshold : float
            B2B 基准概率阈值
        p_outer_threshold : float
            p_outer 最低阈值，高于此值才开仓（识别极端尾部事件）
        """
        data = self._get_merged_data(pred_df)

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # B2B 基准信号
        pos_cond = (
            (data['p_positive'] >= threshold)
            & (data['p_positive'] > data['p_negative'])
        )
        neg_cond = (
            (data['p_negative'] >= threshold)
            & (data['p_negative'] > data['p_positive'])
        )

        data['Signal'] = 0
        data.loc[pos_cond, 'Signal'] = 1
        data.loc[neg_cond, 'Signal'] = -1

        # p_outer 尖峰过滤器：仅在两端极端概率足够高时交易
        if 'p_outer' in data.columns:
            low_outer = (
                (data['Signal'] != 0)
                & (data['p_outer'] < p_outer_threshold)
            )
            data.loc[low_outer, 'Signal'] = 0

        data['Strategy'] = f'B2B_pOuter_{p_outer_threshold}'
        data['Active_Threshold'] = threshold

        if per_execution_hour_costs:
            return self._settle_pnl_per_execution_hour(data)
        else:
            data['Trade_Action'] = data['Signal'].diff().fillna(0).abs()
            data['Commission'] = data['Trade_Action'] * self.fee_per_mwh
            data['Slippage'] = (
                data['Trade_Action'] * np.abs(data['spread_usd_per_mwh'])
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

    def grid_search_b2b_threshold(self, pred_df,
                                   thresholds=None,
                                   strategy='baseline',
                                   per_execution_hour_costs=True,
                                   **strategy_kwargs):
        """
        B2B 概率阈值网格搜索（以夏普比率为优化目标）。

        Parameters
        ----------
        thresholds : list
            B2B 概率阈值搜索空间，默认 [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        strategy : str
            策略类型: 'baseline', 'combined', 'confidence', 'p_outer'
        strategy_kwargs : dict
            传递给具体策略方法的额外参数
        """
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

        best_sharpe = -float('inf')
        best_params = {}
        results = []

        strategy_map = {
            'baseline': self.execute_b2b_baseline,
            'combined': self.execute_b2b_b2a_combined,
            'confidence': self.execute_confidence_scaled,
            'p_outer': self.execute_p_outer_strategy,
        }
        execute_fn = strategy_map.get(strategy, self.execute_b2b_baseline)

        print(f"B2B 阈值网格搜索: strategy={strategy}, "
              f"{len(thresholds)} 个阈值候选")

        for thresh in thresholds:
            res_df = execute_fn(
                pred_df, threshold=thresh,
                per_execution_hour_costs=per_execution_hour_costs,
                **strategy_kwargs
            )
            metrics = self.calculate_metrics(res_df, self.initial_capital)
            metrics['threshold'] = thresh
            metrics['Total_Trades'] = len(res_df[res_df['Signal'] != 0])
            results.append(metrics)

            if metrics['Sharpe_Ratio'] > best_sharpe:
                best_sharpe = metrics['Sharpe_Ratio']
                best_params = {
                    'threshold': thresh,
                    'strategy': strategy,
                    **strategy_kwargs,
                }

        print(f"B2B 阈值搜索完成，最佳夏普: {best_sharpe:.4f} "
              f"(threshold={best_params['threshold']})")
        return pd.DataFrame(results), best_params

    def grid_search(self, pred_df, spread_space, extreme_space=None,
                    direction_filter=True, max_consecutive_hours=48,
                    rolling_window_space=None, std_multiplier_space=None,
                    short_risk_multiplier_space=None,
                    vol_regime_threshold_space=None,
                    min_profit_ratio_space=None,
                    min_trades_for_selection=20):
        """
        参数化网格搜索最佳阈值（以夏普比率为优化目标）

        Parameters
        ----------
        min_trades_for_selection : int
            最低交易数门槛，交易数低于此值的参数组合不会被选为最佳参数。
            防止选出"只做1笔完美交易"的过拟合参数。默认 20。

        Parameters
        ----------
        spread_space : list
            常规阈值搜索空间（已废弃，保留兼容；实际使用自适应阈值）
        extreme_space : list, optional
            极端天气阈值搜索空间，若为 None 则只搜索单一阈值
        rolling_window_space : list, optional
            滚动窗口搜索空间，默认 [72, 168, 336] (3天/7天/14天)
        std_multiplier_space : list, optional
            波动率倍数搜索空间，默认 [1.0, 1.5, 2.0]
        short_risk_multiplier_space : list, optional
            空头风险系数搜索空间，默认 [1.0, 1.3, 1.5]
        vol_regime_threshold_space : list, optional
            市场波动率最低阈值搜索空间，默认 [3.0, 5.0, 8.0]
        min_profit_ratio_space : list, optional
            预期利润/成本最低倍数搜索空间，默认 [1.0, 1.5, 2.0]
        """
        if rolling_window_space is None:
            rolling_window_space = [72, 168, 336]
        if std_multiplier_space is None:
            std_multiplier_space = [1.0, 1.5, 2.0]   # 粗筛: 3个值 (原4个值, 提速1.3x)
        if short_risk_multiplier_space is None:
            short_risk_multiplier_space = [1.0, 1.3, 1.5]  # 粗筛: 3个值
        if vol_regime_threshold_space is None:
            vol_regime_threshold_space = [3.0, 5.0, 8.0]   # 粗筛: 3个值
        if min_profit_ratio_space is None:
            min_profit_ratio_space = [1.0, 1.5, 2.0]       # 粗筛: 3个值

        best_sharpe = -float('inf')
        best_params = {}
        results = []

        total_combinations = (
            len(rolling_window_space)
            * len(std_multiplier_space)
            * len(short_risk_multiplier_space)
            * len(vol_regime_threshold_space)
            * len(min_profit_ratio_space)
        )
        print(f"自适应参数网格搜索: 共 {total_combinations} 种组合")

        count = 0
        for rw in rolling_window_space:
            for sm in std_multiplier_space:
                for srm in short_risk_multiplier_space:
                    for vrt in vol_regime_threshold_space:
                        for mpr in min_profit_ratio_space:
                            count += 1
                            res_df = self.execute_strategy(
                                pred_df,
                                spread_threshold=50.0,  # 静态阈值已不再使用
                                rolling_window=rw,
                                std_multiplier=sm,
                                short_risk_multiplier=srm,
                                vol_regime_threshold=vrt,
                                min_profit_ratio=mpr,
                                direction_filter=direction_filter,
                                max_consecutive_hours=max_consecutive_hours,
                            )
                            metrics = self.calculate_metrics(res_df, self.initial_capital)
                            metrics['rolling_window'] = rw
                            metrics['std_multiplier'] = sm
                            metrics['short_risk_multiplier'] = srm
                            metrics['vol_regime_threshold'] = vrt
                            metrics['min_profit_ratio'] = mpr
                            metrics['Total_Trades'] = len(res_df[res_df['Signal'] != 0])
                            results.append(metrics)

                            if (metrics['Sharpe_Ratio'] > best_sharpe
                                    and metrics['Total_Trades'] >= min_trades_for_selection):
                                best_sharpe = metrics['Sharpe_Ratio']
                                best_params = {
                                    'rolling_window': rw,
                                    'std_multiplier': sm,
                                    'short_risk_multiplier': srm,
                                    'vol_regime_threshold': vrt,
                                    'min_profit_ratio': mpr,
                                    'direction_filter': direction_filter,
                                    'max_consecutive_hours': max_consecutive_hours,
                                    '_best_sharpe': best_sharpe,
                                    '_best_trades': metrics['Total_Trades'],
                                }

                            if count % 50 == 0:
                                print(f"  进度: {count}/{total_combinations}, "
                                      f"当前最佳夏普: {best_sharpe:.4f}")

        print(f"网格搜索完成，最佳夏普: {best_sharpe:.4f}")
        return pd.DataFrame(results), best_params

    def walk_forward_validation(self, pred_df, n_splits=4,
                                 rolling_window_space=None,
                                 std_multiplier_space=None,
                                 short_risk_multiplier_space=None,
                                 vol_regime_threshold_space=None,
                                 min_profit_ratio_space=None,
                                 direction_filter=True,
                                 max_consecutive_hours=48,
                                 min_trades_for_selection=20):
        """
        Walk-Forward 交叉验证：按时间顺序滚动切分数据，每段用前段数据搜索参数，
        后段数据验证，避免过拟合到单一时间段。

        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据
        n_splits : int
            切分份数，默认 4（即 4 折 walk-forward）
        其余参数同 grid_search

        Returns
        -------
        summary_df : pd.DataFrame
            每折的最佳参数与验证集表现
        robust_params : dict
            各折最佳参数的中位数（对整数参数取众数）
        """
        data = self._get_merged_data(pred_df)

        n_total = len(data)
        fold_size = n_total // (n_splits + 1)  # 第一段作为初始训练

        print(f"\nWalk-Forward 交叉验证: {n_splits} 折, 每折约 {fold_size} 小时")
        print("=" * 60)

        fold_results = []
        all_best_params = []

        for fold in range(n_splits):
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min((fold + 2) * fold_size, n_total)

            train_market = data.iloc[:train_end].copy()
            val_market = data.iloc[val_start:val_end].copy()

            train_pred = pred_df[
                pred_df['delivery_hour_utc'].isin(train_market['delivery_hour_utc'])
            ].copy()
            val_pred = pred_df[
                pred_df['delivery_hour_utc'].isin(val_market['delivery_hour_utc'])
            ].copy()

            # 用训练段搜索参数
            train_engine = ERCOTBacktestEngine(
                train_market[['delivery_hour_utc', 'spread_usd_per_mwh']],
                initial_capital=self.initial_capital,
                fee_per_mwh=self.fee_per_mwh,
                slippage_bps=self.slippage_bps,
                capture_rate=self.capture_rate,
            )

            _, best_p = train_engine.grid_search(
                train_pred, [],
                rolling_window_space=rolling_window_space,
                std_multiplier_space=std_multiplier_space,
                short_risk_multiplier_space=short_risk_multiplier_space,
                vol_regime_threshold_space=vol_regime_threshold_space,
                min_profit_ratio_space=min_profit_ratio_space,
                direction_filter=direction_filter,
                max_consecutive_hours=max_consecutive_hours,
                min_trades_for_selection=min_trades_for_selection,
            )

            # 在验证段上用冻结参数回测
            val_res = self.execute_strategy(
                val_pred,
                spread_threshold=50.0,
                rolling_window=best_p['rolling_window'],
                std_multiplier=best_p['std_multiplier'],
                short_risk_multiplier=best_p['short_risk_multiplier'],
                vol_regime_threshold=best_p['vol_regime_threshold'],
                min_profit_ratio=best_p['min_profit_ratio'],
                direction_filter=direction_filter,
                max_consecutive_hours=max_consecutive_hours,
            )
            val_metrics = self.calculate_metrics(val_res, self.initial_capital)

            fold_results.append({
                'Fold': fold + 1,
                'Train_End': data.iloc[train_end - 1]['delivery_hour_utc'],
                'Val_Start': data.iloc[val_start]['delivery_hour_utc'],
                'Val_End': data.iloc[val_end - 1]['delivery_hour_utc'],
                **best_p,
                'Val_Sharpe': val_metrics['Sharpe_Ratio'],
                'Val_Total_Return': val_metrics['Total_Return'],
                'Val_Max_DD': val_metrics['Max_Drawdown'],
                'Val_Trades': val_metrics['Total_Trades'],
                'Val_Win_Rate': val_metrics['Win_Rate'],
            })
            all_best_params.append(best_p)

            train_sharpe = best_p.get('_best_sharpe', None)
            train_trades = best_p.get('_best_trades', None)
            print(f"Fold {fold + 1}: "
                  f"训练集夏普={train_sharpe:.4f} (交易数={train_trades}), "
                  f"验证集夏普={val_metrics['Sharpe_Ratio']:.4f} (交易数={val_metrics['Total_Trades']})")

        # 汇总稳健参数：数值参数取中位数，整数参数取众数
        summary_df = pd.DataFrame(fold_results)

        robust_params = {}
        for key in all_best_params[0]:
            values = [p[key] for p in all_best_params]
            if key in ('rolling_window', 'max_consecutive_hours'):
                robust_params[key] = int(np.median(values))
            elif key == 'direction_filter':
                robust_params[key] = max(set(values), key=values.count)
            else:
                robust_params[key] = np.median(values)

        robust_params['_fold_count'] = n_splits
        robust_params['_val_sharpe_mean'] = summary_df['Val_Sharpe'].mean()
        robust_params['_val_sharpe_std'] = summary_df['Val_Sharpe'].std()

        print(f"\n稳健参数 (各折中位数): {robust_params}")
        print(f"验证集夏普均值: {robust_params['_val_sharpe_mean']:.4f} "
              f"± {robust_params['_val_sharpe_std']:.4f}")

        return summary_df, robust_params


# ==========================================
# 2. 极端小时条件分组统计函数
# ==========================================
def analyze_filter_impact(result_df):
    """
    分析各过滤器对交易信号的拦截效果

    Returns
    -------
    dict : 过滤器统计信息
    """
    if 'low_vol_regime' not in result_df.columns:
        return None

    total_hours = len(result_df)
    raw_signals = (result_df['predicted_spread'].abs() > 0).sum()
    low_vol_blocked = result_df['low_vol_regime'].sum()
    actual_trades = (result_df['Signal'] != 0).sum()

    # 计算过滤掉的交易
    long_raw = (result_df['predicted_spread'] > result_df['dynamic_threshold']).sum()
    short_raw = (result_df['predicted_spread'] < -result_df['dynamic_threshold']).sum()
    total_raw = long_raw + short_raw

    stats = {
        '总小时数': total_hours,
        '原始开仓信号数': total_raw,
        '波动率Regime拦截小时数': low_vol_blocked,
        '最终执行交易数': actual_trades,
        '信号保留率': f"{actual_trades / max(total_raw, 1) * 100:.1f}%",
        '交易频率(笔/天)': f"{actual_trades / (total_hours / 24):.1f}",
    }
    return stats


def analyze_extreme_hour_performance(result_df):
    """
    按 extreme_hour_flag 分组统计策略表现
    """
    if 'extreme_hour_flag' not in result_df.columns:
        print("数据中无 extreme_hour_flag 字段，跳过极端小时分析")
        return None

    group = result_df.groupby('extreme_hour_flag').agg(
        Avg_Hourly_Pnl=('Hourly_Pnl', 'mean'),
        Trade_Count=('Signal', lambda x: (x != 0).sum()),
        Avg_Real_Spread=('spread_usd_per_mwh', 'mean'),
        Max_Spread=('spread_usd_per_mwh', 'max'),
        Min_Spread=('spread_usd_per_mwh', 'min'),
    )
    return group


def analyze_direction_performance(result_df):
    """
    按交易方向分组统计策略表现
    """
    trades = result_df[result_df['Signal'] != 0].copy()
    if len(trades) == 0:
        print("无交易记录")
        return None

    trades['Direction'] = trades['Signal'].map({1: 'Long', -1: 'Short'})
    group = trades.groupby('Direction').agg(
        Trade_Count=('Signal', 'count'),
        Avg_Hourly_Pnl=('Hourly_Pnl', 'mean'),
        Total_PnL=('Hourly_Pnl', 'sum'),
        Win_Rate=('Hourly_Pnl', lambda x: (x > 0).mean()),
        Avg_Real_Spread=('spread_usd_per_mwh', 'mean'),
    )
    return group


def analyze_risk_concentration(result_df):
    """
    C1 风险集中度分析（对应 handoff 第6节 风险提示）。

    分析维度:
      - Top N 交易日集中度（去掉Top 5/Top 10后剩余PnL）
      - 月度 PnL 分布与盈利月数
      - 1月效应（January dependency）
      - 极端天气 vs 正常天气 PnL 分解
      - 各分类（actual_class 1-5）的表现

    Returns
    -------
    risk_report : dict
        风险集中度指标汇总
    monthly_df : pd.DataFrame
        月度 PnL 明细
    daily_df : pd.DataFrame
        每日 PnL 明细（含排名）
    """
    df = result_df.copy()
    df['date'] = df['delivery_hour_utc'].dt.date
    df['month'] = df['delivery_hour_utc'].dt.month
    df['month_label'] = df['delivery_hour_utc'].dt.strftime('%Y-%m')

    # ---- 每日 PnL 与集中度 ----
    daily_pnl = df.groupby('date')['Hourly_Pnl'].sum().sort_values(ascending=False)
    total_pnl = daily_pnl.sum()

    top5_pnl = daily_pnl.head(5).sum()
    top10_pnl = daily_pnl.head(10).sum()

    # ---- 月度分布 ----
    monthly = df.groupby('month_label').agg(
        Monthly_PnL=('Hourly_Pnl', 'sum'),
        Trade_Count=('Signal', lambda x: (x != 0).sum()),
        Avg_Spread=('spread_usd_per_mwh', 'mean'),
        Max_Spread=('spread_usd_per_mwh', 'max'),
    ).sort_index()

    profitable_months = int((monthly['Monthly_PnL'] > 0).sum())
    total_months = len(monthly)

    # ---- 1月效应 ----
    jan_mask = df['month'] == 1
    jan_pnl = df.loc[jan_mask, 'Hourly_Pnl'].sum()
    non_jan_pnl = df.loc[~jan_mask, 'Hourly_Pnl'].sum()

    # ---- 极端天气分解 ----
    if 'fixed_extreme_weather_flag' in df.columns:
        ext_mask = df['fixed_extreme_weather_flag'] == 1
        extreme_pnl = df.loc[ext_mask, 'Hourly_Pnl'].sum()
        normal_pnl = df.loc[~ext_mask, 'Hourly_Pnl'].sum()
        extreme_trades = (df.loc[ext_mask, 'Signal'] != 0).sum()
        normal_trades = (df.loc[~ext_mask, 'Signal'] != 0).sum()
    else:
        extreme_pnl = None
        normal_pnl = None
        extreme_trades = 0
        normal_trades = 0

    # ---- 各分类表现 ----
    if 'actual_class' in df.columns:
        class_perf = df[df['Signal'] != 0].groupby('actual_class').agg(
            Trade_Count=('Signal', 'count'),
            Total_PnL=('Hourly_Pnl', 'sum'),
            Avg_PnL=('Hourly_Pnl', 'mean'),
            Win_Rate=('Hourly_Pnl', lambda x: (x > 0).mean()),
        )
    else:
        class_perf = None

    risk_report = {
        'Total_PnL': total_pnl,
        'Top5_Days_PnL': top5_pnl,
        'Top5_Concentration': top5_pnl / total_pnl if total_pnl != 0 else 0.0,
        'PnL_ex_Top5': total_pnl - top5_pnl,
        'Top10_Days_PnL': top10_pnl,
        'Top10_Concentration': top10_pnl / total_pnl if total_pnl != 0 else 0.0,
        'PnL_ex_Top10': total_pnl - top10_pnl,
        'Profitable_Months': f"{profitable_months}/{total_months}",
        'Profitable_Months_Ratio': profitable_months / total_months if total_months > 0 else 0,
        'January_PnL': jan_pnl,
        'Non_January_PnL': non_jan_pnl,
        'January_Concentration': jan_pnl / total_pnl if total_pnl != 0 else 0.0,
    }

    if extreme_pnl is not None:
        risk_report['Extreme_Weather_PnL'] = extreme_pnl
        risk_report['Normal_Weather_PnL'] = normal_pnl
        risk_report['Extreme_Trades'] = extreme_trades
        risk_report['Normal_Trades'] = normal_trades

    return risk_report, monthly, daily_pnl, class_perf


def print_risk_report(risk_report):
    """格式化打印风险集中度报告（对应 handoff 第6节格式）。"""
    print("\n" + "=" * 60)
    print("风险集中度分析 (Risk Concentration Report)")
    print("=" * 60)

    total = risk_report['Total_PnL']
    print(f"  Total PnL:              ${total:>12,.2f}")
    print(f"  Top 5 Days PnL:         ${risk_report['Top5_Days_PnL']:>12,.2f}"
          f"  ({risk_report['Top5_Concentration']*100:.1f}%)")
    print(f"  PnL ex-Top 5:           ${risk_report['PnL_ex_Top5']:>12,.2f}")
    print(f"  Top 10 Days PnL:        ${risk_report['Top10_Days_PnL']:>12,.2f}"
          f"  ({risk_report['Top10_Concentration']*100:.1f}%)")
    print(f"  PnL ex-Top 10:          ${risk_report['PnL_ex_Top10']:>12,.2f}")
    print(f"  Profitable Months:      {risk_report['Profitable_Months']:>15}")
    print(f"  January PnL:            ${risk_report['January_PnL']:>12,.2f}"
          f"  ({risk_report['January_Concentration']*100:.1f}%)")
    print(f"  Non-January PnL:        ${risk_report['Non_January_PnL']:>12,.2f}")

    if 'Extreme_Weather_PnL' in risk_report:
        print(f"  Extreme-Weather PnL:    ${risk_report['Extreme_Weather_PnL']:>12,.2f}"
              f"  ({risk_report['Extreme_Trades']} trades)")
        print(f"  Normal-Weather PnL:     ${risk_report['Normal_Weather_PnL']:>12,.2f}"
              f"  ({risk_report['Normal_Trades']} trades)")


def analyze_strategy_comparison(strategies_results, initial_capital=100000.0):
    """
    多策略横向对比（生成 handoff 风格对比表）。

    Parameters
    ----------
    strategies_results : dict
        {策略名称: result_df}
    initial_capital : float
        初始资金

    Returns
    -------
    pd.DataFrame
        策略对比表，列: Strategy, Trades, Direction_Precision, Total_PnL,
        PnL_per_Trade, Sharpe, Max_DD, Profitable_Months
    """
    comparison = []
    for name, res_df in strategies_results.items():
        metrics = ERCOTBacktestEngine.calculate_metrics(res_df, initial_capital)
        trades_df = res_df[res_df['Signal'] != 0]
        n_trades = len(trades_df)

        # 方向精度: 信号方向与实际价差方向一致的比例
        if n_trades > 0 and 'actual_class' in res_df.columns:
            # actual_class: 1,2=INC, 3=Neutral, 4,5=DEC
            correct_inc = (
                (trades_df['Signal'] == -1) & (trades_df['actual_class'].isin([1, 2]))
            ).sum()
            correct_dec = (
                (trades_df['Signal'] == 1) & (trades_df['actual_class'].isin([4, 5]))
            ).sum()
            direction_precision = (
                (correct_inc + correct_dec) / n_trades if n_trades > 0 else 0.0
            )
        elif n_trades > 0:
            direction_precision = (trades_df['Hourly_Pnl'] > 0).mean()
        else:
            direction_precision = 0.0

        # 盈利月数
        if 'delivery_hour_utc' in res_df.columns:
            res_df_copy = res_df.copy()
            res_df_copy['month_label'] = (
                res_df_copy['delivery_hour_utc'].dt.strftime('%Y-%m')
            )
            monthly_pnl = res_df_copy.groupby('month_label')['Hourly_Pnl'].sum()
            profitable_months = int((monthly_pnl > 0).sum())
            total_months = len(monthly_pnl)
        else:
            profitable_months = 0
            total_months = 0

        comparison.append({
            'Strategy': name,
            'Trades': n_trades,
            'Direction_Precision': direction_precision,
            'Total_PnL': metrics['Total_Pnl'],
            'PnL_per_Trade': metrics['Total_Pnl'] / n_trades if n_trades > 0 else 0.0,
            'Sharpe': metrics['Sharpe_Ratio'],
            'Sortino': metrics['Sortino_Ratio'],
            'Max_DD': metrics['Max_Drawdown'],
            'Win_Rate': metrics['Win_Rate'],
            'Profit_Factor': metrics['Profit_Factor'],
            'Profitable_Months': f"{profitable_months}/{total_months}",
        })

    return pd.DataFrame(comparison)
import matplotlib.dates as mdates

def plot_backtest_result(result_df, title="ERCOT Arbitrage Strategy Equity Curve",
                         save_path=None):
    """
    标准尺寸回测可视化（修复图表过长、时间轴重叠、柱状图挤压问题）
    """
    result_df = result_df.copy()
    time_col = 'delivery_hour_utc'

    # 固定画布尺寸为标准比例 (宽 14, 高 10)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 0.5, 1.0, 1.0],
                          hspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax_pos = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3])

    # ---- 子图 1: 净值曲线 ----
    ax1.plot(result_df[time_col], result_df['Equity'],
             label='Strategy Equity', color='#1f77b4', linewidth=1.5, zorder=2)

    initial_eq = result_df['Equity'].iloc[0]
    ax1.axhline(y=initial_eq, color='gray', linestyle=':', alpha=0.5,
                label=f'Initial Capital (${initial_eq:,.0f})')

    trade_rows = result_df[result_df['Signal'] != 0]
    if not trade_rows.empty:
        long_trades = trade_rows[trade_rows['Signal'] == 1]
        short_trades = trade_rows[trade_rows['Signal'] == -1]

        if not long_trades.empty:
            ax1.scatter(long_trades[time_col], long_trades['Equity'],
                        color='#2ca02c', s=12, marker=MarkerStyle('^'), zorder=5,
                        alpha=0.7, label=f'LONG (n={len(long_trades)})') # type: ignore
        if not short_trades.empty:
            ax1.scatter(short_trades[time_col], short_trades['Equity'],
                        color='#d62728', s=12, marker=MarkerStyle('v'), zorder=5,
                        alpha=0.7, label=f'SHORT (n={len(short_trades)})') # type: ignore

    total_pnl = result_df['Hourly_Pnl'].sum()
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_ylabel("Equity ($)", fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.25)

    # ---- 子图 2: 持仓带状图 ----
    long_mask = result_df['Signal'] == 1
    short_mask = result_df['Signal'] == -1
    ax_pos.fill_between(result_df[time_col], 0, 1,
                        where=long_mask, color='#2ca02c', alpha=0.7, step='post')
    ax_pos.fill_between(result_df[time_col], 0, 1,
                        where=short_mask, color='#d62728', alpha=0.7, step='post')
    ax_pos.set_yticks([])
    ax_pos.set_ylabel('Pos', fontsize=8, rotation=0, labelpad=15)
    plt.setp(ax_pos.get_xticklabels(), visible=False)

    # ---- 子图 3: 回撤曲线 ----
    drawdown = (result_df['Equity'] - result_df['Equity'].cummax()) / \
        result_df['Equity'].cummax().replace(0, np.nan)
    ax2.fill_between(result_df[time_col], 0, drawdown * 100,
                     color='#d62728', alpha=0.3)
    ax2.plot(result_df[time_col], drawdown * 100, color='#d62728', linewidth=0.8)
    
    # 自动优化 X 轴时间格式（按月显示，防止标签太密）
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.grid(True, alpha=0.25)

    # ---- 子图 4: 逐笔交易 PnL (使用折线或细柱，防止撑爆画布) ----
    trade_pnls = result_df[result_df['Signal'] != 0]['Hourly_Pnl'].values
    if len(trade_pnls) > 0:
        # 交易笔数过多时使用线图/散点图，笔数少时使用柱状图
        if len(trade_pnls) > 500:
            ax3.plot(trade_pnls, color='#1f77b4', linewidth=0.5, alpha=0.7)
        else:
            colors = ['#2ca02c' if p >= 0 else '#d62728' for p in trade_pnls]
            ax3.bar(range(len(trade_pnls)), trade_pnls, color=colors, width=0.8, alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        
        win_count = (trade_pnls > 0).sum()
        ax3.set_title(
            f'Trade PnL Distribution  |  Win Rate: {win_count}/{len(trade_pnls)} '
            f'({win_count/len(trade_pnls)*100:.1f}%)',
            fontsize=9
        )
    ax3.set_xlabel("Trade Sequence", fontsize=10)
    ax3.set_ylabel("PnL ($)", fontsize=10)
    ax3.grid(True, alpha=0.25)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass
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
            print(f"  {label}: {v:.2f}" if v != float('inf') else f"  {label}: ∞")
        elif k in ("Long_Trades", "Short_Trades", "Total_Trades"):
            print(f"  {label}: {v}")
        else:
            print(f"  {label}: {v:.4f}")


# ==========================================
# 4. 主流程：两阶段策略开发与测试
# ==========================================
if __name__ == "__main__":
    import sys
    import time
    from datetime import datetime

    start_time = time.time()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'backtest_{timestamp}.log')

    class TeeOutput:
        def __init__(self, console, log_file):
            self.console = console
            self.log_file = log_file

        def write(self, message):
            self.console.write(message)
            self.log_file.write(message)

        def flush(self):
            self.console.flush()
            self.log_file.flush()

        def close(self):
            self.log_file.close()

    log_fh = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_fh)

    try:
        USE_REAL_DATA = os.path.isdir(MEMBER_B_DATA_DIR)
        USE_C1_DATA = os.path.isdir(C1_DATA_DIR)

        if USE_REAL_DATA:

            # =============================================
            # Phase 1: 2025 验证集 — 策略开发与阈值搜索
            # =============================================
            print("=" * 60)
            print("Phase 1: 2025 验证集 — 策略开发与阈值搜索")
            print("=" * 60)

            market_2025, pred_2025 = load_validation_2025()
            print(f"2025 数据加载完成: {len(market_2025)} 小时")

            engine_2025 = ERCOTBacktestEngine(
                market_2025,
                initial_capital=100000,
                fee_per_mwh=2.0,
                slippage_bps=50.0,
                capture_rate=0.65,
            )

            print("\n--- 自适应参数网格搜索（粗筛阶段: 3^5=243 组合） ---")
            # 粗筛: 使用默认的3值/维度，共 243 种组合
            # 如需精筛: 粗筛找到最优参数后，在其附近缩小搜索范围重新运行
            # 例如: best_rw=168, 则精筛 rolling_window_space=[120, 144, 168, 192, 216]
            rolling_window_space = [72, 168, 336]        # 3天/7天/14天
            std_multiplier_space = [1.0, 1.5, 2.0]       # 波动率倍数 (粗筛3值)
            short_risk_multiplier_space = [1.0, 1.3, 1.5]  # 空头惩罚系数 (粗筛3值)
            vol_regime_threshold_space = [3.0, 5.0, 8.0]   # 市场波动率最低阈值 (粗筛3值)
            min_profit_ratio_space = [1.0, 1.5, 2.0]       # 预期利润/成本倍数 (粗筛3值)

            search_results, best_params = engine_2025.grid_search(
                pred_2025, [],  # spread_space 已废弃，传空列表
                rolling_window_space=rolling_window_space,
                std_multiplier_space=std_multiplier_space,
                short_risk_multiplier_space=short_risk_multiplier_space,
                vol_regime_threshold_space=vol_regime_threshold_space,
                min_profit_ratio_space=min_profit_ratio_space,
                direction_filter=True, max_consecutive_hours=48,
            )
            print(f"最佳参数: {best_params}")
            print("\n网格搜索结果 (按夏普排序 Top 10):")
            top_results = search_results.sort_values('Sharpe_Ratio', ascending=False).head(10)
            print(top_results.to_string(index=False))

            # ---- 可选: Walk-Forward 交叉验证 (更稳健的参数选择) ----
            # 取消下面注释以使用 Walk-Forward 替代单次网格搜索:
            wf_summary, best_params = engine_2025.walk_forward_validation(
                pred_2025, n_splits=4,
                rolling_window_space=rolling_window_space,
                std_multiplier_space=std_multiplier_space,
                short_risk_multiplier_space=short_risk_multiplier_space,
                vol_regime_threshold_space=vol_regime_threshold_space,
                min_profit_ratio_space=min_profit_ratio_space,
                direction_filter=True, max_consecutive_hours=48,
            )
            print("\nWalk-Forward 各折详情:")
            print(wf_summary.to_string(index=False))
            # 清理内部字段，保留纯策略参数
            best_params = {k: v for k, v in best_params.items()
                           if not k.startswith('_')}

            final_res_2025 = engine_2025.execute_strategy(pred_2025, **best_params)
            metrics_2025 = engine_2025.calculate_metrics(final_res_2025, engine_2025.initial_capital)

            print("\n--- 2025 策略核心指标 (开发集) ---")
            print_metrics(metrics_2025)

            print("\n--- 极端小时 vs 正常小时表现对比 ---")
            extreme_analysis = analyze_extreme_hour_performance(final_res_2025)
            if extreme_analysis is not None:
                print(extreme_analysis.to_string())

            print("\n--- 多空方向表现对比 ---")
            dir_analysis = analyze_direction_performance(final_res_2025)
            if dir_analysis is not None:
                print(dir_analysis.to_string())

            print("\n--- 过滤器拦截效果 ---")
            filter_stats = analyze_filter_impact(final_res_2025)
            if filter_stats is not None:
                for k, v in filter_stats.items():
                    print(f"  {k}: {v}")

            plot_backtest_result(
                final_res_2025,
                title="ERCOT Arbitrage Strategy — 2025 Validation (Development)",
                save_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'ercot_backtest_2025.png'
                ),
            )

            # =============================================
            # Phase 2: 2026 H1 — 冻结策略，最终独立测试
            # =============================================
            print("\n" + "=" * 60)
            print("Phase 2: 2026 H1 — 冻结策略，最终独立测试")
            print("=" * 60)

            market_2026, pred_2026 = load_test_2026()
            print(f"2026 数据加载完成: {len(market_2026)} 小时 (已排除 target_available=0)")

            engine_2026 = ERCOTBacktestEngine(
                market_2026,
                initial_capital=100000,
                fee_per_mwh=2.0,
                slippage_bps=50.0,
                capture_rate=0.65,
            )

            final_res_2026 = engine_2026.execute_strategy(pred_2026, **best_params)
            metrics_2026 = engine_2026.calculate_metrics(final_res_2026, engine_2026.initial_capital)

            print("\n--- 2026 策略核心指标 (独立测试集) ---")
            print(f"  使用冻结参数: {best_params}")
            print_metrics(metrics_2026)

            print("\n--- 多空方向表现对比 ---")
            dir_analysis_2026 = analyze_direction_performance(final_res_2026)
            if dir_analysis_2026 is not None:
                print(dir_analysis_2026.to_string())

            if 'week_id' in final_res_2026.columns:
                print("\n--- 各周表现 ---")
                weekly = final_res_2026.groupby('week_id').agg(
                    Weekly_PnL=('Hourly_Pnl', 'sum'),
                    Trade_Count=('Signal', lambda x: (x != 0).sum()),
                )
                print(weekly.to_string())

            plot_backtest_result(
                final_res_2026,
                title="ERCOT Arbitrage Strategy — 2026 H1 (Frozen Test)",
                save_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'ercot_backtest_2026.png'
                ),
            )

            # =============================================
            # 汇总对比
            # =============================================
            print("\n" + "=" * 60)
            print("两阶段对比汇总")
            print("=" * 60)
            print(f"{'指标':<25} {'2025 开发集':>15} {'2026 测试集':>15}")
            print("-" * 55)
            compare_keys = [
                'Total_Return', 'Sharpe_Ratio', 'Sortino_Ratio',
                'Max_Drawdown', 'Calmar_Ratio', 'Win_Rate',
                'Profit_Factor', 'Total_Trades', 'Avg_Trade_Pnl',
            ]
            for key in compare_keys:
                v25 = metrics_2025[key]
                v26 = metrics_2026[key]
                if key in ('Total_Return', 'Max_Drawdown', 'Win_Rate'):
                    print(f"{key:<25} {v25*100:>14.2f}% {v26*100:>14.2f}%")
                elif key == 'Total_PnL' or key == 'Avg_Trade_Pnl':
                    print(f"{key:<25} ${v25:>14,.2f} ${v26:>14,.2f}")
                elif key == 'Profit_Factor':
                    s25 = f"{v25:.2f}" if v25 != float('inf') else "∞"
                    s26 = f"{v26:.2f}" if v26 != float('inf') else "∞"
                    print(f"{key:<25} {s25:>15} {s26:>15}")
                else:
                    print(f"{key:<25} {v25:>15.4f} {v26:>15.4f}")

        # =============================================
        # Phase 3 & 4: C1 统一预测表 — 基准与组合策略
        # =============================================
        if USE_C1_DATA:
            print("\n" + "=" * 60)
            print("Phase 3: C1 统一预测表 — 基准 B2B 策略复现")
            print("=" * 60)

            market_c1, pred_c1 = load_c1_unified_2025()
            print(f"C1 数据加载完成: {len(market_c1)} 小时")

            engine_c1 = ERCOTBacktestEngine(
                market_c1,
                initial_capital=100000,
                fee_per_mwh=2.0,
                slippage_bps=50.0,   # 0.5% = handoff slippage formula
                capture_rate=0.65,    # handoff capture rate
            )

            # ---- 3a: 复现 Handoff 基准策略 (threshold=0.60) ----
            print("\n--- 3a: Handoff 基准 B2B 策略 (threshold=0.60) ---")
            res_b2b_baseline = engine_c1.execute_b2b_baseline(
                pred_c1, threshold=0.60, per_execution_hour_costs=True
            )
            metrics_b2b_baseline = engine_c1.calculate_metrics(
                res_b2b_baseline, engine_c1.initial_capital
            )

            print("C1 基准策略指标:")
            print_metrics(metrics_b2b_baseline)

            # 对比 handoff 报告的 net_pnl
            if 'net_pnl' in pred_c1.columns:
                handoff_total_pnl = pred_c1['net_pnl'].sum()
                print(f"\n  Handoff 报告 net_pnl 总和: ${handoff_total_pnl:,.2f}")
                print(f"  我们计算的 Total_PnL:     ${metrics_b2b_baseline['Total_Pnl']:,.2f}")

            # ---- 3b: B2B 阈值网格搜索 ----
            print("\n--- 3b: B2B 阈值网格搜索 (0.50~0.80) ---")
            b2b_search_results, b2b_best_params = engine_c1.grid_search_b2b_threshold(
                pred_c1,
                thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                strategy='baseline',
            )
            print(f"最佳 B2B 阈值参数: {b2b_best_params}")
            print("\nB2B 阈值搜索结果 (按夏普排序):")
            top_b2b = b2b_search_results.sort_values('Sharpe_Ratio', ascending=False)
            print(top_b2b[['threshold', 'Sharpe_Ratio', 'Total_Pnl',
                            'Total_Trades', 'Win_Rate', 'Max_Drawdown']].to_string(index=False))

            # ---- 3c: 风险集中度分析 ----
            print("\n--- 3c: C1 基准策略 风险集中度分析 ---")
            risk_report, monthly_df, daily_pnl, class_perf = analyze_risk_concentration(
                res_b2b_baseline
            )
            print_risk_report(risk_report)

            if class_perf is not None:
                print("\n各 actual_class 交易表现:")
                print(class_perf.to_string())

            print("\n月度 PnL 明细:")
            print(monthly_df.to_string())

            # ---- 3d: 可视化 ----
            plot_backtest_result(
                res_b2b_baseline,
                title="C1 B2B Baseline Strategy — 2025 OOF (threshold=0.60)",
                save_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'c1_b2b_baseline_2025.png'
                ),
            )

            # =============================================
            # Phase 4: C1 组合策略对比
            # =============================================
            print("\n" + "=" * 60)
            print("Phase 4: C1 多策略对比 — B2B vs B2B+B2A vs Confidence")
            print("=" * 60)

            # 运行各策略变体（使用最佳阈值）
            best_threshold = b2b_best_params.get('threshold', 0.60)

            # 策略1: B2B 基准
            res_s1 = engine_c1.execute_b2b_baseline(pred_c1, threshold=best_threshold)

            # 策略2: B2B + B2A 方向确认
            res_s2 = engine_c1.execute_b2b_b2a_combined(
                pred_c1, threshold=best_threshold,
                use_b2a_direction=True, use_b2a_magnitude=False,
            )

            # 策略3: B2B + B2A 方向 + 幅度过滤
            res_s3 = engine_c1.execute_b2b_b2a_combined(
                pred_c1, threshold=best_threshold,
                use_b2a_direction=True, use_b2a_magnitude=True,
                min_magnitude=5.0,
            )

            # 策略4: B2B + Confidence 仓位缩放
            res_s4 = engine_c1.execute_confidence_scaled(
                pred_c1, threshold=best_threshold,
                use_confidence_sizing=True,
            )

            # 策略5: B2B + p_outer 尖峰过滤
            res_s5 = engine_c1.execute_p_outer_strategy(
                pred_c1, threshold=best_threshold,
                p_outer_threshold=0.15,
            )

            # 策略6: 极端天气风险开关（仅在极端天气交易）
            res_s6 = engine_c1.execute_b2b_b2a_combined(
                pred_c1, threshold=best_threshold,
                use_b2a_direction=True,
                extreme_weather_filter=True,
            )

            # 策略7: B2B + 极端天气时降低阈值
            res_s7 = engine_c1.execute_b2b_baseline(
                pred_c1, threshold=best_threshold,
            )
            # 对极端小时手动降低阈值
            ext_hours = pred_c1['fixed_extreme_weather_flag'] == 1
            if ext_hours.any():
                res_s7_ext = engine_c1.execute_b2b_baseline(
                    pred_c1, threshold=max(0.45, best_threshold - 0.10),
                )
                # 合并：正常小时用基准阈值，极端小时用降低阈值
                res_s7 = res_s7.copy()
                ext_idx = res_s7['fixed_extreme_weather_flag'] == 1
                res_s7.loc[ext_idx, 'Signal'] = res_s7_ext.loc[ext_idx, 'Signal']
                res_s7 = engine_c1._settle_pnl_per_execution_hour(res_s7)

            # 汇总对比
            all_strategies = {
                '1_B2B_Baseline': res_s1,
                '2_B2B+B2A_Dir': res_s2,
                '3_B2B+B2A_Dir+Mag': res_s3,
                '4_ConfidenceScaled': res_s4,
                '5_pOuter_Filter': res_s5,
                '6_ExtremeWx_Only': res_s6,
                '7_ExtremeWx_LowerThresh': res_s7,
            }

            comparison_df = analyze_strategy_comparison(
                all_strategies, initial_capital=engine_c1.initial_capital
            )
            print("\n策略对比表:")
            print(comparison_df.to_string(index=False))

            # 信号分布分析
            print("\n--- 各策略信号分布 ---")
            for name, res in all_strategies.items():
                n_pos = (res['Signal'] == 1).sum()
                n_neg = (res['Signal'] == -1).sum()
                n_total = (res['Signal'] != 0).sum()
                print(f"  {name:<30s}  LONG={n_pos:>4d}  SHORT={n_neg:>4d}  "
                      f"TOTAL={n_total:>4d}  ({n_total/len(res)*100:.1f}%)")

        if not USE_REAL_DATA and not USE_C1_DATA:
            print("未找到成员B数据目录或C1数据目录，使用模拟数据运行演示...")
            print(f"期望路径: {MEMBER_B_DATA_DIR}")
            print(f"        : {C1_DATA_DIR}\n")

        np.random.seed(2026)
        time_len = 2000
        mock_times = pd.date_range(start="2026-06-01", periods=time_len, freq="h")

        da_price = np.random.normal(loc=40, scale=10, size=time_len)
        true_spread = np.random.normal(loc=2.0, scale=15.0, size=time_len)
        spike_mask = np.random.random(time_len) < 0.03
        true_spread[spike_mask] = np.random.uniform(50, 300, spike_mask.sum()) * \
            np.random.choice([1, -1], spike_mask.sum())
        rt_price = np.clip(da_price + true_spread, -30, 5000)

        market_df = pd.DataFrame({
            'delivery_hour_utc': mock_times,
            'spread_usd_per_mwh': true_spread,
        })

        pred_spread = true_spread + np.random.normal(0, 20, time_len)
        pred_direction = np.where(pred_spread > 0, 1, -1)

        pred_df = pd.DataFrame({
            'delivery_hour_utc': mock_times,
            'predicted_spread': pred_spread,
            'predicted_direction': pred_direction,
        })

        engine = ERCOTBacktestEngine(
            market_df, initial_capital=100000,
            fee_per_mwh=2.0, slippage_bps=50.0, capture_rate=0.65,
        )

        print("--- 网格搜索最佳 spread_threshold ---")
        spread_space = [10.0, 30.0, 50.0, 100.0]
        search_results, best_params = engine.grid_search(
            pred_df, spread_space, direction_filter=True,
        )
        print(f"最佳参数: {best_params}\n")

        final_res = engine.execute_strategy(pred_df, **best_params)
        final_metrics = engine.calculate_metrics(final_res, engine.initial_capital)

        print("--- 模拟策略核心量化指标 ---")
        print_metrics(final_metrics)

        print("\n--- 多空方向表现对比 ---")
        dir_analysis = analyze_direction_performance(final_res)
        if dir_analysis is not None:
            print(dir_analysis.to_string())

        plot_backtest_result(final_res)

    finally:
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        elif minutes > 0:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        print(f"\n程序总执行时间: {time_str}")

        sys.stdout = original_stdout
        log_fh.close()
        print(f"日志已保存至: {log_path}")
        print(f"程序总执行时间: {time_str}")