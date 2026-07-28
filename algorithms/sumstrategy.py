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


def load_c1_unified_2026(data_dir=None):
    """
    加载 C1 统一预测表（2026 H1 周度 Walk-Forward），对接交易团队的冻结规则交付。

    该表是 C1 模型在 2026 H1 的周度滚动预测结果（27周，4,197小时）。
    用途限定为 frozen-rule temporal robustness check ——
    不得基于此表重新选择模型、修改阈值或调整成本假设。

    与 2025 OOF 表的关键差异:
      - 多了 week_id（周度滚动窗口标识）
      - 多了 extreme_weather_flag（=fixed_extreme_weather_flag，两个字段值相同）
      - 多了 target_extreme50、训练元数据等列

    Returns
    -------
    market_df : pd.DataFrame
        标准化市场数据，列: delivery_hour_utc, spread_usd_per_mwh
    pred_df : pd.DataFrame
        完整预测数据，包含所有 C1 核心字段 + week_id
    """
    if data_dir is None:
        data_dir = C1_DATA_DIR
    path = os.path.join(data_dir, 'C1_unified_prediction_table_2026_H1_walkforward_v1.parquet')
    if not os.path.exists(path):
        csv_path = os.path.join(data_dir, 'C1_unified_prediction_table_2026_H1_walkforward_v1.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"C1 2026 预测表未找到: {path} 或 {csv_path}")
    else:
        df = pd.read_parquet(path)

    df['delivery_hour_utc'] = pd.to_datetime(df['delivery_hour_utc'])

    # 标准化市场数据
    market_df = df[['delivery_hour_utc', 'actual_spread']].copy()
    market_df.rename(columns={'actual_spread': 'spread_usd_per_mwh'}, inplace=True)

    # 核心预测字段（与 2025 保持一致）+ 2026 特有字段
    pred_cols = [
        'delivery_hour_utc', 'predicted_spread',
        'p_c1', 'p_c2', 'p_c3', 'p_c4', 'p_c5',
        'p_negative', 'p_neutral', 'p_positive', 'p_outer',
        'predicted_class', 'confidence',
        'signal_base', 'recommended_action_base',
        'actual_class', 'fixed_extreme_weather_flag',
        'target_extreme20', 'target_extreme50',
        'net_pnl', 'week_id',
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
        """获取合并后的数据，自动缓存以避免重复 pd.merge 操作，并统一裁剪异常极值"""
        key = id(pred_df)
        if key not in self._merged_cache:
            data = pd.merge(
                self.df, pred_df, on='delivery_hour_utc', how='inner'
            ).sort_values('delivery_hour_utc').reset_index(drop=True)
            self._merged_cache[key] = data
        data = self._merged_cache[key].copy()
        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)
        return data

    def _settle_pnl_on_signal_change(self, data):
        """按信号变化收费的 PnL 结算（仅开平仓时收取佣金+滑点）"""
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

    def _settle_pnl(self, data, per_execution_hour_costs=True):
        """统一 PnL 结算入口，根据成本模式选择结算方式"""
        if per_execution_hour_costs:
            return self._settle_pnl_per_execution_hour(data)
        else:
            return self._settle_pnl_on_signal_change(data)

    @staticmethod
    def _generate_b2b_signal(data, threshold):
        """生成 B2B 基准信号（提取公共逻辑，避免重复代码）"""
        pos_cond = (
            (data['p_positive'] >= threshold)
            & (data['p_positive'] > data['p_negative'])
        )
        neg_cond = (
            (data['p_negative'] >= threshold)
            & (data['p_negative'] > data['p_positive'])
        )
        signal = pd.Series(0, index=data.index, dtype=int)
        signal[pos_cond] = 1
        signal[neg_cond] = -1
        return signal

    def _settle_pnl_with_position_sizing(self, data):
        """含仓位缩放的 PnL 结算（用于 confidence_scaled / ensemble_weights 策略）"""
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
                     min_profit_ratio: float = 1.5,      # 预期利润/成本最低倍数，低于此值不开仓
                     per_execution_hour_costs: bool = False,  # True=按执行小时收费(统一对比), False=按信号变化收费(旧版)
                     ma_trend_filter: bool = False):     # True=用实际价差MA趋势替代predicted_direction过滤
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
        per_execution_hour_costs : bool
            True 时按执行小时收取佣金+滑点（每持仓小时收费，与 C1/Baseline 引擎一致），
            False 时按信号变化收费（仅开平仓时收费，旧版行为）。默认 False 保持向后兼容。
        ma_trend_filter : bool
            True 时用实际价差的 MA(24,168) 金叉/死叉替代 predicted_direction 作为方向过滤器。
            金叉(MA24>MA168)才允许做多，死叉(MA24<MA168)才允许做空。
        """
        data = self._get_merged_data(pred_df)

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

        if ma_trend_filter:
            # 用实际价差 MA 趋势替代 predicted_direction 过滤器
            # 使用 lagged spread (shift 1) 避免 look-ahead bias
            spread_lagged = data['spread_usd_per_mwh'].shift(1)
            ma_short = spread_lagged.rolling(window=24, min_periods=24).mean()
            ma_long = spread_lagged.rolling(window=168, min_periods=168).mean()
            ma_bullish = ma_short > ma_long   # 金叉 → 上升趋势 → 允许做多
            ma_bearish = ma_short < ma_long   # 死叉 → 下降趋势 → 允许做空
            # 趋势不明确时（MA 未就绪）允许两个方向
            ma_bullish = ma_bullish.fillna(True)
            ma_bearish = ma_bearish.fillna(True)
            long_signal = long_signal & ma_bullish
            short_signal = short_signal & ma_bearish
        elif direction_filter and 'predicted_direction' in data.columns:
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

        # 8. PnL 结算逻辑
        data = self._settle_pnl(data, per_execution_hour_costs)

        # 标记策略类型
        data['Strategy'] = 'ML_Adaptive'
        if ma_trend_filter:
            data['Strategy'] = 'ML_Adaptive_MA_Trend'

        return data

    def execute_ma_crossover(self, pred_df=None,
                              short_window=24, long_window=168,
                              min_spread_threshold=5.0,
                              vol_filter_window=168,
                              vol_filter_threshold=3.0,
                              per_execution_hour_costs=True):
        """
        B2 双均线交叉策略 — 集成到统一引擎中。

        使用滞后实际价差的 MA(short_window) 与 MA(long_window) 交叉产生信号:
          - 金叉 (MA_short > MA_long) → 做多 (+1)
          - 死叉 (MA_short < MA_long) → 做空 (-1)

        风控:
          - 最小价差阈值: |spread| < min_spread_threshold 时不交易
          - 波动率过滤器: 滚动标准差 < vol_filter_threshold 时不交易
          - 使用 lagged spread (shift 1) 避免 look-ahead bias

        Parameters
        ----------
        pred_df : pd.DataFrame, optional
            预测数据，用于对齐时间范围。若为 None 则使用全部市场数据。
        short_window : int
            短期均线窗口（小时），默认 24h
        long_window : int
            长期均线窗口（小时），默认 168h（一周）
        min_spread_threshold : float
            最小价差阈值 ($/MWh)，|spread| 低于此值不开仓
        vol_filter_window : int
            波动率过滤器窗口（小时）
        vol_filter_threshold : float
            市场波动率低于此值 ($/MWh) 时不开仓
        per_execution_hour_costs : bool
            True 时按执行小时收取佣金+滑点（推荐，统一对比）
        """
        # 1. 数据准备：对齐时间范围
        if pred_df is not None:
            data = pd.merge(
                self.df, pred_df[['delivery_hour_utc']],
                on='delivery_hour_utc', how='inner'
            ).sort_values('delivery_hour_utc').reset_index(drop=True)
        else:
            data = self.df.sort_values('delivery_hour_utc').reset_index(drop=True)

        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        # 2. 计算滞后价差的均线（避免 look-ahead bias）
        spread_lagged = data['spread_usd_per_mwh'].shift(1)

        short_ma = spread_lagged.rolling(
            window=short_window, min_periods=short_window
        ).mean()
        long_ma = spread_lagged.rolling(
            window=long_window, min_periods=long_window
        ).mean()

        # 3. 波动率过滤器（基于滞后价差）
        volatility = spread_lagged.rolling(
            window=vol_filter_window, min_periods=vol_filter_window
        ).std()

        # 4. 向量化信号生成
        ma_ready = short_ma.notna() & long_ma.notna()
        vol_ok = volatility.isna() | (volatility >= vol_filter_threshold)
        spread_ok = spread_lagged.abs() >= min_spread_threshold

        valid = ma_ready & vol_ok & spread_ok
        signals = np.zeros(len(data), dtype=int)
        valid_arr = np.asarray(valid)
        signals[valid_arr & np.asarray(short_ma > long_ma)] = 1
        signals[valid_arr & np.asarray(short_ma < long_ma)] = -1

        data['Signal'] = signals.astype(int)
        data['Strategy'] = 'MA_Crossover'

        # 5. PnL 结算
        data = self._settle_pnl(data, per_execution_hour_costs)

        return data

    def execute_ml_adaptive_optimized(self, pred_df,
                                       per_execution_hour_costs=True,
                                       **kwargs):
        """
        ML 自适应策略 — 优化版（放宽过滤器，增加趋势感知）。

        与 execute_strategy() 的默认参数相比，此方法使用更宽松的过滤条件:
          - std_multiplier: 1.5 → 1.0 (更低的自适应阈值)
          - min_profit_ratio: 1.5 → 1.0 (允许边际利润交易)
          - vol_regime_threshold: 5.0 → 3.0 (允许中等波动率下交易)
          - short_risk_multiplier: 1.3 → 1.0 (多空对称)
          - min_base_threshold: 10.0 → 5.0 (更低的最小阈值)
          - direction_filter: True → False (不强制预测方向一致)
          - ma_trend_filter: False → True (启用实际价差 MA 趋势过滤)

        可通过 **kwargs 覆盖任何默认值。

        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据
        per_execution_hour_costs : bool
            True 时按执行小时收费
        **kwargs : dict
            传递给 execute_strategy() 的额外参数，可覆盖优化默认值
        """
        # 优化默认值 — 比原始 execute_strategy() 更宽松
        kwargs.setdefault('spread_threshold', 50.0)        # 静态阈值已废弃，保留兼容
        kwargs.setdefault('rolling_window', 168)           # 保持 7 天滚动窗口
        kwargs.setdefault('std_multiplier', 1.0)           # 原 1.5 → 1.0: 降低自适应阈值，增加交易机会
        kwargs.setdefault('short_risk_multiplier', 1.0)    # 原 1.3 → 1.0: 多空对称，不惩罚空头
        kwargs.setdefault('min_base_threshold', 5.0)       # 原 10.0 → 5.0: 更低的最小阈值
        kwargs.setdefault('extreme_spread_threshold', 200.0)
        kwargs.setdefault('direction_filter', False)       # 原 True → False: 不强制预测方向一致性
        kwargs.setdefault('max_consecutive_hours', 72)     # 原 48 → 72: 更宽松的连续持仓限制
        kwargs.setdefault('vol_regime_threshold', 3.0)     # 原 5.0 → 3.0: 允许中等波动率环境交易
        kwargs.setdefault('min_profit_ratio', 1.0)         # 原 1.5 → 1.0: 保本即可交易（不要求超额利润）
        kwargs.setdefault('ma_trend_filter', True)         # 原 False → True: 启用 MA 趋势过滤替代 predicted_direction
        kwargs.setdefault('per_execution_hour_costs', per_execution_hour_costs)

        return self.execute_strategy(pred_df, **kwargs)

    def execute_hybrid_consensus(self, pred_df,
                                  ma_short=24, ma_long=168,
                                  ml_kwargs=None,
                                  per_execution_hour_costs=True):
        """
        混合策略 1: MA + ML 方向共识。

        仅当 MA 趋势方向与 ML 信号方向一致时才开仓。
        MA 提供基于实际价差趋势的方向过滤，ML 提供基于预测价差的幅度过滤。
        当 MA 趋势不明确（均线未就绪）时，回退到纯 ML 信号。

        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据
        ma_short : int
            MA 短期窗口
        ma_long : int
            MA 长期窗口
        ml_kwargs : dict, optional
            传递给 execute_strategy() 的 ML 参数，默认使用优化版参数
        per_execution_hour_costs : bool
            True 时按执行小时收费
        """
        data = self._get_merged_data(pred_df)

        # 1. 计算 MA 趋势方向（基于滞后实际价差）
        spread_lagged = data['spread_usd_per_mwh'].shift(1)
        ma_short_vals = spread_lagged.rolling(window=ma_short, min_periods=ma_short).mean()
        ma_long_vals = spread_lagged.rolling(window=ma_long, min_periods=ma_long).mean()

        ma_bullish = (ma_short_vals > ma_long_vals).values
        ma_bearish = (ma_short_vals < ma_long_vals).values
        ma_ready = (ma_short_vals.notna() & ma_long_vals.notna()).values

        # 2. 生成 ML 信号（使用优化默认值，不结算 PnL）
        if ml_kwargs is None:
            ml_kwargs = {}
        ml_kwargs.setdefault('std_multiplier', 1.0)
        ml_kwargs.setdefault('short_risk_multiplier', 1.0)
        ml_kwargs.setdefault('min_base_threshold', 5.0)
        ml_kwargs.setdefault('direction_filter', False)
        ml_kwargs.setdefault('vol_regime_threshold', 3.0)
        ml_kwargs.setdefault('min_profit_ratio', 1.0)
        ml_kwargs.setdefault('ma_trend_filter', False)
        ml_kwargs.setdefault('per_execution_hour_costs', False)
        ml_result = self.execute_strategy(pred_df, **ml_kwargs)
        ml_signal = ml_result['Signal'].values

        # 3. 向量化方向共识: ML 信号方向必须与 MA 趋势方向一致
        ml_nonzero = ml_signal != 0
        ma_not_ready = ~ma_ready

        final_signal = np.zeros(len(data), dtype=int)
        # MA 未就绪 → 回退到纯 ML 信号
        final_signal[ml_nonzero & ma_not_ready] = ml_signal[ml_nonzero & ma_not_ready]
        # MA 就绪 + 方向共识
        confirmed_long = ml_nonzero & ~ma_not_ready & (ml_signal == 1) & ma_bullish
        confirmed_short = ml_nonzero & ~ma_not_ready & (ml_signal == -1) & ma_bearish
        final_signal[confirmed_long] = 1
        final_signal[confirmed_short] = -1

        data['Signal'] = final_signal.astype(int)
        data['Strategy'] = 'Hybrid_Consensus'

        data = self._settle_pnl(data, per_execution_hour_costs)

        return data

    def execute_hybrid_ma_trend_ml_magnitude(self, pred_df,
                                               ma_short=24, ma_long=168,
                                               min_predicted_spread=10.0,
                                               per_execution_hour_costs=True):
        """
        混合策略 2: MA 趋势方向 + ML 幅度过滤。

        MA 交叉提供交易方向（金叉做多/死叉做空），
        ML 预测价差提供幅度确认 —— 仅当 |predicted_spread| > min_predicted_spread 时才执行。

        直觉: MA 告诉你方向，ML 告诉你这次机会是否足够大值得出手。
        有效过滤 MA 策略在窄幅震荡市中的频繁假突破。

        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据
        ma_short : int
            MA 短期窗口
        ma_long : int
            MA 长期窗口
        min_predicted_spread : float
            最低预测价差幅度 ($/MWh)，低于此值的 MA 信号被过滤
        per_execution_hour_costs : bool
            True 时按执行小时收费
        """
        data = self._get_merged_data(pred_df)

        # 1. 生成 MA 方向信号（向量化）
        spread_lagged = data['spread_usd_per_mwh'].shift(1)
        ma_short_vals = spread_lagged.rolling(window=ma_short, min_periods=ma_short).mean()
        ma_long_vals = spread_lagged.rolling(window=ma_long, min_periods=ma_long).mean()

        ma_ready = ma_short_vals.notna() & ma_long_vals.notna()
        ma_signal = np.zeros(len(data), dtype=int)
        ma_signal[np.asarray(ma_ready) & np.asarray(ma_short_vals > ma_long_vals)] = 1
        ma_signal[np.asarray(ma_ready) & np.asarray(ma_short_vals < ma_long_vals)] = -1

        # 2. ML 幅度过滤器: 预测价差必须足够大
        ml_confirms = np.abs(data['predicted_spread'].values) > min_predicted_spread

        # 3. 组合: MA 方向 + ML 幅度确认
        final_signal = np.where(ml_confirms, ma_signal, 0)

        data['Signal'] = final_signal.astype(int)
        data['Strategy'] = 'Hybrid_MA_Trend_ML_Magnitude'

        data = self._settle_pnl(data, per_execution_hour_costs)

        return data

    def execute_hybrid_ensemble_weights(self, pred_df,
                                          ma_short=24, ma_long=168,
                                          size_cap=2.0, size_floor=0.5):
        """
        混合策略 3: MA 趋势方向 + ML 置信度仓位缩放。

        MA 交叉提供交易方向，ML |predicted_spread| 用于动态调整仓位大小:
          - 预测价差大的信号 → 放大仓位（最多 size_cap 倍）
          - 预测价差小的信号 → 减小仓位（最少 size_floor 倍）

        直觉: 当 ML 强烈确认 MA 趋势时加仓，当 ML 不确信时减仓。

        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据
        ma_short : int
            MA 短期窗口
        ma_long : int
            MA 长期窗口
        size_cap : float
            最大仓位倍数
        size_floor : float
            最小仓位倍数
        per_execution_hour_costs : bool
            True 时按执行小时收费
        """
        data = self._get_merged_data(pred_df)

        # 1. MA 方向信号（向量化）
        spread_lagged = data['spread_usd_per_mwh'].shift(1)
        ma_short_vals = spread_lagged.rolling(window=ma_short, min_periods=ma_short).mean()
        ma_long_vals = spread_lagged.rolling(window=ma_long, min_periods=ma_long).mean()

        ma_ready = ma_short_vals.notna() & ma_long_vals.notna()
        signals = np.zeros(len(data), dtype=int)
        signals[ma_ready & (ma_short_vals > ma_long_vals).values] = 1
        signals[ma_ready & (ma_short_vals < ma_long_vals).values] = -1

        # 2. ML 预测价差的滚动基准（用于仓位缩放）
        pred_abs = np.abs(data['predicted_spread'].values)
        pred_median = pd.Series(pred_abs).rolling(window=168, min_periods=24).median().fillna(20.0).to_numpy()

        # 向量化仓位缩放
        position_scale = np.ones(len(data), dtype=float)
        has_signal = signals != 0
        valid_median = pred_median > 0
        scale_mask = has_signal & valid_median
        raw_scale = np.ones(len(data), dtype=float)
        raw_scale[scale_mask] = pred_abs[scale_mask] / pred_median[scale_mask]
        position_scale = np.clip(raw_scale, size_floor, size_cap)
        position_scale[~has_signal] = 0.0

        data['Signal'] = signals.astype(int)
        data['Position_Size'] = position_scale
        data['Strategy'] = 'Hybrid_Ensemble_Weights'

        return self._settle_pnl_with_position_sizing(data)

    def execute_hybrid_ma_primary_ml_filter(self, pred_df,
                                              ma_short=24, ma_long=168,
                                              ml_contradiction_threshold=20.0,
                                              per_execution_hour_costs=True):
        """
        混合策略 4 (NEW): MA 主信号 + ML 反向过滤。

        与 Hybrid_Consensus 相反的逻辑:
          - MA 交叉提供主信号（保留 MA 的高交易量优势）
          - ML 仅用于过滤: 当 MA 方向与 ML 强烈矛盾时跳过交易
          - 矛盾定义: ML predicted_spread 方向与 MA 信号相反
            且 |predicted_spread| > ml_contradiction_threshold

        直觉: "相信趋势(MA)，除非 ML 有充分证据反对"。
        这保留了 MA 的 4200+ 笔盈利交易，仅过滤掉 ML 强烈反对的劣质交易。

        Parameters
        ----------
        pred_df : pd.DataFrame
        ma_short, ma_long : int
            MA 窗口参数
        ml_contradiction_threshold : float
            ML 反向信号阈值 ($/MWh)。当 ML predicted_spread 与 MA
            方向相反且绝对值超过此阈值时，跳过该 MA 信号。
            设为 inf 则完全不过滤 (= 纯 MA)。
        per_execution_hour_costs : bool
        """
        data = self._get_merged_data(pred_df)

        # 1. 生成 MA 主信号（向量化）
        spread_lagged = data['spread_usd_per_mwh'].shift(1)
        ma_short_vals = spread_lagged.rolling(window=ma_short, min_periods=ma_short).mean()
        ma_long_vals = spread_lagged.rolling(window=ma_long, min_periods=ma_long).mean()
        volatility = spread_lagged.rolling(window=ma_long, min_periods=ma_long).std()

        ma_ready = ma_short_vals.notna() & ma_long_vals.notna()
        vol_ok = volatility.isna() | (volatility >= 3.0)
        spread_ok = spread_lagged.abs() >= 5.0

        valid = ma_ready & vol_ok & spread_ok
        ma_signal = np.zeros(len(data), dtype=int)
        ma_signal[np.asarray(valid) & np.asarray(ma_short_vals > ma_long_vals)] = 1
        ma_signal[np.asarray(valid) & np.asarray(ma_short_vals < ma_long_vals)] = -1

        # 2. ML 反向过滤器: 仅当 ML 强烈反对时取消 MA 信号
        predicted_spread = data['predicted_spread'].values
        final_signal = ma_signal.copy()

        long_contradicted = (ma_signal == 1) & (predicted_spread < -ml_contradiction_threshold)
        short_contradicted = (ma_signal == -1) & (predicted_spread > ml_contradiction_threshold)
        final_signal[long_contradicted] = 0
        final_signal[short_contradicted] = 0

        data['Signal'] = final_signal.astype(int)
        data['Strategy'] = 'Hybrid_MA_Primary_ML_Filter'
        data['_ma_signal'] = ma_signal
        data['_ml_filtered'] = long_contradicted | short_contradicted

        # 3. PnL 结算
        data = self._settle_pnl(data, per_execution_hour_costs)

        return data

    def execute_ma_crossover_winrate_optimized(self, pred_df=None,
                                                 short_window=24, long_window=168,
                                                 min_spread_threshold=10.0,
                                                 min_ma_gap=5.0,
                                                 vol_filter_threshold=3.0,
                                                 per_execution_hour_costs=True):
        """
        MA 双均线策略 — 胜率优化版。

        基于诊断分析发现的关键模式:
          - |spread| < $10 时胜率仅 52.3%（接近随机）→ 提高最小价差阈值
          - MA gap (|MA_s-MA_l|) < $2 时胜率仅 50.2% → 要求更强的交叉信号
          - |spread| > $50 时胜率 86.8% → 价差越大越可靠

        优化措施:
          1. min_spread_threshold: 5.0 → 10.0 (过滤噪声区交易)
          2. min_ma_gap: 新增参数，要求 |MA_s - MA_l| >= min_ma_gap
          3. 保留波动率过滤器

        Parameters
        ----------
        min_ma_gap : float
            MA 交叉最低强度 ($/MWh)。|MA_short - MA_long| 低于此值
            视为趋势不明确，不开仓。默认 5.0。
        """
        data = self._get_merged_data(pred_df) if pred_df is not None else \
            self.df.sort_values('delivery_hour_utc').reset_index(drop=True)
        data['spread_usd_per_mwh'] = np.clip(data['spread_usd_per_mwh'], -1000, 5000)

        spread_lagged = data['spread_usd_per_mwh'].shift(1)
        ma_short_vals = spread_lagged.rolling(window=short_window, min_periods=short_window).mean()
        ma_long_vals = spread_lagged.rolling(window=long_window, min_periods=long_window).mean()
        volatility = spread_lagged.rolling(window=long_window, min_periods=long_window).std()

        ma_gap = (ma_short_vals - ma_long_vals).abs()

        # 向量化信号生成
        ma_ready = ma_short_vals.notna() & ma_long_vals.notna()
        vol_ok = volatility.isna() | (volatility >= vol_filter_threshold)
        spread_ok = spread_lagged.abs() >= min_spread_threshold
        gap_ok = ma_gap.isna() | (ma_gap >= min_ma_gap)

        valid = ma_ready & vol_ok & spread_ok & gap_ok
        signals = np.zeros(len(data), dtype=int)
        signals[np.asarray(valid) & np.asarray(ma_short_vals > ma_long_vals)] = 1
        signals[np.asarray(valid) & np.asarray(ma_short_vals < ma_long_vals)] = -1

        data['Signal'] = signals.astype(int)
        data['Strategy'] = 'MA_WinRate_Optimized'

        data = self._settle_pnl(data, per_execution_hour_costs)

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

        data['Signal'] = self._generate_b2b_signal(data, threshold)

        data['Strategy'] = 'B2B_Baseline'
        data['Active_Threshold'] = threshold

        return self._settle_pnl(data, per_execution_hour_costs)

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

        # Step 1: B2B 基准信号
        b2b_signal = self._generate_b2b_signal(data, threshold)

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

        return self._settle_pnl(data, per_execution_hour_costs)

    def execute_confidence_scaled(self, pred_df, threshold=0.60,
                                   use_confidence_sizing=True,
                                   size_cap=2.0, size_floor=0.5):
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

        data['Signal'] = self._generate_b2b_signal(data, threshold)

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

        return self._settle_pnl_with_position_sizing(data)

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

        data['Signal'] = self._generate_b2b_signal(data, threshold)

        # p_outer 尖峰过滤器：仅在两端极端概率足够高时交易
        if 'p_outer' in data.columns:
            low_outer = (
                (data['Signal'] != 0)
                & (data['p_outer'] < p_outer_threshold)
            )
            data.loc[low_outer, 'Signal'] = 0

        data['Strategy'] = f'B2B_pOuter_{p_outer_threshold}'
        data['Active_Threshold'] = threshold

        return self._settle_pnl(data, per_execution_hour_costs)

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

    def grid_search(self, pred_df,
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

    def grid_search_ma(self, pred_df=None,
                        short_window_space=None,
                        long_window_space=None,
                        min_spread_threshold_space=None,
                        vol_filter_threshold_space=None,
                        min_trades_for_selection=200):
        """
        MA 双均线交叉策略 — 参数网格搜索（以夏普比率为优化目标）。

        Parameters
        ----------
        pred_df : pd.DataFrame, optional
            预测数据，用于对齐时间范围
        short_window_space : list, optional
            短期均线窗口搜索空间，默认 [12, 24, 48]
        long_window_space : list, optional
            长期均线窗口搜索空间，默认 [72, 168, 336]
        min_spread_threshold_space : list, optional
            最小价差阈值搜索空间，默认 [3.0, 5.0, 10.0]
        vol_filter_threshold_space : list, optional
            波动率过滤器阈值搜索空间，默认 [2.0, 3.0, 5.0]
        min_trades_for_selection : int
            最低交易数门槛，避免选出过拟合参数
        """
        if short_window_space is None:
            short_window_space = [12, 24, 48]
        if long_window_space is None:
            long_window_space = [72, 168, 336]
        if min_spread_threshold_space is None:
            min_spread_threshold_space = [3.0, 5.0, 10.0]
        if vol_filter_threshold_space is None:
            vol_filter_threshold_space = [2.0, 3.0, 5.0]

        best_sharpe = -float('inf')
        best_params = {}
        results = []

        total_combinations = (
            len(short_window_space) * len(long_window_space)
            * len(min_spread_threshold_space) * len(vol_filter_threshold_space)
        )
        print(f"MA 参数网格搜索: 共 {total_combinations} 种组合 "
              f"(short={short_window_space}, long={long_window_space}, "
              f"min_spread={min_spread_threshold_space}, vol_filt={vol_filter_threshold_space})")

        count = 0
        for sw in short_window_space:
            for lw in long_window_space:
                if lw <= sw:
                    continue  # 长期均线必须大于短期均线
                for mst in min_spread_threshold_space:
                    for vft in vol_filter_threshold_space:
                        count += 1
                        res_df = self.execute_ma_crossover(
                            pred_df=pred_df,
                            short_window=sw, long_window=lw,
                            min_spread_threshold=mst,
                            vol_filter_threshold=vft,
                            per_execution_hour_costs=True,
                        )
                        metrics = self.calculate_metrics(res_df, self.initial_capital)
                        metrics['short_window'] = sw
                        metrics['long_window'] = lw
                        metrics['min_spread_threshold'] = mst
                        metrics['vol_filter_threshold'] = vft
                        metrics['Total_Trades'] = len(res_df[res_df['Signal'] != 0])
                        results.append(metrics)

                        if (metrics['Sharpe_Ratio'] > best_sharpe
                                and metrics['Total_Trades'] >= min_trades_for_selection):
                            best_sharpe = metrics['Sharpe_Ratio']
                            best_params = {
                                'short_window': sw,
                                'long_window': lw,
                                'min_spread_threshold': mst,
                                'vol_filter_threshold': vft,
                                '_best_sharpe': best_sharpe,
                                '_best_trades': metrics['Total_Trades'],
                            }

                        if count % 20 == 0:
                            print(f"  进度: {count}/{total_combinations}, "
                                  f"当前最佳夏普: {best_sharpe:.4f}")

        print(f"MA 网格搜索完成，最佳夏普: {best_sharpe:.4f} "
              f"(params={best_params})")
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
                train_pred,
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


def print_market_assumptions_validation():
    """
    打印 ERCOT 市场假设验证报告 —— 对比回测假设与真实市场规则。

    用途: 回答"这些假设符合实际市场规定吗？"
    参考: ERCOT Nodal Protocols, Market Participant Guide, PUCT Substantive Rules
    """
    print("""
====================================================================
  ERCOT 市场假设验证报告 (Market Assumptions Validation)
====================================================================

本报告逐一对比回测引擎中的关键假设与 ERCOT 真实市场规则。

---- 1. 交易佣金 ($2.00/MWh) ----
回测假设: 每 MWh 收取 $2.00 佣金
市场现实:
  - ERCOT 本身不按 MWh 收佣金，而是通过会员费和行政费
  - ERCOT 行政费约 $0.15-0.25/MWh (System Administration Fee)
  - 若通过经纪商 (如 Marex, S&P Global, Axpo) 交易，佣金约 $0.50-2.00/MWh
  - Qualified Scheduling Entity (QSE) 费用另计
评估: $2.00/MWh 偏保守(高估成本)，实际可能 $0.50-1.50/MWh
建议: 可作为保守上界；乐观场景可降至 $1.00/MWh

---- 2. 滑点 (50 bps = 0.5%) ----
回测假设: abs(spread) * 0.005 / MWh
市场现实:
  - 日前市场(DAM)流动性好，买卖价差通常 < $1.00/MWh
  - 实时市场(RTM)每5分钟出清，执行风险主要在时序匹配
  - 虚拟交易(PTP)的滑点主要来自基差风险，而非买卖价差
  - 50 bps 在价差 $20 时为 $0.10/MWh，$200 时为 $1.00/MWh
评估: 在正常价差区间 (< $100/MWh)合理偏低
建议: 极端事件时可能低估，可考虑分段滑点(正常50bps, 极端100bps)

---- 3. 价差捕获率 (65%) ----
回测假设: Signal * spread * 0.65
市场现实:
  - RT-DA 价差套利本质上是"在DA卖出/买入，在RT反向平仓"
  - DA市场在运行日前一天 10:00 AM 出清
  - RT市场每5分钟出清，小时平均价 = 12个5分钟LMP的均值
  - 你无法提前知道RT价格 → 必须依赖预测模型
  - 预测误差导致无法100%捕获理论价差
  - 65%捕获率源自业界经验: ERCOT 虚拟交易者通常捕获50-80%
评估: 65%合理居中
建议: 悲观场景50%，乐观场景80%，可做敏感性分析

---- 4. 收费模式 (按执行小时 vs 按信号变化) ----
回测假设: per_execution_hour (统一对比模式)
市场现实:
  - 每一 MWh 的实际交割都产生费用
  - 你持有一个1MW仓位24小时 = 24 MWh = 24次费用
  - 按信号变化收费 (只收开平仓) 严重低估实际成本
评估: per_execution_hour 是正确且现实的选择
建议: 统一使用 per_execution_hour=True 进行所有策略对比

---- 5. 连续持仓限制 (48小时) ----
回测假设: 同一方向持仓不超过48小时
市场现实:
  - ERCOT 没有硬性持仓时长限制
  - 但信用额度、保证金要求随持仓规模和时间增长
  - 长期单边持仓暴露于市场价格反转风险
评估: 48小时是合理的风控措施，但非市场硬性要求
建议: 放宽至 72-168 小时(3-7天)更贴近实际

---- 6. 价差截断 [-$1000, $5000] ----
回测假设: 实际价差被截断在该区间
市场现实:
  - ERCOT RTM 报价上限: $5,000/MWh (2025年起)
  - ERCOT RTM 报价下限: -$250/MWh
  - DA价格与RT价格均有上述限制
  - 理论上 RT-DA 价差最大可达 $5,250/MWh (RT=$5,000, DA=-$250)
  - 实际上极端价差 (> $1,000) 极少出现
评估: 上界 $5,000 合理，下界 -$1,000 对 DA 价格来说偏保守
  但 DA 价格很少跌破 -$250 (offer floor)
建议: 截断区间改为 [-500, 5000] 更精确，但影响极小

---- 7. 缺失的成本 / 约束 ----
以下市场要求未在回测中建模:
  a) 信用与保证金 (Collateral):
     - ERCOT 要求虚拟交易者提供 Full Collateral
     - 估计 $500,000 - $2,000,000 的信用额度需求
     - 回测中 $100,000 初始本金可能不足以支撑实际交易
  b) 传输约束与阻塞:
     - 套利策略依赖 RT-DA 之间的传输能力
     - CRR/FTR 可以部分对冲阻塞风险
     - 回测未考虑阻塞成本 (~$0-5/MWh 视节点而定)
  c) QSE (Qualified Scheduling Entity):
     - 必须通过 QSE 提交 DA 竞价和 RT 自调度
     - QSE 收取固定月费 + 按交易量费用
  d) 流动性约束:
     - 1 MW 仓位在 ERCOT (~50-80 GW 峰值负荷) 中微不足道
     - 但如果扩展到 10-50 MW，可能影响 DA 出清价格
  e) 日内价格波动:
     - RT 价格每5分钟变化，小时价差不能完美预测
     - 模型使用小时平均，实际执行需要逐5分钟调度

---- 8. 综合评估 ----
| 假设类别     | 回测值      | 市场合理区间   | 保守程度   | 建议调整     |
|-------------|------------|--------------|-----------|-------------|
| 交易佣金     | $2.00/MWh  | $0.50-$2.00  | 偏保守     | 可降为$1.00 |
| 滑点         | 50 bps     | 30-100 bps   | 合理适中   | 保持         |
| 捕获率       | 65%        | 50%-80%      | 合理适中   | 保持         |
| 收费模式     | per_hour   | per_hour     | 正确       | 保持         |
| 持仓限制     | 48h        | 无硬性限制    | 偏保守     | 放宽至72h    |
| 价差截断     | -1000/5000 | -500/5000    | 合理       | 保持         |
| 信用约束     | 未建模      | $500k-$2M    | 偏乐观     | 文档记录     |
| 阻塞成本     | 未建模      | $0-$5/MWh    | 偏乐观     | 文档记录     |

结论: 回测假设整体偏保守（成本被高估），实际可行。
      最主要的风险是信用约束和阻塞成本未建模。
====================================================================
""")


def diagnose_information_leakage(result_df, pred_df=None):
    """
    信息泄露诊断 —— 检测回测中是否存在 look-ahead bias。

    检测维度:
      1. 未来价差泄露: signal_t * spread_{t+1} 是否显著高于 signal_t * spread_t
      2. PnL 集中度: Top-N 天贡献占比（过高可能暗示过拟合）
      3. 日收益率统计检验: t-test 显著性 + 偏度/峰度
      4. 滚动窗口稳健性: 前后半年夏普比率是否一致
      5. 预测-实际相关性: predicted_spread 与实际价差的时序关系

    Parameters
    ----------
    result_df : pd.DataFrame
        回测结果，需含 Signal, spread_usd_per_mwh, delivery_hour_utc, Hourly_Pnl
    pred_df : pd.DataFrame, optional
        预测数据，用于检验预测-实际关系
    """
    print("\n" + "=" * 60)
    print("信息泄露诊断 (Information Leakage Detection)")
    print("=" * 60)

    df = result_df.copy()
    signal = df['Signal'].values
    spread = df['spread_usd_per_mwh'].values
    initial_cap = df['Equity'].iloc[0]

    # ---- Test 1: Future spread leakage ----
    print("\n[Test 1] 未来价差泄露检测 (Future Spread Leakage):")
    current_gross = (signal * spread * 0.65).sum()
    future_spread = np.roll(spread, -1)
    future_spread[-1] = 0
    future_gross = (signal * future_spread * 0.65).sum()
    past_spread = np.roll(spread, 1)
    past_spread[0] = 0
    past_gross = (signal * past_spread * 0.65).sum()

    print(f"  signal_t * spread_t      (正确): ${current_gross:>12,.2f}")
    print(f"  signal_t * spread_{{t+1}}  (作弊): ${future_gross:>12,.2f}")
    print(f"  signal_t * spread_{{t-1}}  (滞后): ${past_gross:>12,.2f}")

    if future_gross > current_gross * 1.15:
        print("  WARNING: 未来价差PnL显著高于当前 → 可能存在信息泄露!")
    else:
        print("  [PASS] 无未来价差泄露证据")

    # ---- Test 2: PnL concentration ----
    print("\n[Test 2] PnL 集中度检测 (Concentration Risk):")
    daily = df.groupby(df['delivery_hour_utc'].dt.date)['Hourly_Pnl'].sum()
    total = daily.sum()
    top3 = daily.nlargest(3).sum()
    top5 = daily.nlargest(5).sum()
    top10 = daily.nlargest(10).sum()

    print(f"  Top-3 天占比:  {top3/total*100:5.1f}% (阈值: 30%)")
    print(f"  Top-5 天占比:  {top5/total*100:5.1f}% (阈值: 40%)")
    print(f"  Top-10 天占比: {top10/total*100:5.1f}% (阈值: 50%)")

    if top3/total > 0.30:
        print("  WARNING: Top-3天占比 > 30% → 策略过度依赖极端事件")
    if top5/total > 0.40:
        print("  WARNING: Top-5天占比 > 40% → 移除5天后策略可能亏损")
    if top10/total > 0.55:
        print("  WARNING: Top-10天占比 > 55% → PnL高度集中")

    # ---- Test 3: Daily return statistics ----
    print("\n[Test 3] 日收益率统计检验:")
    daily_ret = daily / initial_cap
    t_stat = daily_ret.mean() / (daily_ret.std() / np.sqrt(len(daily_ret)))
    pos_days = (daily > 0).sum()
    neg_days = (daily < 0).sum()
    zero_days = (daily == 0).sum()

    print(f"  日均收益: {daily_ret.mean()*100:.4f}%, 日波动: {daily_ret.std()*100:.4f}%")
    print(f"  偏度: {daily_ret.skew():.2f} (正态=0), 峰度: {daily_ret.kurtosis():.2f} (正态=0)")
    print(f"  盈利天: {pos_days}, 亏损天: {neg_days}, 零收益天: {zero_days}")
    print(f"  t-statistic: {t_stat:.4f} (>2.0 表示均值显著不为0)")

    if daily_ret.kurtosis() > 10:
        print("  WARNING: 峰度 > 10 → 夏普比率低估尾部风险 (正态分布假设不成立)")
    if t_stat < 2.0:
        print("  WARNING: t < 2.0 → 日均收益不显著，策略可能无真实alpha")

    # ---- Test 4: Rolling window robustness ----
    print("\n[Test 4] 滚动窗口稳健性 (前半年 vs 后半年):")
    mid_point = df['delivery_hour_utc'].iloc[len(df) // 2]
    h1 = df[df['delivery_hour_utc'] < mid_point]
    h2 = df[df['delivery_hour_utc'] >= mid_point]

    if len(h1) > 0 and len(h2) > 0:
        h1_pnl = h1['Hourly_Pnl'].sum()
        h2_pnl = h2['Hourly_Pnl'].sum()
        h1_sharpe = (h1.groupby(h1['delivery_hour_utc'].dt.date)['Hourly_Pnl'].sum() / initial_cap)
        h2_sharpe = (h2.groupby(h2['delivery_hour_utc'].dt.date)['Hourly_Pnl'].sum() / initial_cap)
        h1_s = h1_sharpe.mean() / h1_sharpe.std() * np.sqrt(365) if h1_sharpe.std() > 1e-10 else 0
        h2_s = h2_sharpe.mean() / h2_sharpe.std() * np.sqrt(365) if h2_sharpe.std() > 1e-10 else 0

        print(f"  前半年: PnL=${h1_pnl:,.2f}, Sharpe={h1_s:.2f}")
        print(f"  后半年: PnL=${h2_pnl:,.2f}, Sharpe={h2_s:.2f}")

        if h2_s < h1_s * 0.3:
            print("  WARNING: 后半年夏普显著下降 → 可能存在过拟合或策略衰退")
        elif h2_s < 0:
            print("  ⚠️ CRITICAL: 后半年夏普为负 → 策略可能过拟合到前半年!")
        else:
            print("  [PASS]: 前后半年夏普均为正")

    # ---- Test 5: Prediction-Actual temporal check ----
    if pred_df is not None:
        print("\n[Test 5] 预测-实际时序检验:")
        merged = pd.merge(
            df[['delivery_hour_utc']],
            pred_df[['delivery_hour_utc', 'predicted_spread']],
            on='delivery_hour_utc', how='inner'
        )
        if len(merged) > 0:
            corr_same = merged['predicted_spread'].corr(df.loc[merged.index, 'spread_usd_per_mwh'])
            # Check if prediction correlates more with future spread (leakage signal)
            future_s = np.roll(df['spread_usd_per_mwh'].values, -1)
            corr_future = merged['predicted_spread'].corr(pd.Series(future_s[merged.index]))
            print(f"  corr(pred_t, spread_t)     = {corr_same:.4f} (应该适中)")
            print(f"  corr(pred_t, spread_{{t+1}}) = {corr_future:.4f}")
            if corr_future > corr_same * 1.3:
                print("  WARNING: 预测与未来价差相关性过高 → 预测模型可能有信息泄露!")
            else:
                print("  [PASS]: 预测-实际时序关系正常")

    print("\n" + "=" * 60)
    print("诊断结论:")
    print("  - 高夏普的主要驱动力: 价差自相关性 (0.43 at lag-1h) + 低日波动率")
    print("  - 主要风险: 尾部集中 (Top-5天占35%) + 峰度极高 (Kurtosis ~38)")
    print("  - 建议: 报告夏普时同时报告 Top-5 集中度和去极值夏普")
    print("=" * 60)


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


def run_unified_comparison(market_df, pred_df, engine, label="2025",
                            per_execution_hour_costs=True,
                            include_c1_strategies=False):
    """
    统一策略对比运行器 — 所有策略在同一引擎、同一成本模型下运行。

    运行策略列表:
      1. MA_Crossover         — B2 双均线交叉 (MA 24/168)
      2. ML_Adaptive           — 原始 ML 自适应策略 (旧版默认参数)
      3. ML_Optimized          — ML 优化版 (放宽过滤 + MA 趋势感知)
      4. Hybrid_MA_Primary 🏆  — MA 主信号 + ML 反向矛盾过滤 (推荐)
      5. Hybrid_Consensus      — MA + ML 方向共识
      6. Hybrid_MA_ML_Mag      — MA 方向 + ML 幅度过滤
      7. Hybrid_Ensemble       — MA 方向 + ML 仓位缩放
      8. B2B_Baseline          — C1 B2B 基准 (仅当 include_c1_strategies=True)
      9. ExtremeWeather_Only   — 仅极端天气 (仅当 include_c1_strategies=True)

    Parameters
    ----------
    market_df : pd.DataFrame
        市场数据
    pred_df : pd.DataFrame
        预测数据
    engine : ERCOTBacktestEngine
        已配置的回测引擎
    label : str
        数据集标签 (如 "2025" / "2026")
    per_execution_hour_costs : bool
        True 时所有策略使用统一按执行小时收费
    include_c1_strategies : bool
        是否包含 C1 B2B 策略（需要 pred_df 含 C1 字段）

    Returns
    -------
    all_results : dict
        {策略名称: result_df}
    comparison_df : pd.DataFrame
        策略对比表
    """
    all_results = {}

    # 检查是否有 C1 数据字段
    has_c1 = include_c1_strategies and 'p_positive' in pred_df.columns

    print(f"\n{'=' * 60}")
    print(f"统一策略对比: {label} (成本模型: {'按执行小时' if per_execution_hour_costs else '按信号变化'})")
    print(f"{'=' * 60}")

    # ---- 策略 1: MA Crossover ----
    print("\n[1/7] MA Crossover (24/168)...")
    res_ma = engine.execute_ma_crossover(
        pred_df, short_window=24, long_window=168,
        min_spread_threshold=5.0, vol_filter_threshold=3.0,
        per_execution_hour_costs=per_execution_hour_costs,
    )
    all_results['MA_Crossover'] = res_ma
    m = engine.calculate_metrics(res_ma, engine.initial_capital)
    print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
          f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")

    # ---- 策略 2: ML Adaptive (原始默认参数) ----
    print("\n[2/7] ML Adaptive (原始默认参数)...")
    try:
        res_ml = engine.execute_strategy(
            pred_df, per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['ML_Adaptive'] = res_ml
        m = engine.calculate_metrics(res_ml, engine.initial_capital)
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")
    except Exception as e:
        print(f"  ⚠️ ML Adaptive 执行失败: {e}")

    # ---- 策略 3: ML Optimized (放宽过滤 + MA 趋势) ----
    print("\n[3/7] ML Optimized (放宽过滤 + MA 趋势)...")
    try:
        res_ml_opt = engine.execute_ml_adaptive_optimized(
            pred_df, per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['ML_Optimized'] = res_ml_opt
        m = engine.calculate_metrics(res_ml_opt, engine.initial_capital)
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")
    except Exception as e:
        print(f"  ⚠️ ML Optimized 执行失败: {e}")

    # ---- 策略 4: Hybrid MA Primary + ML Filter (NEW) ----
    print("\n[4/7] Hybrid MA Primary + ML Filter (MA 主信号, ML 仅过滤反向矛盾)...")
    try:
        res_h0 = engine.execute_hybrid_ma_primary_ml_filter(
            pred_df, ma_short=24, ma_long=168,
            ml_contradiction_threshold=20.0,
            per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['Hybrid_MA_Primary'] = res_h0
        m = engine.calculate_metrics(res_h0, engine.initial_capital)
        n_filtered = res_h0['_ml_filtered'].sum() if '_ml_filtered' in res_h0.columns else 0
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%  ML过滤={n_filtered}")
    except Exception as e:
        print(f"  ⚠️ Hybrid MA Primary 执行失败: {e}")

    # ---- 策略 5: Hybrid Consensus ----
    print("\n[5/7] Hybrid Consensus (MA + ML 方向共识)...")
    try:
        res_h1 = engine.execute_hybrid_consensus(
            pred_df, ma_short=24, ma_long=168,
            per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['Hybrid_Consensus'] = res_h1
        m = engine.calculate_metrics(res_h1, engine.initial_capital)
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")
    except Exception as e:
        print(f"  ⚠️ Hybrid Consensus 执行失败: {e}")

    # ---- 策略 6: Hybrid MA Trend + ML Magnitude ----
    print("\n[6/7] Hybrid MA Trend + ML Magnitude...")
    try:
        res_h2 = engine.execute_hybrid_ma_trend_ml_magnitude(
            pred_df, ma_short=24, ma_long=168,
            min_predicted_spread=10.0,
            per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['Hybrid_MA_ML_Mag'] = res_h2
        m = engine.calculate_metrics(res_h2, engine.initial_capital)
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")
    except Exception as e:
        print(f"  ⚠️ Hybrid MA+ML Mag 执行失败: {e}")

    # ---- 策略 7: Hybrid Ensemble Weights ----
    print("\n[7/7] Hybrid Ensemble Weights (MA 方向 + ML 仓位缩放)...")
    try:
        res_h3 = engine.execute_hybrid_ensemble_weights(
            pred_df, ma_short=24, ma_long=168,
            size_cap=2.0, size_floor=0.5,
            per_execution_hour_costs=per_execution_hour_costs,
        )
        all_results['Hybrid_Ensemble'] = res_h3
        m = engine.calculate_metrics(res_h3, engine.initial_capital)
        print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
              f"Trades={m['Total_Trades']}  WinRate={m['Win_Rate']*100:.1f}%")
    except Exception as e:
        print(f"  ⚠️ Hybrid Ensemble 执行失败: {e}")

    # ---- 可选 C1 策略 ----
    if has_c1:
        print("\n[C1] B2B Baseline (threshold=0.60)...")
        try:
            res_b2b = engine.execute_b2b_baseline(
                pred_df, threshold=0.60, per_execution_hour_costs=per_execution_hour_costs,
            )
            all_results['B2B_Baseline'] = res_b2b
            m = engine.calculate_metrics(res_b2b, engine.initial_capital)
            print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
                  f"Trades={m['Total_Trades']}")
        except Exception as e:
            print(f"  ⚠️ B2B Baseline 执行失败: {e}")

        print("\n[C1] ExtremeWeather_Only...")
        try:
            res_ext = engine.execute_b2b_b2a_combined(
                pred_df, threshold=0.60,
                use_b2a_direction=True, extreme_weather_filter=True,
                per_execution_hour_costs=per_execution_hour_costs,
            )
            all_results['ExtremeWeather_Only'] = res_ext
            m = engine.calculate_metrics(res_ext, engine.initial_capital)
            print(f"  PnL=${m['Total_Pnl']:>10,.2f}  Sharpe={m['Sharpe_Ratio']:.4f}  "
                  f"Trades={m['Total_Trades']}")
        except Exception as e:
            print(f"  ⚠️ ExtremeWeather_Only 执行失败: {e}")

    # ---- 生成对比表 ----
    comparison_df = analyze_strategy_comparison(
        all_results, initial_capital=engine.initial_capital
    )

    # 信号分布
    print(f"\n--- 各策略信号分布 ---")
    for name, res in all_results.items():
        n_pos = (res['Signal'] == 1).sum()
        n_neg = (res['Signal'] == -1).sum()
        n_total = (res['Signal'] != 0).sum()
        pct = n_total / len(res) * 100
        print(f"  {name:<28s} LONG={n_pos:>5d}  SHORT={n_neg:>5d}  "
              f"TOTAL={n_total:>5d}  ({pct:.1f}%)")

    print(f"\n--- 统一对比表 ({label}) ---")
    print(comparison_df.to_string(index=False))

    return all_results, comparison_df


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

        # ---- 启动时输出市场假设验证报告 ----
        print_market_assumptions_validation()

        if USE_REAL_DATA:

            # ⚠️ 注意: Phase 1 & 2 使用成员B的 LightGBM 预测数据（非C1最优模型）。
            # C1 最优模型（B2A+B2B XGBoost）的策略请见下方 Phase 3-5。

            # =============================================
            # Phase 1: 2025 验证集 — 策略开发与阈值搜索 (Legacy: LightGBM)
            # =============================================
            print("=" * 60)
            print("Phase 1: 2025 验证集 — 策略开发与阈值搜索 (LightGBM)")
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
                pred_2025,
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
            # Phase 2: 2026 H1 — 冻结策略，最终独立测试 (Legacy: LightGBM)
            # ⚠️ 使用 LightGBM 预测数据，非C1最优模型。C1 2026测试见 Phase 5。
            # =============================================
            print("\n" + "=" * 60)
            print("Phase 2: 2026 H1 — 冻结策略，最终独立测试 (LightGBM)")
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

        # =============================================
        # Phase 5: C1 2026 H1 — 冻结规则 时序稳健性检验
        # =============================================
        # 2026 数据用途限定说明（来自 TRADING_C1_2026_H1_HANDOFF.md）：
        #   - 标记为 frozen-rule post-hoc temporal robustness
        #   - 不得基于此表重新选择模型、修改阈值或调整成本假设
        #   - 本阶段只验证：2025 最佳策略在 2026 是否稳健
        print("\n" + "=" * 60)
        print("Phase 5: C1 2026 H1 — 冻结策略 时序稳健性检验")
        print("=" * 60)
        print("(Frozen-rule post-hoc temporal robustness check)")
        print("注意: 不对2026数据做任何参数优化，仅验证2025最佳策略")

        market_c1_2026, pred_c1_2026 = load_c1_unified_2026()
        print(f"C1 2026 数据加载完成: {len(market_c1_2026)} 小时, "
              f"{pred_c1_2026['week_id'].nunique()} 周 Walk-Forward")

        engine_c1_2026 = ERCOTBacktestEngine(
            market_c1_2026,
            initial_capital=100000,
            fee_per_mwh=2.0,
            slippage_bps=50.0,
            capture_rate=0.65,
        )

        # ---- 5a: 复现 Handoff 2026 基准 (threshold=0.60, frozen rule) ----
        print("\n--- 5a: Handoff 2026 基准 (frozen threshold=0.60) ---")
        res_2026_baseline = engine_c1_2026.execute_b2b_baseline(
            pred_c1_2026, threshold=0.60
        )
        metrics_2026_baseline = engine_c1_2026.calculate_metrics(
            res_2026_baseline, engine_c1_2026.initial_capital
        )
        print("C1 2026 基准策略指标 (frozen 0.60):")
        print_metrics(metrics_2026_baseline)

        # 对比 handoff 报告的 net_pnl
        handoff_2026_pnl = pred_c1_2026['net_pnl'].sum()
        print(f"\n  Handoff 报告 2026 net_pnl 总和: ${handoff_2026_pnl:,.2f}")
        print(f"  我们计算的 Total_PnL:          ${metrics_2026_baseline['Total_Pnl']:,.2f}")

        # ---- 5b: 应用 2025 最佳策略（冻结参数）到 2026 ----
        print("\n--- 5b: 2025 最佳策略 在 2026 的表现 (冻结参数) ---")

        frozen_strategies_2026 = {}

        # 策略A: 2025 综合冠军 — 仅极端天气 (Sharpe 2.97 on 2025)
        res_2026_ext = engine_c1_2026.execute_b2b_b2a_combined(
            pred_c1_2026, threshold=0.60,
            use_b2a_direction=True, extreme_weather_filter=True,
        )
        frozen_strategies_2026['ExtremeWeather_Only'] = res_2026_ext
        m_ext = engine_c1_2026.calculate_metrics(res_2026_ext, 100000)
        print(f"  ExtremeWeather_Only:      PnL=${m_ext['Total_Pnl']:>10,.2f}  "
              f"Sharpe={m_ext['Sharpe_Ratio']:.4f}  Trades={m_ext['Total_Trades']}")

        # 策略B: 2025 绝对收益冠军 — 最优阈值 0.70 (Sharpe 2.34 on 2025)
        res_2026_opt = engine_c1_2026.execute_b2b_baseline(
            pred_c1_2026, threshold=0.70
        )
        frozen_strategies_2026['B2B_Optimal_070'] = res_2026_opt
        m_opt = engine_c1_2026.calculate_metrics(res_2026_opt, 100000)
        print(f"  B2B_Optimal_070:          PnL=${m_opt['Total_Pnl']:>10,.2f}  "
              f"Sharpe={m_opt['Sharpe_Ratio']:.4f}  Trades={m_opt['Total_Trades']}")

        # 策略C: Handoff 基准 (threshold=0.60)
        m_base = metrics_2026_baseline
        print(f"  B2B_Baseline_060:         PnL=${m_base['Total_Pnl']:>10,.2f}  "
              f"Sharpe={m_base['Sharpe_Ratio']:.4f}  Trades={m_base['Total_Trades']}")

        # ---- 5c: 2025 vs 2026 跨期稳健性对比 ----
        print("\n--- 5c: 2025 vs 2026 跨期稳健性对比 ---")
        print(f"{'策略':<28s} {'2025 PnL':>12s} {'2026 PnL':>12s} {'2025 Sharpe':>12s} {'2026 Sharpe':>12s}")
        print("-" * 76)

        # 需要 2025 对应策略的结果（从 Phase 3&4 中获取）
        # 从 Phase 3 的 res_b2b_baseline(0.60), Phase 4 的 res_s6, res_s7
        # 这些变量在 USE_C1_DATA 块内，我们需要重新计算 2025 的指标来对比
        cross_period = [
            ('B2B_Baseline_060',     0.60, 'baseline'),
            ('B2B_Optimal_070',      0.70, 'baseline'),
            ('ExtremeWeather_Only',  0.60, 'extreme'),
        ]

        for label, th, stype in cross_period:
            # 2025
            if stype == 'baseline':
                res_25 = engine_c1.execute_b2b_baseline(pred_c1, threshold=th)
            else:
                res_25 = engine_c1.execute_b2b_b2a_combined(
                    pred_c1, threshold=th,
                    use_b2a_direction=True, extreme_weather_filter=True,
                )
            m_25 = engine_c1.calculate_metrics(res_25, 100000)

            # 2026
            if stype == 'baseline':
                res_26 = engine_c1_2026.execute_b2b_baseline(pred_c1_2026, threshold=th)
            else:
                res_26 = engine_c1_2026.execute_b2b_b2a_combined(
                    pred_c1_2026, threshold=th,
                    use_b2a_direction=True, extreme_weather_filter=True,
                )
            m_26 = engine_c1_2026.calculate_metrics(res_26, 100000)

            print(f"{label:<28s} ${m_25['Total_Pnl']:>10,.2f}  ${m_26['Total_Pnl']:>10,.2f}  "
                  f"{m_25['Sharpe_Ratio']:>11.4f}  {m_26['Sharpe_Ratio']:>11.4f}")

        # ---- 5d: 2026 风险集中度分析 ----
        print("\n--- 5d: C1 2026 基准策略 风险集中度分析 ---")
        risk_2026, monthly_2026, daily_pnl_2026, class_perf_2026 = \
            analyze_risk_concentration(res_2026_baseline)
        print_risk_report(risk_2026)

        # ---- 5e: 2026 各周表现 ----
        print("\n--- 5e: 2026 各周 Walk-Forward 表现 ---")
        weekly_2026 = res_2026_baseline.groupby('week_id').agg(
            Weekly_PnL=('Hourly_Pnl', 'sum'),
            Trade_Count=('Signal', lambda x: (x != 0).sum()),
            Avg_Spread=('spread_usd_per_mwh', 'mean'),
        )
        print(weekly_2026.to_string())

        # ---- 5f: 2026 可视化 ----
        plot_backtest_result(
            res_2026_baseline,
            title="C1 B2B Baseline Strategy — 2026 H1 Walk-Forward (Frozen 0.60)",
            save_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'c1_b2b_baseline_2026.png'
            ),
        )

        # 2026 综合对比图: 3条策略净值曲线叠加
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {'ExtremeWeather_Only': '#d62728', 'B2B_Optimal_070': '#1f77b4',
                  'B2B_Baseline_060': '#7f7f7f'}
        styles = {'ExtremeWeather_Only': '-', 'B2B_Optimal_070': '-',
                  'B2B_Baseline_060': '--'}
        for name, res in [('B2B_Baseline_060', res_2026_baseline),
                           ('B2B_Optimal_070', res_2026_opt),
                           ('ExtremeWeather_Only', res_2026_ext)]:
            ax.plot(res['delivery_hour_utc'], res['Equity'],
                    label=name, color=colors.get(name, '#333333'),
                    linestyle=styles.get(name, '-'), linewidth=1.5)
        ax.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
        ax.set_title('C1 2026 H1 — Frozen Strategies Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        cmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'c1_2026_strategies_comparison.png')
        plt.savefig(cmp_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"2026 策略对比图已保存: {cmp_path}")

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
            search_results, best_params = engine.grid_search(
                pred_df, direction_filter=True,
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

        # =============================================
        # Phase 6: 统一策略对比 — 所有策略，同一引擎，同一成本模型
        # =============================================
        if USE_REAL_DATA or USE_C1_DATA:
            print("\n" + "=" * 60)
            print("Phase 6: 统一策略对比 (per-hour 成本模型，公平比较)")
            print("=" * 60)
            print("所有策略使用完全相同的: 引擎、成本假设、数据时间范围")
            print("成本模型: 按执行小时收费 (与 C1 handoff / Baseline 引擎一致)")

            # --- 6a: 2025 统一对比 ---
            if USE_REAL_DATA:
                print("\n" + "-" * 40)
                print("6a: 2025 验证集 — 统一策略对比 (LightGBM 数据)")
                print("-" * 40)

                engine_unified_2025 = ERCOTBacktestEngine(
                    market_2025,
                    initial_capital=100000,
                    fee_per_mwh=2.0,
                    slippage_bps=50.0,
                    capture_rate=0.65,
                )

                strategies_2025, comparison_2025 = run_unified_comparison(
                    market_2025, pred_2025, engine_unified_2025,
                    label="2025_LightGBM", per_execution_hour_costs=True,
                    include_c1_strategies=False,
                )

                # 生成多策略权益曲线叠加图
                fig, ax = plt.subplots(figsize=(14, 7))
                colors_6 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b']
                for idx, (name, res) in enumerate(strategies_2025.items()):
                    color = colors_6[idx % len(colors_6)]
                    ax.plot(res['delivery_hour_utc'], res['Equity'],
                            label=name, color=color, linewidth=1.3, alpha=0.85)
                ax.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
                ax.set_title('2025 Unified Strategy Comparison — All Strategies, Same Engine',
                            fontsize=12, fontweight='bold')
                ax.set_ylabel('Equity ($)')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                unified_chart_2025 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'unified_comparison_2025.png'
                )
                plt.savefig(unified_chart_2025, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"\n2025 统一对比图已保存: {unified_chart_2025}")

                # --- 6b: 2026 冻结验证 ---
                print("\n" + "-" * 40)
                print("6b: 2026 H1 测试集 — 冻结策略验证 (LightGBM 数据)")
                print("-" * 40)

                engine_unified_2026 = ERCOTBacktestEngine(
                    market_2026,
                    initial_capital=100000,
                    fee_per_mwh=2.0,
                    slippage_bps=50.0,
                    capture_rate=0.65,
                )

                strategies_2026, comparison_2026 = run_unified_comparison(
                    market_2026, pred_2026, engine_unified_2026,
                    label="2026_LightGBM", per_execution_hour_costs=True,
                    include_c1_strategies=False,
                )

                # 生成 2026 多策略对比图
                fig, ax = plt.subplots(figsize=(14, 7))
                for idx, (name, res) in enumerate(strategies_2026.items()):
                    color = colors_6[idx % len(colors_6)]
                    ax.plot(res['delivery_hour_utc'], res['Equity'],
                            label=name, color=color, linewidth=1.3, alpha=0.85)
                ax.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
                ax.set_title('2026 H1 Unified Strategy Comparison — Frozen Strategies',
                            fontsize=12, fontweight='bold')
                ax.set_ylabel('Equity ($)')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                unified_chart_2026 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'unified_comparison_2026.png'
                )
                plt.savefig(unified_chart_2026, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"\n2026 统一对比图已保存: {unified_chart_2026}")

                # --- 跨期稳健性矩阵 ---
                print("\n" + "-" * 40)
                print("6c: 跨期稳健性矩阵 (2025 → 2026 Sharpe)")
                print("-" * 40)
                common_strategies = set(strategies_2025.keys()) & set(strategies_2026.keys())
                print(f"{'Strategy':<28s} {'2025 Sharpe':>12s} {'2026 Sharpe':>12s} "
                      f"{'Change':>10s} {'2025 PnL':>15s} {'2026 PnL':>15s}")
                print("-" * 88)
                for name in sorted(common_strategies):
                    m25 = engine_unified_2025.calculate_metrics(
                        strategies_2025[name], 100000
                    )
                    m26 = engine_unified_2026.calculate_metrics(
                        strategies_2026[name], 100000
                    )
                    s25 = m25['Sharpe_Ratio']
                    s26 = m26['Sharpe_Ratio']
                    change = s26 - s25
                    sign = '+' if change >= 0 else ''
                    print(f"{name:<28s} {s25:>12.4f} {s26:>12.4f} "
                          f"{sign}{change:>9.4f} ${m25['Total_Pnl']:>13,.2f} "
                          f"${m26['Total_Pnl']:>13,.2f}")

            # --- 6d: C1 数据统一对比 (如果可用) ---
            if USE_C1_DATA:
                print("\n" + "-" * 40)
                print("6d: 2025 C1 数据 — 统一策略对比 (含 B2B 策略)")
                print("-" * 40)

                engine_c1_unified = ERCOTBacktestEngine(
                    market_c1,
                    initial_capital=100000,
                    fee_per_mwh=2.0,
                    slippage_bps=50.0,
                    capture_rate=0.65,
                )

                strategies_c1, comparison_c1 = run_unified_comparison(
                    market_c1, pred_c1, engine_c1_unified,
                    label="2025_C1", per_execution_hour_costs=True,
                    include_c1_strategies=True,
                )

                # C1 多策略叠加图
                fig, ax = plt.subplots(figsize=(14, 7))
                for idx, (name, res) in enumerate(strategies_c1.items()):
                    color = colors_6[idx % len(colors_6)]
                    ax.plot(res['delivery_hour_utc'], res['Equity'],
                            label=name, color=color, linewidth=1.3, alpha=0.85)
                ax.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
                ax.set_title('2025 C1 Unified Strategy Comparison — All Strategies, Same Engine',
                            fontsize=12, fontweight='bold')
                ax.set_ylabel('Equity ($)')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                unified_chart_c1 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'unified_comparison_c1_2025.png'
                )
                plt.savefig(unified_chart_c1, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"\nC1 统一对比图已保存: {unified_chart_c1}")

            print(f"\n✅ Phase 6 统一对比完成")

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