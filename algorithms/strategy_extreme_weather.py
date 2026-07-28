"""
C1 ExtremeWeather_Only — 生产级无状态交易信号函数
===================================================

C1 回测系统验证通过的冠军策略，封装为 FastAPI 可直接调用的纯函数。

策略规则（三要素）：
  1. 极端天气开关：仅在 extreme_weather_flag == True/1 时允许交易
  2. B2B 分类信号：p_positive >= 0.60 → DEC, p_negative >= 0.60 → INC
  3. B2A 方向确认：B2A predicted_spread 符号必须与 B2B 信号方向一致

2025 OOF 验证：Sharpe 2.97, PnL $3,617, Max DD -0.62%
2026 H1 Walk-Forward 验证：Sharpe 2.06, PnL $6,532, Max DD -0.17%

交付对象：成员 D — FastAPI 后端 / 前端实时信号展示
"""

from typing import Dict, List, Union, Optional
import pandas as pd
import numpy as np


# ============================================================
# 核心函数：单小时交易信号
# ============================================================

def get_trade_signal(
    p_negative: float,
    p_positive: float,
    extreme_weather_flag: Union[bool, int],
    predicted_spread: Optional[float] = None,
    confidence: Optional[float] = None,
    threshold: float = 0.60,
) -> Dict:
    """
    对单个小时返回 ExtremeWeather_Only 策略的交易指令。

    Parameters
    ----------
    p_negative : float
        INC 方向概率（= p_c1 + p_c2），取值范围 [0, 1]。
    p_positive : float
        DEC 方向概率（= p_c4 + p_c5），取值范围 [0, 1]。
    extreme_weather_flag : bool or int
        极端天气触发标志。True/1 表示极端天气小时。
    predicted_spread : float, optional
        B2A 回归头预测的连续价差 ($/MWh)。用于方向确认。
        若不提供则跳过 B2A 确认，仅用 B2B 信号。
    confidence : float, optional
        最大类别概率 (= max(p_c1..p_c5))。用于仓位缩放。
        若不提供则使用 max(p_negative, 1-p_negative-p_positive, p_positive) 近似。
    threshold : float, default 0.60
        B2B 概率阈值。默认值已经 2025 OOF + 2026 H1 双时段验证。

    Returns
    -------
    dict
        {
            "strategy_action": str,       # "INC" | "DEC" | "NO_TRADE"
            "strategy_confidence": float, # 信号置信度 (0.0 ~ 1.0)
            "position_size": float,       # 建议仓位倍数（1.0 = 1 MWh）
            "signal_details": {
                "b2b_signal": int,        # B2B 原始信号 (-1/0/+1)
                "b2a_confirmed": bool,    # B2A 方向是否确认
                "extreme_weather": bool,  # 是否为极端天气小时
                "rule_version": str,      # 策略版本号
            }
        }
    """
    # ── 输入校验 ──
    for name, val in [("p_negative", p_negative), ("p_positive", p_positive)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} 必须在 [0, 1] 范围内，收到 {val}")

    is_extreme = bool(extreme_weather_flag)

    # ── 规则 1：极端天气开关 ──
    if not is_extreme:
        return {
            "strategy_action": "NO_TRADE",
            "strategy_confidence": 0.0,
            "position_size": 0.0,
            "signal_details": {
                "b2b_signal": 0,
                "b2a_confirmed": False,
                "extreme_weather": False,
                "rule_version": "C1_EWO_v1",
            },
        }

    # ── 规则 2：B2B 分类信号 ──
    b2b_signal = 0
    if p_positive >= threshold and p_positive > p_negative:
        b2b_signal = 1   # DEC
    elif p_negative >= threshold and p_negative > p_positive:
        b2b_signal = -1  # INC

    if b2b_signal == 0:
        return {
            "strategy_action": "NO_TRADE",
            "strategy_confidence": 0.0,
            "position_size": 0.0,
            "signal_details": {
                "b2b_signal": 0,
                "b2a_confirmed": False,
                "extreme_weather": True,
                "rule_version": "C1_EWO_v1",
            },
        }

    # ── 规则 3：B2A 方向确认 ──
    b2a_confirmed = True
    if predicted_spread is not None:
        b2a_direction = 1 if predicted_spread > 0 else (-1 if predicted_spread < 0 else 0)
        if b2a_direction != 0 and b2a_direction != b2b_signal:
            b2a_confirmed = False

    if not b2a_confirmed:
        return {
            "strategy_action": "NO_TRADE",
            "strategy_confidence": 0.0,
            "position_size": 0.0,
            "signal_details": {
                "b2b_signal": int(b2b_signal),
                "b2a_confirmed": False,
                "extreme_weather": True,
                "rule_version": "C1_EWO_v1",
            },
        }

    # ── 计算置信度与仓位 ──
    if confidence is None:
        confidence = max(p_negative, p_positive, 1.0 - p_negative - p_positive)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # 仓位缩放：confidence / threshold，钳制在 [0.5, 2.0]
    position_size = float(np.clip(confidence / threshold, 0.5, 2.0))

    action = "INC" if b2b_signal == -1 else "DEC"

    return {
        "strategy_action": action,
        "strategy_confidence": round(confidence, 4),
        "position_size": round(position_size, 4),
        "signal_details": {
            "b2b_signal": int(b2b_signal),
            "b2a_confirmed": True,
            "extreme_weather": True,
            "rule_version": "C1_EWO_v1",
        },
    }


# ============================================================
# 便捷函数：DataFrame 批量调用
# ============================================================

def get_trade_signals_batch(
    df: pd.DataFrame,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """
    对 DataFrame 中每一行批量调用 get_trade_signal。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列：p_negative, p_positive, extreme_weather_flag
        可选列：predicted_spread, confidence
    threshold : float, default 0.60

    Returns
    -------
    pd.DataFrame
        原 DataFrame 附加三列：
        - strategy_action  ("INC" | "DEC" | "NO_TRADE")
        - strategy_confidence (float)
        - position_size (float)
    """
    required = ["p_negative", "p_positive", "extreme_weather_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必需列: {missing}")

    has_spread = "predicted_spread" in df.columns
    has_conf = "confidence" in df.columns

    actions, confidences, positions = [], [], []
    for _, row in df.iterrows():
        result = get_trade_signal(
            p_negative=float(row["p_negative"]),
            p_positive=float(row["p_positive"]),
            extreme_weather_flag=row["extreme_weather_flag"],
            predicted_spread=float(row["predicted_spread"]) if has_spread else None,
            confidence=float(row["confidence"]) if has_conf else None,
            threshold=threshold,
        )
        actions.append(result["strategy_action"])
        confidences.append(result["strategy_confidence"])
        positions.append(result["position_size"])

    out = df.copy()
    out["strategy_action"] = actions
    out["strategy_confidence"] = confidences
    out["position_size"] = positions
    return out


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=== strategy_extreme_weather 自测 ===\n")

    # 测试 1：极端天气 INC 信号
    r1 = get_trade_signal(
        p_negative=0.75, p_positive=0.15,
        extreme_weather_flag=True,
        predicted_spread=-50.0,
        confidence=0.75,
    )
    print(f"Test 1 (极端天气 + INC + B2A确认): {r1['strategy_action']}")
    assert r1["strategy_action"] == "INC"
    assert r1["strategy_confidence"] > 0
    print("  PASS\n")

    # 测试 2：极端天气 但 B2B 不满足
    r2 = get_trade_signal(
        p_negative=0.30, p_positive=0.25,
        extreme_weather_flag=True,
    )
    print(f"Test 2 (极端天气 但概率不足): {r2['strategy_action']}")
    assert r2["strategy_action"] == "NO_TRADE"
    print("  PASS\n")

    # 测试 3：非极端天气 → 强制空仓
    r3 = get_trade_signal(
        p_negative=0.90, p_positive=0.05,
        extreme_weather_flag=False,
        predicted_spread=-100.0,
    )
    print(f"Test 3 (非极端天气 强制空仓): {r3['strategy_action']}")
    assert r3["strategy_action"] == "NO_TRADE"
    assert r3["signal_details"]["extreme_weather"] is False
    print("  PASS\n")

    # 测试 4：B2A 方向冲突
    r4 = get_trade_signal(
        p_negative=0.70, p_positive=0.20,
        extreme_weather_flag=True,
        predicted_spread=+80.0,  # positive spread → DEC direction
    )
    print(f"Test 4 (B2B=INC 但 B2A=DEC方向): {r4['strategy_action']}")
    assert r4["strategy_action"] == "NO_TRADE"
    assert r4["signal_details"]["b2a_confirmed"] is False
    print("  PASS\n")

    # 测试 5：DEC 信号
    r5 = get_trade_signal(
        p_negative=0.10, p_positive=0.85,
        extreme_weather_flag=True,
        predicted_spread=+30.0,
    )
    print(f"Test 5 (极端天气 + DEC): {r5['strategy_action']}")
    assert r5["strategy_action"] == "DEC"
    print("  PASS\n")

    # 测试 6：批量 DataFrame
    df_test = pd.DataFrame({
        "p_negative":           [0.75, 0.30, 0.90, 0.70, 0.10],
        "p_positive":           [0.15, 0.25, 0.05, 0.20, 0.85],
        "extreme_weather_flag": [True, True, False, True, True],
        "predicted_spread":     [-50.0, 10.0, -100.0, 80.0, 30.0],
        "confidence":           [0.75, 0.40, 0.90, 0.70, 0.85],
    })
    df_out = get_trade_signals_batch(df_test)
    print("Test 6 (批量 DataFrame):")
    print(df_out[["p_negative", "p_positive", "strategy_action", "position_size"]].to_string())
    expected = ["INC", "NO_TRADE", "NO_TRADE", "NO_TRADE", "DEC"]
    assert df_out["strategy_action"].tolist() == expected
    print("  PASS\n")

    # 测试 7：阈值可配置
    r7 = get_trade_signal(
        p_negative=0.65, p_positive=0.20,
        extreme_weather_flag=True,
        predicted_spread=-30.0,
        threshold=0.70,  # 更高的阈值
    )
    print(f"Test 7 (threshold=0.70, p_negative=0.65): {r7['strategy_action']}")
    assert r7["strategy_action"] == "NO_TRADE"  # 0.65 < 0.70 → 不满足
    print("  PASS\n")

    print("=== 全部 7 项测试通过 ===")
