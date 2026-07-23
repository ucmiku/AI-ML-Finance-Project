from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "C_combination_systems"
METRICS = BASE / "outputs" / "prediction_quality_gate"
ECON = BASE / "outputs" / "economic_value_gate"
REPORTS = BASE / "reports"
RUN_ID = "C4_head_swap_exploratory_v3_f5730506"
PROB_THRESHOLD = 0.60
CAPTURE_RATE = 0.65
COMMISSION = 2.0
SLIPPAGE = 0.005
INITIAL_CAPITAL = 100000.0


def class_int(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        nums = pd.to_numeric(s, errors="coerce")
        if nums.min() >= 0 and nums.max() <= 4:
            return nums.astype(int) + 1
        return nums.astype(int)
    return s.astype(str).str.extract(r"(\d+)", expand=False).astype(int)


def direction_from_class(c: pd.Series) -> pd.Series:
    return pd.Series(np.select([c <= 2, c == 3, c >= 4], [-1, 0, 1], default=0), index=c.index)


def local_day(df: pd.DataFrame) -> pd.Series:
    if "delivery_date_local" in df.columns:
        return pd.to_datetime(df["delivery_date_local"], errors="coerce").dt.date
    if "delivery_time_local" in df.columns:
        return pd.to_datetime(df["delivery_time_local"], errors="coerce", utc=True).dt.date
    return pd.to_datetime(df["delivery_hour_utc"], utc=True).dt.date


def signal_from_probs(df: pd.DataFrame) -> pd.Series:
    p_pos = pd.to_numeric(df["p_positive"], errors="coerce")
    p_neg = pd.to_numeric(df["p_negative"], errors="coerce")
    sig = np.where((p_pos >= PROB_THRESHOLD) & (p_pos > p_neg), 1, 0)
    sig = np.where((p_neg >= PROB_THRESHOLD) & (p_neg > p_pos), -1, sig)
    return pd.Series(sig, index=df.index).astype(int)


def apply_trade(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = signal_from_probs(out)
    spread = pd.to_numeric(out["actual_spread"], errors="coerce")
    clipped = spread.clip(-1000.0, 5000.0)
    traded = out["signal"].ne(0)
    out["net_pnl"] = out["signal"] * clipped * CAPTURE_RATE
    out["net_pnl"] -= traded.astype(float) * COMMISSION
    out["net_pnl"] -= traded.astype(float) * clipped.abs() * SLIPPAGE
    out["gross_pnl"] = out["signal"] * clipped * CAPTURE_RATE
    out["commission"] = traded.astype(float) * COMMISSION
    out["slippage"] = traded.astype(float) * clipped.abs() * SLIPPAGE
    return out


def prediction_metrics(df: pd.DataFrame, system_id: str) -> dict:
    y = class_int(df["actual_class"])
    pred = class_int(df["predicted_class"])
    p = df[[f"p_c{i}" for i in range(1, 6)]].astype(float).to_numpy()
    row_sum = p.sum(axis=1)
    valid = np.isfinite(p).all(axis=1) & (row_sum > 0)
    p_norm = p[valid] / row_sum[valid, None]
    true_dir = direction_from_class(y)
    pred_dir = direction_from_class(pred)
    catastrophic = ((y <= 2) & (pred >= 4)) | ((y >= 4) & (pred <= 2))
    extreme = pd.to_numeric(df.get("fixed_extreme_weather_flag", 0), errors="coerce").fillna(0).astype(bool)
    return {
        "system_id": system_id,
        "run_id": RUN_ID,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y.iloc[valid], p_norm, labels=[1, 2, 3, 4, 5])) if valid.any() else np.nan,
        "mean_abs_class_distance": float(np.abs(y.to_numpy() - pred.to_numpy()).mean()),
        "catastrophic_reversal_rate": float(catastrophic.mean()),
        "direction_accuracy": float((true_dir == pred_dir).mean()),
        "extreme_weather_macro_f1": float(f1_score(y[extreme], pred[extreme], labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)) if extreme.any() else np.nan,
        "normal_weather_macro_f1": float(f1_score(y[~extreme], pred[~extreme], labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)) if (~extreme).any() else np.nan,
    }


def cvar(daily: pd.Series, level: float = 0.95) -> float:
    if daily.empty:
        return np.nan
    q = daily.quantile(1.0 - level)
    tail = daily[daily <= q]
    return float(tail.mean()) if not tail.empty else np.nan


def economic_metrics(df: pd.DataFrame, system_id: str) -> dict:
    d = df.copy()
    d["delivery_hour_utc"] = pd.to_datetime(d["delivery_hour_utc"], utc=True)
    pnl = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
    sig = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int)
    traded = sig.ne(0)
    day = local_day(d)
    daily = pnl.groupby(day).sum()
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    downside = daily[daily < 0]
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(365.0)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    sortino = float(daily.mean() / downside.std(ddof=1) * math.sqrt(365.0)) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    gp = pnl[traded & (pnl > 0)].sum()
    gl = -pnl[traded & (pnl < 0)].sum()
    months = pnl.groupby(d["delivery_hour_utc"].dt.to_period("M").astype(str)).sum()
    top_hours = pnl[traded].sort_values(ascending=False)
    day_pnl = daily.sort_values(ascending=False)
    total = float(pnl.sum())
    default_zero = pd.Series(0, index=d.index)
    extreme = pd.to_numeric(d["fixed_extreme_weather_flag"] if "fixed_extreme_weather_flag" in d.columns else default_zero, errors="coerce").fillna(0).astype(bool)
    tail20 = pd.to_numeric(d["target_extreme20"] if "target_extreme20" in d.columns else default_zero, errors="coerce").fillna(0).astype(bool)
    tail50 = pd.to_numeric(d["target_extreme50"] if "target_extreme50" in d.columns else default_zero, errors="coerce").fillna(0).astype(bool)
    return {
        "system_id": system_id,
        "run_id": RUN_ID,
        "total_pnl": total,
        "total_return": total / INITIAL_CAPITAL,
        "trade_count": int(traded.sum()),
        "coverage": float(traded.mean()),
        "direction_precision": float((np.sign(pd.to_numeric(d.loc[traded, "actual_spread"], errors="coerce")) == sig[traded]).mean()) if traded.any() else np.nan,
        "pnl_per_mwh": float(pnl[traded].mean()) if traded.any() else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "cvar_95_daily": cvar(daily),
        "win_rate": float((pnl[traded] > 0).mean()) if traded.any() else np.nan,
        "profit_factor": float(gp / gl) if gl > 0 else np.nan,
        "maximum_single_loss": float(pnl.min()) if len(pnl) else np.nan,
        "profitable_months": int(months.gt(0).sum()),
        "inc_count": int((sig == -1).sum()),
        "dec_count": int((sig == 1).sum()),
        "inc_pnl": float(pnl[sig == -1].sum()),
        "dec_pnl": float(pnl[sig == 1].sum()),
        "extreme_weather_pnl": float(pnl[extreme].sum()),
        "normal_weather_pnl": float(pnl[~extreme].sum()),
        "extreme_weather_tail20_pnl": float(pnl[extreme & tail20].sum()),
        "extreme_weather_tail50_pnl": float(pnl[extreme & tail50].sum()),
        "pnl_ex_top5_days": float(total - day_pnl.head(5).sum()) if len(day_pnl) else total,
        "top1_day_share": float(day_pnl.head(1).sum() / total) if total != 0 and len(day_pnl) else np.nan,
        "top5_day_share": float(day_pnl.head(5).sum() / total) if total != 0 and len(day_pnl) else np.nan,
        "top1_hour_share": float(top_hours.head(1).sum() / total) if total != 0 and len(top_hours) else np.nan,
        "top5_hour_share": float(top_hours.head(5).sum() / total) if total != 0 and len(top_hours) else np.nan,
    }


def strict_align(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    for name, df in [(left_name, left), (right_name, right)]:
        if df["delivery_hour_utc"].duplicated().any():
            dupes = df.loc[df["delivery_hour_utc"].duplicated(), "delivery_hour_utc"].head(10).astype(str).tolist()
            raise RuntimeError(f"{name} has duplicate delivery_hour_utc: {dupes}")
    l = pd.to_datetime(left["delivery_hour_utc"], utc=True).sort_values().reset_index(drop=True)
    r = pd.to_datetime(right["delivery_hour_utc"], utc=True).sort_values().reset_index(drop=True)
    if len(l) != len(r) or not l.equals(r):
        missing_l = sorted(set(r.astype(str)) - set(l.astype(str)))[:10]
        missing_r = sorted(set(l.astype(str)) - set(r.astype(str)))[:10]
        raise RuntimeError(
            f"Timestamp alignment failed: {left_name} rows={len(l)}, {right_name} rows={len(r)}, "
            f"missing_in_{left_name}={missing_l}, missing_in_{right_name}={missing_r}"
        )


def markdown_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    return "\n".join(
        [
            "| " + " | ".join(out.columns) + " |",
            "| " + " | ".join(["---"] * len(out.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in out.astype(str).to_numpy()],
        ]
    )


def main() -> None:
    for d in [OUT, METRICS, ECON, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    b4a = pd.read_parquet(
        BASE / "outputs" / "B_single_models" / "B4A_seq2seq_continuous_v3_f5730506" / "oof_2025_B4A_seq2seq_continuous_v3.parquet"
    )
    b2b = pd.read_parquet(
        BASE / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "predictions.parquet"
    )
    c1 = pd.read_parquet(OUT / "C1_best_boosting_complete_system_oof_2025_v3.parquet")
    for df in [b4a, b2b, c1]:
        df["delivery_hour_utc"] = pd.to_datetime(df["delivery_hour_utc"], utc=True)

    strict_align(b4a, b2b, "B4A", "B2B")
    strict_align(c1, b2b, "C1", "B2B")

    b4a_sorted = b4a.sort_values("delivery_hour_utc").reset_index(drop=True)
    b2b_sorted = b2b.sort_values("delivery_hour_utc").reset_index(drop=True)
    c1_sorted = c1.sort_values("delivery_hour_utc").reset_index(drop=True)
    if not np.allclose(pd.to_numeric(b4a_sorted["actual_spread"], errors="coerce"), pd.to_numeric(b2b_sorted["actual_spread"], errors="coerce"), equal_nan=True):
        raise RuntimeError("B4A and B2B actual_spread mismatch; refusing head-swap")

    c4 = b2b_sorted.copy()
    c4["predicted_spread"] = pd.to_numeric(b4a_sorted["predicted_spread"], errors="coerce")
    c4["system_id"] = "C4_exploratory_B4A_continuous_B2B_probabilities"
    c4["model_family"] = "LSTM_continuous_head_plus_XGBoost_classifier_head"
    c4["system_run_id"] = RUN_ID
    c4["p_negative"] = pd.to_numeric(c4["p_c1"], errors="coerce") + pd.to_numeric(c4["p_c2"], errors="coerce")
    c4["p_no_trade"] = pd.to_numeric(c4["p_c3"], errors="coerce")
    c4["p_positive"] = pd.to_numeric(c4["p_c4"], errors="coerce") + pd.to_numeric(c4["p_c5"], errors="coerce")
    c4["p_outer"] = pd.to_numeric(c4["p_c1"], errors="coerce") + pd.to_numeric(c4["p_c5"], errors="coerce")
    c4 = apply_trade(c4)
    c4_path = OUT / "C4_exploratory_head_swap_oof_2025_v3.parquet"
    c4.to_parquet(c4_path, index=False)

    c1_for_compare = c1_sorted.copy()
    c1_for_compare["system_id"] = "C1_best_boosting_complete_system"
    if "net_pnl" not in c1_for_compare.columns or c1_for_compare["net_pnl"].isna().all():
        c1_for_compare = apply_trade(c1_for_compare)
    else:
        c1_for_compare = apply_trade(c1_for_compare)

    pred_metrics = pd.DataFrame(
        [
            prediction_metrics(c1_for_compare, "C1_best_boosting_complete_system"),
            prediction_metrics(c4, "C4_exploratory_head_swap"),
        ]
    )
    econ_metrics = pd.DataFrame(
        [
            economic_metrics(c1_for_compare, "C1_best_boosting_complete_system"),
            economic_metrics(c4, "C4_exploratory_head_swap"),
        ]
    )
    pred_path = METRICS / "C4_head_swap_prediction_metrics_2025_v3.csv"
    econ_path = ECON / "C4_head_swap_economic_metrics_2025_v3.csv"
    pred_metrics.to_csv(pred_path, index=False)
    econ_metrics.to_csv(econ_path, index=False)

    combined = pd.concat(
        [
            c1_for_compare.assign(compare_system="C1"),
            c4.assign(compare_system="C4"),
        ],
        ignore_index=True,
    )
    combined["month"] = pd.to_datetime(combined["delivery_hour_utc"], utc=True).dt.to_period("M").astype(str)
    monthly = combined.groupby(["compare_system", "month"], as_index=False)["net_pnl"].sum()
    monthly.columns = ["system_id", "month", "net_pnl"]
    monthly_path = ECON / "C4_head_swap_monthly_pnl_2025_v3.csv"
    monthly.to_csv(monthly_path, index=False)

    c1_sig = pd.to_numeric(c1_for_compare["signal"], errors="coerce").fillna(0).astype(int)
    c4_sig = pd.to_numeric(c4["signal"], errors="coerce").fillna(0).astype(int)
    overlap = pd.DataFrame(
        [
            {
                "comparison": "C1_vs_C4",
                "hours": len(c4),
                "both_trade": int(c1_sig.ne(0).mul(c4_sig.ne(0)).sum()),
                "both_same_signal": int((c1_sig == c4_sig).sum()),
                "c1_only_trade": int(c1_sig.ne(0).mul(c4_sig.eq(0)).sum()),
                "c4_only_trade": int(c4_sig.ne(0).mul(c1_sig.eq(0)).sum()),
                "opposite_trade": int((c1_sig * c4_sig == -1).sum()),
                "trade_jaccard": float((c1_sig.ne(0) & c4_sig.ne(0)).sum() / (c1_sig.ne(0) | c4_sig.ne(0)).sum()),
            }
        ]
    )
    overlap_path = ECON / "C4_head_swap_trade_overlap_2025_v3.csv"
    overlap.to_csv(overlap_path, index=False)

    disagreement = pd.DataFrame(
        {
            "delivery_hour_utc": c4["delivery_hour_utc"],
            "actual_spread": c4["actual_spread"],
            "c1_predicted_spread": c1_for_compare["predicted_spread"],
            "c4_predicted_spread": c4["predicted_spread"],
            "p_negative": c4["p_negative"],
            "p_positive": c4["p_positive"],
            "c1_signal": c1_sig,
            "c4_signal": c4_sig,
            "c1_net_pnl": c1_for_compare["net_pnl"],
            "c4_net_pnl": c4["net_pnl"],
            "fixed_extreme_weather_flag": c4.get("fixed_extreme_weather_flag", 0),
            "target_extreme20": c4.get("target_extreme20", 0),
            "target_extreme50": c4.get("target_extreme50", 0),
        }
    )
    disagreement["spread_head_abs_diff"] = (pd.to_numeric(disagreement["c1_predicted_spread"], errors="coerce") - pd.to_numeric(disagreement["c4_predicted_spread"], errors="coerce")).abs()
    disagreement["signal_disagreement"] = disagreement["c1_signal"].ne(disagreement["c4_signal"])
    disagreement_path = ECON / "C4_head_swap_disagreement_hours_2025_v3.csv"
    disagreement.to_csv(disagreement_path, index=False)

    report_path = REPORTS / "C4_EXPLORATORY_HEAD_SWAP_REPORT.md"
    report_path.write_text(
        "# C4 exploratory head-swap报告\n\n"
        "- C4 = B4A LSTM continuous OOF predictions + B2B XGBoost 5/20 classifier OOF probabilities。\n"
        "- 未重新训练模型，未运行Optuna，未修改阈值。\n"
        "- 严格检查B4A、B2B、C1的2025 OOF UTC时间戳一一对齐；未使用inner join静默删样本。\n"
        "- 标记：exploratory head-swap，不覆盖正式C1。\n\n"
        "## Prediction Metrics\n\n"
        f"{markdown_table(pred_metrics)}\n\n"
        "## Economic Metrics\n\n"
        f"{markdown_table(econ_metrics)}\n\n"
        "## Trade Overlap\n\n"
        f"{markdown_table(overlap)}\n\n"
        "## 文件\n\n"
        f"- `{c4_path}`\n"
        f"- `{pred_path}`\n"
        f"- `{econ_path}`\n"
        f"- `{monthly_path}`\n"
        f"- `{overlap_path}`\n"
        f"- `{disagreement_path}`\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_EXPLORATORY",
        "no_training": True,
        "timestamp_alignment": "strict_one_to_one_passed",
        "formal_c1_overwritten": False,
        "generated_files": [str(p) for p in [c4_path, pred_path, econ_path, monthly_path, overlap_path, disagreement_path, report_path]],
    }
    (OUT / "C4_exploratory_head_swap_manifest_v3.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("C4 exploratory head-swap completed")
    print(f"prediction_metrics={pred_path}")
    print(f"economic_metrics={econ_path}")
    print(f"report={report_path}")
    print(econ_metrics[["system_id", "trade_count", "direction_precision", "total_pnl", "pnl_per_mwh", "sharpe", "max_drawdown", "cvar_95_daily", "profitable_months", "pnl_ex_top5_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
