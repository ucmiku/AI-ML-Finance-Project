from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
REPORTS = BASE / "reports"
PROGRESS = BASE / "progress"
RUN_ID = "C_system_quality_econ_v3_f5730506"
DATA_HASH = "f5730506707c2f227f6208bb6bc00ca4c0c45fe5a23c3148c1e9c2c04cfa0717"
FEATURE_HASH = "0c6903cdbaca034b74f59dac8d014c57a04b3bfabb80b7c42b3ba7d6b3fd59de"
INITIAL_CAPITAL = 100000.0
CAPTURE_RATE = 0.65
COMMISSION = 2.0
SLIPPAGE_BPS = 50.0
PROB_THRESHOLD = 0.60


def class_int(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        nums = pd.to_numeric(s, errors="coerce")
        if nums.min() >= 0 and nums.max() <= 4:
            return nums.astype(int) + 1
        return nums.astype(int)
    return s.astype(str).str.extract(r"(\d+)", expand=False).astype(int)


def class_from_spread(v: pd.Series) -> pd.Series:
    return pd.cut(v.astype(float), [-np.inf, -20.0, -5.0, 5.0, 20.0, np.inf], labels=[1, 2, 3, 4, 5], right=True).astype(int)


def direction_from_class(c: pd.Series) -> pd.Series:
    return pd.Series(np.select([c <= 2, c == 3, c >= 4], [-1, 0, 1], default=0), index=c.index)


def signal_from_probs(df: pd.DataFrame) -> pd.Series:
    p_pos = pd.to_numeric(df["p_positive"], errors="coerce")
    p_neg = pd.to_numeric(df["p_negative"], errors="coerce")
    sig = np.where((p_pos >= PROB_THRESHOLD) & (p_pos > p_neg), 1, 0)
    sig = np.where((p_neg >= PROB_THRESHOLD) & (p_neg > p_pos), -1, sig)
    return pd.Series(sig, index=df.index).astype(int)


def apply_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = signal_from_probs(out)
    spread = pd.to_numeric(out["actual_spread"], errors="coerce")
    clipped = spread.clip(-1000.0, 5000.0)
    traded = out["signal"].ne(0)
    gross = out["signal"] * clipped * CAPTURE_RATE
    commission = traded.astype(float) * COMMISSION
    slippage = traded.astype(float) * clipped.abs() * (SLIPPAGE_BPS / 10000.0)
    out["net_pnl"] = gross - commission - slippage
    out["gross_pnl"] = gross
    out["commission"] = commission
    out["slippage"] = slippage
    out["p_outer"] = pd.to_numeric(out["p_c1"], errors="coerce") + pd.to_numeric(out["p_c5"], errors="coerce")
    out["confidence"] = out[[f"p_c{i}" for i in range(1, 6)]].max(axis=1)
    return out


def normalize_system(df: pd.DataFrame, system_id: str, family: str) -> pd.DataFrame:
    out = df.copy()
    out["delivery_hour_utc"] = pd.to_datetime(out["delivery_hour_utc"], utc=True)
    if "delivery_date_local" not in out.columns:
        if "delivery_time_local" in out.columns:
            out["delivery_date_local"] = pd.to_datetime(out["delivery_time_local"], utc=True, errors="coerce").dt.date.astype(str)
        else:
            out["delivery_date_local"] = out["delivery_hour_utc"].dt.date.astype(str)
    if "actual_class" not in out.columns:
        out["actual_class"] = class_from_spread(out["actual_spread"])
    if "predicted_class" not in out.columns:
        proba = out[[f"p_c{i}" for i in range(1, 6)]].to_numpy(float)
        out["predicted_class"] = np.nanargmax(proba, axis=1) + 1
    out["actual_class"] = class_int(out["actual_class"])
    out["predicted_class"] = class_int(out["predicted_class"])
    out["p_negative"] = pd.to_numeric(out["p_c1"], errors="coerce") + pd.to_numeric(out["p_c2"], errors="coerce")
    out["p_no_trade"] = pd.to_numeric(out["p_c3"], errors="coerce")
    out["p_positive"] = pd.to_numeric(out["p_c4"], errors="coerce") + pd.to_numeric(out["p_c5"], errors="coerce")
    out["system_id"] = system_id
    out["model_family"] = family
    out["system_run_id"] = RUN_ID
    out = apply_benchmark(out)
    keep = [
        "delivery_hour_utc",
        "delivery_date_local",
        "fold_id",
        "system_id",
        "model_family",
        "system_run_id",
        "predicted_spread",
        "p_c1",
        "p_c2",
        "p_c3",
        "p_c4",
        "p_c5",
        "p_negative",
        "p_no_trade",
        "p_positive",
        "p_outer",
        "predicted_class",
        "confidence",
        "actual_class",
        "actual_spread",
        "fixed_extreme_weather_flag",
        "target_extreme20",
        "signal",
        "net_pnl",
        "gross_pnl",
        "commission",
        "slippage",
    ]
    optional = [c for c in ["target_extreme50"] if c in out.columns]
    return out[[c for c in keep + optional if c in out.columns]].sort_values("delivery_hour_utc").reset_index(drop=True)


def build_c1() -> pd.DataFrame:
    reg = pd.read_parquet(OUT / "experiments" / "B2_regression_formal_v3r1_f5730506" / "predictions.parquet")
    clf = pd.read_parquet(OUT / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "predictions.parquet")
    keys = ["delivery_hour_utc", "fold_id"]
    merged = clf.drop(columns=["predicted_spread"], errors="ignore").merge(
        reg[keys + ["predicted_spread"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(clf) or len(merged) != len(reg):
        raise ValueError(f"C1 merge row mismatch: reg={len(reg)} clf={len(clf)} merged={len(merged)}")
    if not np.allclose(merged["actual_spread"], clf["actual_spread"], equal_nan=True):
        raise ValueError("C1 actual_spread alignment failed")
    return normalize_system(merged, "C1_best_boosting_complete_system", "XGBoost")


def build_c2() -> pd.DataFrame:
    df = pd.read_parquet(
        OUT
        / "B_single_models"
        / "B4C_seq2seq_multitask_v3_f5730506"
        / "oof_2025_B4C_seq2seq_multitask_v3.parquet"
    )
    return normalize_system(df, "C2_multitask_lstm_complete_system", "Seq2Seq_LSTM")


def prediction_metrics(df: pd.DataFrame) -> dict:
    y = class_int(df["actual_class"])
    pred = class_int(df["predicted_class"])
    p = df[[f"p_c{i}" for i in range(1, 6)]].astype(float).to_numpy()
    row_sum = p.sum(axis=1)
    valid_p = np.isfinite(p).all(axis=1) & (row_sum > 0)
    p_norm = p[valid_p] / row_sum[valid_p, None]
    true_dir = direction_from_class(y)
    pred_dir = direction_from_class(pred)
    catastrophic = ((y <= 2) & (pred >= 4)) | ((y >= 4) & (pred <= 2))
    outer = y.isin([1, 5])
    extreme_weather = pd.to_numeric(df.get("fixed_extreme_weather_flag", 0), errors="coerce").fillna(0).astype(bool)
    return {
        "system_id": df["system_id"].iloc[0],
        "model_family": df["model_family"].iloc[0],
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y.iloc[valid_p], p_norm, labels=[1, 2, 3, 4, 5])) if valid_p.any() else np.nan,
        "mean_abs_class_distance": float(np.abs(y.to_numpy() - pred.to_numpy()).mean()),
        "catastrophic_reversal_rate": float(catastrophic.mean()),
        "direction_accuracy": float((true_dir == pred_dir).mean()),
        "outer_macro_f1": float(f1_score(y[outer], pred[outer], labels=[1, 5], average="macro", zero_division=0)) if outer.any() else np.nan,
        "extreme_weather_macro_f1": float(f1_score(y[extreme_weather], pred[extreme_weather], labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)) if extreme_weather.any() else np.nan,
        "normal_weather_macro_f1": float(f1_score(y[~extreme_weather], pred[~extreme_weather], labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)) if (~extreme_weather).any() else np.nan,
    }


def economic_metrics(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["delivery_hour_utc"] = pd.to_datetime(d["delivery_hour_utc"], utc=True)
    pnl = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
    signal = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int)
    traded = signal.ne(0)
    daily = pnl.groupby(pd.to_datetime(d["delivery_date_local"]).dt.date).sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(365.0)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std(ddof=1) * math.sqrt(365.0)) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    gross_profit = pnl[(traded) & (pnl > 0)].sum()
    gross_loss = -pnl[(traded) & (pnl < 0)].sum()
    extreme_weather = pd.to_numeric(d.get("fixed_extreme_weather_flag", 0), errors="coerce").fillna(0).astype(bool)
    return {
        "system_id": d["system_id"].iloc[0],
        "total_pnl": float(pnl.sum()),
        "total_return": float(pnl.sum() / INITIAL_CAPITAL),
        "trade_count": int(traded.sum()),
        "coverage": float(traded.mean()),
        "pnl_per_mwh": float(pnl[traded].mean()) if traded.any() else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "win_rate": float((pnl[traded] > 0).mean()) if traded.any() else np.nan,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "maximum_single_loss": float(pnl.min()) if len(pnl) else np.nan,
        "inc_trade_count": int((signal == -1).sum()),
        "dec_trade_count": int((signal == 1).sum()),
        "inc_pnl": float(pnl[signal == -1].sum()),
        "dec_pnl": float(pnl[signal == 1].sum()),
        "profitable_months": int(pnl.groupby(d["delivery_hour_utc"].dt.to_period("M").astype(str)).sum().gt(0).sum()),
        "extreme_weather_pnl": float(pnl[extreme_weather].sum()),
        "normal_weather_pnl": float(pnl[~extreme_weather].sum()),
    }


def markdown_table(df: pd.DataFrame) -> str:
    txt = df.copy()
    for c in txt.columns:
        if pd.api.types.is_float_dtype(txt[c]):
            txt[c] = txt[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            txt[c] = txt[c].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(txt.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(txt.columns)) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in txt.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def update_progress(files: list[Path], selected: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with (PROGRESS / "PHASE_C_PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n\n## {stamp} - C systems, quality gate and economic gate completed\n"
            f"- selected_2025_candidate: {selected}\n"
            f"- next_task: Z0-Z3 ablation and explainability preparation\n"
            + "".join(f"\n- generated: {p}" for p in files)
        )
    state_path = PROGRESS / "progress_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except json.JSONDecodeError:
        state = {}
    completed = list(dict.fromkeys(state.get("completed_runs", []) + [RUN_ID, "C1", "C2", "Prediction Quality Gate", "Economic Value Gate"]))
    state.update(
        {
            "updated_at_utc": stamp,
            "current_phase": "post_C_gates",
            "current_task": "ablation_and_explainability_ready",
            "running": [],
            "completed_runs": completed,
            "last_completed_task": "Economic Value Gate",
            "next": "Z0_Z3_ablation",
            "frozen_2026_access": "prohibited",
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for d in [
        OUT / "C_combination_systems",
        OUT / "prediction_quality_gate",
        OUT / "economic_value_gate",
        OUT / "model_freeze",
        REPORTS,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    c1 = build_c1()
    c2 = build_c2()
    c1_path = OUT / "C_combination_systems" / "C1_best_boosting_complete_system_oof_2025_v3.parquet"
    c2_path = OUT / "C_combination_systems" / "C2_multitask_lstm_complete_system_oof_2025_v3.parquet"
    all_path = OUT / "C_combination_systems" / "C_systems_oof_2025_v3.parquet"
    c1.to_parquet(c1_path, index=False)
    c2.to_parquet(c2_path, index=False)
    all_df = pd.concat([c1, c2], ignore_index=True)
    all_df.to_parquet(all_path, index=False)

    quality = pd.DataFrame([prediction_metrics(c1), prediction_metrics(c2)])
    econ = pd.DataFrame([economic_metrics(c1), economic_metrics(c2)])
    quality_path = OUT / "prediction_quality_gate" / "system_prediction_metrics_2025_v3.csv"
    econ_path = OUT / "economic_value_gate" / "system_economic_metrics_2025_v3.csv"
    daily_path = OUT / "economic_value_gate" / "system_daily_pnl_2025_v3.parquet"
    quality.to_csv(quality_path, index=False)
    econ.to_csv(econ_path, index=False)
    all_df.assign(delivery_date_local=pd.to_datetime(all_df["delivery_date_local"]).dt.date.astype(str)).groupby(
        ["system_id", "delivery_date_local"], as_index=False
    )["net_pnl"].sum().to_parquet(daily_path, index=False)

    ranked = quality.merge(econ, on="system_id", how="left")
    ranked = ranked.sort_values(
        ["macro_f1", "log_loss", "catastrophic_reversal_rate", "sharpe", "max_drawdown"],
        ascending=[False, True, True, False, False],
    )
    selected = str(ranked.iloc[0]["system_id"])
    freeze = {
        "run_id": RUN_ID,
        "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        "selection_based_on": "2025_OOF_only",
        "selected_prediction_agent": selected,
        "candidate_systems": ["C1_best_boosting_complete_system", "C2_multitask_lstm_complete_system"],
        "failed_archived_systems": ["C3_multitask_tft_complete_system"],
        "source_data_hash": DATA_HASH,
        "feature_whitelist_hash": FEATURE_HASH,
        "probability_threshold": PROB_THRESHOLD,
        "trading_cost_rule": {
            "capture_rate": CAPTURE_RATE,
            "commission_usd_per_mwh": COMMISSION,
            "slippage_bps": SLIPPAGE_BPS,
        },
        "ready_for_2026": False,
        "notes": "2026 remains prohibited until explicit authorization after freeze review.",
    }
    freeze_path = OUT / "model_freeze" / "final_prediction_agent_selection_2025_v3.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    c3_fail_dir = OUT / "failed_experiments" / "C3_multitask_tft_complete_system_v3_f5730506"
    c3_fail_dir.mkdir(parents=True, exist_ok=True)
    (c3_fail_dir / "failure_summary.md").write_text(
        "# C3 Multi-task TFT完整系统失败归档\n\n"
        "B5C true multi-task TFT trainer 不存在且未成功运行，因此 C3 不进入正式比较。\n",
        encoding="utf-8",
    )

    report_c = REPORTS / "C_GROUP_COMPLETE_SYSTEM_COMPARISON.md"
    report_q = REPORTS / "PREDICTION_QUALITY_GATE.md"
    report_e = REPORTS / "ECONOMIC_VALUE_GATE.md"
    report_f = REPORTS / "FINAL_PREDICTION_AGENT_SELECTION.md"
    report_c.write_text(
        "# C组完整预测系统比较报告\n\n"
        f"- run_id: `{RUN_ID}`\n"
        "- C1：B2A XGBoost连续预测 + B2B XGBoost 5/20概率。\n"
        "- C2：B4C Seq2Seq LSTM multi-task。\n"
        "- C3：TFT multi-task失败归档，不进入比较。\n\n"
        "## Prediction Quality\n\n"
        f"{markdown_table(quality)}\n\n"
        "## Economic Gate\n\n"
        f"{markdown_table(econ)}\n",
        encoding="utf-8",
    )
    report_q.write_text(
        "# Prediction Quality Gate\n\n"
        "主排序使用 Macro-F1，其次 Log Loss、灾难性反转率。普通 Accuracy 不作为主排名依据。\n\n"
        f"{markdown_table(quality)}\n",
        encoding="utf-8",
    )
    report_e.write_text(
        "# Economic Value Gate\n\n"
        "统一使用固定交易成本、0.60概率阈值、65% capture rate；未做阈值优化，未访问2026。\n\n"
        f"{markdown_table(econ)}\n",
        encoding="utf-8",
    )
    report_f.write_text(
        "# final_prediction_agent选择报告\n\n"
        f"- 2025 OOF选出：`{selected}`。\n"
        "- 经济指标仅作为第二阶段检验，不用于回头修改模型或阈值。\n"
        "- 2026仍未运行。\n\n"
        f"冻结文件：`{freeze_path}`\n",
        encoding="utf-8",
    )

    files = [c1_path, c2_path, all_path, quality_path, econ_path, daily_path, freeze_path, report_c, report_q, report_e, report_f]
    update_progress(files, selected)
    print("C组与2025质量/经济门完成")
    print(f"selected={selected}")
    print(f"quality={quality_path}")
    print(f"economic={econ_path}")
    print(f"freeze={freeze_path}")


if __name__ == "__main__":
    main()
