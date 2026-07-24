from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "outputs" / "B_single_models"
REPORT_DIR = BASE / "reports"
PROGRESS_DIR = BASE / "progress"
RUN_ID = "B6_single_model_comparison_v3_f5730506"
DATA_HASH = "f5730506707c2f227f6208bb6bc00ca4c0c45fe5a23c3148c1e9c2c04cfa0717"


CANDIDATES = [
    {
        "experiment_id": "B1A",
        "model_name": "Ridge Regression",
        "task": "continuous",
        "status": "COMPLETED",
        "comparable": True,
        "path": BASE / "outputs" / "experiments" / "B1_ridge_formal_v3_f5730506" / "predictions.parquet",
    },
    {
        "experiment_id": "B1B",
        "model_name": "Logistic 5/20",
        "task": "classification_5_20",
        "status": "COMPLETED",
        "comparable": True,
        "path": BASE / "outputs" / "experiments" / "B1_logistic_5_20_formal_v3_f5730506" / "predictions.parquet",
    },
    {
        "experiment_id": "B2A",
        "model_name": "XGBoost Regression",
        "task": "continuous",
        "status": "COMPLETED",
        "comparable": True,
        "path": BASE / "outputs" / "experiments" / "B2_regression_formal_v3r1_f5730506" / "predictions.parquet",
    },
    {
        "experiment_id": "B2B",
        "model_name": "XGBoost 5/20",
        "task": "classification_5_20",
        "status": "COMPLETED",
        "comparable": True,
        "path": BASE / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "predictions.parquet",
    },
    {
        "experiment_id": "B3A",
        "model_name": "LightGBM Regression Import",
        "task": "continuous",
        "status": "COMPLETED_IMPORTED",
        "comparable": True,
        "path": OUT_DIR / "B3_lightgbm_import" / "B3A_lightgbm_regression_oof_2025_normalized.parquet",
    },
    {
        "experiment_id": "B3B",
        "model_name": "LightGBM 5/20 Import",
        "task": "classification_5_20",
        "status": "COMPLETED_IMPORTED",
        "comparable": True,
        "path": OUT_DIR / "B3_lightgbm_import" / "B3B_lightgbm_5_20_classifier_oof_2025_normalized.parquet",
    },
    {
        "experiment_id": "B4A",
        "model_name": "Seq2Seq LSTM Continuous",
        "task": "continuous",
        "status": "COMPLETED",
        "comparable": True,
        "path": OUT_DIR / "B4A_seq2seq_continuous_v3_f5730506" / "oof_2025_B4A_seq2seq_continuous_v3.parquet",
    },
    {
        "experiment_id": "B4B",
        "model_name": "Seq2Seq LSTM 5/20",
        "task": "classification_5_20",
        "status": "COMPLETED",
        "comparable": True,
        "path": OUT_DIR / "B4B_seq2seq_classifier_5_20_v3_f5730506" / "oof_2025_B4B_seq2seq_classifier_5_20_v3.parquet",
    },
    {
        "experiment_id": "B4C",
        "model_name": "Seq2Seq LSTM Multi-task",
        "task": "multitask",
        "status": "COMPLETED",
        "comparable": True,
        "path": OUT_DIR / "B4C_seq2seq_multitask_v3_f5730506" / "oof_2025_B4C_seq2seq_multitask_v3.parquet",
    },
    {
        "experiment_id": "B5A",
        "model_name": "TFT Distribution",
        "task": "continuous",
        "status": "COMPLETED",
        "comparable": True,
        "path": OUT_DIR / "B5A_tft_distribution_v3_f5730506" / "oof_2025_B5A_tft_distribution_v3.parquet",
    },
    {
        "experiment_id": "B5B",
        "model_name": "TFT 5/20",
        "task": "classification_5_20",
        "status": "COMPLETED",
        "comparable": True,
        "path": OUT_DIR / "B5B_tft_classifier_5_20_v3_f5730506" / "oof_2025_B5B_tft_classifier_5_20_v3.parquet",
    },
    {
        "experiment_id": "B5C",
        "model_name": "TFT Multi-task",
        "task": "multitask",
        "status": "FAILED_ARCHIVED",
        "comparable": False,
        "path": BASE / "outputs" / "failed_experiments" / "B5C_tft_multitask_v3_f5730506" / "status.json",
    },
]


def class_5_20_from_spread(values: pd.Series) -> pd.Series:
    bins = [-np.inf, -20.0, -5.0, 5.0, 20.0, np.inf]
    return pd.cut(values.astype(float), bins=bins, labels=[1, 2, 3, 4, 5], right=True).astype(int)


def to_class_int(series: pd.Series, actual_spread: pd.Series | None = None) -> pd.Series:
    if series is None:
        if actual_spread is None:
            raise ValueError("class series and actual_spread are both missing")
        return class_5_20_from_spread(actual_spread)
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        mn = float(pd.to_numeric(s, errors="coerce").min())
        mx = float(pd.to_numeric(s, errors="coerce").max())
        if mn >= 0 and mx <= 4:
            return pd.to_numeric(s, errors="coerce").astype(int) + 1
        return pd.to_numeric(s, errors="coerce").astype(int)
    return s.astype(str).str.extract(r"(\d+)", expand=False).astype(int)


def direction_from_class(c: pd.Series) -> pd.Series:
    return np.select([c <= 2, c == 3, c >= 4], [-1, 0, 1], default=np.nan)


def safe_log_loss(y_true: pd.Series, df: pd.DataFrame) -> float:
    proba_cols = [f"p_c{i}" for i in range(1, 6)]
    if not set(proba_cols).issubset(df.columns):
        return np.nan
    p = df[proba_cols].astype(float).to_numpy()
    row_sum = p.sum(axis=1)
    valid = np.isfinite(p).all(axis=1) & (row_sum > 0)
    if valid.sum() == 0:
        return np.nan
    p = p[valid] / row_sum[valid, None]
    return float(log_loss(y_true.loc[valid].to_numpy(), p, labels=[1, 2, 3, 4, 5]))


def pnl_metrics(df: pd.DataFrame) -> dict[str, float]:
    if "net_pnl" not in df.columns:
        return {"total_pnl": np.nan, "trade_count": np.nan, "pnl_per_trade": np.nan, "daily_sharpe": np.nan}
    pnl = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
    if "trade_signal" in df.columns:
        trades = pd.to_numeric(df["trade_signal"], errors="coerce").fillna(0).ne(0)
    elif "signal" in df.columns:
        trades = pd.to_numeric(df["signal"], errors="coerce").fillna(0).ne(0)
    else:
        trades = pnl.ne(0)
    if "delivery_date_local" in df.columns:
        day = pd.to_datetime(df["delivery_date_local"]).dt.date
    elif "delivery_time_local" in df.columns:
        day = pd.to_datetime(df["delivery_time_local"], errors="coerce", utc=True).dt.date
    else:
        day = pd.to_datetime(df["delivery_hour_utc"], errors="coerce").dt.date
    daily = pnl.groupby(day).sum()
    sharpe = np.nan
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.0))
    trade_count = int(trades.sum())
    return {
        "total_pnl": float(pnl.sum()),
        "trade_count": trade_count,
        "pnl_per_trade": float(pnl[trades].mean()) if trade_count else np.nan,
        "daily_sharpe": sharpe,
    }


def max_drawdown(df: pd.DataFrame) -> float:
    if "net_pnl" not in df.columns:
        return np.nan
    d = df.copy()
    d["delivery_hour_utc"] = pd.to_datetime(d["delivery_hour_utc"], utc=True, errors="coerce")
    equity = pd.to_numeric(d.sort_values("delivery_hour_utc")["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    if equity.empty:
        return np.nan
    return float((equity - equity.cummax()).min())


def summarize_candidate(candidate: dict) -> dict:
    row = {
        "experiment_id": candidate["experiment_id"],
        "model_name": candidate["model_name"],
        "task": candidate["task"],
        "status": candidate["status"],
        "comparable": candidate["comparable"],
        "source_path": str(candidate["path"]),
        "run_id": np.nan,
        "rows": 0,
        "fold_count": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "direction_accuracy": np.nan,
        "accuracy_5_20": np.nan,
        "macro_f1_5_20": np.nan,
        "balanced_accuracy_5_20": np.nan,
        "log_loss_5_20": np.nan,
        "mean_abs_class_distance_5_20": np.nan,
        "catastrophic_reversal_rate": np.nan,
        "extreme_hour_mae": np.nan,
        "total_pnl": np.nan,
        "trade_count": np.nan,
        "pnl_per_trade": np.nan,
        "daily_sharpe": np.nan,
        "max_drawdown": np.nan,
        "notes": "",
    }
    path = candidate["path"]
    if not path.exists():
        row.update(status="MISSING", comparable=False, notes="source file missing")
        return row
    if not candidate["comparable"]:
        row["notes"] = "archived failure; excluded from formal B6 comparison"
        return row

    df = pd.read_parquet(path)
    row["rows"] = int(len(df))
    if "run_id" in df.columns:
        run_ids = sorted(df["run_id"].dropna().astype(str).unique())
        row["run_id"] = ";".join(run_ids[:3])
    if "fold_id" in df.columns:
        row["fold_count"] = int(df["fold_id"].nunique())

    actual_spread = pd.to_numeric(df["actual_spread"], errors="coerce")
    y_true = to_class_int(df["actual_class"] if "actual_class" in df.columns else None, actual_spread)

    if "predicted_spread" in df.columns:
        pred_spread = pd.to_numeric(df["predicted_spread"], errors="coerce")
        valid_reg = actual_spread.notna() & pred_spread.notna()
        if valid_reg.any():
            yt = actual_spread.loc[valid_reg]
            yp = pred_spread.loc[valid_reg]
            row["mae"] = float(mean_absolute_error(yt, yp))
            row["rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
            denom = float(((yt - yt.mean()) ** 2).sum())
            row["r2"] = float(1.0 - ((yt - yp) ** 2).sum() / denom) if denom > 0 else np.nan
            row["direction_accuracy"] = float((np.sign(yt) == np.sign(yp)).mean())
            if "target_extreme20" in df.columns:
                extreme = pd.to_numeric(df.loc[valid_reg, "target_extreme20"], errors="coerce").fillna(0).astype(bool)
            else:
                extreme = yt.abs() > 20
            row["extreme_hour_mae"] = float(mean_absolute_error(yt.loc[extreme], yp.loc[extreme])) if extreme.any() else np.nan

    if "predicted_class" in df.columns and df["predicted_class"].notna().any():
        y_pred = to_class_int(df["predicted_class"], None)
    elif "predicted_spread" in df.columns:
        y_pred = class_5_20_from_spread(pd.to_numeric(df["predicted_spread"], errors="coerce"))
    else:
        proba_cols = [f"p_c{i}" for i in range(1, 6)]
        p = df[proba_cols].astype(float).to_numpy()
        y_pred = pd.Series(np.nanargmax(p, axis=1) + 1, index=df.index)

    valid_cls = y_true.notna() & y_pred.notna()
    if valid_cls.any():
        yt_cls = y_true.loc[valid_cls].astype(int)
        yp_cls = y_pred.loc[valid_cls].astype(int)
        row["accuracy_5_20"] = float(accuracy_score(yt_cls, yp_cls))
        row["macro_f1_5_20"] = float(f1_score(yt_cls, yp_cls, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0))
        row["balanced_accuracy_5_20"] = float(balanced_accuracy_score(yt_cls, yp_cls))
        row["log_loss_5_20"] = safe_log_loss(yt_cls, df.loc[valid_cls])
        row["mean_abs_class_distance_5_20"] = float(np.abs(yt_cls.to_numpy() - yp_cls.to_numpy()).mean())
        true_dir = direction_from_class(yt_cls)
        pred_dir = direction_from_class(yp_cls)
        row["direction_accuracy"] = float((true_dir == pred_dir).mean())
        catastrophic = ((yt_cls <= 2) & (yp_cls >= 4)) | ((yt_cls >= 4) & (yp_cls <= 2))
        row["catastrophic_reversal_rate"] = float(catastrophic.mean())

    row.update(pnl_metrics(df))
    row["max_drawdown"] = max_drawdown(df)
    return row


def update_progress(summary_path: Path, winners: dict) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    progress_md = PROGRESS_DIR / "PHASE_C_PROGRESS.md"
    stamp = datetime.now(timezone.utc).isoformat()
    with progress_md.open("a", encoding="utf-8") as f:
        f.write(
            f"\n\n## {stamp} - B6 single-model comparison completed\n"
            f"- comparison_table: {summary_path}\n"
            f"- best_continuous: {winners.get('best_continuous')}\n"
            f"- best_classifier: {winners.get('best_classifier')}\n"
            f"- best_deep: {winners.get('best_deep')}\n"
            f"- next_task: C1 best boosting complete system\n"
        )
    state_path = PROGRESS_DIR / "progress_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except json.JSONDecodeError:
        state = {}
    completed = list(dict.fromkeys(state.get("completed_runs", []) + [RUN_ID, "B6_single_model_comparison"]))
    state.update(
        {
            "updated_at_utc": stamp,
            "current_phase": "C_group_ready",
            "current_model": None,
            "current_task": "B6 completed; next C1",
            "running": [],
            "completed_runs": completed,
            "last_completed_task": "B6_single_model_comparison",
            "next": "C1_best_boosting_complete_system",
            "frozen_2026_access": "prohibited_until_final_2025_model_freeze",
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_candidate(c) for c in CANDIDATES]
    metrics = pd.DataFrame(rows)

    comparable = metrics[metrics["comparable"].eq(True)].copy()
    continuous = comparable[comparable["task"].isin(["continuous", "multitask"])].copy()
    classifiers = comparable[comparable["task"].isin(["classification_5_20", "multitask"])].copy()
    deep = comparable[comparable["experiment_id"].str.startswith(("B4", "B5"))].copy()

    winners = {}
    if not continuous.empty:
        winners["best_continuous"] = continuous.sort_values(["mae", "rmse"], ascending=[True, True]).iloc[0]["experiment_id"]
    if not classifiers.empty:
        winners["best_classifier"] = classifiers.sort_values(
            ["macro_f1_5_20", "log_loss_5_20", "catastrophic_reversal_rate"],
            ascending=[False, True, True],
        ).iloc[0]["experiment_id"]
    if not deep.empty:
        deep_rank = deep.copy()
        deep_rank["deep_score"] = deep_rank["macro_f1_5_20"].fillna(-np.inf)
        winners["best_deep"] = deep_rank.sort_values(
            ["deep_score", "mae", "log_loss_5_20"], ascending=[False, True, True]
        ).iloc[0]["experiment_id"]

    metrics_path = OUT_DIR / "B6_single_model_comparison_v3.csv"
    winners_path = OUT_DIR / "B6_task_winners_v3.json"
    metrics.to_csv(metrics_path, index=False)
    winners_path.write_text(json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8")

    table_cols = [
        "experiment_id",
        "model_name",
        "task",
        "status",
        "comparable",
        "rows",
        "mae",
        "rmse",
        "r2",
        "direction_accuracy",
        "macro_f1_5_20",
        "log_loss_5_20",
        "catastrophic_reversal_rate",
        "total_pnl",
        "trade_count",
        "daily_sharpe",
        "max_drawdown",
    ]
    display = metrics[table_cols].copy()
    report = [
        "# B组单模型比较报告",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- data_hash: `{DATA_HASH}`",
        "- 范围：仅2025 OOF；未访问2026。",
        "- 规则：失败或 NOT_COMPARABLE 的实验只归档，不进入正式排名。",
        "",
        "## 单模型汇总表",
        "",
        markdown_table(display),
        "",
        "## B6结论",
        "",
        f"- 最佳连续/分布单任务：`{winners.get('best_continuous', 'NA')}`。",
        f"- 最佳五分类单任务：`{winners.get('best_classifier', 'NA')}`。",
        f"- 最佳深度候选：`{winners.get('best_deep', 'NA')}`。",
        "- `B5C` 已按失败归档，原因是当前可运行 TFT baseline 不支持真正单模型 multi-task trainer；未用拼接结果冒充。",
        "",
        "## 生成文件",
        "",
        f"- `{metrics_path}`",
        f"- `{winners_path}`",
    ]
    report_path = REPORT_DIR / "B_GROUP_SINGLE_MODEL_COMPARISON.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    update_progress(metrics_path, winners)
    print("B6单模型比较完成")
    print(f"summary_csv={metrics_path}")
    print(f"report={report_path}")
    print(f"winners={winners_path}")
    print(json.dumps(winners, ensure_ascii=False))


if __name__ == "__main__":
    main()
