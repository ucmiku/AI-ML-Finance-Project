from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phase_C_model_selection_validation"
DATA_PATH = ROOT / "data" / "model_ready" / "model_input_frozen_v2.parquet"
WHITELIST_PATH = ROOT / "config" / "feature_whitelist_v2.csv"
REG_PARAMS_PATH = PHASE / "outputs" / "experiments" / "B2_regression_formal_v3r1_f5730506" / "best_params.json"
CLF_PARAMS_PATH = PHASE / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "best_params.json"
OUT = PHASE / "outputs" / "ablation"
MODEL_DIR = PHASE / "models" / "C1_Z0_Z3_ablation_2025_v3_f5730506"
REPORTS = PHASE / "reports"
RUN_ID = "C1_Z0_Z3_ablation_2025_v3_f5730506"
FOLDS = ["validation_fold_1", "validation_fold_2", "validation_fold_3"]
CLASSES = [1, 2, 3, 4, 5]


def emit(message: str) -> None:
    print(message, flush=True)


def class_5_20(spread: pd.Series | np.ndarray) -> np.ndarray:
    v = np.asarray(spread, dtype=float)
    return np.select([v < -20, v < -5, v <= 5, v <= 20], [1, 2, 3, 4], default=5).astype(int)


def feature_set(df: pd.DataFrame, wl: pd.DataFrame, z: str) -> list[str]:
    allowed = wl.loc[
        wl["feature_set"].eq(z)
        & wl["fold_fitted"].eq(0)
        & wl["leakage_status"].astype(str).str.contains("approved", case=False, na=False),
        "feature_name",
    ].drop_duplicates()
    exclusions = ("target", "eligible", "issue_time", "source_product", "split_name", "research_period", "validation_fold_id", "availability", "qc")
    features = [
        c for c in allowed
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(token in c.lower() for token in exclusions)
    ]
    if not features:
        raise RuntimeError(f"No usable features for {z}")
    return features


def fold_masks(df: pd.DataFrame, fold: str) -> tuple[pd.Series, pd.Series]:
    val = df["validation_fold_id"].eq(fold) & df["evaluation_eligible"].eq(1)
    start = df.loc[val, "delivery_hour_utc"].min()
    train = df["delivery_hour_utc"].lt(start) & df["evaluation_eligible"].eq(1)
    if not val.any() or not train.any() or df.loc[train, "delivery_hour_utc"].max() >= start:
        raise RuntimeError(f"Invalid expanding fold: {fold}")
    return train, val


def output_frame(df: pd.DataFrame, val: pd.Series, z: str, fold: str, pred_spread: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    part = df.loc[val].copy()
    actual = part["spread"].to_numpy(float)
    out = pd.DataFrame({
        "delivery_hour_utc": part["delivery_hour_utc"].to_numpy(),
        "delivery_time_local": part["delivery_time_local"].to_numpy(),
        "delivery_date_local": part["delivery_date_local"].to_numpy(),
        "fold_id": fold,
        "feature_set": z,
        "system_id": f"C1_ablation_{z}",
        "run_id": RUN_ID,
        "predicted_spread": pred_spread,
        "p_c1": proba[:, 0],
        "p_c2": proba[:, 1],
        "p_c3": proba[:, 2],
        "p_c4": proba[:, 3],
        "p_c5": proba[:, 4],
        "p_negative": proba[:, 0] + proba[:, 1],
        "p_no_trade": proba[:, 2],
        "p_positive": proba[:, 3] + proba[:, 4],
        "p_outer": proba[:, 0] + proba[:, 4],
        "predicted_class": proba.argmax(axis=1) + 1,
        "confidence": proba.max(axis=1),
        "actual_class": class_5_20(actual),
        "actual_spread": actual,
        "fixed_extreme_weather_flag": part["fixed_extreme_weather_flag"].to_numpy(),
        "target_extreme20": part["target_extreme20"].to_numpy(),
        "target_extreme50": part["target_extreme50"].to_numpy(),
    })
    out["signal"] = np.where((out.p_positive >= 0.60) & (out.p_positive > out.p_negative), 1,
                             np.where((out.p_negative >= 0.60) & (out.p_negative > out.p_positive), -1, 0))
    clipped = out["actual_spread"].clip(-1000, 5000)
    out["net_pnl"] = out["signal"] * clipped * 0.65 - out["signal"].ne(0).astype(float) * 2.0 - out["signal"].ne(0).astype(float) * clipped.abs() * 0.005
    if float(np.abs(out[["p_c1", "p_c2", "p_c3", "p_c4", "p_c5"]].sum(axis=1) - 1).max()) > 1e-6:
        raise RuntimeError(f"Probability sum failed for {z} {fold}")
    return out


def fit_fold(df: pd.DataFrame, features: list[str], z: str, fold: str, reg_params: dict, clf_params: dict) -> tuple[pd.DataFrame, dict]:
    train, val = fold_masks(df, fold)
    reg_imp = SimpleImputer(strategy="median")
    clf_imp = SimpleImputer(strategy="median")
    x_train_reg = reg_imp.fit_transform(df.loc[train, features])
    x_val_reg = reg_imp.transform(df.loc[val, features])
    reg = XGBRegressor(**reg_params)
    reg.fit(x_train_reg, df.loc[train, "spread"].to_numpy(float), eval_set=[(x_val_reg, df.loc[val, "spread"].to_numpy(float))], verbose=False)
    pred_spread = reg.predict(x_val_reg)

    x_train_clf = clf_imp.fit_transform(df.loc[train, features])
    x_val_clf = clf_imp.transform(df.loc[val, features])
    y_train = class_5_20(df.loc[train, "spread"])
    y_val = class_5_20(df.loc[val, "spread"])
    counts = pd.Series(y_train).value_counts()
    weights = pd.Series(y_train).map({c: len(y_train) / (5 * counts.get(c, 1)) for c in CLASSES}).to_numpy(float)
    clf = XGBClassifier(**clf_params)
    clf.fit(x_train_clf, y_train - 1, sample_weight=weights, eval_set=[(x_val_clf, y_val - 1)], verbose=False)
    raw = clf.predict_proba(x_val_clf)
    proba = np.zeros((len(x_val_clf), 5))
    proba[:, clf.classes_.astype(int)] = raw
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = output_frame(df, val, z, fold, pred_spread, proba)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": reg_imp, "model": reg, "features": features}, MODEL_DIR / f"{z}_{fold}_reg.joblib")
    joblib.dump({"imputer": clf_imp, "model": clf, "features": features}, MODEL_DIR / f"{z}_{fold}_clf.joblib")
    metric = metrics_prediction(pred, z, fold)
    metric.update({
        "feature_count": len(features),
        "train_rows": int(train.sum()),
        "validation_rows": int(val.sum()),
        "best_iteration_reg": getattr(reg, "best_iteration", None),
        "best_iteration_clf": getattr(clf, "best_iteration", None),
    })
    return pred, metric


def metrics_prediction(pred: pd.DataFrame, z: str, fold: str) -> dict:
    y = pred["actual_class"].astype(int)
    p = pred["predicted_class"].astype(int)
    proba = pred[[f"p_c{i}" for i in range(1, 6)]].to_numpy(float)
    extreme = pd.to_numeric(pred["fixed_extreme_weather_flag"], errors="coerce").fillna(0).astype(bool)
    catastrophic = ((y <= 2) & (p >= 4)) | ((y >= 4) & (p <= 2))
    pred_dir = np.where(p >= 4, 1, np.where(p <= 2, -1, 0))
    return {
        "feature_set": z,
        "fold_id": fold,
        "rows": int(len(pred)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "log_loss": float(log_loss(y, proba, labels=CLASSES)),
        "mean_abs_class_distance": float(np.abs(y.to_numpy() - p.to_numpy()).mean()),
        "catastrophic_reversal_rate": float(catastrophic.mean()),
        "direction_accuracy": float((np.sign(pred["actual_spread"]) == pred_dir).mean()),
        "spread_mae": float(mean_absolute_error(pred["actual_spread"], pred["predicted_spread"])),
        "spread_rmse": float(np.sqrt(mean_squared_error(pred["actual_spread"], pred["predicted_spread"]))),
        "spread_r2": float(r2_score(pred["actual_spread"], pred["predicted_spread"])),
        "extreme_weather_macro_f1": float(f1_score(y[extreme], p[extreme], labels=CLASSES, average="macro", zero_division=0)) if extreme.any() else np.nan,
        "normal_weather_macro_f1": float(f1_score(y[~extreme], p[~extreme], labels=CLASSES, average="macro", zero_division=0)) if (~extreme).any() else np.nan,
    }


def cvar(daily: pd.Series, level: float = 0.95) -> float:
    q = daily.quantile(1 - level)
    tail = daily[daily <= q]
    return float(tail.mean()) if not tail.empty else np.nan


def metrics_economic(pred: pd.DataFrame, z: str) -> dict:
    d = pred.sort_values("delivery_hour_utc").copy()
    pnl = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
    sig = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int)
    traded = sig.ne(0)
    day = pd.to_datetime(d["delivery_date_local"], errors="coerce").dt.date
    daily = pnl.groupby(day).sum()
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    downside = daily[daily < 0]
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(365)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    sortino = float(daily.mean() / downside.std(ddof=1) * math.sqrt(365)) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    gp = pnl[traded & (pnl > 0)].sum()
    gl = -pnl[traded & (pnl < 0)].sum()
    extreme = pd.to_numeric(d["fixed_extreme_weather_flag"], errors="coerce").fillna(0).astype(bool)
    tail20 = pd.to_numeric(d["target_extreme20"], errors="coerce").fillna(0).astype(bool)
    tail50 = pd.to_numeric(d["target_extreme50"], errors="coerce").fillna(0).astype(bool)
    total = float(pnl.sum())
    day_sorted = daily.sort_values(ascending=False)
    return {
        "feature_set": z,
        "total_pnl": total,
        "trade_count": int(traded.sum()),
        "coverage": float(traded.mean()),
        "direction_precision": float((np.sign(d.loc[traded, "actual_spread"]) == sig[traded]).mean()) if traded.any() else np.nan,
        "pnl_per_mwh": float(pnl[traded].mean()) if traded.any() else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "cvar_95_daily": cvar(daily),
        "win_rate": float((pnl[traded] > 0).mean()) if traded.any() else np.nan,
        "profit_factor": float(gp / gl) if gl > 0 else np.nan,
        "maximum_single_loss": float(pnl.min()) if len(pnl) else np.nan,
        "profitable_months": int(pnl.groupby(pd.to_datetime(d["delivery_hour_utc"], utc=True).dt.to_period("M").astype(str)).sum().gt(0).sum()),
        "inc_count": int((sig == -1).sum()),
        "dec_count": int((sig == 1).sum()),
        "inc_pnl": float(pnl[sig == -1].sum()),
        "dec_pnl": float(pnl[sig == 1].sum()),
        "extreme_weather_pnl": float(pnl[extreme].sum()),
        "normal_weather_pnl": float(pnl[~extreme].sum()),
        "extreme_weather_tail20_pnl": float(pnl[extreme & tail20].sum()),
        "extreme_weather_tail50_pnl": float(pnl[extreme & tail50].sum()),
        "pnl_ex_top5_days": float(total - day_sorted.head(5).sum()) if len(day_sorted) else total,
        "top5_day_share": float(day_sorted.head(5).sum() / total) if total != 0 and len(day_sorted) else np.nan,
    }


def markdown_table(df: pd.DataFrame) -> str:
    t = df.copy()
    for col in t.columns:
        if pd.api.types.is_float_dtype(t[col]):
            t[col] = t[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            t[col] = t[col].map(lambda x: "" if pd.isna(x) else str(x))
    return "\n".join([
        "| " + " | ".join(t.columns) + " |",
        "| " + " | ".join(["---"] * len(t.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in t.astype(str).to_numpy()],
    ])


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH, filters=[("delivery_hour_utc", "<", pd.Timestamp("2026-01-01T06:00:00Z"))])
    df["delivery_hour_utc"] = pd.to_datetime(df["delivery_hour_utc"], utc=True)
    df = df.sort_values("delivery_hour_utc").reset_index(drop=True)
    wl = pd.read_csv(WHITELIST_PATH)
    reg_params = json.loads(REG_PARAMS_PATH.read_text(encoding="utf-8"))
    clf_params = json.loads(CLF_PARAMS_PATH.read_text(encoding="utf-8"))
    all_predictions, fold_rows, economic_rows = [], [], []
    inventory = []
    for z in ["Z0", "Z1", "Z2", "Z3"]:
        features = feature_set(df, wl, z)
        inventory.append({"feature_set": z, "feature_count": len(features), "features": ";".join(features)})
        emit(f"[{RUN_ID}] {z} started | features={len(features)}")
        parts = []
        for fold in FOLDS:
            emit(f"[{RUN_ID}] {z} {fold} training fixed C1 heads")
            pred, metric = fit_fold(df, features, z, fold, reg_params, clf_params)
            parts.append(pred)
            fold_rows.append(metric)
            emit(f"[{RUN_ID}] {z} {fold} macro_f1={metric['macro_f1']:.6f} log_loss={metric['log_loss']:.6f} pnl={pred['net_pnl'].sum():.2f}")
        z_pred = pd.concat(parts, ignore_index=True).sort_values("delivery_hour_utc").reset_index(drop=True)
        z_pred.to_parquet(OUT / f"C1_ablation_{z}_oof_2025_v3.parquet", index=False)
        all_predictions.append(z_pred)
        oof_metric = metrics_prediction(z_pred, z, "2025_OOF")
        fold_rows.append(oof_metric)
        economic_rows.append(metrics_economic(z_pred, z))
        emit(f"[{RUN_ID}] {z} completed OOF macro_f1={oof_metric['macro_f1']:.6f}")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(OUT / "C1_Z0_Z3_ablation_predictions_2025_v3.parquet", index=False)
    pred_metrics = pd.DataFrame(fold_rows)
    econ_metrics = pd.DataFrame(economic_rows)
    inventory_df = pd.DataFrame(inventory)
    pred_metrics.to_csv(OUT / "C1_Z0_Z3_ablation_prediction_metrics_2025_v3.csv", index=False)
    econ_metrics.to_csv(OUT / "C1_Z0_Z3_ablation_economic_metrics_2025_v3.csv", index=False)
    inventory_df.to_csv(OUT / "C1_Z0_Z3_ablation_feature_inventory_2025_v3.csv", index=False)
    summary = pred_metrics[pred_metrics["fold_id"].eq("2025_OOF")].merge(econ_metrics, on="feature_set", how="left")
    summary.to_csv(OUT / "C1_Z0_Z3_ablation_summary_2025_v3.csv", index=False)
    report = REPORTS / "C1_Z0_Z3_ABLATION_2025_REPORT.md"
    cols = ["feature_set", "rows", "macro_f1", "log_loss", "direction_accuracy", "spread_mae", "extreme_weather_macro_f1", "total_pnl", "trade_count", "sharpe", "max_drawdown", "extreme_weather_pnl"]
    report.write_text(
        "# C1 Z0-Z3消融实验报告\n\n"
        f"- run_id: `{RUN_ID}`\n"
        "- 范围：2025三折OOF；未访问2026。\n"
        "- 方法：固定C1的B2A/B2B XGBoost超参数，不运行Optuna；每个feature set在各fold内重新拟合imputer、类别权重和模型。\n"
        "- 注意：本实现只使用 `fold_fitted=0` 且已在冻结表中存在的特征，避免在全表提前拟合fold-specific天气异常。\n\n"
        "## Summary\n\n"
        f"{markdown_table(summary[cols])}\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "no_optuna": True,
        "no_2026_access": True,
        "fixed_params": {
            "regression": str(REG_PARAMS_PATH),
            "classifier": str(CLF_PARAMS_PATH),
        },
    }
    (OUT / "C1_Z0_Z3_ablation_manifest_2025_v3.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"[{RUN_ID}] completed runtime_seconds={manifest['runtime_seconds']:.2f}")
    emit(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
