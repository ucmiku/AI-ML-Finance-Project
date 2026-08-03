from __future__ import annotations

import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phase_C_model_selection_validation"
DATA_PATH = ROOT / "data" / "model_ready" / "model_input_frozen_v2.parquet"
WHITELIST_PATH = ROOT / "config" / "feature_whitelist_v2.csv"
REG_PARAMS = PHASE / "outputs" / "experiments" / "B2_regression_formal_v3r1_f5730506" / "best_params.json"
CLF_PARAMS = PHASE / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "best_params.json"
RUN_ID = "C1_C4_weekly_2026_v3_f5730506"
SEED = 20260722
CLASSES = [1, 2, 3, 4, 5]
CLASS_NAMES = ["C1", "C2", "C3", "C4", "C5"]
ENCODER_LENGTH = 168
MAX_HORIZON = 25
HIDDEN_SIZE = 32
EPOCHS = 12
PATIENCE = 4
DEVICE = torch.device("cpu")

OUT_PRED = PHASE / "outputs" / "final_2026"
OUT_METRIC = PHASE / "outputs" / "final_2026"
MODEL_DIR = PHASE / "models" / RUN_ID
LOG_PATH = PHASE / "logs" / f"{RUN_ID}.log"


class Seq2Seq(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, HIDDEN_SIZE, batch_first=True)
        self.decoder = nn.LSTM(HIDDEN_SIZE + n_features, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, encoder_x: torch.Tensor, known_x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(encoder_x)
        context = hidden[-1].unsqueeze(1).expand(-1, known_x.shape[1], -1)
        out, _ = self.decoder(torch.cat([context, known_x], dim=-1), (hidden, cell))
        return self.head(out).squeeze(-1)


def emit(msg: str) -> None:
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def class_5_20(spread: pd.Series | np.ndarray) -> np.ndarray:
    v = np.asarray(spread, dtype=float)
    return np.select([v < -20, v < -5, v <= 5, v <= 20], [1, 2, 3, 4], default=5).astype(int)


def approved_features(df: pd.DataFrame) -> list[str]:
    wl = pd.read_csv(WHITELIST_PATH)
    allowed = wl.loc[(wl["feature_set"].eq("Z3")) & (wl["fold_fitted"].eq(0)), "feature_name"].drop_duplicates()
    exclusions = ("target", "eligible", "issue_time", "source_product", "split_name", "research_period", "validation_fold_id", "availability", "qc")
    features = [
        c for c in allowed
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not any(tok in c.lower() for tok in exclusions)
    ]
    if not features:
        raise RuntimeError("No approved Z3 numeric features available")
    return list(features)


def weekly_windows(df: pd.DataFrame) -> list[dict]:
    days = pd.to_datetime(df["delivery_date_local"]).dt.date
    h1 = df.loc[(days >= date(2026, 1, 1)) & (days <= date(2026, 6, 30)) & df["evaluation_eligible"].eq(1)].copy()
    all_days = sorted(pd.to_datetime(h1["delivery_date_local"]).dt.date.unique())
    if not all_days:
        raise RuntimeError("No eligible 2026 H1 local delivery days")
    first_monday = all_days[0] - timedelta(days=all_days[0].weekday())
    last_day = all_days[-1]
    windows = []
    start = first_monday
    while start <= last_day:
        end = start + timedelta(days=6)
        mask_days = (pd.to_datetime(df["delivery_date_local"]).dt.date >= max(start, date(2026, 1, 1))) & (
            pd.to_datetime(df["delivery_date_local"]).dt.date <= min(end, date(2026, 6, 30))
        )
        val = mask_days & df["evaluation_eligible"].eq(1)
        if val.any():
            first_utc = df.loc[val, "delivery_hour_utc"].min()
            windows.append({"week_id": f"{start.isoformat()}_{end.isoformat()}", "week_start": start, "week_end": end, "val_mask": val, "train_cutoff_utc": first_utc})
        start += timedelta(days=7)
    return windows


def split_history(df: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    hist = df["delivery_hour_utc"].lt(cutoff) & df["evaluation_eligible"].eq(1)
    hist_days = pd.Series(pd.to_datetime(df.loc[hist, "delivery_date_local"]).dt.date.unique()).sort_values()
    eval_days = set(hist_days.tail(7).tolist())
    internal_eval = hist & pd.to_datetime(df["delivery_date_local"]).dt.date.isin(eval_days)
    train_core = hist & ~internal_eval
    if train_core.sum() < 1000 or internal_eval.sum() < 24:
        raise RuntimeError(f"Insufficient historical train/eval split before {cutoff}: train={train_core.sum()} eval={internal_eval.sum()}")
    return train_core, internal_eval


def fit_xgb_week(df: pd.DataFrame, features: list[str], train_core: pd.Series, internal_eval: pd.Series, val_mask: pd.Series, week_id: str) -> tuple[pd.DataFrame, dict, dict]:
    reg_params = json.loads(REG_PARAMS.read_text(encoding="utf-8"))
    clf_params = json.loads(CLF_PARAMS.read_text(encoding="utf-8"))
    reg_imputer = SimpleImputer(strategy="median")
    clf_imputer = SimpleImputer(strategy="median")
    x_train_reg = reg_imputer.fit_transform(df.loc[train_core, features])
    x_eval_reg = reg_imputer.transform(df.loc[internal_eval, features])
    x_val_reg = reg_imputer.transform(df.loc[val_mask, features])
    reg = XGBRegressor(**reg_params)
    reg.fit(x_train_reg, df.loc[train_core, "spread"].to_numpy(float), eval_set=[(x_eval_reg, df.loc[internal_eval, "spread"].to_numpy(float))], verbose=False)
    pred_spread = reg.predict(x_val_reg)

    x_train_clf = clf_imputer.fit_transform(df.loc[train_core, features])
    x_eval_clf = clf_imputer.transform(df.loc[internal_eval, features])
    x_val_clf = clf_imputer.transform(df.loc[val_mask, features])
    y_train = class_5_20(df.loc[train_core, "spread"])
    counts = pd.Series(y_train).value_counts()
    weights = pd.Series(y_train).map({c: len(y_train) / (5 * counts.get(c, 1)) for c in CLASSES}).to_numpy(float)
    clf = XGBClassifier(**clf_params)
    clf.fit(x_train_clf, y_train - 1, sample_weight=weights, eval_set=[(x_eval_clf, class_5_20(df.loc[internal_eval, "spread"]) - 1)], verbose=False)
    raw = clf.predict_proba(x_val_clf)
    proba = np.zeros((len(x_val_clf), 5))
    proba[:, clf.classes_.astype(int)] = raw
    proba = proba / proba.sum(axis=1, keepdims=True)

    part = df.loc[val_mask].copy()
    out = pd.DataFrame({
        "delivery_hour_utc": part["delivery_hour_utc"].to_numpy(),
        "delivery_time_local": part["delivery_time_local"].to_numpy(),
        "delivery_date_local": part["delivery_date_local"].to_numpy(),
        "week_id": week_id,
        "fold_id": week_id,
        "model_name": "C1_best_boosting_complete_system",
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
        "actual_class": class_5_20(part["spread"]),
        "actual_spread": part["spread"].to_numpy(float),
        "fixed_extreme_weather_flag": part["fixed_extreme_weather_flag"].to_numpy(),
        "target_extreme20": part["target_extreme20"].to_numpy(),
        "target_extreme50": part["target_extreme50"].to_numpy(),
    })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": reg_imputer, "model": reg, "features": features}, MODEL_DIR / f"C1_B2A_reg_{week_id}.joblib")
    joblib.dump({"imputer": clf_imputer, "model": clf, "features": features}, MODEL_DIR / f"C1_B2B_clf_{week_id}.joblib")
    return out, {"best_iteration_reg": getattr(reg, "best_iteration", None), "best_iteration_clf": getattr(clf, "best_iteration", None)}, {"reg": reg, "clf": clf}


def day_samples(frame: pd.DataFrame, values: np.ndarray, eligible_days: set[str]) -> list[dict]:
    samples = []
    dates = frame["delivery_date_local"].astype(str).to_numpy()
    for day in sorted(eligible_days):
        idx = np.flatnonzero(dates == day)
        if not 23 <= len(idx) <= 25 or idx[0] < ENCODER_LENGTH:
            continue
        start = idx[0]
        encoder_idx = np.arange(start - ENCODER_LENGTH, start)
        if len(encoder_idx) != ENCODER_LENGTH:
            continue
        samples.append({"day": day, "target_idx": idx, "encoder": values[encoder_idx], "known": values[idx], "labels": frame.iloc[idx]["spread"].astype(float).to_numpy()})
    return samples


def batch(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, f = len(samples), samples[0]["encoder"].shape[1]
    enc = np.zeros((n, ENCODER_LENGTH, f), dtype=np.float32)
    known = np.zeros((n, MAX_HORIZON, f), dtype=np.float32)
    y = np.zeros((n, MAX_HORIZON), dtype=np.float32)
    mask = np.zeros((n, MAX_HORIZON), dtype=np.bool_)
    for i, sample in enumerate(samples):
        h = len(sample["labels"])
        enc[i], known[i, :h], y[i, :h], mask[i, :h] = sample["encoder"], sample["known"], sample["labels"], True
    return tuple(torch.from_numpy(x).to(DEVICE) for x in (enc, known, y, mask))


def validation_loss(model: Seq2Seq, samples: list[dict]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            enc, known, y, mask = batch(samples[start:start + 8])
            pred = model(enc, known)
            losses.append(float(nn.functional.smooth_l1_loss(pred[mask], y[mask]).detach()))
    return float(np.mean(losses))


def fit_b4a_week(df: pd.DataFrame, features: list[str], train_core: pd.Series, internal_eval: pd.Series, val_mask: pd.Series, week_id: str) -> pd.Series:
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    scaler.fit(imputer.fit_transform(df.loc[train_core, features]))
    values = scaler.transform(imputer.transform(df[features]))
    train_days = set(df.loc[train_core, "delivery_date_local"].astype(str))
    eval_days = set(df.loc[internal_eval, "delivery_date_local"].astype(str))
    val_days = set(df.loc[val_mask, "delivery_date_local"].astype(str))
    train_samples = day_samples(df, values, train_days)
    eval_samples = day_samples(df, values, eval_days)
    val_samples = day_samples(df, values, val_days)
    if not train_samples or not eval_samples or not val_samples:
        raise RuntimeError(f"B4A weekly samples missing for {week_id}: train={len(train_samples)} eval={len(eval_samples)} val={len(val_samples)}")
    torch.manual_seed(SEED)
    model = Seq2Seq(len(features)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_val, wait, best_state, best_epoch = np.inf, 0, None, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(train_samples)
        losses = []
        for start in range(0, len(train_samples), 8):
            enc, known, y, mask = batch(train_samples[start:start + 8])
            pred = model(enc, known)
            loss = nn.functional.smooth_l1_loss(pred[mask], y[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = validation_loss(model, eval_samples)
        train_loss = float(np.mean(losses))
        improved = val_loss < best_val - 1e-7
        if improved:
            best_val, wait, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        emit(f"[C4/B4A] {week_id} epoch {epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f} wait={wait}/{PATIENCE}")
        if wait >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": best_state, "imputer": imputer, "scaler": scaler, "features": features, "best_epoch": best_epoch}, MODEL_DIR / f"C4_B4A_lstm_{week_id}.pt")
    model.eval()
    pred_by_idx = {}
    with torch.no_grad():
        for start in range(0, len(val_samples), 8):
            chunk = val_samples[start:start + 8]
            enc, known, _, _ = batch(chunk)
            pred = model(enc, known).cpu().numpy()
            for sample, p in zip(chunk, pred):
                for idx, value in zip(sample["target_idx"], p[:len(sample["target_idx"])]):
                    pred_by_idx[idx] = float(value)
    val_indices = list(np.flatnonzero(val_mask.to_numpy()))
    if sorted(pred_by_idx) != sorted(val_indices):
        missing = sorted(set(val_indices) - set(pred_by_idx))[:10]
        extra = sorted(set(pred_by_idx) - set(val_indices))[:10]
        raise RuntimeError(f"C4 B4A prediction row mismatch for {week_id}: missing={missing}, extra={extra}")
    return pd.Series([pred_by_idx[i] for i in val_indices], index=df.index[val_indices])


def apply_trade(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sig = np.where((out["p_positive"] >= 0.60) & (out["p_positive"] > out["p_negative"]), 1, 0)
    sig = np.where((out["p_negative"] >= 0.60) & (out["p_negative"] > out["p_positive"]), -1, sig)
    out["signal"] = sig.astype(int)
    spread = pd.to_numeric(out["actual_spread"], errors="coerce")
    clipped = spread.clip(-1000, 5000)
    traded = out["signal"].ne(0)
    out["net_pnl"] = out["signal"] * clipped * 0.65 - traded.astype(float) * 2.0 - traded.astype(float) * clipped.abs() * 0.005
    return out


def metrics_pred(df: pd.DataFrame, system: str) -> dict:
    y = df["actual_class"].astype(int)
    pred = df["predicted_class"].astype(int)
    p = df[[f"p_c{i}" for i in range(1, 6)]].astype(float).to_numpy()
    catastrophic = ((y <= 2) & (pred >= 4)) | ((y >= 4) & (pred <= 2))
    extreme = pd.to_numeric(df["fixed_extreme_weather_flag"], errors="coerce").fillna(0).astype(bool)
    return {
        "period": "2026_H1",
        "system_id": system,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "mean_abs_class_distance": float(np.abs(y.to_numpy() - pred.to_numpy()).mean()),
        "catastrophic_reversal_rate": float(catastrophic.mean()),
        "direction_accuracy": float((np.sign(df["actual_spread"]) == np.where(pred >= 4, 1, np.where(pred <= 2, -1, 0))).mean()),
        "spread_mae": float(mean_absolute_error(df["actual_spread"], df["predicted_spread"])),
        "spread_rmse": float(np.sqrt(mean_squared_error(df["actual_spread"], df["predicted_spread"]))),
        "spread_r2": float(r2_score(df["actual_spread"], df["predicted_spread"])),
        "extreme_weather_macro_f1": float(f1_score(y[extreme], pred[extreme], labels=CLASSES, average="macro", zero_division=0)) if extreme.any() else np.nan,
    }


def metrics_econ(df: pd.DataFrame, system: str) -> dict:
    d = df.copy()
    d["delivery_hour_utc"] = pd.to_datetime(d["delivery_hour_utc"], utc=True)
    pnl = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
    sig = d["signal"].astype(int)
    traded = sig.ne(0)
    day = pd.to_datetime(d["delivery_date_local"]).dt.date
    daily = pnl.groupby(day).sum()
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    downside = daily[daily < 0]
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(365.0)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    sortino = float(daily.mean() / downside.std(ddof=1) * math.sqrt(365.0)) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    gp = pnl[traded & (pnl > 0)].sum()
    gl = -pnl[traded & (pnl < 0)].sum()
    months = pnl.groupby(d["delivery_hour_utc"].dt.to_period("M").astype(str)).sum()
    day_sorted = daily.sort_values(ascending=False)
    top_hours = pnl[traded].sort_values(ascending=False)
    total = float(pnl.sum())
    extreme = pd.to_numeric(d["fixed_extreme_weather_flag"], errors="coerce").fillna(0).astype(bool)
    tail20 = pd.to_numeric(d["target_extreme20"], errors="coerce").fillna(0).astype(bool)
    tail50 = pd.to_numeric(d["target_extreme50"], errors="coerce").fillna(0).astype(bool)
    q = daily.quantile(0.05)
    return {
        "period": "2026_H1",
        "system_id": system,
        "total_pnl": total,
        "total_return": total / 100000.0,
        "trade_count": int(traded.sum()),
        "coverage": float(traded.mean()),
        "direction_precision": float((np.sign(d.loc[traded, "actual_spread"]) == sig[traded]).mean()) if traded.any() else np.nan,
        "pnl_per_mwh": float(pnl[traded].mean()) if traded.any() else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "cvar_95_daily": float(daily[daily <= q].mean()) if len(daily) else np.nan,
        "win_rate": float((pnl[traded] > 0).mean()) if traded.any() else np.nan,
        "profit_factor": float(gp / gl) if gl > 0 else np.nan,
        "maximum_single_loss": float(pnl.min()) if len(pnl) else np.nan,
        "profitable_months": int(months.gt(0).sum()),
        "inc_count": int((sig == -1).sum()),
        "dec_count": int((sig == 1).sum()),
        "inc_pnl": float(pnl[sig == -1].sum()),
        "dec_pnl": float(pnl[sig == 1].sum()),
        "pnl_ex_january": float(pnl[d["delivery_hour_utc"].dt.month != 1].sum()),
        "pnl_ex_top5_days": float(total - day_sorted.head(5).sum()) if len(day_sorted) else total,
        "top1_day_share": float(day_sorted.head(1).sum() / total) if total != 0 and len(day_sorted) else np.nan,
        "top5_day_share": float(day_sorted.head(5).sum() / total) if total != 0 and len(day_sorted) else np.nan,
        "top1_hour_share": float(top_hours.head(1).sum() / total) if total != 0 and len(top_hours) else np.nan,
        "top5_hour_share": float(top_hours.head(5).sum() / total) if total != 0 and len(top_hours) else np.nan,
        "extreme_weather_pnl": float(pnl[extreme].sum()),
        "normal_weather_pnl": float(pnl[~extreme].sum()),
        "extreme_weather_tail20_pnl": float(pnl[extreme & tail20].sum()),
        "extreme_weather_tail50_pnl": float(pnl[extreme & tail50].sum()),
    }


def main() -> None:
    started = time.perf_counter()
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).sort_values("delivery_hour_utc").reset_index(drop=True)
    df["delivery_hour_utc"] = pd.to_datetime(df["delivery_hour_utc"], utc=True)
    features = approved_features(df)
    windows = weekly_windows(df)
    emit(f"[{RUN_ID}] started | features={len(features)} weeks={len(windows)}")
    c1_parts, c4_parts, manifest = [], [], []
    for i, win in enumerate(windows, 1):
        week_id = win["week_id"]
        val_mask = win["val_mask"]
        train_core, internal_eval = split_history(df, win["train_cutoff_utc"])
        emit(f"[{RUN_ID}] week {i:02d}/{len(windows)} {week_id} train={int(train_core.sum())} internal_eval={int(internal_eval.sum())} predict={int(val_mask.sum())}")
        c1_pred, xgb_info, _ = fit_xgb_week(df, features, train_core, internal_eval, val_mask, week_id)
        b4a_pred = fit_b4a_week(df, features, train_core, internal_eval, val_mask, week_id)
        c1_pred = apply_trade(c1_pred)
        c4_pred = c1_pred.copy()
        c4_pred["model_name"] = "C4_exploratory_head_swap"
        c4_pred["predicted_spread"] = b4a_pred.to_numpy(float)
        c4_pred = apply_trade(c4_pred)
        c1_parts.append(c1_pred)
        c4_parts.append(c4_pred)
        manifest.append({
            "week_id": week_id,
            "train_cutoff_utc": str(win["train_cutoff_utc"]),
            "train_rows": int(train_core.sum()),
            "internal_eval_rows": int(internal_eval.sum()),
            "prediction_rows": int(val_mask.sum()),
            **xgb_info,
        })
        pd.concat(c1_parts, ignore_index=True).to_parquet(OUT_PRED / "C1_weekly_walkforward_2026_partial_v3.parquet", index=False)
        pd.concat(c4_parts, ignore_index=True).to_parquet(OUT_PRED / "C4_weekly_walkforward_2026_partial_v3.parquet", index=False)
        pd.DataFrame(manifest).to_csv(OUT_PRED / "C1_C4_weekly_manifest_2026_v3.csv", index=False)

    c1 = pd.concat(c1_parts, ignore_index=True).sort_values("delivery_hour_utc").reset_index(drop=True)
    c4 = pd.concat(c4_parts, ignore_index=True).sort_values("delivery_hour_utc").reset_index(drop=True)
    if not c1["delivery_hour_utc"].equals(c4["delivery_hour_utc"]):
        raise RuntimeError("C1/C4 2026 timestamp mismatch")
    c1.to_parquet(OUT_PRED / "C1_weekly_walkforward_2026_v3.parquet", index=False)
    c4.to_parquet(OUT_PRED / "C4_weekly_walkforward_2026_v3.parquet", index=False)
    pred_metrics = pd.DataFrame([metrics_pred(c1, "C1_best_boosting_complete_system"), metrics_pred(c4, "C4_exploratory_head_swap")])
    econ_metrics = pd.DataFrame([metrics_econ(c1, "C1_best_boosting_complete_system"), metrics_econ(c4, "C4_exploratory_head_swap")])
    pred_metrics.to_csv(OUT_METRIC / "C1_C4_prediction_metrics_2026_v3.csv", index=False)
    econ_metrics.to_csv(OUT_METRIC / "C1_C4_economic_metrics_2026_v3.csv", index=False)
    monthly = pd.concat([c1.assign(system_id="C1"), c4.assign(system_id="C4")], ignore_index=True)
    monthly["month"] = pd.to_datetime(monthly["delivery_hour_utc"], utc=True).dt.to_period("M").astype(str)
    monthly.groupby(["system_id", "month"], as_index=False)["net_pnl"].sum().to_csv(OUT_METRIC / "C1_C4_monthly_pnl_2026_v3.csv", index=False)
    overlap = pd.DataFrame([{
        "comparison": "C1_vs_C4_2026",
        "hours": len(c1),
        "both_same_signal": int((c1["signal"].astype(int) == c4["signal"].astype(int)).sum()),
        "both_trade": int((c1["signal"].astype(int).ne(0) & c4["signal"].astype(int).ne(0)).sum()),
        "c1_only_trade": int((c1["signal"].astype(int).ne(0) & c4["signal"].astype(int).eq(0)).sum()),
        "c4_only_trade": int((c4["signal"].astype(int).ne(0) & c1["signal"].astype(int).eq(0)).sum()),
        "opposite_trade": int((c1["signal"].astype(int) * c4["signal"].astype(int) == -1).sum()),
    }])
    overlap.to_csv(OUT_METRIC / "C1_C4_trade_overlap_2026_v3.csv", index=False)
    report = PHASE / "reports" / "C1_C4_WEEKLY_2026_REPORT.md"
    report.write_text(
        "# C1与C4 2026 weekly walk-forward报告\n\n"
        "- C1：B2A XGBoost continuous + B2B XGBoost 5/20 probabilities。\n"
        "- C4：B4A LSTM continuous + B2B XGBoost 5/20 probabilities。\n"
        "- C4为exploratory head-swap，不覆盖正式C1。\n"
        "- 未运行Optuna，未修改阈值、成本、仓位或特征白名单。\n"
        f"- weeks: {len(windows)}；rows: {len(c1)}；runtime_seconds: {time.perf_counter() - started:.2f}\n\n"
        "## Prediction Metrics\n\n"
        f"{pred_metrics.to_string(index=False)}\n\n"
        "## Economic Metrics\n\n"
        f"{econ_metrics.to_string(index=False)}\n",
        encoding="utf-8",
    )
    emit(f"[{RUN_ID}] completed runtime_seconds={time.perf_counter() - started:.2f}")
    emit(str(econ_metrics[["system_id", "trade_count", "direction_precision", "total_pnl", "pnl_per_mwh", "sharpe", "max_drawdown", "cvar_95_daily", "profitable_months", "pnl_ex_top5_days"]]))


if __name__ == "__main__":
    main()
