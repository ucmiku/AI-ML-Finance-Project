# -*- coding: utf-8 -*-
"""Phase C B1A/B1B smoke tests on the frozen first expanding validation fold."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phase_C_model_selection_validation"
DATA_PATH = ROOT / "data/model_ready/model_input_frozen_v2.parquet"
WHITELIST_PATH = ROOT / "config/feature_whitelist_v2.csv"
REGISTRY_PATH = PHASE / "outputs/experiment_registry_phase_C.csv"
STATE_PATH = PHASE / "progress/progress_state.json"
PROGRESS_PATH = PHASE / "progress/PHASE_C_PROGRESS.md"
MASTER_LOG = PHASE / "logs/phase_C_master.log"
SEED = 20260722
CLASSES = np.array([1, 2, 3, 4, 5])
PHASE_CUTOFF_UTC = pd.Timestamp("2026-01-01T06:00:00Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_joblib(value: object, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    joblib.dump(value, temporary)
    temporary.replace(path)


def class_5_20(spread: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(spread, dtype=float)
    return np.select([values < -20, values < -5, values <= 5, values <= 20], [1, 2, 3, 4], default=5).astype(int)


def approved_features(frame: pd.DataFrame, whitelist: pd.DataFrame) -> list[str]:
    allowed = whitelist.loc[(whitelist["feature_set"] == "Z3") & (whitelist["fold_fitted"] == 0), "feature_name"]
    exclusions = ("target", "eligible", "issue_time", "source_product", "split_name", "research_period", "validation_fold_id", "availability", "qc")
    return [
        name for name in allowed.drop_duplicates()
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]) and not any(token in name.lower() for token in exclusions)
    ]


def validate_inputs(frame: pd.DataFrame, whitelist: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    report = (ROOT / "data/model_ready/model_input_freeze_report_v2.md").read_text(encoding="utf-8")
    if "model_dataset_status=GO" not in report:
        raise ValueError("Frozen model dataset is not GO")
    if frame["delivery_hour_utc"].duplicated().any():
        raise ValueError("Duplicate UTC primary key")
    if frame["delivery_hour_utc"].ge(PHASE_CUTOFF_UTC).any():
        raise ValueError("Phase C 2025 development input includes sealed 2026 rows")
    if not frame["evaluation_eligible"].isin([0, 1]).all():
        raise ValueError("evaluation_eligible must be binary")
    train = frame["research_period"].eq("initial_train") & frame["evaluation_eligible"].eq(1)
    validation = frame["validation_fold_id"].eq("validation_fold_1") & frame["evaluation_eligible"].eq(1)
    if not train.any() or not validation.any():
        raise ValueError("Missing initial training or validation fold 1 rows")
    if frame.loc[train, "delivery_hour_utc"].max() >= frame.loc[validation, "delivery_hour_utc"].min():
        raise ValueError("Training/validation time overlap")
    if frame.loc[validation, "delivery_hour_utc"].dt.year.min() < 2025:
        raise ValueError("Validation fold is not in 2025")
    features = approved_features(frame, whitelist)
    if not features:
        raise ValueError("No approved numeric Z3 features")
    return train, validation, features


def make_run_id(task: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = hashlib.sha256(f"{task}|{timestamp}|{sha256(DATA_PATH)}".encode()).hexdigest()[:6]
    return f"B1_{task}_{timestamp}_{short_hash}"


def output_frame(frame: pd.DataFrame, validation: pd.Series, model_name: str, task_type: str, run_id: str, predicted_spread: np.ndarray, probabilities: np.ndarray | None) -> pd.DataFrame:
    part = frame.loc[validation].copy()
    actual_spread = part["spread"].to_numpy(float)
    actual_class = class_5_20(actual_spread)
    if probabilities is None:
        predicted_class = class_5_20(predicted_spread)
        probability_values = np.full((len(part), 5), np.nan)
        p_negative = np.full(len(part), np.nan)
        p_positive = np.full(len(part), np.nan)
    else:
        predicted_class = probabilities.argmax(axis=1) + 1
        probability_values = probabilities
        p_negative = probabilities[:, 0] + probabilities[:, 1]
        p_positive = probabilities[:, 3] + probabilities[:, 4]
    return pd.DataFrame({
        "delivery_hour_utc": part["delivery_hour_utc"].to_numpy(),
        "delivery_date_local": pd.to_datetime(part["delivery_time_local"], utc=True).dt.date.astype(str).to_numpy(),
        "model_name": model_name,
        "task_type": task_type,
        "run_id": run_id,
        "fold_id": "validation_fold_1",
        "dataset_split": "validation",
        "actual_spread": actual_spread,
        "predicted_spread": predicted_spread,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "p_c1": probability_values[:, 0],
        "p_c2": probability_values[:, 1],
        "p_c3": probability_values[:, 2],
        "p_c4": probability_values[:, 3],
        "p_c5": probability_values[:, 4],
        "predicted_direction": np.sign(predicted_spread).astype(int),
        "actual_direction": np.sign(actual_spread).astype(int),
        # Smoke tests do not define or optimize a trading policy.
        "trade_signal": np.zeros(len(part), dtype=int),
        "net_pnl": np.full(len(part), np.nan),
        "fixed_extreme_weather_flag": part["fixed_extreme_weather_flag"].to_numpy(),
        "target_extreme20": (np.abs(actual_spread) > 20).astype(int),
        "p_negative": p_negative,
        "p_positive": p_positive,
    })


def metrics(predictions: pd.DataFrame, probabilities_available: bool) -> dict[str, float | int | str]:
    actual = predictions["actual_class"].to_numpy(int)
    predicted = predictions["predicted_class"].to_numpy(int)
    result: dict[str, float | int | str] = {
        "rows": len(predictions),
        "macro_f1_5_20": f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0),
        "balanced_accuracy_5_20": balanced_accuracy_score(actual, predicted),
        "direction_accuracy": float((predictions["actual_direction"] == predictions["predicted_direction"]).mean()),
        "mae": mean_absolute_error(predictions["actual_spread"], predictions["predicted_spread"]),
        "rmse": mean_squared_error(predictions["actual_spread"], predictions["predicted_spread"]) ** 0.5,
        "r2": r2_score(predictions["actual_spread"], predictions["predicted_spread"]),
    }
    if probabilities_available:
        probabilities = predictions[["p_c1", "p_c2", "p_c3", "p_c4", "p_c5"]].to_numpy()
        result.update({
            "accuracy_5_20": accuracy_score(actual, predicted),
            "log_loss_5_20": log_loss(actual, probabilities, labels=CLASSES),
            "max_probability_sum_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
        })
    return result


def update_phase_state(run_id: str, task: str, status: str, notes: str, data_hash: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    registry = pd.read_csv(REGISTRY_PATH)
    registry = pd.concat([registry, pd.DataFrame([{
        "run_id": run_id, "status": status, "task": task, "timestamp_utc": now,
        "source_data_hash": data_hash, "notes": notes,
    }])], ignore_index=True)
    atomic_csv(registry, REGISTRY_PATH)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update({"updated_at_utc": now, "current_run_id": run_id, "current_phase": "B1_smoke_tests"})
    completed = set(state.get("completed", []))
    if status == "COMPLETED":
        completed.add(task)
    state["completed"] = sorted(completed)
    state["running"] = ["B1A_ridge_smoke", "B1B_logistic_smoke"] if status == "RUNNING" else []
    state["next"] = "B1_formal_2025_expanding_window_after_both_smokes_pass" if status == "COMPLETED" else "inspect_archived_failure_then_continue_next_independent_task"
    atomic_text(STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False))
    progress = f"""# Phase C 进度

- 当前时间：{now}
- 当前阶段：B1 经验与线性基线冒烟测试
- 当前运行编号：{run_id}
- 已发现 B 阶段模型：M1–M5、M7、M8、M11；M9/M10/M12 为历史组合。
- 可比来源：8 个完整 2025 OOF 已通过只读结构审计；M6 缺少完整 OOF，保持受阻。
- 当前任务：{task}，状态：{status}。
- 已完成：Phase C 初始化、B 输入清单、OOF/LightGBM 审计、{task if status == 'COMPLETED' else '无新增完成项'}。
- 关键约束：仅使用 2024 训练和 2025 Fold 1 验证；2026 未读取、未参与选择。
- 下一项：{state['next']}。
- 失败模型：无；若当前任务失败，将在 failed_experiments 下归档并继续下一独立任务。
- 当前可继续：是。
- 2026 配置已冻结：否。
"""
    atomic_text(PROGRESS_PATH, progress)
    with MASTER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "B1_smoke_state", "timestamp_utc": now, "run_id": run_id, "task": task, "status": status, "notes": notes}, ensure_ascii=False) + "\n")


def execute(task: str) -> None:
    run_id = make_run_id(task)
    output_dir = PHASE / "outputs/experiments" / run_id
    model_dir = PHASE / "models" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    data_hash = sha256(DATA_PATH)
    update_phase_state(run_id, task, "RUNNING", "Starting first-fold smoke test.", data_hash)
    try:
        # Parquet predicate prevents materializing any sealed 2026 row for Phase C selection work.
        frame = pd.read_parquet(DATA_PATH, filters=[("delivery_hour_utc", "<", PHASE_CUTOFF_UTC)])
        frame["delivery_hour_utc"] = pd.to_datetime(frame["delivery_hour_utc"], utc=True)
        whitelist = pd.read_csv(WHITELIST_PATH)
        train, validation, features = validate_inputs(frame, whitelist)
        X_train, X_validation = frame.loc[train, features], frame.loc[validation, features]
        y_train = frame.loc[train, "spread"].to_numpy(float)
        config = {
            "run_id": run_id, "task": task, "seed": SEED, "data_hash": data_hash,
            "feature_whitelist_hash": sha256(WHITELIST_PATH), "feature_set": "Z3_non_fold_fitted",
            "train_period": "initial_train / 2024", "validation_period": "validation_fold_1 / 2025-01 to 2025-04",
            "label_5_20": "C1<-20; C2[-20,-5); C3[-5,5]; C4(5,20]; C5>20",
        }
        if task == "ridge_smoke":
            pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
            pipeline.fit(X_train, y_train)
            predicted_spread = pipeline.predict(X_validation)
            predictions = output_frame(frame, validation, "B1A_ridge", "continuous_regression", run_id, predicted_spread, None)
            probabilities_available = False
            config["model_parameters"] = {"alpha": 10.0}
        elif task == "logistic_smoke":
            y_class = class_5_20(y_train)
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=SEED)),
            ])
            pipeline.fit(X_train, y_class)
            raw = pipeline.predict_proba(X_validation)
            probabilities = np.zeros((len(X_validation), 5))
            probabilities[:, pipeline.named_steps["model"].classes_ - 1] = raw
            # For a classifier-only smoke, the expected value is deliberately omitted from scoring.
            predicted_spread = np.zeros(len(X_validation))
            predictions = output_frame(frame, validation, "B1B_logistic", "five_class_5_20", run_id, predicted_spread, probabilities)
            predictions["predicted_direction"] = np.where(predictions["predicted_class"].isin([4, 5]), 1, np.where(predictions["predicted_class"].isin([1, 2]), -1, 0))
            probabilities_available = True
            config["model_parameters"] = {"C": 0.1, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000}
        else:
            raise ValueError(f"Unknown task: {task}")

        metric = metrics(predictions, probabilities_available)
        if task == "logistic_smoke":
            metric.pop("mae")
            metric.pop("rmse")
            metric.pop("r2")
        config["feature_count"] = len(features)
        atomic_text(output_dir / "config.json", json.dumps(config, indent=2, ensure_ascii=False))
        atomic_text(output_dir / "environment.json", json.dumps({"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__}, indent=2, ensure_ascii=False))
        atomic_csv(pd.DataFrame({"feature_name": features}), output_dir / "feature_list.csv")
        atomic_csv(pd.DataFrame([{"train_rows": int(train.sum()), "validation_rows": int(validation.sum()), "feature_count": len(features)}]), output_dir / "data_summary.csv")
        atomic_csv(pd.DataFrame([{**{"run_id": run_id, "model_name": predictions.model_name.iloc[0], "fold_id": "validation_fold_1"}, **metric}]), output_dir / "fold_metrics.csv")
        atomic_csv(pd.DataFrame([{**{"run_id": run_id, "model_name": predictions.model_name.iloc[0]}, **metric}]), output_dir / "aggregate_metrics.csv")
        atomic_parquet(predictions, output_dir / "predictions.parquet")
        atomic_csv(predictions, output_dir / "predictions.csv")
        atomic_csv(pd.DataFrame(columns=["not_applicable_for_smoke_test"]), output_dir / "trading_metrics.csv")
        atomic_csv(pd.DataFrame(columns=["trial", "state"]), output_dir / "optuna_trials.csv")
        atomic_csv(pd.DataFrame(columns=["epoch", "loss"]), output_dir / "training_history.csv")
        atomic_csv(pd.DataFrame(columns=["actual_class", "predicted_class", "count"]), output_dir / "confusion_matrix_counts.csv")
        atomic_csv(pd.DataFrame(columns=["bin", "mean_prediction", "observed_rate"]), output_dir / "calibration_bins.csv")
        atomic_csv(pd.DataFrame(columns=["threshold", "precision", "recall"]), output_dir / "pr_curve_points.csv")
        atomic_text(output_dir / "warnings.log", "Smoke test only; no Optuna, no trading selection, no 2026 access.\n")
        atomic_joblib(pipeline, model_dir / "pipeline.joblib")
        atomic_text(output_dir / "run_summary.json", json.dumps({"status": "COMPLETED", "run_id": run_id, "metrics": metric}, indent=2, ensure_ascii=False))
        update_phase_state(run_id, task, "COMPLETED", json.dumps(metric, ensure_ascii=False), data_hash)
        print(json.dumps({"run_id": run_id, "task": task, "status": "COMPLETED", "metrics": metric}, ensure_ascii=False))
    except Exception as exc:
        failure = PHASE / "outputs/failed_experiments" / run_id
        failure.mkdir(parents=True, exist_ok=True)
        atomic_text(failure / "failure_summary.md", f"# {run_id}\n\n{type(exc).__name__}: {exc}\n")
        update_phase_state(run_id, task, "FAILED_ARCHIVED", f"{type(exc).__name__}: {exc}", data_hash)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"ridge_smoke", "logistic_smoke"}:
        raise SystemExit("Usage: run_b1_smoke_v3.py [ridge_smoke|logistic_smoke]")
    execute(sys.argv[1])
