# -*- coding: utf-8 -*-
"""Resumable Phase C B1 formal models: Ridge regression and 5/20 Logistic."""
from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_b1_smoke_v3 as common


ROOT = common.ROOT
PHASE = common.PHASE
SEED = 20260722
FOLDS = ["validation_fold_1", "validation_fold_2", "validation_fold_3"]
DATABASE = PHASE / "outputs/optuna/model_studies_v3.db"


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.stem + f".{uuid.uuid4().hex}.tmp" + path.suffix)
    temporary.write_text(content, encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + f".{uuid.uuid4().hex}.tmp" + path.suffix)
    frame.to_csv(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + f".{uuid.uuid4().hex}.tmp" + path.suffix)
    frame.to_parquet(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_joblib(value: object, path: Path) -> None:
    temporary = path.with_name(path.stem + f".{uuid.uuid4().hex}.tmp" + path.suffix)
    joblib.dump(value, temporary)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def run_id(task: str) -> str:
    suffix = common.sha256(common.DATA_PATH)[:8]
    return f"B1_{task}_formal_v3_{suffix}"


def set_progress(run: str, task: str, event: str, **details: object) -> None:
    now = datetime.now(timezone.utc).isoformat()
    state = json.loads(common.STATE_PATH.read_text(encoding="utf-8"))
    state.update({"updated_at_utc": now, "current_phase": "B1_formal", "current_run_id": run, "current_model": task, "current_event": event, **details})
    atomic_text(common.STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False))
    with common.MASTER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "timestamp_utc": now, "run_id": run, "task": task, **details}, ensure_ascii=False) + "\n")


def append_registry(run: str, task: str, status: str, note: str) -> None:
    registry = pd.read_csv(common.REGISTRY_PATH)
    row = pd.DataFrame([{
        "run_id": run, "status": status, "task": task,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_data_hash": common.sha256(common.DATA_PATH), "notes": note,
    }])
    atomic_csv(pd.concat([registry, row], ignore_index=True), common.REGISTRY_PATH)


def data_and_features(task: str) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, pd.Series]]:
    # The predicate is the Phase C hard barrier: no sealed 2026 row is materialized.
    frame = pd.read_parquet(common.DATA_PATH, filters=[("delivery_hour_utc", "<", common.PHASE_CUTOFF_UTC)])
    frame["delivery_hour_utc"] = pd.to_datetime(frame["delivery_hour_utc"], utc=True)
    whitelist = pd.read_csv(common.WHITELIST_PATH)
    train_initial, _, base_features = common.validate_inputs(frame, whitelist)
    if task == "logistic_5_20":
        subset_path = ROOT / "configs/model_search/M2_logistic_feature_subset_v2.csv"
        subset = pd.read_csv(subset_path)["feature_name"].drop_duplicates().tolist()
        missing = sorted(set(subset) - set(base_features))
        if missing:
            raise ValueError(f"M2 feature subset includes non-approved/missing fields: {missing[:8]}")
        features = subset
    else:
        features = base_features
    masks: dict[str, pd.Series] = {}
    for fold in FOLDS:
        validation = frame["validation_fold_id"].eq(fold) & frame["evaluation_eligible"].eq(1)
        start = frame.loc[validation, "delivery_hour_utc"].min()
        training = frame["delivery_hour_utc"].lt(start) & frame["evaluation_eligible"].eq(1)
        if not validation.any() or not training.any() or frame.loc[training, "delivery_hour_utc"].max() >= start:
            raise ValueError(f"Invalid expanding split: {fold}")
        masks[f"train_{fold}"] = training
        masks[f"validation_{fold}"] = validation
    return frame, train_initial, features, masks


def classifier_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    values = np.zeros((len(x), 5))
    values[:, model.named_steps["model"].classes_ - 1] = raw
    if np.max(np.abs(values.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("Classifier probability class order/sum check failed")
    return values


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "median_absolute_error": median_absolute_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "pearson_correlation": float(pd.Series(actual).corr(pd.Series(predicted), method="pearson")),
        "spearman_correlation": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
        "direction_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
        "macro_f1_5_20_from_thresholds": f1_score(common.class_5_20(actual), common.class_5_20(predicted), labels=common.CLASSES, average="macro", zero_division=0),
    }


def evaluate_classifier(actual: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    onehot = np.eye(5)[actual - 1]
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=common.CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(actual, predicted, labels=common.CLASSES, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "log_loss": log_loss(actual, probabilities, labels=common.CLASSES),
        "multiclass_brier": float(np.mean((probabilities - onehot) ** 2)),
        "mean_class_distance": float(np.abs(actual - predicted).mean()),
        "spearman_correlation": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
        "c1_pr_auc": average_precision_score(actual == 1, probabilities[:, 0]),
        "c5_pr_auc": average_precision_score(actual == 5, probabilities[:, 4]),
    }


def fit_fold(task: str, params: dict[str, float], frame: pd.DataFrame, features: list[str], train: pd.Series, validation: pd.Series, fold: str, run: str) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    x_train, x_validation = frame.loc[train, features], frame.loc[validation, features]
    actual_spread = frame.loc[validation, "spread"].to_numpy(float)
    started = time.perf_counter()
    if task == "ridge":
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=float(params["alpha"])))])
        pipeline.fit(x_train, frame.loc[train, "spread"].to_numpy(float))
        predicted_spread = pipeline.predict(x_validation)
        predictions = common.output_frame(frame, validation, "B1A_ridge", "continuous_regression", run, predicted_spread, None)
        metric = evaluate_regression(actual_spread, predicted_spread)
    else:
        y_train = common.class_5_20(frame.loc[train, "spread"])
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=float(params["C"]), penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=1500, random_state=SEED)),
        ])
        pipeline.fit(x_train, y_train)
        probabilities = classifier_probabilities(pipeline, x_validation)
        medians = pd.Series(frame.loc[train, "spread"].to_numpy(float)).groupby(y_train).median().reindex(common.CLASSES).to_numpy(float)
        predicted_spread = probabilities @ medians
        predictions = common.output_frame(frame, validation, "B1B_logistic", "five_class_5_20", run, predicted_spread, probabilities)
        predictions["predicted_direction"] = np.where(predictions["predicted_class"].isin([4, 5]), 1, np.where(predictions["predicted_class"].isin([1, 2]), -1, 0))
        metric = evaluate_classifier(common.class_5_20(actual_spread), predictions["predicted_class"].to_numpy(int), probabilities)
        metric["class_medians_training_fold"] = json.dumps({f"C{i}": float(value) for i, value in enumerate(medians, 1)})
    metric.update({"fold_id": fold, "prediction_seconds": time.perf_counter() - started, "train_rows": int(train.sum()), "validation_rows": int(validation.sum())})
    return pipeline, predictions, metric


def export_trials(study: optuna.Study, output_dir: Path) -> None:
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete", "user_attrs"))
    atomic_csv(trials, output_dir / "optuna_trials.csv")


def diagnostics(predictions: pd.DataFrame, task: str, output_dir: Path) -> None:
    actual = predictions["actual_class"].to_numpy(int)
    predicted = predictions["predicted_class"].to_numpy(int)
    matrix = confusion_matrix(actual, predicted, labels=common.CLASSES)
    rows = [{"actual_class": actual_class, "predicted_class": predicted_class, "count": int(matrix[i, j])} for i, actual_class in enumerate(common.CLASSES) for j, predicted_class in enumerate(common.CLASSES)]
    atomic_csv(pd.DataFrame(rows), output_dir / "confusion_matrix_counts.csv")
    if task == "logistic_5_20":
        calibration_rows, pr_rows = [], []
        for class_id in common.CLASSES:
            probability = predictions[f"p_c{class_id}"].to_numpy(float)
            observed = (actual == class_id).astype(int)
            for lower in np.linspace(0, 0.9, 10):
                upper = lower + 0.1
                bucket = (probability >= lower) & ((probability < upper) if upper < 1 else (probability <= upper))
                if bucket.any():
                    calibration_rows.append({"class": int(class_id), "bin_lower": lower, "bin_upper": upper, "mean_prediction": probability[bucket].mean(), "observed_rate": observed[bucket].mean(), "count": int(bucket.sum())})
            from sklearn.metrics import precision_recall_curve
            precision, recall, threshold = precision_recall_curve(observed, probability)
            pr_rows.extend({"class": int(class_id), "threshold": float(threshold[index]) if index < len(threshold) else np.nan, "precision": float(precision[index]), "recall": float(recall[index])} for index in range(len(precision)))
        atomic_csv(pd.DataFrame(calibration_rows), output_dir / "calibration_bins.csv")
        atomic_csv(pd.DataFrame(pr_rows), output_dir / "pr_curve_points.csv")
    else:
        atomic_csv(pd.DataFrame(columns=["class", "bin_lower", "bin_upper", "mean_prediction", "observed_rate", "count"]), output_dir / "calibration_bins.csv")
        atomic_csv(pd.DataFrame(columns=["class", "threshold", "precision", "recall"]), output_dir / "pr_curve_points.csv")


def main(task: str) -> None:
    if task not in {"ridge", "logistic_5_20"}:
        raise SystemExit("Usage: run_b1_formal_v3.py [ridge|logistic_5_20]")
    run = run_id(task)
    output_dir = PHASE / "outputs/experiments" / run
    model_dir = PHASE / "models" / run
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "COMPLETED":
        print(json.dumps({"run_id": run, "status": "ALREADY_COMPLETED"}, ensure_ascii=False))
        return
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    append_registry(run, task, "RUNNING", "Formal 2025-only expanding-window B1 run started.")
    set_progress(run, task, "formal_run_started", current_trial=None, current_fold=None)
    try:
        frame, _, features, masks = data_and_features(task)
        data_hash = common.sha256(common.DATA_PATH)
        config = {
            "run_id": run, "task": task, "seed": SEED, "trial_budget": 12, "data_hash": data_hash,
            "feature_whitelist_hash": common.sha256(common.WHITELIST_PATH), "features": "Z3_non_fold_fitted" if task == "ridge" else "verified_M2_logistic_feature_subset_v2",
            "2026_access": "blocked_by_parquet_predicate", "folds": FOLDS,
            "objective": "mean_MAE" if task == "ridge" else "mean_Macro_F1",
            "class_weights": "not_applicable" if task == "ridge" else "sklearn balanced weights computed from each training fold",
        }
        atomic_text(output_dir / "config.json", json.dumps(config, indent=2, ensure_ascii=False))
        atomic_text(output_dir / "environment.json", json.dumps({"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "optuna": optuna.__version__}, indent=2, ensure_ascii=False))
        atomic_csv(pd.DataFrame({"feature_name": features}), output_dir / "feature_list.csv")
        atomic_csv(pd.DataFrame([{"fold_id": fold, "train_rows": int(masks[f"train_{fold}"].sum()), "validation_rows": int(masks[f"validation_{fold}"].sum()), "feature_count": len(features)} for fold in FOLDS]), output_dir / "data_summary.csv")
        study = optuna.create_study(
            study_name=run, storage=f"sqlite:///{DATABASE.resolve().as_posix()}", load_if_exists=True,
            direction="minimize" if task == "ridge" else "maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        def objective(trial: optuna.Trial) -> float:
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)} if task == "ridge" else {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
            values = []
            for fold in FOLDS:
                set_progress(run, task, "trial_fold_started", current_trial=trial.number, current_fold=fold)
                _, prediction, metric = fit_fold(task, params, frame, features, masks[f"train_{fold}"], masks[f"validation_{fold}"], fold, run)
                value = metric["mae"] if task == "ridge" else metric["macro_f1"]
                values.append(float(value))
                trial.set_user_attr(f"{fold}_{'mae' if task == 'ridge' else 'macro_f1'}", float(value))
                set_progress(run, task, "trial_fold_completed", current_trial=trial.number, current_fold=fold, last_fold_value=float(value))
            return float(np.mean(values))
        def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            export_trials(study_, output_dir)
            set_progress(run, task, "trial_completed", current_trial=trial.number, current_fold=None, trial_value=trial.value)
        remaining = max(0, 12 - len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]))
        if remaining:
            study.optimize(objective, n_trials=remaining, callbacks=[callback], n_jobs=1, show_progress_bar=False)
        export_trials(study, output_dir)
        params = study.best_params
        atomic_text(output_dir / "best_params.json", json.dumps(params, indent=2, ensure_ascii=False))
        pipelines, predictions, metrics = [], [], []
        for fold in FOLDS:
            set_progress(run, task, "final_fold_started", current_trial=None, current_fold=fold)
            pipeline, prediction, metric = fit_fold(task, params, frame, features, masks[f"train_{fold}"], masks[f"validation_{fold}"], fold, run)
            atomic_joblib(pipeline, model_dir / f"{fold}_pipeline.joblib")
            atomic_parquet(prediction, output_dir / f"predictions_{fold}.parquet")
            predictions.append(prediction)
            metrics.append(metric | {"run_id": run, "model_name": prediction["model_name"].iloc[0]})
            set_progress(run, task, "final_fold_completed", current_trial=None, current_fold=fold)
        all_predictions = pd.concat(predictions, ignore_index=True)
        atomic_parquet(all_predictions, output_dir / "predictions.parquet")
        atomic_csv(all_predictions, output_dir / "predictions.csv")
        metric_frame = pd.DataFrame(metrics)
        atomic_csv(metric_frame, output_dir / "fold_metrics.csv")
        numeric = metric_frame.select_dtypes(include=[np.number]).mean().to_dict()
        atomic_csv(pd.DataFrame([{"run_id": run, "model_name": all_predictions["model_name"].iloc[0], **numeric}]), output_dir / "aggregate_metrics.csv")
        diagnostics(all_predictions, task, output_dir)
        atomic_csv(pd.DataFrame(columns=["not_evaluated_before_prediction_quality_gate"]), output_dir / "trading_metrics.csv")
        atomic_csv(pd.DataFrame(columns=["epoch", "loss"]), output_dir / "training_history.csv")
        atomic_text(output_dir / "warnings.log", "Formal B1 model: selection uses only 2025 expanding-window OOF. Trading and 2026 are not evaluated here.\n")
        atomic_text(summary_path, json.dumps({"status": "COMPLETED", "run_id": run, "best_params": params, "aggregate_metrics": numeric}, indent=2, ensure_ascii=False))
        append_registry(run, task, "COMPLETED", f"Formal completed with 12 SQLite trials; best_params={json.dumps(params)}")
        set_progress(run, task, "formal_run_completed", current_trial=None, current_fold=None)
        print(json.dumps({"run_id": run, "status": "COMPLETED", "best_params": params, "aggregate_metrics": numeric}, ensure_ascii=False))
    except Exception as exc:
        failure_dir = PHASE / "outputs/failed_experiments" / run
        failure_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(failure_dir / "failure_summary.md", f"# {run}\n\n{type(exc).__name__}: {exc}\n")
        append_registry(run, task, "FAILED_ARCHIVED", f"{type(exc).__name__}: {exc}")
        set_progress(run, task, "formal_run_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_b1_formal_v3.py [ridge|logistic_5_20]")
    main(sys.argv[1])
