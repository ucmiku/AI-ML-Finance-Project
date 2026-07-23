# -*- coding: utf-8 -*-
"""Resumable Phase C B2 XGBoost formal 2025-only experiments."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

import run_b1_formal_v3 as b1
import run_b1_smoke_v3 as common


PHASE = common.PHASE
DATABASE = PHASE / "outputs/optuna/model_studies_v3.db"
FOLDS = b1.FOLDS
SEED = b1.SEED


def run_id(task: str) -> str:
    return f"B2_{task}_formal_v3r1_{common.sha256(common.DATA_PATH)[:8]}"


def params_from_trial(trial: optuna.Trial, classifier: bool) -> dict[str, float | int]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 15.0, log=True),
        "random_state": SEED, "n_jobs": 2, "tree_method": "hist", "early_stopping_rounds": 50,
        **({"objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss"} if classifier else {"objective": "reg:squarederror", "eval_metric": "mae"}),
    }


def fit_fold(task: str, params: dict, frame: pd.DataFrame, features: list[str], train: pd.Series, validation: pd.Series, fold: str, run: str):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(frame.loc[train, features])
    x_validation = imputer.transform(frame.loc[validation, features])
    actual_spread = frame.loc[validation, "spread"].to_numpy(float)
    started = time.perf_counter()
    if task == "regression":
        model = XGBRegressor(**params)
        model.fit(x_train, frame.loc[train, "spread"].to_numpy(float), eval_set=[(x_validation, actual_spread)], verbose=False)
        predicted_spread = model.predict(x_validation)
        prediction = common.output_frame(frame, validation, "B2A_xgboost_regression", "continuous_regression", run, predicted_spread, None)
        metric = b1.evaluate_regression(actual_spread, predicted_spread)
    else:
        target = common.class_5_20(frame.loc[train, "spread"])
        counts = pd.Series(target).value_counts()
        weights = pd.Series(target).map({class_id: len(target) / (5 * counts[class_id]) for class_id in common.CLASSES}).to_numpy(float)
        model = XGBClassifier(**params)
        model.fit(x_train, target - 1, sample_weight=weights, eval_set=[(x_validation, common.class_5_20(actual_spread) - 1)], verbose=False)
        raw = model.predict_proba(x_validation)
        probabilities = np.zeros((len(x_validation), 5))
        probabilities[:, model.classes_.astype(int)] = raw
        if np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1e-6:
            raise ValueError("XGBoost probability class order/sum check failed")
        medians = pd.Series(frame.loc[train, "spread"].to_numpy(float)).groupby(target).median().reindex(common.CLASSES).to_numpy(float)
        predicted_spread = probabilities @ medians
        prediction = common.output_frame(frame, validation, "B2B_xgboost_5_20", "five_class_5_20", run, predicted_spread, probabilities)
        prediction["predicted_direction"] = np.where(prediction["predicted_class"].isin([4, 5]), 1, np.where(prediction["predicted_class"].isin([1, 2]), -1, 0))
        metric = b1.evaluate_classifier(common.class_5_20(actual_spread), prediction["predicted_class"].to_numpy(int), probabilities)
        metric["class_weight_rule"] = "n_train/(5*n_train_class), recomputed per fold"
    metric.update({"fold_id": fold, "prediction_seconds": time.perf_counter() - started, "train_rows": int(train.sum()), "validation_rows": int(validation.sum()), "best_iteration": getattr(model, "best_iteration", None)})
    return {"imputer": imputer, "model": model, "features": features}, prediction, metric


def main(task: str) -> None:
    if task not in {"regression", "classifier_5_20"}:
        raise SystemExit("Usage: run_b2_xgboost_v3.py [regression|classifier_5_20]")
    classifier = task == "classifier_5_20"
    run = run_id(task)
    output = PHASE / "outputs/experiments" / run
    model_dir = PHASE / "models" / run
    lock = PHASE / "outputs" / f"{run}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Run lock exists; do not start a concurrent recovery: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary = output / "run_summary.json"
    if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "COMPLETED":
        lock.unlink(missing_ok=True)
        print(json.dumps({"run_id": run, "status": "ALREADY_COMPLETED"}, ensure_ascii=False)); return
    b1.append_registry(run, task, "RUNNING", "Formal B2 2025-only expanding-window XGBoost started.")
    b1.set_progress(run, task, "formal_run_started", current_trial=None, current_fold=None)
    try:
        frame, _, features, masks = b1.data_and_features("ridge")
        config = {"run_id": run, "task": task, "trial_budget": 27, "data_hash": common.sha256(common.DATA_PATH), "feature_set": "Z3_non_fold_fitted", "2026_access": "blocked_by_parquet_predicate", "objective": "mean_MAE" if not classifier else "mean_Macro_F1", "early_stopping_rounds": 50, "parameter_space": "fixed_before_results"}
        b1.atomic_text(output / "config.json", json.dumps(config, indent=2, ensure_ascii=False))
        b1.atomic_text(output / "environment.json", json.dumps({"python": sys.version, "xgboost": __import__("xgboost").__version__, "optuna": optuna.__version__}, indent=2, ensure_ascii=False))
        b1.atomic_csv(pd.DataFrame({"feature_name": features}), output / "feature_list.csv")
        b1.atomic_csv(pd.DataFrame([{"fold_id": fold, "train_rows": int(masks[f"train_{fold}"].sum()), "validation_rows": int(masks[f"validation_{fold}"].sum()), "feature_count": len(features)} for fold in FOLDS]), output / "data_summary.csv")
        DATABASE.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(study_name=run, storage=f"sqlite:///{DATABASE.resolve().as_posix()}", load_if_exists=True, direction="minimize" if not classifier else "maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        def objective(trial: optuna.Trial) -> float:
            params = params_from_trial(trial, classifier)
            values = []
            for fold in FOLDS:
                b1.set_progress(run, task, "trial_fold_started", current_trial=trial.number, current_fold=fold)
                _, _, metric = fit_fold("classifier" if classifier else "regression", params, frame, features, masks[f"train_{fold}"], masks[f"validation_{fold}"], fold, run)
                value = metric["macro_f1"] if classifier else metric["mae"]
                values.append(float(value)); trial.set_user_attr(f"{fold}_{'macro_f1' if classifier else 'mae'}", float(value))
                b1.set_progress(run, task, "trial_fold_completed", current_trial=trial.number, current_fold=fold, last_fold_value=float(value))
            return float(np.mean(values))
        def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            b1.export_trials(study_, output); b1.set_progress(run, task, "trial_completed", current_trial=trial.number, current_fold=None, trial_value=trial.value)
        complete = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        if complete < 27: study.optimize(objective, n_trials=27 - complete, callbacks=[callback], n_jobs=1, show_progress_bar=False)
        b1.export_trials(study, output); params = study.best_params | {"random_state": SEED, "n_jobs": 2, "tree_method": "hist", "early_stopping_rounds": 50, **({"objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss"} if classifier else {"objective": "reg:squarederror", "eval_metric": "mae"})}
        b1.atomic_text(output / "best_params.json", json.dumps(params, indent=2, ensure_ascii=False))
        predictions, metrics = [], []
        for fold in FOLDS:
            b1.set_progress(run, task, "final_fold_started", current_trial=None, current_fold=fold)
            fitted, prediction, metric = fit_fold("classifier" if classifier else "regression", params, frame, features, masks[f"train_{fold}"], masks[f"validation_{fold}"], fold, run)
            b1.atomic_joblib(fitted, model_dir / f"{fold}_pipeline.joblib"); b1.atomic_parquet(prediction, output / f"predictions_{fold}.parquet")
            predictions.append(prediction); metrics.append(metric | {"run_id": run, "model_name": prediction.model_name.iloc[0]})
            b1.set_progress(run, task, "final_fold_completed", current_trial=None, current_fold=fold)
        all_predictions = pd.concat(predictions, ignore_index=True); metric_frame = pd.DataFrame(metrics)
        b1.atomic_parquet(all_predictions, output / "predictions.parquet"); b1.atomic_csv(all_predictions, output / "predictions.csv"); b1.atomic_csv(metric_frame, output / "fold_metrics.csv")
        aggregate = metric_frame.select_dtypes(include=[np.number]).mean().to_dict(); b1.atomic_csv(pd.DataFrame([{"run_id": run, "model_name": all_predictions.model_name.iloc[0], **aggregate}]), output / "aggregate_metrics.csv")
        b1.diagnostics(all_predictions, "logistic_5_20" if classifier else "ridge", output)
        b1.atomic_csv(pd.DataFrame(columns=["not_evaluated_before_prediction_quality_gate"]), output / "trading_metrics.csv"); b1.atomic_csv(pd.DataFrame(columns=["epoch", "loss"]), output / "training_history.csv")
        b1.atomic_text(output / "warnings.log", "2025-only B2 run. 2026, economic selection, and combinations are excluded.\n")
        b1.atomic_text(summary, json.dumps({"status": "COMPLETED", "run_id": run, "best_params": params, "aggregate_metrics": aggregate}, indent=2, ensure_ascii=False))
        b1.append_registry(run, task, "COMPLETED", f"Formal completed with 27 SQLite trials; best_params={json.dumps(params)}")
        b1.set_progress(run, task, "formal_run_completed", current_trial=None, current_fold=None)
        lock.unlink(missing_ok=True)
        print(json.dumps({"run_id": run, "status": "COMPLETED", "best_params": params, "aggregate_metrics": aggregate}, ensure_ascii=False))
    except Exception as exc:
        failed = PHASE / "outputs/failed_experiments" / run; failed.mkdir(parents=True, exist_ok=True)
        b1.atomic_text(failed / "failure_summary.md", f"# {run}\n\n{type(exc).__name__}: {exc}\n")
        b1.append_registry(run, task, "FAILED_ARCHIVED", f"{type(exc).__name__}: {exc}")
        b1.set_progress(run, task, "formal_run_failed", error=f"{type(exc).__name__}: {exc}")
        lock.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("Usage: run_b2_xgboost_v3.py [regression|classifier_5_20]")
    main(sys.argv[1])
