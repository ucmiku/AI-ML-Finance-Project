from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[0]
DATA_PATH = ROOT / "data" / "model_ready" / "model_input_frozen_v2.parquet"
OUT = BASE / "outputs"
REPORTS = BASE / "reports"
PROGRESS = BASE / "progress"
RUN_ID = "explainability_v3_f5730506"
DATA_HASH = "f5730506707c2f227f6208bb6bc00ca4c0c45fe5a23c3148c1e9c2c04cfa0717"
FEATURE_VERSION = "feature_whitelist_v2"


def ensure_dirs() -> None:
    for d in [
        OUT / "ablation",
        OUT / "explainability" / "global",
        OUT / "explainability" / "grouped",
        OUT / "explainability" / "interactions",
        OUT / "explainability" / "local_cases",
        OUT / "explainability" / "source_tables",
        REPORTS,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def feature_group(feature: str) -> str:
    f = feature.lower()
    if "spread" in f:
        return "Historical Spread"
    if f.startswith("load_") or "_load" in f:
        return "Load"
    if f.startswith("wind_") or "wind" in f:
        return "Wind"
    if "solar" in f or "renewable" in f or "net_load" in f:
        return "Solar and Net Load"
    if any(x in f for x in ["temperature", "humidity", "precipitation", "radiation", "cloud", "weather"]):
        return "Raw Weather"
    if any(x in f for x in ["extreme", "freezing", "rainy", "heat", "duration", "coverage"]):
        return "Extreme Weather"
    if "gas" in f:
        return "Gas"
    if any(x in f for x in ["hour", "month", "dow", "weekend", "dst", "peak", "calendar"]):
        return "Calendar"
    return "Other"


def save_bar(table: pd.DataFrame, value_col: str, label_col: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    d = table.head(20).iloc[::-1]
    ax.barh(d[label_col].astype(str), d[value_col].astype(float), color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_fold_model(run: str, fold: str = "validation_fold_3") -> dict:
    return joblib.load(BASE / "models" / run / f"{fold}_pipeline.joblib")


def validation_frame(features: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    t = pd.to_datetime(df["delivery_hour_utc"], utc=True)
    mask = (t >= pd.Timestamp("2025-09-01T00:00:00Z")) & (t <= pd.Timestamp("2025-12-31T23:00:00Z"))
    cols = list(dict.fromkeys(["delivery_hour_utc", "spread", "fixed_extreme_weather_flag", *features]))
    return df.loc[mask, [c for c in cols if c in df.columns]].copy()


def shap_matrix(model, x: np.ndarray, class_index: int | None = None) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        if class_index is None:
            return np.asarray(values[0])
        return np.asarray(values[class_index])
    arr = np.asarray(values)
    if arr.ndim == 3:
        if class_index is None:
            return arr[:, :, 0]
        return arr[:, :, class_index]
    return arr


def top20_from_shap(values: np.ndarray, features: list[str], output_head: str) -> pd.DataFrame:
    score = np.nanmean(np.abs(values), axis=0)
    df = pd.DataFrame({"feature_name": features, "mean_abs_shap": score})
    df["feature_group"] = df["feature_name"].map(feature_group)
    df["model_name"] = "C1_best_boosting_complete_system"
    df["run_id"] = RUN_ID
    df["output_head"] = output_head
    df["feature_version"] = FEATURE_VERSION
    df["data_hash"] = DATA_HASH
    df["evaluation_period"] = "2025_validation_fold_3_sample"
    df["explainer_name"] = "TreeSHAP"
    df["explainer_config"] = "fold3 pipeline; median-imputed features; sample capped at 800 rows"
    return df.sort_values("mean_abs_shap", ascending=False).head(20).reset_index(drop=True)


def fallback_importance(pipe: dict, output_head: str) -> pd.DataFrame:
    model = pipe["model"]
    features = pipe["features"]
    values = getattr(model, "feature_importances_", np.zeros(len(features)))
    df = pd.DataFrame({"feature_name": features, "mean_abs_shap": np.asarray(values, dtype=float)})
    df["feature_group"] = df["feature_name"].map(feature_group)
    df["model_name"] = "C1_best_boosting_complete_system"
    df["run_id"] = RUN_ID
    df["output_head"] = output_head
    df["feature_version"] = FEATURE_VERSION
    df["data_hash"] = DATA_HASH
    df["evaluation_period"] = "2025_validation_fold_3_sample"
    df["explainer_name"] = "XGBoostGainFallback"
    df["explainer_config"] = "TreeSHAP failed; using model.feature_importances_"
    return df.sort_values("mean_abs_shap", ascending=False).head(20).reset_index(drop=True)


def make_global_explanations() -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    warnings: list[str] = []
    reg_pipe = load_fold_model("B2_regression_formal_v3r1_f5730506")
    clf_pipe = load_fold_model("B2_classifier_5_20_formal_v3r1_f5730506")
    reg_features = list(reg_pipe["features"])
    clf_features = list(clf_pipe["features"])
    features = list(dict.fromkeys(reg_features + clf_features))
    frame = validation_frame(features).dropna(subset=["spread"])
    sample = frame.sample(n=min(800, len(frame)), random_state=20260722)
    x_reg = reg_pipe["imputer"].transform(sample[reg_features].to_numpy())
    x_clf = clf_pipe["imputer"].transform(sample[clf_features].to_numpy())

    specs = [
        ("q50_continuous_spread", reg_pipe, x_reg, reg_features, None),
        ("negative_probability_C1_logit", clf_pipe, x_clf, clf_features, 0),
        ("no_trade_probability_C3_logit", clf_pipe, x_clf, clf_features, 2),
        ("positive_probability_C5_logit", clf_pipe, x_clf, clf_features, 4),
    ]
    all_top = []
    for output_head, pipe, x, ordered_features, class_index in specs:
        try:
            vals = shap_matrix(pipe["model"], x, class_index)
            top = top20_from_shap(vals, ordered_features, output_head)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{output_head}: TreeSHAP failed: {type(exc).__name__}: {exc}")
            top = fallback_importance(pipe, output_head)
        csv = OUT / "explainability" / "source_tables" / f"global_top20_{output_head}_v3.csv"
        png = OUT / "explainability" / "global" / f"global_top20_{output_head}_v3.png"
        top.to_csv(csv, index=False)
        save_bar(top, "mean_abs_shap", "feature_name", png, f"Top 20 - {output_head}")
        files.extend([csv, png])
        all_top.append(top)

    combined = pd.concat(all_top, ignore_index=True)
    grouped = combined.groupby(["output_head", "feature_group"], as_index=False)["mean_abs_shap"].sum()
    grouped_csv = OUT / "explainability" / "source_tables" / "grouped_feature_importance_v3.csv"
    grouped.to_csv(grouped_csv, index=False)
    files.append(grouped_csv)
    for head, group_df in grouped.groupby("output_head"):
        png = OUT / "explainability" / "grouped" / f"grouped_importance_{head}_v3.png"
        save_bar(group_df.sort_values("mean_abs_shap", ascending=False), "mean_abs_shap", "feature_group", png, f"Feature groups - {head}")
        files.append(png)
    return files, warnings


def local_cases() -> list[Path]:
    files: list[Path] = []
    pred = pd.read_parquet(OUT / "C_combination_systems" / "C1_best_boosting_complete_system_oof_2025_v3.parquet")
    pred["abs_spread"] = pd.to_numeric(pred["actual_spread"], errors="coerce").abs()
    pred["trade_profitable"] = pd.to_numeric(pred["net_pnl"], errors="coerce") > 0
    pred["direction_ok"] = np.sign(pred["actual_spread"]) == pred["signal"]
    traded = pred[pred["signal"].ne(0)].copy()
    cases = []
    def add_case(case_type: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        row = df.sort_values("abs_spread", ascending=False).iloc[0].to_dict()
        row["case_type"] = case_type
        cases.append(row)

    add_case("correct_extreme_profitable_opportunity", traded[(traded["abs_spread"] > 20) & traded["trade_profitable"] & traded["direction_ok"]])
    add_case("missed_large_spread_opportunity", pred[(pred["abs_spread"] > 20) & pred["signal"].eq(0)])
    add_case("wrong_direction_large_loss", traded[(traded["net_pnl"] < 0) & (~traded["direction_ok"])])
    add_case("normal_weather_day", pred[(pred["fixed_extreme_weather_flag"].eq(0)) & (pred["abs_spread"] <= 5)])
    out = pd.DataFrame(cases)
    if not out.empty:
        out["model_name"] = "C1_best_boosting_complete_system"
        out["run_id"] = RUN_ID
        out["feature_version"] = FEATURE_VERSION
        out["data_hash"] = DATA_HASH
        out["evaluation_period"] = "2025_OOF"
        out["explainer_name"] = "case_selection_table"
        out["explainer_config"] = "four archetypal cases selected from C1 OOF predictions"
    csv = OUT / "explainability" / "local_cases" / "local_case_selection_v3.csv"
    out.to_csv(csv, index=False)
    files.append(csv)
    return files


def write_ablation_limitations() -> list[Path]:
    rows = [
        {
            "experiment": "best_boosting_Z0_Z3_ablation",
            "status": "NOT_STARTED",
            "ready_for_comparison": False,
            "reason": "No completed Z0-Z3 retraining outputs exist for C1; B2 formal runner has no frozen feature-set argument.",
            "required_next_action": "Run fixed-hyperparameter Z0/Z1/Z2/Z3 retraining without Optuna before reporting ablation deltas.",
        },
        {
            "experiment": "multitask_tft_Z0_Z3_ablation",
            "status": "FAILED_ARCHIVED",
            "ready_for_comparison": False,
            "reason": "B5C true multi-task TFT failed/archived, so no valid multi-task TFT ablation target exists.",
            "required_next_action": "Implement true TFT multi-task trainer first, then run fixed-hyperparameter ablation.",
        },
    ]
    table = pd.DataFrame(rows)
    csv = OUT / "ablation" / "Z0_Z3_ablation_scope_status_v3.csv"
    table.to_csv(csv, index=False)
    report = REPORTS / "Z0_Z3_ABLATION_REPORT.md"
    report.write_text(
        "# Z0至Z3消融报告\n\n"
        "本轮未生成新的消融训练结果，因此不报告任何Z0-Z3性能差异。\n\n"
        "| experiment | status | ready_for_comparison | reason |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {r['experiment']} | {r['status']} | {r['ready_for_comparison']} | {r['reason']} |"
            for r in rows
        )
        + "\n\n消融不能用主实验结果反推，后续需使用已冻结超参数单独训练Z0/Z1/Z2/Z3配置。\n",
        encoding="utf-8",
    )
    return [csv, report]


def update_progress(files: list[Path], warnings: list[str]) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    with (PROGRESS / "PHASE_C_PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n\n## {stamp} - Explainability and limitations completed\n"
            f"- ready_for_explainability: true\n"
            f"- ablation_ready_for_comparison: false\n"
            f"- warnings: {len(warnings)}\n"
            + "".join(f"\n- generated: {p}" for p in files)
        )
    state_path = PROGRESS / "progress_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except json.JSONDecodeError:
        state = {}
    completed = list(dict.fromkeys(state.get("completed_runs", []) + [RUN_ID, "Explainability and Error Attribution"]))
    state.update(
        {
            "updated_at_utc": stamp,
            "current_phase": "STOP_BEFORE_2026",
            "current_task": "completed_through_explainability",
            "running": [],
            "completed_runs": completed,
            "last_completed_task": "Explainability and Error Attribution",
            "next": "2026 weekly walk-forward requires explicit authorization",
            "ready_for_explainability": True,
            "ablation_ready_for_comparison": False,
            "workflow_status": "STOP_BEFORE_2026",
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    files: list[Path] = []
    ablation_files = write_ablation_limitations()
    files.extend(ablation_files)
    global_files, warnings = make_global_explanations()
    files.extend(global_files)
    files.extend(local_cases())

    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- 无"
    report = REPORTS / "EXPLAINABILITY_AND_ERROR_ATTRIBUTION.md"
    report.write_text(
        "# Explainability and Error Attribution\n\n"
        f"- run_id: `{RUN_ID}`\n"
        "- final_prediction_agent: `C1_best_boosting_complete_system`\n"
        "- evaluation_period: `2025_OOF / validation_fold_3解释样本`\n"
        "- 解释器：TreeSHAP；若某输出失败，已在源表中标记为XGBoostGainFallback。\n"
        "- 注意：SHAP及梯度归因属于模型解释，不构成天气导致电价变化的因果证明。\n\n"
        "## 生成内容\n\n"
        "- Global SHAP / feature importance top-20源表与PNG。\n"
        "- 固定特征聚合组：Historical Spread, Load, Wind, Solar and Net Load, Raw Weather, Extreme Weather, Gas, Calendar。\n"
        "- 本地案例选择表：正确极端盈利机会、漏报大价差机会、方向错误亏损、正常天气日。\n\n"
        "## 警告与限制\n\n"
        f"{warning_text}\n\n"
        "## 消融限制\n\n"
        "Z0-Z3消融没有已完成训练输出，本轮只保存范围缩减说明，未把空结果加入正式比较。\n",
        encoding="utf-8",
    )
    files.append(report)
    update_progress(files, warnings)
    print("解释与限制报告完成")
    print(f"report={report}")
    print(f"warnings={len(warnings)}")
    print("workflow_status=STOP_BEFORE_2026")


if __name__ == "__main__":
    main()
