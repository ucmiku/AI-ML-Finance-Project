#!/usr/bin/env python3
"""Read-only Phase 0 and Reviewer 1.1 evidence audit for PESIM 2027.

This script does not train models and never modifies source artifacts. It
reconciles the submitted economic results and describes the fixed five-class
spread partition from existing local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_VERSION = "1.0.0"
RUN_PREFIX = "pesim2027_phase0_r1_1"
CAPTURE_RATE = 0.65
COMMISSION_USD_PER_MWH = 2.0
SLIPPAGE_RATE = 0.005
OLD_SPREAD_CLIP = (-1000.0, 5000.0)
CLASS_ORDER = ["C1", "C2", "C3", "C4", "C5"]
PERIOD_ORDER = ["2024 development", "2025 OOF", "2026 H1 OOT"]
STRATEGIES = ["S1", "S3", "S5"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_5_20(spread: pd.Series) -> pd.Series:
    values = pd.to_numeric(spread, errors="raise").to_numpy(float)
    labels = np.select(
        [values < -20.0, values < -5.0, values <= 5.0, values <= 20.0],
        CLASS_ORDER[:4],
        default="C5",
    )
    return pd.Series(labels, index=spread.index, name="spread_class")


def research_period_label(series: pd.Series) -> pd.Series:
    labels = np.select(
        [
            series.eq("initial_train"),
            series.astype(str).str.startswith("validation_fold"),
            series.eq("final_test"),
        ],
        PERIOD_ORDER,
        default="excluded",
    )
    return pd.Series(labels, index=series.index, name="period")


def load_hourly(path: Path, period: str, expected_hours: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "delivery_hour_utc",
        "period",
        "strategy_id",
        "strategy_name",
        "signal",
        "quantity_mwh",
        "actual_spread",
        "gross_pnl",
        "commission",
        "slippage",
        "net_pnl",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing columns {missing}")
    frame = frame.copy()
    frame["delivery_hour_utc"] = pd.to_datetime(frame["delivery_hour_utc"], utc=True)
    if len(frame) != expected_hours * 8:
        raise AssertionError(f"{period}: expected {expected_hours * 8} rows, found {len(frame)}")
    if frame[["delivery_hour_utc", "strategy_id"]].duplicated().any():
        raise AssertionError(f"{period}: duplicate strategy-hour rows")
    if frame["delivery_hour_utc"].nunique() != expected_hours:
        raise AssertionError(f"{period}: expected {expected_hours} unique hours")
    if set(frame["strategy_id"].unique()) != {f"S{i}" for i in range(1, 9)}:
        raise AssertionError(f"{period}: unexpected strategy set")
    return frame


def strategy_summary(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_id in STRATEGIES:
        part = frame.loc[frame["strategy_id"].eq(strategy_id)].copy()
        active = part["signal"].ne(0)
        traded_mwh = float(part.loc[active, "quantity_mwh"].sum())
        rows.append(
            {
                "period": period,
                "strategy_id": strategy_id,
                "strategy_name": str(part["strategy_name"].iloc[0]),
                "eligible_hours": int(len(part)),
                "trades": int(active.sum()),
                "coverage": float(active.mean()),
                "total_pnl_usd": float(part["net_pnl"].sum()),
                "traded_mwh": traded_mwh,
                "pnl_per_mwh": float(part["net_pnl"].sum() / traded_mwh),
            }
        )
    return pd.DataFrame(rows)


def settle(signal: pd.Series, spread: pd.Series, clip: tuple[float, float] | None) -> pd.Series:
    values = pd.to_numeric(spread, errors="raise").astype(float)
    if clip is not None:
        values = values.clip(*clip)
    action = pd.to_numeric(signal, errors="raise").astype(int)
    active = action.ne(0).astype(float)
    return (
        CAPTURE_RATE * action * values
        - COMMISSION_USD_PER_MWH * active
        - SLIPPAGE_RATE * values.abs() * active
    )


def stable_run_id(input_hashes: dict[str, str]) -> str:
    payload = {
        "audit_version": AUDIT_VERSION,
        "inputs": input_hashes,
        "parameters": {
            "capture_rate": CAPTURE_RATE,
            "commission_usd_per_mwh": COMMISSION_USD_PER_MWH,
            "slippage_rate": SLIPPAGE_RATE,
            "old_spread_clip": list(OLD_SPREAD_CLIP),
            "class_boundaries_usd_per_mwh": [-20.0, -5.0, 5.0, 20.0],
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{RUN_PREFIX}_{digest[:16]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--submitted-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    submitted_pdf = args.submitted_pdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_c = project_root / "02_prediction_agent_v2" / "phase_C_model_selection_validation"
    canonical = project_root / "04_trading_strategy_evaluation" / "section_iv_canonical_v1"
    files = {
        "submitted_pdf": submitted_pdf,
        "model_input_frozen": project_root / "02_prediction_agent_v2" / "data" / "model_ready" / "model_input_frozen_v2.parquet",
        "hourly_strategy_2025": canonical / "outputs" / "hourly_strategy_results_2025.parquet",
        "hourly_strategy_2026": canonical / "outputs" / "hourly_strategy_results_2026.parquet",
        "table7_2025": canonical / "outputs" / "table7_2025_strategy_evaluation.csv",
        "table8_2026": canonical / "outputs" / "table8_2026_strategy_evaluation.csv",
        "risk_diagnostics": canonical / "outputs" / "tableS_strategy_risk_diagnostics.csv",
        "canonical_execution_manifest": canonical / "outputs" / "execution_manifest.json",
        "canonical_code": canonical / "src" / "canonical_trading.py",
        "old_2026_prediction_package": phase_c / "delivery_packages" / "trading_handoff_C1_2026_H1_v1" / "C1_unified_prediction_table_2026_H1_walkforward_v1.csv",
        "old_2026_code": phase_c / "scripts" / "run_c1_c4_weekly_2026_v3.py",
        "initial_threshold_readme": project_root / "02_prediction_agent_v2" / "README.md",
        "initial_label_builder": project_root / "02_prediction_agent_v2" / "data_audit" / "refresh_data_architecture_v2.py",
        "m8_5_20_classifier_code": project_root / "02_prediction_agent_v2" / "models" / "formal_m8_spike_classifier_v2.py",
        "m11_multiresolution": project_root / "02_prediction_agent_v2" / "outputs" / "metrics" / "M11_multiresolution_metrics_v2.csv",
        "b2_classifier_config": phase_c / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "config.json",
        "b2_classifier_summary": phase_c / "outputs" / "experiments" / "B2_classifier_5_20_formal_v3r1_f5730506" / "run_summary.json",
        "2026_prediction_manifest": phase_c
        / "delivery_packages"
        / "trading_handoff_C1_2026_H1_v1"
        / "C1_unified_prediction_table_2026_H1_manifest_v1.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required audit inputs: {missing}")

    input_hashes = {name: sha256(path) for name, path in files.items()}
    run_id = stable_run_id(input_hashes)

    hourly_2025 = load_hourly(files["hourly_strategy_2025"], "2025 OOF", 8760)
    hourly_2026 = load_hourly(files["hourly_strategy_2026"], "2026 H1 OOT", 4197)
    strategy = pd.concat(
        [strategy_summary(hourly_2025, "2025 OOF"), strategy_summary(hourly_2026, "2026 H1 OOT")],
        ignore_index=True,
    )
    strategy.insert(0, "run_id", run_id)
    strategy.to_csv(output_dir / "phase0_strategy_summary.csv", index=False)

    expected = {
        ("2026 H1 OOT", "S1"): (441, 6991.486025),
        ("2026 H1 OOT", "S3"): (353, 8045.90695),
        ("2026 H1 OOT", "S5"): (158, 7818.6134375),
        ("2025 OOF", "S1"): (1174, 3438.9964),
        ("2025 OOF", "S3"): (933, 2986.866225),
        ("2025 OOF", "S5"): (425, 3617.17815),
    }
    for key, (trades, pnl) in expected.items():
        period, strategy_id = key
        row = strategy.loc[
            strategy["period"].eq(period) & strategy["strategy_id"].eq(strategy_id)
        ].iloc[0]
        if int(row["trades"]) != trades or not np.isclose(float(row["total_pnl_usd"]), pnl, atol=1e-9):
            raise AssertionError(f"Canonical strategy mismatch for {key}: {row.to_dict()}")

    old = pd.read_csv(files["old_2026_prediction_package"])
    old["delivery_hour_utc"] = pd.to_datetime(old["delivery_hour_utc"], utc=True)
    s1 = hourly_2026.loc[hourly_2026["strategy_id"].eq("S1")].copy()
    merged = old.merge(
        s1[["delivery_hour_utc", "signal", "actual_spread", "net_pnl"]],
        on="delivery_hour_utc",
        suffixes=("_old", "_canonical"),
        validate="one_to_one",
    )
    if len(merged) != 4197:
        raise AssertionError("Old and canonical 2026 hourly sets do not match")
    if (merged["signal_old"].astype(int) != merged["signal_canonical"].astype(int)).any():
        raise AssertionError("Old and canonical 2026 signals differ")
    if not np.allclose(merged["actual_spread_old"], merged["actual_spread_canonical"], atol=1e-12):
        raise AssertionError("Old and canonical realized spreads differ")

    merged["pnl_old_recomputed"] = settle(
        merged["signal_canonical"], merged["actual_spread_canonical"], OLD_SPREAD_CLIP
    )
    merged["pnl_canonical_recomputed"] = settle(
        merged["signal_canonical"], merged["actual_spread_canonical"], None
    )
    old_sum = float(pd.to_numeric(old["net_pnl"], errors="raise").sum())
    old_recomputed_sum = float(merged["pnl_old_recomputed"].sum())
    canonical_sum = float(s1["net_pnl"].sum())
    canonical_recomputed_sum = float(merged["pnl_canonical_recomputed"].sum())
    if not np.isclose(old_sum, 5705.04965, atol=1e-9):
        raise AssertionError(f"Unexpected old PnL: {old_sum}")
    if not np.isclose(old_sum, old_recomputed_sum, atol=1e-9):
        raise AssertionError("Old stored PnL does not reproduce clipped settlement")
    if not np.isclose(canonical_sum, 6991.486025, atol=1e-9):
        raise AssertionError(f"Unexpected canonical PnL: {canonical_sum}")
    if not np.isclose(canonical_sum, canonical_recomputed_sum, atol=1e-9):
        raise AssertionError("Canonical stored PnL does not reproduce unclipped settlement")

    difference = merged.loc[
        ~np.isclose(merged["pnl_old_recomputed"], merged["pnl_canonical_recomputed"], atol=1e-12),
        [
            "delivery_hour_utc",
            "signal_canonical",
            "actual_spread_canonical",
            "pnl_old_recomputed",
            "pnl_canonical_recomputed",
        ],
    ].copy()
    difference["old_clipped_spread"] = difference["actual_spread_canonical"].clip(*OLD_SPREAD_CLIP)
    difference["pnl_difference_usd"] = (
        difference["pnl_canonical_recomputed"] - difference["pnl_old_recomputed"]
    )
    difference.insert(0, "run_id", run_id)
    difference.to_csv(output_dir / "phase0_clipping_difference_rows.csv", index=False)
    if len(difference) != 5 or not np.isclose(difference["pnl_difference_usd"].sum(), canonical_sum - old_sum, atol=1e-9):
        raise AssertionError("Clipping difference is not fully explained by the five affected hours")

    reconciliation = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "comparison": "old_prediction_package_clipped",
                "eligible_hours": len(old),
                "trades": int(old["signal"].astype(int).ne(0).sum()),
                "total_pnl_usd": old_sum,
                "spread_treatment": "clip[-1000,5000]",
                "status": "reproduced",
            },
            {
                "run_id": run_id,
                "comparison": "submitted_canonical_unclipped",
                "eligible_hours": len(s1),
                "trades": int(s1["signal"].ne(0).sum()),
                "total_pnl_usd": canonical_sum,
                "spread_treatment": "unclipped observed spread",
                "status": "reproduced",
            },
            {
                "run_id": run_id,
                "comparison": "canonical_minus_old",
                "eligible_hours": len(difference),
                "trades": len(difference),
                "total_pnl_usd": canonical_sum - old_sum,
                "spread_treatment": "difference caused only by clipping",
                "status": "fully reconciled",
            },
        ]
    )
    reconciliation.to_csv(output_dir / "phase0_pnl_reconciliation.csv", index=False)

    model = pd.read_parquet(files["model_input_frozen"])
    required_model = {"spread", "research_period", "evaluation_eligible", "delivery_hour_utc"}
    if not required_model.issubset(model.columns):
        raise AssertionError(f"Frozen model input lacks {sorted(required_model - set(model.columns))}")
    model = model.loc[pd.to_numeric(model["evaluation_eligible"], errors="coerce").eq(1)].copy()
    model["period"] = research_period_label(model["research_period"])
    model = model.loc[model["period"].isin(PERIOD_ORDER)].copy()
    model["spread_class"] = class_5_20(model["spread"])

    counts = (
        model.groupby(["period", "spread_class"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = model.groupby("period").size().rename("period_total")
    counts["period_total"] = counts["period"].map(totals)
    counts["proportion"] = counts["count"] / counts["period_total"]
    counts["period"] = pd.Categorical(counts["period"], PERIOD_ORDER, ordered=True)
    counts["spread_class"] = pd.Categorical(counts["spread_class"], CLASS_ORDER, ordered=True)
    counts = counts.sort_values(["period", "spread_class"]).reset_index(drop=True)
    counts.insert(0, "run_id", run_id)
    counts.to_csv(output_dir / "r1_1_class_counts.csv", index=False)

    class_stats = (
        model.groupby(["period", "spread_class"], observed=False)["spread"]
        .agg(
            count="size",
            mean="mean",
            minimum="min",
            q25=lambda x: x.quantile(0.25),
            median="median",
            q75=lambda x: x.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    class_stats["period"] = pd.Categorical(class_stats["period"], PERIOD_ORDER, ordered=True)
    class_stats["spread_class"] = pd.Categorical(class_stats["spread_class"], CLASS_ORDER, ordered=True)
    class_stats = class_stats.sort_values(["period", "spread_class"]).reset_index(drop=True)
    class_stats.insert(0, "run_id", run_id)
    class_stats.to_csv(output_dir / "r1_1_class_spread_statistics.csv", index=False)

    boundary_rows: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        spread = pd.to_numeric(model.loc[model["period"].eq(period), "spread"], errors="raise").astype(float)
        class_counts = model.loc[model["period"].eq(period), "spread_class"].value_counts()
        boundary_rows.append(
            {
                "run_id": run_id,
                "period": period,
                "rows": len(spread),
                "q01": spread.quantile(0.01),
                "q05": spread.quantile(0.05),
                "q25": spread.quantile(0.25),
                "median": spread.quantile(0.50),
                "q75": spread.quantile(0.75),
                "q95": spread.quantile(0.95),
                "q99": spread.quantile(0.99),
                "central_abs_le_5_proportion": float((spread.abs() <= 5.0).mean()),
                "tail_abs_gt_20_proportion": float((spread.abs() > 20.0).mean()),
                "negative_tail_lt_minus20_proportion": float((spread < -20.0).mean()),
                "positive_tail_gt_20_proportion": float((spread > 20.0).mean()),
                "moderate_negative_c2_proportion": float(class_counts.get("C2", 0) / len(spread)),
                "moderate_positive_c4_proportion": float(class_counts.get("C4", 0) / len(spread)),
            }
        )
    pd.DataFrame(boundary_rows).to_csv(output_dir / "r1_1_boundary_context.csv", index=False)

    inventory = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "artifact_key": name,
                "path": str(path),
                "sha256": input_hashes[name],
                "bytes": path.stat().st_size,
            }
            for name, path in files.items()
        ]
    )
    inventory.to_csv(output_dir / "evidence_inventory_sha256.csv", index=False)

    command = (
        f'"{Path(sys.executable)}" "{Path(__file__).resolve()}" '
        f'--project-root "{project_root}" --submitted-pdf "{submitted_pdf}" '
        f'--output-dir "{output_dir}"'
    )
    manifest = {
        "run_id": run_id,
        "audit_version": AUDIT_VERSION,
        "status": "COMPLETE_READ_ONLY_AUDIT",
        "scope": ["Phase 0 economic baseline", "Reviewer 1.1 threshold evidence"],
        "no_model_training": True,
        "source_artifacts_modified": False,
        "parameters": {
            "spread_definition": "RT_minus_DA",
            "capture_rate": CAPTURE_RATE,
            "commission_usd_per_mwh": COMMISSION_USD_PER_MWH,
            "slippage_rate": SLIPPAGE_RATE,
            "old_spread_clip": list(OLD_SPREAD_CLIP),
            "canonical_spread_treatment": "unclipped",
            "class_boundaries_usd_per_mwh": [-20.0, -5.0, 5.0, 20.0],
            "evaluation_population": "evaluation_eligible == 1 in frozen model input",
        },
        "input_artifacts": {
            name: {"path": str(path), "sha256": input_hashes[name], "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "audit_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "version": AUDIT_VERSION,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "reproduce_command": command,
        "acceptance_checks": {
            "canonical_2026_rows": len(hourly_2026),
            "canonical_2026_unique_hours": hourly_2026["delivery_hour_utc"].nunique(),
            "old_and_canonical_signal_mismatch_rows": int(
                (merged["signal_old"].astype(int) != merged["signal_canonical"].astype(int)).sum()
            ),
            "old_total_pnl_usd": old_sum,
            "canonical_s1_total_pnl_usd": canonical_sum,
            "canonical_s5_total_pnl_usd": float(
                strategy.loc[
                    strategy["period"].eq("2026 H1 OOT") & strategy["strategy_id"].eq("S5"),
                    "total_pnl_usd",
                ].iloc[0]
            ),
            "clipping_difference_rows": len(difference),
            "clipping_explained_difference_usd": float(difference["pnl_difference_usd"].sum()),
        },
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
