"""Read-only audit of the PES IM 2027 canonical Weather Gate.

This script reads frozen feature, prediction-handoff, and canonical hourly-result
artifacts. It does not train models or modify any source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_VERSION = "pesim2027-r2.3-weather-gate-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def period_slice(frozen: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frozen.loc[frozen["evaluation_eligible"].eq(1)].copy()
    local_date = pd.to_datetime(frame["delivery_date_local"], errors="raise")
    if year == 2025:
        keep = local_date.between("2025-01-01", "2025-12-31")
    elif year == 2026:
        keep = local_date.between("2026-01-01", "2026-06-30")
    else:
        raise ValueError(year)
    frame = frame.loc[keep].copy()
    frame["delivery_hour_utc"] = as_utc(frame["delivery_hour_utc"])
    return frame.sort_values("delivery_hour_utc").reset_index(drop=True)


def signal_counts(prediction: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_plus = pd.to_numeric(prediction["p_positive"]).to_numpy(float)
    p_minus = pd.to_numeric(prediction["p_negative"]).to_numpy(float)
    predicted = pd.to_numeric(prediction["predicted_spread"]).to_numpy(float)
    s1 = np.zeros(len(prediction), dtype=np.int8)
    s1[(p_plus >= 0.60) & (p_plus > p_minus)] = 1
    s1[(p_minus >= 0.60) & (p_minus > p_plus)] = -1
    agrees = ((s1 == 1) & (predicted > 0)) | ((s1 == -1) & (predicted < 0))
    s3 = np.where(agrees, s1, 0).astype(np.int8)
    return s1, s3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--submitted-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "submitted_pdf": args.submitted_pdf.resolve(),
        "frozen_model_input": root / "02_prediction_agent_v2/data/model_ready/model_input_frozen_v2.parquet",
        "source_model_wide_csv": root / "02_prediction_agent_v2/model_wide_hourly_2024_2026_final(2).csv",
        "base_feature_table": root / "02_prediction_agent_v2/data/data_architecture/base_feature_table_leakage_safe_v2_partial.parquet",
        "weather_refresh_code": root / "02_prediction_agent_v2/data_audit/refresh_data_architecture_v2.py",
        "freeze_model_dataset_code": root / "02_prediction_agent_v2/data_audit/freeze_model_dataset_v2.py",
        "weather_design_document": root / "02_prediction_agent_v2/model_wide_explanation_zh_2024_2026(1).md",
        "feature_whitelist": root / "02_prediction_agent_v2/config/feature_whitelist_v2.csv",
        "canonical_execution_code": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/src/canonical_trading.py",
        "canonical_run_audit_code": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/src/run_section_iv.py",
        "strategy_specification": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/config/strategy_implementation_spec.csv",
        "prediction_2025": root / "02_prediction_agent_v2/phase_C_model_selection_validation/delivery_packages/trading_handoff_C1_v3/C1_unified_prediction_table_2025_oof_v3.parquet",
        "prediction_2026": root / "02_prediction_agent_v2/phase_C_model_selection_validation/delivery_packages/trading_handoff_C1_2026_H1_v1/C1_unified_prediction_table_2026_H1_walkforward_v1.parquet",
        "canonical_hourly_2025": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/outputs/hourly_strategy_results_2025.parquet",
        "canonical_hourly_2026": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/outputs/hourly_strategy_results_2026.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    run_payload = AUDIT_VERSION + "\n" + "\n".join(f"{name}:{hashes[name]}" for name in sorted(hashes))
    run_id = f"R2_3_WG_{hashlib.sha256(run_payload.encode('utf-8')).hexdigest()[:16].upper()}"

    frozen = pd.read_parquet(paths["frozen_model_input"])
    required = [
        "delivery_hour_utc", "delivery_date_local", "ercot_local_hour", "evaluation_eligible",
        "freezing_city_count", "extreme_heat_city_count", "high_wind_city_count", "rainy_city_count",
        "temperature_dfw_min_c", "temperature_dfw_max_c", "temperature_wichita_c",
        "wind_gust_dfw_max_ms", "wind_gust_wichita_ms",
        "precipitation_dfw_max_mm", "precipitation_wichita_mm",
        "freezing_hour_flag", "extreme_heat_hour_flag", "high_wind_hour_flag", "rainy_hour_flag",
        "fixed_extreme_weather_flag", "fixed_compound_extreme_count",
        "weather_forecast_run_time_utc", "weather_decision_cutoff_utc", "decision_time_utc",
        "project_cutoff_time_utc", "weather_forecast_lead_hours", "weather_availability_assumption",
        "weather_source_version", "dfw_city_count",
    ]
    absent = [column for column in required if column not in frozen.columns]
    if absent:
        raise AssertionError(f"Frozen model input lacks required columns: {absent}")

    identity_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    for year, label in [(2025, "2025 OOF"), (2026, "2026 H1 OOT")]:
        frame = period_slice(frozen, year)
        prediction = pd.read_parquet(paths[f"prediction_{year}"]).copy()
        prediction["delivery_hour_utc"] = as_utc(prediction["delivery_hour_utc"])
        prediction = prediction.sort_values("delivery_hour_utc").reset_index(drop=True)
        hourly = pd.read_parquet(paths[f"canonical_hourly_{year}"]).copy()
        hourly["delivery_hour_utc"] = as_utc(hourly["delivery_hour_utc"])

        expected_flags = {
            "freezing_hour_flag": frame["freezing_city_count"].gt(0).astype(int),
            "extreme_heat_hour_flag": frame["extreme_heat_city_count"].gt(0).astype(int),
            "high_wind_hour_flag": frame["high_wind_city_count"].gt(0).astype(int),
            "rainy_hour_flag": frame["rainy_city_count"].gt(0).astype(int),
        }
        expected_gate = (
            expected_flags["freezing_hour_flag"].eq(1)
            | expected_flags["extreme_heat_hour_flag"].eq(1)
            | expected_flags["high_wind_hour_flag"].eq(1)
        ).astype(int)
        expected_compound = (
            expected_flags["freezing_hour_flag"]
            + expected_flags["extreme_heat_hour_flag"]
            + expected_flags["high_wind_hour_flag"]
        )
        checks = {
            **{name: int((frame[name].astype(int) != expected).sum()) for name, expected in expected_flags.items()},
            "fixed_extreme_weather_flag": int((frame["fixed_extreme_weather_flag"].astype(int) != expected_gate).sum()),
            "fixed_compound_extreme_count": int((frame["fixed_compound_extreme_count"].astype(int) != expected_compound).sum()),
            "freezing_trigger_matches_dfw_min_or_wichita": int((
                frame["freezing_city_count"].gt(0)
                != (frame["temperature_dfw_min_c"].le(0) | frame["temperature_wichita_c"].le(0))
            ).sum()),
            "heat_trigger_matches_dfw_max_or_wichita": int((
                frame["extreme_heat_city_count"].gt(0)
                != (frame["temperature_dfw_max_c"].ge(35) | frame["temperature_wichita_c"].ge(35))
            ).sum()),
            "wind_trigger_matches_dfw_max_or_wichita": int((
                frame["high_wind_city_count"].gt(0)
                != (frame["wind_gust_dfw_max_ms"].ge(15) | frame["wind_gust_wichita_ms"].ge(15))
            ).sum()),
            "rain_trigger_matches_dfw_max_or_wichita": int((
                frame["rainy_city_count"].gt(0)
                != (frame["precipitation_dfw_max_mm"].gt(0.1) | frame["precipitation_wichita_mm"].gt(0.1))
            ).sum()),
        }
        for check, mismatches in checks.items():
            identity_rows.append({"run_id": run_id, "period": label, "check": check, "mismatched_hours": mismatches, "status": "PASS" if mismatches == 0 else "FAIL"})

        if len(frame) != len(prediction):
            raise AssertionError(f"{label}: frozen/prediction row mismatch {len(frame)} != {len(prediction)}")
        merged = frame[["delivery_hour_utc", "fixed_extreme_weather_flag"]].merge(
            prediction[["delivery_hour_utc", "fixed_extreme_weather_flag"]],
            on="delivery_hour_utc", how="outer", validate="one_to_one", suffixes=("_frozen", "_prediction"), indicator=True,
        )
        layer_rows.append({
            "run_id": run_id, "period": label, "comparison": "frozen_to_prediction_handoff",
            "rows": len(merged), "timestamp_mismatches": int(merged["_merge"].ne("both").sum()),
            "flag_mismatches": int((merged["fixed_extreme_weather_flag_frozen"] != merged["fixed_extreme_weather_flag_prediction"]).fillna(True).sum()),
        })

        s1, s3 = signal_counts(prediction)
        gate = prediction["fixed_extreme_weather_flag"].astype(int).to_numpy() == 1
        expected_s5 = np.where(gate, s3, 0).astype(np.int8)
        canonical_s3 = hourly.loc[hourly["strategy_id"].eq("S3")].sort_values("delivery_hour_utc").reset_index(drop=True)
        canonical_s5 = hourly.loc[hourly["strategy_id"].eq("S5")].sort_values("delivery_hour_utc").reset_index(drop=True)
        if not np.array_equal(canonical_s3["delivery_hour_utc"].to_numpy(), prediction["delivery_hour_utc"].to_numpy()):
            raise AssertionError(f"{label}: canonical S3 timestamps differ from prediction handoff")
        if not np.array_equal(canonical_s5["delivery_hour_utc"].to_numpy(), prediction["delivery_hour_utc"].to_numpy()):
            raise AssertionError(f"{label}: canonical S5 timestamps differ from prediction handoff")
        layer_rows.append({
            "run_id": run_id, "period": label, "comparison": "recomputed_S5_to_canonical_hourly",
            "rows": len(prediction), "timestamp_mismatches": 0,
            "flag_mismatches": int((canonical_s5["signal"].to_numpy(int) != expected_s5).sum()),
        })

        count_rows.append({
            "run_id": run_id, "period": label, "eligible_hours": len(frame),
            "freezing_trigger_hours": int(frame["freezing_hour_flag"].sum()),
            "extreme_heat_trigger_hours": int(frame["extreme_heat_hour_flag"].sum()),
            "high_wind_trigger_hours": int(frame["high_wind_hour_flag"].sum()),
            "rainy_hours_not_in_gate": int(frame["rainy_hour_flag"].sum()),
            "weather_gate_trigger_hours": int(frame["fixed_extreme_weather_flag"].sum()),
            "base_probability_trades_S1": int(np.count_nonzero(s1)),
            "direction_confirmed_trades_S3": int(np.count_nonzero(s3)),
            "weather_gate_trades_S5": int(np.count_nonzero(expected_s5)),
            "canonical_S5_trades": int(canonical_s5["signal"].ne(0).sum()),
        })

        issue = as_utc(frame["weather_forecast_run_time_utc"])
        weather_cutoff = as_utc(frame["weather_decision_cutoff_utc"])
        project_cutoff = as_utc(frame["project_cutoff_time_utc"])
        decision = as_utc(frame["decision_time_utc"])
        delivery = as_utc(frame["delivery_hour_utc"])
        assumptions = sorted(frame["weather_availability_assumption"].astype(str).unique().tolist())
        versions = sorted(frame["weather_source_version"].astype(str).unique().tolist())
        local_hour = pd.to_numeric(frame["ercot_local_hour"]).astype(int)
        lead = pd.to_numeric(frame["weather_forecast_lead_hours"])
        expected_lead = np.where(local_hour.le(8), 24.0, 48.0)
        time_rows.append({
            "run_id": run_id, "period": label, "hours": len(frame),
            "issue_not_before_delivery_hours": int(issue.ge(delivery).sum()),
            "issue_not_before_project_cutoff_hours": int(issue.ge(project_cutoff).sum()),
            "issue_after_weather_cutoff_hours": int(issue.gt(weather_cutoff).sum()),
            "issue_after_decision_time_hours": int(issue.gt(decision).sum()),
            "stored_lead_hours_min": float(pd.to_numeric(frame["weather_forecast_lead_hours"]).min()),
            "stored_lead_hours_max": float(pd.to_numeric(frame["weather_forecast_lead_hours"]).max()),
            "lead_24_hours": int(lead.eq(24).sum()),
            "lead_48_hours": int(lead.eq(48).sum()),
            "hybrid_local_hour_mapping_mismatches": int((lead.to_numpy(float) != expected_lead).sum()),
            "decision_time_not_0955_local_hours": int((decision.dt.tz_convert("US/Central").dt.strftime("%H:%M") != "09:55").sum()),
            "project_cutoff_not_1000_local_hours": int((project_cutoff.dt.tz_convert("US/Central").dt.strftime("%H:%M") != "10:00").sum()),
            "forecast_run_utc_hours": ";".join(map(str, sorted(issue.dt.hour.unique().tolist()))),
            "availability_assumptions": ";".join(assumptions),
            "weather_source_versions": ";".join(versions),
            "dfw_city_count_values": ";".join(map(str, sorted(frame["dfw_city_count"].unique().tolist()))),
        })

    identities = pd.DataFrame(identity_rows)
    counts = pd.DataFrame(count_rows)
    times = pd.DataFrame(time_rows)
    layers = pd.DataFrame(layer_rows)
    if identities["status"].ne("PASS").any() or layers[["timestamp_mismatches", "flag_mismatches"]].to_numpy().any():
        raise AssertionError("Weather Gate layer or identity audit failed")
    expected_trades = {"2025 OOF": 425, "2026 H1 OOT": 158}
    for _, row in counts.iterrows():
        if int(row["canonical_S5_trades"]) != expected_trades[row["period"]]:
            raise AssertionError(f"{row['period']}: canonical S5 trades differ from expected paper count")

    identities.to_csv(out / "weather_gate_field_identity.csv", index=False)
    counts.to_csv(out / "weather_gate_counts.csv", index=False)
    times.to_csv(out / "weather_gate_time_audit.csv", index=False)
    layers.to_csv(out / "weather_gate_layer_reconciliation.csv", index=False)
    inventory = pd.DataFrame([
        {"run_id": run_id, "artifact": name, "path": str(path), "sha256": hashes[name]}
        for name, path in paths.items()
    ])
    inventory.to_csv(out / "evidence_inventory_sha256.csv", index=False)
    output_names = [
        "weather_gate_field_identity.csv", "weather_gate_counts.csv", "weather_gate_time_audit.csv",
        "weather_gate_layer_reconciliation.csv", "evidence_inventory_sha256.csv",
    ]
    manifest = {
        "run_id": run_id,
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "read_only": True,
        "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "project_root": str(root),
        "output_dir": str(out),
        "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
        "outputs": {name: {"path": str(out / name), "sha256": sha256_file(out / name)} for name in output_names},
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": "PASS", "counts": count_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
