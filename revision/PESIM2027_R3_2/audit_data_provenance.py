"""Read-only provenance and forecast-availability audit for PES IM 2027 R3.2.

The script does not call weather APIs, retrain models, or alter source data. It
hashes the datacollection branch content through ``git show`` and reconciles the
surviving local model-wide, frozen-input, prediction, and canonical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_VERSION = "pesim2027-r3.2-data-provenance-v1"
BRANCH_REF = "origin/datacollection"
BRANCH_FILES = (
    "01_data_collection_cleaning/scripts/collectors/openmeteo_collector.py",
    "01_data_collection_cleaning/scripts/collectors/common.py",
    "01_data_collection_cleaning/scripts/collect_data.py",
    "01_data_collection_cleaning/scripts/recover_single_run_weather.py",
    "01_data_collection_cleaning/scripts/build_clean.py",
    "01_data_collection_cleaning/scripts/build_model_wide.py",
    "01_data_collection_cleaning/scripts/import_pre_dam_forecast_features.py",
    "01_data_collection_cleaning/DATA_COLLECTION_STATUS.md",
    "01_data_collection_cleaning/DATA_INVENTORY.md",
    "01_data_collection_cleaning/README.md",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(repo: Path, spec: str) -> bytes:
    return subprocess.check_output(["git", "show", spec], cwd=repo)


def git_text(repo: Path, spec: str) -> str:
    return git_bytes(repo, spec).decode("utf-8")


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def evaluation_slice(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    local_date = pd.to_datetime(frame["delivery_date_local"], errors="raise")
    eligible = frame["evaluation_eligible"].eq(1)
    if year == 2025:
        keep = local_date.between("2025-01-01", "2025-12-31")
    elif year == 2026:
        keep = local_date.between("2026-01-01", "2026-06-30")
    else:
        raise ValueError(year)
    result = frame.loc[eligible & keep].copy()
    result["delivery_hour_utc"] = as_utc(result["delivery_hour_utc"])
    return result.sort_values("delivery_hour_utc").reset_index(drop=True)


def compare_columns(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            mismatch = ~np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=0.0, atol=1e-12, equal_nan=True)
        else:
            mismatch = a.fillna("<NA>").astype(str).to_numpy() != b.fillna("<NA>").astype(str).to_numpy()
        rows.append({"field": column, "rows": len(left), "mismatched_rows": int(mismatch.sum()), "status": "PASS" if not mismatch.any() else "FAIL"})
    return rows


def find_local_provenance_files(search_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__"}
    for current, directories, files in os.walk(search_root):
        directories[:] = [name for name in directories if name not in skip_dirs]
        current_path = Path(current)
        for name in files:
            lower_name = name.lower()
            lower_path = str(current_path / name).lower()
            kind = None
            if lower_name.endswith(".metadata.json") and "openmeteo" in lower_path:
                kind = "openmeteo_metadata"
            elif lower_name in {"ercot_analytics.sqlite", "ercot_data.sqlite"}:
                kind = "sqlite"
            elif "previous-runs" in lower_path or "single-run" in lower_path or "historical-forecast" in lower_path:
                kind = "weather_raw_or_named_artifact"
            elif lower_name.startswith("model_wide_hourly_2024_2026"):
                kind = "model_wide"
            if kind is not None:
                path = current_path / name
                rows.append({"kind": kind, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows, columns=["kind", "path", "size_bytes", "sha256"]).sort_values(["kind", "path"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--submitted-pdf", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    branch_commit = git_value(repo, "rev-parse", BRANCH_REF)
    branch_payloads = {path: git_bytes(repo, f"{BRANCH_REF}:{path}") for path in BRANCH_FILES}
    branch_hashes = {path: sha256_bytes(content) for path, content in branch_payloads.items()}
    collector = branch_payloads[BRANCH_FILES[0]].decode("utf-8")
    build_clean = branch_payloads[BRANCH_FILES[4]].decode("utf-8")
    status_doc = branch_payloads[BRANCH_FILES[7]].decode("utf-8")
    inventory_doc = branch_payloads[BRANCH_FILES[8]].decode("utf-8")

    previous_params = collector.split("def collect_openmeteo_previous_runs_weather", 1)[1].split("def _delivery_day_bounds_utc", 1)[0]
    assertions = {
        "previous_runs_endpoint_present": '"previous-runs": "https://previous-runs-api.open-meteo.com/v1/forecast"' in collector,
        "previous_runs_request_omits_models_parameter": '"models"' not in previous_params,
        "previous_runs_requests_day1_and_day2": "for day in ((1, 2) if lead_mode == \"hybrid\" else (2,))" in previous_params,
        "forecast_run_time_is_derived_target_minus_lead": 'iso_utc(target - timedelta(hours=lead_hours))' in collector,
        "clean_layer_selects_previous_runs_hybrid_only": "f.dataset = 'previous-runs-hybrid'" in build_clean,
        "clean_layer_excludes_historical_by_exact_dataset_filter": "f.dataset = 'previous-runs-hybrid'" in build_clean and "f.dataset IN" not in build_clean.split("WEATHER_INSERT_SQL", 1)[1].split('"""', 2)[1],
        "early_status_document_describes_historical_forecast": "Open-Meteo Historical Forecast API" in status_doc,
        "early_inventory_warns_no_issued_at": "没有精确的历史发布时间" in inventory_doc,
    }
    if not all(assertions.values()):
        raise AssertionError({key: value for key, value in assertions.items() if not value})

    paths = {
        "submitted_pdf": args.submitted_pdf.resolve(),
        "final_model_wide_previous_runs": root / "02_prediction_agent_v2/model_wide_hourly_2024_2026_final(2).csv",
        "frozen_model_input": root / "02_prediction_agent_v2/data/model_ready/model_input_frozen_v2.parquet",
        "deprecated_model_wide_historical": root / "02_prediction_agent_v1_deprecated/data/model_wide_hourly_2024_2026_final.csv",
        "canonical_hourly_2025": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/outputs/hourly_strategy_results_2025.parquet",
        "canonical_hourly_2026": root / "04_trading_strategy_evaluation/section_iv_canonical_v1/outputs/hourly_strategy_results_2026.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    local_hashes = {name: sha256_file(path) for name, path in paths.items()}
    run_material = AUDIT_VERSION + "\n" + branch_commit + "\n"
    run_material += "\n".join(f"branch:{path}:{branch_hashes[path]}" for path in sorted(branch_hashes))
    run_material += "\n" + "\n".join(f"local:{name}:{local_hashes[name]}" for name in sorted(local_hashes))
    run_id = f"R3_2_PROV_{hashlib.sha256(run_material.encode('utf-8')).hexdigest()[:16].upper()}"

    final_source = pd.read_csv(paths["final_model_wide_previous_runs"])
    frozen = pd.read_parquet(paths["frozen_model_input"])
    historical = pd.read_csv(paths["deprecated_model_wide_historical"])
    for frame in (final_source, frozen, historical):
        frame["delivery_hour_utc"] = as_utc(frame["delivery_hour_utc"])

    if set(final_source["weather_availability_assumption"].dropna().unique()) != {"openmeteo_previous_day1_local_hours_00_08_else_day2_before_pre_dam_cutoff"}:
        raise AssertionError("Final model-wide source is not uniformly previous-runs hybrid")
    if set(final_source["weather_forecast_model"].dropna().unique()) != {"openmeteo_previous_runs_default_model"}:
        raise AssertionError("Unexpected final weather model label")
    if set(historical["weather_availability_assumption"].dropna().unique()) != {"historical_forecast_without_publish_time"}:
        raise AssertionError("Deprecated file is not uniformly historical-forecast")

    weather_columns = [
        "dfw_city_count", "weather_forecast_run_time_utc", "weather_decision_cutoff_utc",
        "weather_forecast_model", "weather_forecast_lead_hours",
        "temperature_dfw_mean_c", "temperature_dfw_min_c", "temperature_dfw_max_c",
        "temperature_wichita_c", "temperature_wichita_minus_dfw_c",
        "humidity_dfw_mean_pct", "humidity_wichita_pct", "humidity_wichita_minus_dfw_pct",
        "wind_speed_dfw_mean_ms", "wind_speed_dfw_max_ms", "wind_speed_wichita_ms", "wind_speed_wichita_minus_dfw_ms",
        "wind_gust_dfw_mean_ms", "wind_gust_dfw_max_ms", "wind_gust_wichita_ms", "wind_gust_wichita_minus_dfw_ms",
        "cloud_cover_dfw_mean_pct", "cloud_cover_wichita_pct", "cloud_cover_wichita_minus_dfw_pct",
        "radiation_dfw_mean_wm2", "radiation_wichita_wm2", "radiation_wichita_minus_dfw_wm2",
        "precipitation_dfw_mean_mm", "precipitation_dfw_max_mm", "precipitation_wichita_mm", "precipitation_wichita_minus_dfw_mm",
        "north_temperature_min_c", "north_temperature_max_c", "north_wind_gust_max_ms", "north_precipitation_max_mm",
        "freezing_city_count", "extreme_heat_city_count", "high_wind_city_count", "rainy_city_count",
        "weather_availability_assumption",
    ]
    joined = frozen[["delivery_hour_utc", *weather_columns]].merge(
        final_source[["delivery_hour_utc", *weather_columns]], on="delivery_hour_utc",
        how="left", validate="one_to_one", suffixes=("_frozen", "_source"), indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise AssertionError("Frozen timestamps missing from final model-wide source")
    left = joined[[f"{column}_frozen" for column in weather_columns]].copy()
    left.columns = weather_columns
    right = joined[[f"{column}_source" for column in weather_columns]].copy()
    right.columns = weather_columns
    layer_rows = compare_columns(left, right, weather_columns)
    layer = pd.DataFrame(layer_rows)
    layer.insert(0, "run_id", run_id)
    layer.insert(1, "comparison", "final_model_wide_to_frozen_input")
    if layer["status"].ne("PASS").any():
        raise AssertionError("Final source/frozen weather fields differ")

    transition_fields = ["temperature_dfw_mean_c", "wind_gust_dfw_max_ms", "precipitation_dfw_mean_mm"]
    transition = historical[["delivery_hour_utc", *transition_fields]].merge(
        final_source[["delivery_hour_utc", *transition_fields]], on="delivery_hour_utc",
        how="inner", suffixes=("_historical", "_previous_runs"), validate="one_to_one",
    )
    transition_rows = []
    for field in transition_fields:
        old = transition[f"{field}_historical"].to_numpy(float)
        new = transition[f"{field}_previous_runs"].to_numpy(float)
        unequal = ~np.isclose(old, new, rtol=0.0, atol=1e-12, equal_nan=True)
        transition_rows.append({
            "run_id": run_id, "field": field, "common_hours": len(transition),
            "different_hours": int(unequal.sum()), "mean_absolute_difference": float(np.nanmean(np.abs(old - new))),
        })

    provenance_rows = [
        {
            "run_id": run_id, "layer": "deprecated_local_model_wide", "rows": len(historical),
            "utc_start": historical["delivery_hour_utc"].min().isoformat(), "utc_end": historical["delivery_hour_utc"].max().isoformat(),
            "source_marker": "historical_forecast_without_publish_time", "model_marker": "not recorded",
        },
        {
            "run_id": run_id, "layer": "final_local_model_wide", "rows": len(final_source),
            "utc_start": final_source["delivery_hour_utc"].min().isoformat(), "utc_end": final_source["delivery_hour_utc"].max().isoformat(),
            "source_marker": final_source["weather_availability_assumption"].iloc[0], "model_marker": final_source["weather_forecast_model"].iloc[0],
        },
        {
            "run_id": run_id, "layer": "frozen_model_input", "rows": len(frozen),
            "utc_start": frozen["delivery_hour_utc"].min().isoformat(), "utc_end": frozen["delivery_hour_utc"].max().isoformat(),
            "source_marker": frozen["weather_availability_assumption"].iloc[0], "model_marker": frozen["weather_forecast_model"].iloc[0],
        },
    ]

    availability_rows: list[dict[str, object]] = []
    margin_distribution_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for year, period in [(2025, "2025 OOF"), (2026, "2026 H1 OOT")]:
        frame = evaluation_slice(frozen, year)
        target = as_utc(frame["delivery_hour_utc"])
        reference = as_utc(frame["weather_forecast_run_time_utc"])
        cutoff = as_utc(frame["weather_decision_cutoff_utc"])
        lead = pd.to_numeric(frame["weather_forecast_lead_hours"])
        local_hour = pd.to_numeric(frame["ercot_local_hour"]).astype(int)
        constructed_mismatch = ((target - reference).dt.total_seconds() / 3600.0 - lead).abs().gt(1e-9)
        margin_minutes = (cutoff - reference).dt.total_seconds() / 60.0
        hourly = pd.read_parquet(paths[f"canonical_hourly_{year}"])
        s5 = hourly.loc[hourly["strategy_id"].eq("S5")].copy()
        s5["delivery_hour_utc"] = as_utc(s5["delivery_hour_utc"])
        signal_lookup = s5.set_index("delivery_hour_utc")["signal"]
        frame["canonical_s5_signal"] = frame["delivery_hour_utc"].map(signal_lookup).astype(int)
        edge = margin_minutes.le(120)
        availability_rows.append({
            "run_id": run_id, "period": period, "eligible_hours": len(frame),
            "lead_24_hours": int(lead.eq(24).sum()), "lead_48_hours": int(lead.eq(48).sum()),
            "constructed_reference_after_cutoff_hours": int(reference.ge(cutoff).sum()),
            "target_minus_reference_identity_mismatches": int(constructed_mismatch.sum()),
            "nominal_margin_min_minutes": float(margin_minutes.min()),
            "nominal_margin_max_minutes": float(margin_minutes.max()),
            "local_08_hours": int(local_hour.eq(8).sum()),
            "local_08_nominal_margin_min_minutes": float(margin_minutes[local_hour.eq(8)].min()),
            "nominal_margin_le_60_minutes": int(margin_minutes.le(60).sum()),
            "nominal_margin_le_120_minutes": int(edge.sum()),
            "actual_publish_timestamp_available": False,
            "actual_participant_availability_timestamp_available": False,
            "weather_trigger_hours": int(frame["fixed_extreme_weather_flag"].sum()),
            "canonical_weather_gate_trades": int(s5["signal"].ne(0).sum()),
            "lead_24_weather_trigger_hours": int((lead.eq(24) & frame["fixed_extreme_weather_flag"].eq(1)).sum()),
            "lead_24_weather_gate_trades": int((lead.eq(24) & frame["canonical_s5_signal"].ne(0)).sum()),
            "edge_weather_trigger_hours": int((edge & frame["fixed_extreme_weather_flag"].eq(1)).sum()),
            "edge_weather_gate_trades": int((edge & frame["canonical_s5_signal"].ne(0)).sum()),
        })
        margin_frame = pd.DataFrame({
            "local_hour": local_hour, "lead_hours": lead, "nominal_margin_minutes": margin_minutes,
        })
        grouped = margin_frame.groupby(["local_hour", "lead_hours"], as_index=False)["nominal_margin_minutes"].agg(["count", "min", "median", "max"]).reset_index()
        for _, row in grouped.iterrows():
            margin_distribution_rows.append({
                "run_id": run_id, "period": period, "local_hour": int(row["local_hour"]),
                "lead_hours": float(row["lead_hours"]), "hours": int(row["count"]),
                "margin_min_minutes": float(row["min"]), "margin_median_minutes": float(row["median"]),
                "margin_max_minutes": float(row["max"]),
            })
        edge_part = frame.loc[edge, [
            "delivery_hour_utc", "delivery_time_local", "delivery_date_local", "ercot_local_hour",
            "weather_forecast_run_time_utc", "weather_decision_cutoff_utc", "weather_forecast_lead_hours",
            "fixed_extreme_weather_flag", "canonical_s5_signal",
        ]].copy()
        edge_part["run_id"] = run_id
        edge_part["period"] = period
        edge_part["nominal_margin_minutes"] = margin_minutes.loc[edge].to_numpy()
        edge_rows.extend(edge_part.to_dict("records"))

    request_spec = {
        "run_id": run_id,
        "datacollection_commit": branch_commit,
        "endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "models_parameter": None,
        "model_interpretation": "Open-Meteo default configuration; exact NWP model not recoverable from surviving filtered payload or local metadata",
        "locations": [
            {"name": "Dallas", "latitude": 32.7767, "longitude": -96.7970},
            {"name": "Fort Worth", "latitude": 32.7555, "longitude": -97.3308},
            {"name": "Denton", "latitude": 33.2148, "longitude": -97.1331},
            {"name": "McKinney", "latitude": 33.1972, "longitude": -96.6397},
            {"name": "Arlington", "latitude": 32.7357, "longitude": -97.1081},
            {"name": "Wichita Falls", "latitude": 33.9137, "longitude": -98.4934},
        ],
        "hourly_base_variables": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "shortwave_radiation", "precipitation"],
        "hourly_requested_variants": "each base variable with _previous_day1 and _previous_day2 in hybrid mode",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "date_parameters": "start_date and end_date in 31-day chunks, with query window expanded by one day on each side",
        "lead_rule": "local delivery hours 00:00-08:00 select previous_day1 (24 h); 09:00-23:00 select previous_day2 (48 h)",
        "decision_cutoff": "D-1 09:55 America/Chicago",
        "forecast_run_time_warning": "constructed as target_hour_utc minus 24/48 h; not a supplier publication timestamp",
        "api_key": "optional and omitted from saved public_params",
    }

    local_inventory = find_local_provenance_files(args.search_root.resolve())
    local_inventory.insert(0, "run_id", run_id)
    branch_inventory = pd.DataFrame([
        {"run_id": run_id, "branch_ref": BRANCH_REF, "branch_commit": branch_commit, "path": path, "sha256": branch_hashes[path]}
        for path in BRANCH_FILES
    ])
    local_evidence = pd.DataFrame([
        {"run_id": run_id, "artifact": name, "path": str(path), "sha256": local_hashes[name]}
        for name, path in paths.items()
    ])
    checks = pd.DataFrame([{"run_id": run_id, "check": key, "status": "PASS" if value else "FAIL"} for key, value in assertions.items()])

    outputs = {
        "provenance_layers.csv": pd.DataFrame(provenance_rows),
        "source_transition_comparison.csv": pd.DataFrame(transition_rows),
        "weather_layer_reconciliation.csv": layer,
        "forecast_availability_summary.csv": pd.DataFrame(availability_rows),
        "availability_margin_distribution.csv": pd.DataFrame(margin_distribution_rows),
        "availability_edge_hours.csv": pd.DataFrame(edge_rows),
        "branch_evidence_sha256.csv": branch_inventory,
        "local_evidence_sha256.csv": local_evidence,
        "local_provenance_file_inventory.csv": local_inventory,
        "source_code_assertions.csv": checks,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    (out / "recovered_request_spec.json").write_text(json.dumps(request_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_paths = [out / name for name in outputs] + [out / "recovered_request_spec.json"]
    manifest = {
        "run_id": run_id,
        "audit_version": AUDIT_VERSION,
        "status": "PASS_WITH_AVAILABILITY_LIMITATION",
        "read_only": True,
        "datacollection_ref": BRANCH_REF,
        "datacollection_commit": branch_commit,
        "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "branch_files": {path: {"sha256": branch_hashes[path]} for path in BRANCH_FILES},
            "local_files": {name: {"path": str(path), "sha256": local_hashes[name]} for name, path in paths.items()},
        },
        "outputs": {path.name: {"path": str(path), "sha256": sha256_file(path)} for path in output_paths},
        "limitations": [
            "No original previous-runs raw response, sibling metadata JSON, or collection SQLite database was found under the searched local root.",
            "forecast_run_time_utc is constructed from target time and lag; it is not an observed publication timestamp.",
            "No participant-side availability timestamp survives, so no hour can be independently certified against the DAM cutoff.",
        ],
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "datacollection_commit": branch_commit, "availability": availability_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
