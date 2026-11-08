from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_VERSION = "1.0.0"
COMMISSION = 2.0
SLIPPAGE = 0.005
FROZEN_TAU = 0.60
FROZEN_GAMMA = 0.65
TAU_GRID = [round(value, 3) for value in np.arange(0.50, 0.8001, 0.025)]
GAMMA_GRID = [0.50, 0.65, 0.80, 1.00]
CLIP_BOUNDS = (-1000.0, 5000.0)
STRATEGY_NAMES = {"S1": "Dual-XGB", "S3": "Direction Confirmation", "S5": "Weather Gate"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def probability_signal(frame: pd.DataFrame, tau: float) -> np.ndarray:
    p_positive = frame["p_positive"].to_numpy(float)
    p_negative = frame["p_negative"].to_numpy(float)
    signal = np.zeros(len(frame), dtype=np.int8)
    signal[(p_positive >= tau) & (p_positive > p_negative)] = 1
    signal[(p_negative >= tau) & (p_negative > p_positive)] = -1
    return signal


def strategy_signals(frame: pd.DataFrame, tau: float) -> dict[str, np.ndarray]:
    s1 = probability_signal(frame, tau)
    predicted = frame["predicted_spread"].to_numpy(float)
    agrees = ((s1 == 1) & (predicted > 0.0)) | ((s1 == -1) & (predicted < 0.0))
    s3 = np.where(agrees, s1, 0).astype(np.int8)
    weather = frame["fixed_extreme_weather_flag"].to_numpy(int) == 1
    s5 = np.where(weather, s3, 0).astype(np.int8)
    return {"S1": s1, "S3": s3, "S5": s5}


def period_calendar(period: str) -> pd.Index:
    if period == "2025 OOF":
        return pd.Index(pd.date_range("2025-01-01", "2025-12-31", freq="D").date)
    if period == "2026 H1 OOT":
        return pd.Index(pd.date_range("2026-01-01", "2026-06-30", freq="D").date)
    raise ValueError(period)


def daily_series(local_date: pd.Series, net: np.ndarray, period: str) -> pd.Series:
    values = pd.Series(net, index=pd.Index(local_date), dtype=float).groupby(level=0).sum()
    values.index = pd.to_datetime(values.index).date
    return values.reindex(period_calendar(period), fill_value=0.0).astype(float)


def sharpe(daily: pd.Series) -> float:
    standard_deviation = float(daily.std(ddof=1))
    if len(daily) < 2 or standard_deviation <= 0:
        return float("nan")
    return float(daily.mean() / standard_deviation * math.sqrt(365.0))


def max_drawdown(net: np.ndarray) -> float:
    equity = np.concatenate([[0.0], np.cumsum(net.astype(float))])
    drawdown = equity - np.maximum.accumulate(equity)
    return abs(float(drawdown.min()))


def settlement(
    spread: np.ndarray,
    signal: np.ndarray,
    gamma: float,
    spread_treatment: str = "unclipped",
    capture_treatment: str = "symmetric",
) -> dict[str, np.ndarray]:
    used_spread = spread.astype(float)
    if spread_treatment == "clipped[-1000,5000]":
        used_spread = np.clip(used_spread, *CLIP_BOUNDS)
    elif spread_treatment != "unclipped":
        raise ValueError(spread_treatment)
    active = signal != 0
    raw_gross = signal.astype(float) * used_spread
    if capture_treatment == "symmetric":
        captured_gross = gamma * raw_gross
    elif capture_treatment == "profits_only":
        captured_gross = np.where(raw_gross > 0.0, gamma * raw_gross, raw_gross)
    else:
        raise ValueError(capture_treatment)
    commission = COMMISSION * active.astype(float)
    slippage = SLIPPAGE * np.abs(used_spread) * active.astype(float)
    net = np.where(active, captured_gross - commission - slippage, 0.0)
    return {
        "spread_used": used_spread,
        "raw_gross": np.where(active, raw_gross, 0.0),
        "captured_gross": np.where(active, captured_gross, 0.0),
        "commission": commission,
        "slippage": slippage,
        "net": net,
    }


def metric_row(frame: pd.DataFrame, signal: np.ndarray, settled: dict[str, np.ndarray], period: str) -> dict[str, float | int]:
    active = signal != 0
    actual = frame["actual_spread"].to_numpy(float)
    correct = ((signal == 1) & (actual > 0.0)) | ((signal == -1) & (actual < 0.0))
    net = settled["net"]
    daily = daily_series(frame["local_date"], net, period)
    trades = int(active.sum())
    return {
        "eligible_hours": int(len(frame)),
        "trades": trades,
        "coverage": float(trades / len(frame)),
        "direction_precision": float(correct[active].mean()) if trades else float("nan"),
        "total_pnl_usd": float(net.sum()),
        "pnl_per_traded_mwh_usd": float(net.sum() / trades) if trades else float("nan"),
        "daily_sharpe": sharpe(daily),
        "max_drawdown_usd": max_drawdown(net),
    }


def load_predictions(path: Path, period: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["delivery_hour_utc"] = pd.to_datetime(frame["delivery_hour_utc"], utc=True)
    frame = frame.sort_values("delivery_hour_utc").reset_index(drop=True)
    frame["local_date"] = frame["delivery_hour_utc"].dt.tz_convert("US/Central").dt.date
    required = ["actual_spread", "predicted_spread", "p_negative", "p_positive", "fixed_extreme_weather_flag"]
    if frame["delivery_hour_utc"].duplicated().any():
        raise AssertionError(f"{period}: duplicate UTC hours")
    if frame[required].isna().any().any():
        raise AssertionError(f"{period}: missing required data")
    expected = 8760 if period == "2025 OOF" else 4197
    if len(frame) != expected:
        raise AssertionError(f"{period}: expected {expected} rows, got {len(frame)}")
    return frame


def select_source_rows(path: Path, timestamps: pd.DatetimeIndex, source: str, mapping: dict[str, str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(["delivery_hour_utc", *mapping.values()]))
    frame = pd.read_parquet(path, columns=columns) if path.suffix.lower() == ".parquet" else pd.read_csv(path, usecols=columns)
    frame["delivery_hour_utc"] = pd.to_datetime(frame["delivery_hour_utc"], utc=True)
    frame = frame[frame["delivery_hour_utc"].isin(timestamps)].copy()
    if len(frame) != len(timestamps):
        raise AssertionError(f"{source}: expected {len(timestamps)} extreme rows, got {len(frame)}")
    output = pd.DataFrame({"delivery_hour_utc": frame["delivery_hour_utc"], "source_layer": source})
    for standard_name, source_name in mapping.items():
        output[standard_name] = pd.to_numeric(frame[source_name], errors="coerce")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only PESIM2027 R1.2/R1.4/R1.5 audit")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--submitted-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    submitted_pdf = args.submitted_pdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_c = project / "02_prediction_agent_v2" / "phase_C_model_selection_validation"
    canonical = project / "04_trading_strategy_evaluation" / "section_iv_canonical_v1"
    files = {
        "submitted_pdf": submitted_pdf,
        "predictions_2025_oof": phase_c / "delivery_packages" / "trading_handoff_C1_v3" / "C1_unified_prediction_table_2025_oof_v3.parquet",
        "predictions_2026_h1": phase_c / "delivery_packages" / "trading_handoff_C1_2026_H1_v1" / "C1_unified_prediction_table_2026_H1_walkforward_v1.parquet",
        "prediction_manifest_2025": phase_c / "delivery_packages" / "trading_handoff_C1_v3" / "C1_unified_prediction_table_manifest_v3.json",
        "prediction_manifest_2026": phase_c / "delivery_packages" / "trading_handoff_C1_2026_H1_v1" / "C1_unified_prediction_table_2026_H1_manifest_v1.json",
        "canonical_hourly_2025": canonical / "outputs" / "hourly_strategy_results_2025.parquet",
        "canonical_hourly_2026": canonical / "outputs" / "hourly_strategy_results_2026.parquet",
        "canonical_code": canonical / "src" / "canonical_trading.py",
        "canonical_execution_manifest": canonical / "outputs" / "execution_manifest.json",
        "tau_provenance_code": phase_c / "scripts" / "run_c_systems_quality_econ_v3.py",
        "tau_provenance_report": phase_c / "reports" / "ECONOMIC_VALUE_GATE.md",
        "source_final_export": project / "02_prediction_agent_v2" / "model_wide_hourly_2024_2026_final(2).csv",
        "source_provenance_document": project / "02_prediction_agent_v2" / "model_wide_explanation_zh_2024_2026(1).md",
        "raw_architecture": project / "02_prediction_agent_v2" / "data" / "data_architecture" / "raw_input_leakage_safe_v2_partial.csv",
        "base_feature": project / "02_prediction_agent_v2" / "data" / "data_architecture" / "base_feature_table_leakage_safe_v2_partial.parquet",
        "frozen_model_input": project / "02_prediction_agent_v2" / "data" / "model_ready" / "model_input_frozen_v2.parquet",
        "existing_extreme_source_report": canonical / "reports" / "FIVE_EXTREME_SPREAD_SOURCE_SANITY_CHECK.md",
        "existing_extreme_source_rows": canonical / "reports" / "five_extreme_spread_source_sanity_check.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    inventory = {name: artifact(path) for name, path in files.items()}
    run_payload = {
        "audit_version": AUDIT_VERSION,
        "input_sha256": {name: item["sha256"] for name, item in inventory.items()},
        "parameters": {
            "tau_grid": TAU_GRID, "frozen_tau": FROZEN_TAU, "gamma_grid": GAMMA_GRID,
            "frozen_gamma": FROZEN_GAMMA, "commission": COMMISSION, "slippage": SLIPPAGE,
            "clip_bounds": CLIP_BOUNDS, "spread_definition": "RT_minus_DA",
        },
    }
    run_hash = hashlib.sha256(json.dumps(run_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    run_id = f"pesim2027_r1_2_r1_4_r1_5_{run_hash}"

    frame_2025 = load_predictions(files["predictions_2025_oof"], "2025 OOF")
    frame_2026 = load_predictions(files["predictions_2026_h1"], "2026 H1 OOT")

    tau_rows: list[dict[str, object]] = []
    for tau in TAU_GRID:
        for strategy_id, signal in strategy_signals(frame_2025, tau).items():
            settled = settlement(frame_2025["actual_spread"].to_numpy(float), signal, FROZEN_GAMMA)
            tau_rows.append({
                "run_id": run_id, "period": "2025 OOF", "tau": tau,
                "is_frozen_reference": tau == FROZEN_TAU, "strategy_id": strategy_id,
                "strategy_name": STRATEGY_NAMES[strategy_id],
                **metric_row(frame_2025, signal, settled, "2025 OOF"),
            })
    tau_output = pd.DataFrame(tau_rows)
    tau_output.to_csv(output_dir / "r1_2_tau_sensitivity_2025_oof.csv", index=False)

    gamma_rows: list[dict[str, object]] = []
    for period, frame in {"2025 OOF": frame_2025, "2026 H1 OOT": frame_2026}.items():
        for strategy_id, signal in strategy_signals(frame, FROZEN_TAU).items():
            for spread_treatment in ["unclipped", "clipped[-1000,5000]"]:
                for capture_treatment in ["symmetric", "profits_only"]:
                    for gamma in GAMMA_GRID:
                        settled = settlement(frame["actual_spread"].to_numpy(float), signal, gamma, spread_treatment, capture_treatment)
                        raw, captured = settled["raw_gross"], settled["captured_gross"]
                        gamma_rows.append({
                            "run_id": run_id, "period": period, "strategy_id": strategy_id,
                            "strategy_name": STRATEGY_NAMES[strategy_id], "spread_treatment": spread_treatment,
                            "capture_treatment": capture_treatment, "gamma": gamma,
                            "raw_positive_gross_usd": float(raw[raw > 0.0].sum()),
                            "raw_negative_gross_usd": float(raw[raw < 0.0].sum()),
                            "captured_positive_gross_usd": float(captured[raw > 0.0].sum()),
                            "captured_negative_gross_usd": float(captured[raw < 0.0].sum()),
                            "commission_usd": float(settled["commission"].sum()),
                            "slippage_usd": float(settled["slippage"].sum()),
                            **metric_row(frame, signal, settled, period),
                        })
    gamma_output = pd.DataFrame(gamma_rows)
    gamma_output.to_csv(output_dir / "r1_4_gamma_capture_clip_sensitivity.csv", index=False)

    s5_signal = strategy_signals(frame_2026, FROZEN_TAU)["S5"]
    s5_settled = settlement(frame_2026["actual_spread"].to_numpy(float), s5_signal, FROZEN_GAMMA)
    s5_daily = daily_series(frame_2026["local_date"], s5_settled["net"], "2026 H1 OOT")
    positive_days = s5_daily[s5_daily > 0.0].sort_values(ascending=False)
    positive_sum = float(positive_days.sum())
    concentration_rows = []
    for top_k in [1, 3, 5]:
        selected = positive_days.head(top_k)
        concentration_rows.append({
            "run_id": run_id, "top_k_positive_dates": top_k,
            "dates": ";".join(str(value) for value in selected.index),
            "top_k_positive_pnl_usd": float(selected.sum()), "all_positive_day_pnl_usd": positive_sum,
            "concentration_share_of_positive_day_pnl": float(selected.sum() / positive_sum),
            "share_of_net_total_pnl": float(selected.sum() / s5_daily.sum()),
        })
    concentration_output = pd.DataFrame(concentration_rows)
    concentration_output.to_csv(output_dir / "r1_5_weather_gate_concentration.csv", index=False)

    removal_rows = []
    for ranking in ["positive_pnl", "absolute_pnl"]:
        ranked = positive_days if ranking == "positive_pnl" else s5_daily.abs().sort_values(ascending=False)
        for top_k in [1, 3, 5]:
            dates = ranked.head(top_k).index
            remaining = s5_daily.drop(index=dates)
            removal_rows.append({
                "run_id": run_id, "ranking": ranking, "top_k_removed": top_k,
                "removed_dates": ";".join(str(value) for value in dates),
                "removed_signed_pnl_usd": float(s5_daily.loc[dates].sum()),
                "remaining_calendar_days": int(len(remaining)), "remaining_total_pnl_usd": float(remaining.sum()),
                "remaining_daily_sharpe": sharpe(remaining), "baseline_total_pnl_usd": float(s5_daily.sum()),
                "baseline_daily_sharpe": sharpe(s5_daily),
            })
    removal_output = pd.DataFrame(removal_rows)
    removal_output.to_csv(output_dir / "r1_5_weather_gate_top_date_removal.csv", index=False)

    daily_output = pd.DataFrame({"run_id": run_id, "local_date": [str(value) for value in s5_daily.index], "net_pnl_usd": s5_daily.to_numpy(float)})
    daily_output["positive_pnl_rank"] = daily_output["net_pnl_usd"].where(daily_output["net_pnl_usd"] > 0.0).rank(method="first", ascending=False)
    daily_output["absolute_pnl_rank"] = daily_output["net_pnl_usd"].abs().rank(method="first", ascending=False)
    daily_output.to_csv(output_dir / "r1_5_weather_gate_daily_pnl.csv", index=False)

    extreme_mask = frame_2026["actual_spread"].to_numpy(float) < -1000.0
    extreme_times = pd.DatetimeIndex(frame_2026.loc[extreme_mask, "delivery_hour_utc"])
    if len(extreme_times) != 5:
        raise AssertionError(f"Expected five sub--1000 spread hours, got {len(extreme_times)}")
    traces = [
        select_source_rows(files["source_final_export"], extreme_times, "highest_available_final_export", {
            "da_price_usd_per_mwh": "da_price_usd_per_mwh", "rt_price_usd_per_mwh": "rt_price_usd_per_mwh", "spread_usd_per_mwh": "spread_usd_per_mwh"}),
        select_source_rows(files["raw_architecture"], extreme_times, "v2_raw_architecture", {
            "da_price_usd_per_mwh": "da_price", "rt_price_usd_per_mwh": "rt_price", "spread_usd_per_mwh": "spread"}),
        select_source_rows(files["base_feature"], extreme_times, "v2_base_feature", {
            "da_price_usd_per_mwh": "da_price", "rt_price_usd_per_mwh": "rt_price", "spread_usd_per_mwh": "spread"}),
        select_source_rows(files["frozen_model_input"], extreme_times, "v2_frozen_model_input", {
            "da_price_usd_per_mwh": "da_price", "rt_price_usd_per_mwh": "rt_price", "spread_usd_per_mwh": "spread", "source_payload_present": "source_payload_present"}),
        select_source_rows(files["predictions_2026_h1"], extreme_times, "c1_2026_handoff", {"spread_usd_per_mwh": "actual_spread"}),
    ]
    trace_output = pd.concat(traces, ignore_index=True).sort_values(["delivery_hour_utc", "source_layer"])
    trace_output.insert(0, "run_id", run_id)
    trace_output["identity_error_usd_per_mwh"] = trace_output["rt_price_usd_per_mwh"] - trace_output["da_price_usd_per_mwh"] - trace_output["spread_usd_per_mwh"]
    trace_output.to_csv(output_dir / "r1_5_extreme_hour_source_trace.csv", index=False)

    pd.DataFrame([{"run_id": run_id, "artifact_name": name, **item} for name, item in inventory.items()]).to_csv(output_dir / "evidence_inventory_sha256.csv", index=False)
    script_path = Path(__file__).resolve()
    frozen_tau_s1 = tau_output[tau_output["tau"].eq(FROZEN_TAU) & tau_output["strategy_id"].eq("S1")].iloc[0]
    if not np.isclose(frozen_tau_s1["total_pnl_usd"], 3438.9964, atol=1e-9):
        raise AssertionError("2025 tau=0.60 S1 does not reproduce canonical PnL")
    expected = {
        ("2025 OOF", "S1"): 3438.9964, ("2025 OOF", "S3"): 2986.866225, ("2025 OOF", "S5"): 3617.17815,
        ("2026 H1 OOT", "S1"): 6991.486025, ("2026 H1 OOT", "S3"): 8045.90695, ("2026 H1 OOT", "S5"): 7818.6134375,
    }
    for (period, strategy_id), expected_pnl in expected.items():
        row = gamma_output[
            gamma_output["period"].eq(period) & gamma_output["strategy_id"].eq(strategy_id)
            & gamma_output["spread_treatment"].eq("unclipped") & gamma_output["capture_treatment"].eq("symmetric")
            & gamma_output["gamma"].eq(FROZEN_GAMMA)
        ].iloc[0]
        if not np.isclose(row["total_pnl_usd"], expected_pnl, atol=1e-9):
            raise AssertionError(f"Canonical mismatch for {(period, strategy_id)}")
    top5 = concentration_output[concentration_output["top_k_positive_dates"].eq(5)].iloc[0]
    if not np.isclose(top5["concentration_share_of_positive_day_pnl"], 0.9764019825371117, atol=1e-12):
        raise AssertionError("Weather Gate top-5 concentration mismatch")
    max_identity_error = float(trace_output["identity_error_usd_per_mwh"].abs().max(skipna=True))
    cross_layer_range = trace_output.groupby("delivery_hour_utc")["spread_usd_per_mwh"].max() - trace_output.groupby("delivery_hour_utc")["spread_usd_per_mwh"].min()
    if max_identity_error > 1e-9 or float(cross_layer_range.max()) > 1e-9:
        raise AssertionError("Extreme-hour source trace failed")
    command = f'"{sys.executable}" "{script_path}" --project-root "{project}" --submitted-pdf "{submitted_pdf}" --output-dir "{output_dir}"'
    manifest = {
        "run_id": run_id, "audit_version": AUDIT_VERSION, "status": "COMPLETE_READ_ONLY_AUDIT",
        "scope": ["Reviewer 1.2", "Reviewer 1.4", "Reviewer 1.5"], "no_model_training": True,
        "source_artifacts_modified": False, "parameters": run_payload["parameters"], "input_artifacts": inventory,
        "audit_code": {"path": str(script_path), "sha256": sha256(script_path), "version": AUDIT_VERSION},
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__},
        "reproduce_command": command,
        "provenance_boundary": {
            "highest_available_price_source": str(files["source_final_export"].resolve()),
            "documented_upstream_sqlite": "01_data_collection_cleaning/interim/ercot_analytics.sqlite",
            "upstream_sqlite_present": False, "raw_15_minute_ercot_records_present": False,
        },
        "acceptance_checks": {
            "tau_sensitivity_uses_2025_only": True, "tau_grid_rows": int(len(tau_output)),
            "gamma_sensitivity_rows": int(len(gamma_output)), "canonical_pnl_checks_passed": len(expected),
            "weather_gate_total_pnl_usd": float(s5_daily.sum()),
            "weather_gate_top5_positive_day_concentration": float(top5["concentration_share_of_positive_day_pnl"]),
            "extreme_hour_count": int(len(extreme_times)), "max_rt_minus_da_identity_error_usd_per_mwh": max_identity_error,
            "max_cross_layer_spread_range_usd_per_mwh": float(cross_layer_range.max()),
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
