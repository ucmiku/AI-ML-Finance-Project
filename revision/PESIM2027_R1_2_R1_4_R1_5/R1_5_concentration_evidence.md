# Reviewer 1.5 — Weather Gate concentration and extreme-hour provenance

**Run ID:** `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b`

## Exact concentration definition

Weather Gate has 2026 H1 total PnL of `USD 7,818.613` and daily Sharpe `1.978`. Its positive delivery-date PnL sums to `USD 8,192.303`.

| Positive-date ranking | Dates | Sum (USD) | Share of positive-date PnL | Share of total net PnL |
|---|---|---:|---:|---:|
| Top 1 | 2026-01-26 | 4,726.000 | 57.688% | 60.446% |
| Top 3 | 2026-01-26, 2026-01-25, 2026-01-27 | 7,946.808 | 97.003% | 101.640% |
| Top 5 | plus 2026-04-03 and 2026-03-21 | 7,998.981 | **97.640198%** | 102.307% |

Therefore, the submitted `97.6%` is **Top-5 positive-day PnL divided by all positive-day PnL**. It is not Top-5 PnL divided by total net PnL; that ratio exceeds 100% because the remaining dates have negative net PnL.

## Date-removal experiment

| Removed dates | Ranking rule | Remaining PnL (USD) | Remaining Sharpe |
|---|---|---:|---:|
| Top 1 | Positive PnL | 3,092.613 | 1.442 |
| Top 3 | Positive PnL | -128.195 | -1.217 |
| Top 5 | Positive PnL | -180.368 | -1.783 |
| Top 5 | Largest absolute PnL | **27.288** | **0.383258** |

The submitted de-extremized Sharpe `0.383` uses the five largest **absolute-PnL** delivery dates: `2026-01-26`, `2026-01-25`, `2026-01-27`, `2026-06-23`, and `2026-01-31`. This is a different date set from the five largest positive-PnL dates. The distinction must be preserved.

## 2026-01-26 extreme-hour source trace

The highest available finalized price export records HB_NORTH DA prices, hourly RT prices constructed as the mean of four complete 15-minute RT SPP intervals, and `Spread = RT − DA`:

| Delivery hour UTC | Local time | DA | RT hourly mean | RT−DA spread (USD/MWh) |
|---|---|---:|---:|---:|
| 2026-01-26 05:00 | 2026-01-25 23:00 CST | 1,500.00 | 461.9950 | -1,038.0050 |
| 2026-01-26 11:00 | 2026-01-26 05:00 CST | 1,899.19 | 334.8475 | -1,564.3425 |
| 2026-01-26 12:00 | 2026-01-26 06:00 CST | 1,656.00 | 275.2350 | -1,380.7650 |
| 2026-01-26 13:00 | 2026-01-26 07:00 CST | 1,773.12 | 180.9425 | -1,592.1775 |
| 2026-01-26 14:00 | 2026-01-26 08:00 CST | 1,556.00 | 136.8150 | -1,419.1850 |

The values agree across the finalized export, v2 raw architecture, base-feature table, frozen model input, and 2026 prediction handoff. The maximum `RT − DA − spread` error is `2.27×10⁻¹³ USD/MWh`, and the cross-layer spread range is zero.

**Provenance limit:** the project documentation points to `01_data_collection_cleaning/interim/ercot_analytics.sqlite`, but that database and its underlying 15-minute ERCOT rows are absent locally. The audit verifies exact consistency from the highest available finalized export downstream; it is not an independent re-download of the raw ERCOT intervals.

Machine-readable evidence is in [`results/r1_5_weather_gate_concentration.csv`](results/r1_5_weather_gate_concentration.csv), [`results/r1_5_weather_gate_top_date_removal.csv`](results/r1_5_weather_gate_top_date_removal.csv), and [`results/r1_5_extreme_hour_source_trace.csv`](results/r1_5_extreme_hour_source_trace.csv).

## Claim–evidence audit

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R1.5-C1 | Reviewer 1.5 | The five largest positive-PnL dates account for 97.640198% of all positive-date PnL. | numerical empirical | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_5_weather_gate_concentration.csv` | — | pass | Denominator is positive-date PnL, not net total PnL. |
| R1.5-C2 | Reviewer 1.5 | Removing the five largest absolute-PnL dates leaves Sharpe 0.383258. | numerical empirical | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_5_weather_gate_top_date_removal.csv` | — | pass | Remaining PnL is USD 27.288. |
| R1.5-C3 | Reviewer 1.5 | The five extreme spreads agree exactly across all locally available downstream data layers. | reproducibility | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_5_extreme_hour_source_trace.csv` | — | pass | Maximum cross-layer spread range is zero. |
| R1.5-C4 | Reviewer 1.5 | The original four 15-minute ERCOT rows were independently reverified. | provenance | unsupported | — | — | — | stop | Upstream SQLite and raw intervals are not present locally. |
