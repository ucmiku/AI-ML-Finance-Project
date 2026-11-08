# Phase 0 — Economic evaluation baseline

**Run ID:** `pesim2027_phase0_r1_1_c093d1119f27e414`

**Status:** read-only audit complete; no model retraining and no source artifact modification.

**Submitted-PDF SHA256:** `47dfc0b8bb67fa840b7da2b93a855838789f70f7ae9527e2a774753dcf10a6f4`

## Locked benchmark

The submitted-PDF economic convention is the canonical benchmark: `Spread = RT - DA`, capture rate `γ = 0.65`, commission `$2/MWh`, slippage `0.5%`, and settlement on the observed **unclipped** spread.

The local 2026 canonical hourly file contains exactly **33,576 rows = 4,197 eligible hours × 8 strategies**. The three strategies requested for the audit reconcile as follows:

| Period | Strategy ID | Strategy | Trades | Total PnL (USD) | PnL/traded MWh (USD) |
|---|---:|---|---:|---:|---:|
| 2025 OOF | S1 | Dual-XGB-060 | 1,174 | 3,438.996 | 2.929 |
| 2025 OOF | S3 | Direction Confirmation | 933 | 2,986.866 | 3.201 |
| 2025 OOF | S5 | Extreme-Weather-Only / Weather Gate | 425 | 3,617.178 | 8.511 |
| 2026 H1 OOT | S1 | Dual-XGB-060 | 441 | **6,991.486** | 15.854 |
| 2026 H1 OOT | S3 | Direction Confirmation | 353 | 8,045.907 | 22.793 |
| 2026 H1 OOT | S5 | Extreme-Weather-Only / Weather Gate | 158 | **7,818.613** | 49.485 |

## Reconciliation of USD 5,705.050 versus USD 6,991.486

The old prediction package and canonical S1 file have the same 4,197 timestamps, the same 441 non-zero signals, and identical realized spreads. The difference is only the settlement convention:

| Package | Realized-spread treatment | Total S1 PnL (USD) |
|---|---|---:|
| Old prediction package | Clip to `[-1000, 5000]` before settlement | 5,705.050 |
| Submitted/canonical calculation | No clipping | 6,991.486 |
| Difference | Five negative-spread hours on 2026-01-26 UTC | **1,286.436** |

All five affected hours had S1 signal `-1` and realized spreads below `-1000 USD/MWh`. The row-level reconciliation is in [`results/phase0_clipping_difference_rows.csv`](results/phase0_clipping_difference_rows.csv). Therefore, USD 5,705.050 is not a different signal set or strategy; it is the earlier clipped-settlement version of the same S1 predictions.

## Reproducibility record

Complete absolute input paths, SHA256 values, parameters, software versions, acceptance checks, audit-code hash, and the reproducible command are preserved in [`results/audit_manifest.json`](results/audit_manifest.json) and [`results/evidence_inventory_sha256.csv`](results/evidence_inventory_sha256.csv). Machine-readable summaries are in [`results/phase0_strategy_summary.csv`](results/phase0_strategy_summary.csv) and [`results/phase0_pnl_reconciliation.csv`](results/phase0_pnl_reconciliation.csv).

Reproduce without training or overwriting source data:

```powershell
& 'D:\py\Python3\python.exe' 'D:\NUS SOC2026\AI-ML-Finance-Project\revision\PESIM2027_R1_1\audit_phase0_r1_1.py' --project-root 'D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project' --submitted-pdf 'D:\NUS SOC2026\会议论文\PESIM2027_ERCOT_ExtremeWeather_VirtualBidding.pdf' --output-dir 'D:\NUS SOC2026\AI-ML-Finance-Project\revision\PESIM2027_R1_1\results'
```
