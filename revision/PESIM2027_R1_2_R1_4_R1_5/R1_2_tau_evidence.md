# Reviewer 1.2 — Evidence for τ = 0.60

**Run ID:** `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b`

## Selection record

The contemporaneous Phase-C code and `ECONOMIC_VALUE_GATE.md` describe `τ=0.60` as a **fixed common probability rule**, explicitly state that no threshold optimization was performed, and state that 2026 was not accessed in that economic gate. No earlier optimization log or formal derivation for `0.60` was found. Therefore, `0.60` should not be described as the optimum selected by a threshold search.

## 2025 OOF sensitivity

The frozen 8,760-row 2025 OOF prediction table was evaluated over `τ=0.50–0.80` in increments of `0.025`. No 2026 observation entered this calculation. The submitted economic convention was retained: `γ=0.65`, `$2/MWh` commission, `0.5%` slippage, one MWh, and observed unclipped spread.

| τ | Trades | Coverage | Direction precision | Total PnL (USD) | PnL/MWh (USD) | Daily Sharpe |
|---:|---:|---:|---:|---:|---:|---:|
| 0.550 | 1,789 | 20.4% | 0.630 | 700.203 | 0.391 | 0.224 |
| 0.600 | 1,174 | 13.4% | 0.665 | **3,438.996** | 2.929 | **1.290** |
| 0.650 | 762 | 8.7% | 0.696 | 3,457.082 | 4.537 | 1.381 |
| 0.675 | 603 | 6.9% | 0.715 | **4,235.803** | 7.025 | 1.850 |
| 0.700 | 474 | 5.4% | 0.728 | 4,129.864 | 8.713 | 1.867 |
| 0.725 | 349 | 4.0% | 0.759 | 4,173.346 | 11.958 | **1.949** |
| 0.750 | 260 | 3.0% | 0.781 | 3,919.469 | 15.075 | 1.868 |

The full grid, including Direction Confirmation and Weather Gate, is in [`results/r1_2_tau_sensitivity_2025_oof.csv`](results/r1_2_tau_sensitivity_2025_oof.csv).

## Conclusion

Increasing τ produces the expected precision–coverage trade-off. `τ=0.60` gives materially greater coverage than stricter thresholds and positive 2025 OOF economics, but it is **not** the ex-post maximizer of either total PnL (`τ=0.675`) or Sharpe (`τ=0.725`) on this grid. The defensible response is that `0.60` was a pre-fixed common operating point and that the new 2025-only sensitivity shows the result is positive across a neighborhood of stricter thresholds. The analysis must not use the sensitivity table to reselect τ, and no 2026 result should be cited as threshold-selection evidence.

## Claim–evidence audit

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R1.2-C1 | Reviewer 1.2 | The τ sensitivity calculation used only the 8,760-row 2025 OOF prediction table. | reproducibility | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/audit_manifest.json` | — | pass | No 2026 rows were used in the sweep. |
| R1.2-C2 | Reviewer 1.2 | At τ=0.60, S1 has 1,174 trades, USD 3,438.996 PnL, and Sharpe 1.290. | numerical empirical | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_2_tau_sensitivity_2025_oof.csv` | — | pass | Reproduces the canonical 2025 result. |
| R1.2-C3 | Reviewer 1.2 | τ=0.60 was the threshold-search optimum. | provenance | unsupported | — | — | — | stop | Contemporaneous records say no threshold optimization was performed. |
