# Reviewer 1.4 — γ = 0.65 execution sensitivity

**Run ID:** `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b`

## Submitted convention

The submitted calculation applies capture rate symmetrically:

`net PnL = γ × signal × observed spread − 2 − 0.005 × |observed spread|`

Thus, `γ=0.65` multiplies both favorable and unfavorable gross outcomes; commission and slippage are not multiplied by γ. The frozen `γ=0.65`, symmetric, unclipped calculation exactly reproduces all six canonical S1/S3/S5 totals for 2025 OOF and 2026 H1 OOT.

## γ sensitivity under symmetric capture and unclipped spread

| 2026 H1 strategy | γ=0.50 PnL | γ=0.65 PnL | γ=0.80 PnL | γ=1.00 PnL | Sharpe at γ=0.65 |
|---|---:|---:|---:|---:|---:|
| S1 Dual-XGB | 5,147.436 | **6,991.486** | 8,835.536 | 11,294.269 | 1.687 |
| S3 Direction Confirmation | 6,005.839 | **8,045.907** | 10,085.974 | 12,806.064 | 2.029 |
| S5 Weather Gate | 5,925.685 | **7,818.613** | 9,711.542 | 12,235.446 | 1.978 |

All three remain positive across this γ grid, but this does not resolve how capture should treat adverse execution outcomes.

## Effect of scaling losses

As a diagnostic alternative, `profits_only` applies γ to positive gross outcomes while leaving negative gross outcomes unscaled. This is not the submitted convention.

| Period/strategy | Symmetric γ=0.65 PnL | Profits-only γ=0.65 PnL | Difference caused by scaling losses |
|---|---:|---:|---:|
| 2025 OOF S1 | 3,438.996 | -598.600 | **4,037.597** |
| 2025 OOF S3 | 2,986.866 | -584.982 | **3,571.848** |
| 2025 OOF S5 | 3,617.178 | 2,383.739 | 1,233.439 |
| 2026 H1 S1 | 6,991.486 | 5,034.010 | 1,957.477 |
| 2026 H1 S3 | 8,045.907 | 7,332.491 | 713.416 |
| 2026 H1 S5 | 7,818.613 | 7,644.363 | 174.250 |

The 2025 S1/S3 profitability conclusion is therefore sensitive to the symmetric-loss interpretation. This must be stated as an execution-model assumption, not as a generic market capture fact.

## Clipped versus observed spread at γ=0.65

| Period/strategy | Unclipped PnL | Clipped PnL | Difference | Unclipped Sharpe | Clipped Sharpe |
|---|---:|---:|---:|---:|---:|
| 2025 OOF S1 | 3,438.996 | 3,438.996 | 0.000 | 1.290 | 1.290 |
| 2025 OOF S3 | 2,986.866 | 2,986.866 | 0.000 | 1.170 | 1.170 |
| 2025 OOF S5 | 3,617.178 | 3,617.178 | 0.000 | 2.151 | 2.151 |
| 2026 H1 S1 | 6,991.486 | 5,705.050 | 1,286.436 | 1.687 | 1.649 |
| 2026 H1 S3 | 8,045.907 | 6,759.471 | 1,286.436 | 2.029 | 2.081 |
| 2026 H1 S5 | 7,818.613 | 6,532.177 | 1,286.436 | 1.978 | 2.021 |

Clipping affects only five 2026 hours in these strategies. It lowers total PnL but can raise Sharpe by reducing the variance contribution of the extreme profitable episode. Full decompositions are in [`results/r1_4_gamma_capture_clip_sensitivity.csv`](results/r1_4_gamma_capture_clip_sensitivity.csv).

## Claim–evidence audit

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R1.4-C1 | Reviewer 1.4 | The submitted γ=0.65 convention symmetrically scales favorable and unfavorable gross spreads. | project implementation | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/audit_manifest.json`; local canonical code | — | pass | Costs remain unscaled. |
| R1.4-C2 | Reviewer 1.4 | The frozen symmetric-unclipped calculation reproduces the six canonical S1/S3/S5 totals. | numerical empirical | supported | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_4_gamma_capture_clip_sensitivity.csv` | — | pass | Acceptance checks passed. |
| R1.4-C3 | Reviewer 1.4 | The economic conclusions are invariant to whether γ scales losses. | comparative | conflict | `pesim2027_r1_2_r1_4_r1_5_3b05d7185029717b` | `results/r1_4_gamma_capture_clip_sensitivity.csv` | — | stop | 2025 S1/S3 become negative under profits-only capture. |
