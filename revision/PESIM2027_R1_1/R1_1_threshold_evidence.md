# Reviewer 1.1 — Five-class threshold evidence

**Run ID:** `pesim2027_phase0_r1_1_c093d1119f27e414`

**Audit conclusion:** the local record supports a practical distributional interpretation of the five classes, but it does **not** preserve a contemporaneous record proving how `±5` and `±20 USD/MWh` were originally selected.

## 1. Original evidence found

| Evidence | What it establishes | Limitation |
|---|---|---|
| `02_prediction_agent_v2/README.md` (2026-07-21; SHA256 starts `3b09e6f7`) | Earliest explicit five-class definition found used `±5` and **`±10`**, not `±20`. | Does not justify the later `±20` boundary. |
| `models/formal_m8_spike_classifier_v2.py` (2026-07-22 01:54 UTC; SHA256 starts `3e6ae4f4`) | First located implementation of the `±5/±20` five-class “spike” classifier. | Contains the rule, not a documented derivation or optimization record. |
| `outputs/metrics/M11_multiresolution_metrics_v2.csv` (2026-07-22 04:45 UTC) | A later comparison shows `5/20` exceeds `5/10` on 2025 OOF Macro-F1 (`0.3189` vs `0.2957`), balanced accuracy (`0.3424` vs `0.3146`), and log loss (`1.3806` vs `1.4943`). | This file postdates the first `5/20` implementation, so it cannot be presented as the original selection rationale. Its 2026 rows are explicitly labelled post-hoc exploratory. |
| Phase-C B2 config | Hyperparameter selection used mean 2025 Macro-F1 and records `2026_access: blocked_by_parquet_predicate`. | Supports no 2026 use in B2 model tuning, but does not prove the earlier class-boundary choice was made without seeing 2026. |
| 2026 delivery manifest | Explicitly states that 2026 had already been viewed and was **not** an untouched holdout. | Prevents a strong claim that every threshold-related decision was prospectively isolated from 2026. |

No threshold-selection memo, grid-search log, economic loss function, market-rule source, or immutable pre-2026 decision record was found for the transition from `±10` to `±20`.

## 2. Class counts under the submitted `±5/±20` rule

Classes are `C1: S < -20`, `C2: -20 ≤ S < -5`, `C3: -5 ≤ S ≤ 5`, `C4: 5 < S ≤ 20`, and `C5: S > 20`, with `S = RT - DA`.

| Period | C1 | C2 | C3 | C4 | C5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| 2024 development | 382 (4.7%) | 1,693 (20.8%) | 4,595 (56.5%) | 1,128 (13.9%) | 338 (4.2%) | 8,136 |
| 2025 OOF | 501 (5.7%) | 2,510 (28.7%) | 4,109 (46.9%) | 1,160 (13.2%) | 480 (5.5%) | 8,760 |
| 2026 H1 OOT | 224 (5.3%) | 1,004 (23.9%) | 2,126 (50.7%) | 647 (15.4%) | 196 (4.7%) | 4,197 |

Exact values are in [`results/r1_1_class_counts.csv`](results/r1_1_class_counts.csv).

## 3. Distributional interpretation that the evidence can support

| Period | 5th percentile | 25th percentile | Median | 75th percentile | 95th percentile |
|---|---:|---:|---:|---:|---:|
| 2024 development | -19.101 | -5.098 | -0.701 | 3.148 | 16.624 |
| 2025 OOF | -21.415 | -7.368 | -2.143 | 2.780 | 21.390 |
| 2026 H1 OOT | -20.828 | -6.110 | -1.433 | 3.478 | 18.851 |

The observed evidence supports this limited explanation:

- `±5` defines a near-zero central band containing 46.9%–56.5% of eligible hours. It separates small spreads from larger directional deviations.
- `±20` is close to the empirical 5th/95th-percentile tail region in all three periods. The two extreme classes together contain 8.8%–11.2% of observations, leaving non-trivial samples while emphasizing price spikes.
- Five ordered classes preserve both spread direction and magnitude: large negative, moderate negative, near zero, moderate positive, and large positive.

These are descriptive properties of the frozen data, not proof that the thresholds were originally optimized or selected from those quantiles.

## 4. What can and cannot be stated in the response

**Supported:** the five classes encode direction plus severity; `±5` forms a central near-zero band; `±20` approximately isolates distribution tails; every class retains hundreds of samples in 2024 and 2025; a later 2025-only diagnostic favored `5/20` over `5/10` on several classification metrics; Phase-C B2 hyperparameter tuning blocked 2026 access.

**Not established:** an original formal optimization of `±5/±20`; a market-design mandate for these exact values; prospective quantile calibration; an immutable record showing the `±20` boundary was selected before any inspection of 2026; or a valid claim that 2026 was an untouched holdout for every prior analytical decision.

Accordingly, the revision should describe the thresholds as pragmatic, interpretable fixed bins and disclose the absence of an original threshold-optimization record. It should not claim that the exact cutoffs were optimized or that their initial selection was demonstrably independent of all 2026 inspection.

## Claim–evidence audit

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R1.1-C1 | Reviewer 1.1 | The submitted five-class rule yields the reported 2024, 2025, and 2026 H1 counts. | numerical empirical | supported | `pesim2027_phase0_r1_1_c093d1119f27e414` | `results/r1_1_class_counts.csv` | — | pass | Recomputed from the frozen eligible population. |
| R1.1-C2 | Reviewer 1.1 | The ±20 boundaries lie near the empirical 5th and 95th percentile tail region. | numerical empirical | supported | `pesim2027_phase0_r1_1_c093d1119f27e414` | `results/r1_1_boundary_context.csv` | — | pass | Descriptive interpretation only. |
| R1.1-C3 | Reviewer 1.1 | A later 2025 OOF diagnostic favored 5/20 over 5/10 on Macro-F1, balanced accuracy, and log loss. | numerical empirical | supported | `pesim2027_phase0_r1_1_c093d1119f27e414` | `evidence_inventory_sha256.csv`; local `M11_multiresolution_metrics_v2.csv` | — | pass | Postdates first 5/20 implementation. |
| R1.1-C4 | Reviewer 1.1 | The exact ±5/±20 boundaries were originally selected by a documented optimization. | provenance | unsupported | — | — | — | stop | No original selection record found; earliest README used ±5/±10. |
| R1.1-C5 | Reviewer 1.1 | The B2 hyperparameter search used no 2026 observations. | reproducibility | supported | `B2_classifier_5_20_formal_v3r1_f5730506` | local B2 `config.json` | — | pass | Config records blocked 2026 access. |
| R1.1-C6 | Reviewer 1.1 | The class boundaries were selected before any 2026 result was viewed. | provenance | unsupported | — | — | — | stop | The delivery manifest says 2026 had already been viewed; no immutable boundary-selection record was found. |

Complete paths and SHA256 values are preserved in [`results/audit_manifest.json`](results/audit_manifest.json) and [`results/evidence_inventory_sha256.csv`](results/evidence_inventory_sha256.csv).
