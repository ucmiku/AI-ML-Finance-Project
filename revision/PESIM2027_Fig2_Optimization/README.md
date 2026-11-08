# PES IM 2027 Fig. 2 readability optimization

## Outcome

The exact source package for the submitted Fig. 2 was found locally. The latest Word file embeds `word/media/image2.tiff`; its SHA256 is identical to the local 600-dpi source TIFF:

`A2E10D44240946D789867DC7342457FB85B2126ECD2AF0B0D8A08F617075D2FC`

Therefore, this optimization uses the final local manuscript figure rather than the older SHAP assets on GitHub `main`.

## Source evidence

- Submitted manuscript: `D:\NUS SOC2026\会议论文\ERCOT_PESIM2027_V7.3_Reviewer3_Content.docx`
- Submitted PDF: `D:\NUS SOC2026\会议论文\PESIM2027_ERCOT_ExtremeWeather_VirtualBidding.pdf`
- Exact source figure: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\outputs\manuscript_figures\oof_shap_v2\figures\Fig4_SHAP_StrictOOF_Beeswarm_v3.tiff`
- Original plotting code: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\outputs\manuscript_figures\oof_shap_v2\code\build_strict_oof_shap_fig4.py`, especially lines 631–765.
- IEEE export mapping: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\outputs\manuscript_figures\code\export_v17_ieee_figures.py`, lines 60–66.
- Frozen SHAP data: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\outputs\manuscript_figures\oof_shap_v2\data\`.

The original figure uses 2025 chronological OOF SHAP values for 8,760 hours and the frozen model runs `B2_regression_formal_v3r1_f5730506` and `B2_classifier_5_20_formal_v3r1_f5730506`. No model was retrained.

## Delivered versions

| Version | Intended use | Final size | Text treatment |
|---|---|---:|---|
| A — balanced | More compact replacement | 7.16 × 3.28 in | 7.0–8.1 pt; moderate line wrapping |
| B — readability | Recommended reviewer-response version | 7.16 × 3.68 in | 7.0–8.5 pt; additional semantic-preserving line wrapping |

Each version is exported as 600-dpi PNG, LZW-compressed 600-dpi TIFF, and SVG with live text. The comparison preview places the submitted figure and both alternatives at the same display width.

## Evidence-preservation checks

| Check | Result |
|---|---|
| A/B/C/D remain in one horizontal row | Pass |
| Family heatmap values and family order | Unchanged |
| Top-six feature order in Spread, C1, and C5 | Exact match to source ranking CSV |
| Mean absolute SHAP values | Exact match within `1e-10` |
| 99.7% display clipping thresholds | Exact match within `1e-12` |
| Feature-value color normalization and blue–purple–magenta meaning | Unchanged |
| Per-output sample count | 8,760 |
| Model training or source-data modification | None |

Long labels are only split across lines; wording and meaning are unchanged. Full source and output hashes are recorded in `figure_manifest.json`.

## Recommendation

Use **Version B (`Fig2_SHAP_B_readability`)**. At the intended 7.16-inch IEEE double-column width it provides the clearest feature labels, axes, panel titles, and color scales. Version A is acceptable if vertical space becomes the overriding constraint.

## Reproduction

Run from the repository root:

```powershell
python revision/PESIM2027_Fig2_Optimization/optimize_fig2_shap.py
```

Stable run ID: `FIG2OPT-20261108-95BE40EA1E65`.

## Files

- `Fig2_SHAP_A_balanced.{png,tiff,svg}`
- `Fig2_SHAP_B_readability.{png,tiff,svg}`
- `Fig2_comparison_preview.png`
- `Fig2_submitted_source.png`
- `optimize_fig2_shap.py`
- `figure_manifest.json`
