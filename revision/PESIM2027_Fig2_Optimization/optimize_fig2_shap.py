from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont


IEEE_TWO_COLUMN_IN = 7.16
FAMILY_ORDER = [
    "Calendar & Historical Spread",
    "Natural Gas",
    "Load & Net Load",
    "Renewable Energy",
    "Raw Weather",
    "Engineered Extreme Weather",
    "Operational & Availability Controls",
]
FAMILY_ABBR = {
    "Calendar & Historical Spread": "CHS",
    "Natural Gas": "NG",
    "Load & Net Load": "LNL",
    "Renewable Energy": "RE",
    "Raw Weather": "RW",
    "Engineered Extreme Weather": "EEW",
    "Operational & Availability Controls": "OAC",
}
HEADS = ["B2A spread", "B2B C1 margin", "B2B C5 margin"]
DISPLAY_HEADS = ["Spread", "C1 logit", "C5 logit"]
HEAD_LOOKUP = dict(zip(DISPLAY_HEADS, HEADS))
SHAP_FILENAMES = {
    "Spread": "shap_b2a_spread_oof_2025.parquet",
    "C1 logit": "shap_b2b_c1_margin_oof_2025.parquet",
    "C5 logit": "shap_b2b_c5_margin_oof_2025.parquet",
}
DISPLAY_NAMES = {
    "load_southern_mw": "Southern load",
    "load_south_central_mw": "South Central load",
    "net_load_ramp_1h_mw": "1-h net-load ramp",
    "net_load_ramp_3h_mw": "3-h net-load ramp",
    "solar_ramp_1h_mw": "1-h solar ramp",
    "net_load_z30_same_hour": "Net-load anomaly\n(same hour)",
    "load_system_z30_same_hour": "System-load anomaly\n(same hour)",
    "hour_sin": "Hour-of-day sine",
    "spread_asof_roll_std24": "24-h spread SD",
    "spread_asof_roll_mad24": "24-h spread MAD",
    "is_evening_peak": "Evening-peak indicator",
    "cloud_cover_dfw_mean_pct": "DFW cloud cover",
    "solar_pvgrpp_system_mw": "System solar potential",
    "gas_price": "Henry Hub gas price",
    "spread_asof_lag168": "Spread lag (168 h)",
}
WRAPPED_NAMES = {
    **DISPLAY_NAMES,
    "load_south_central_mw": "South Central\nload",
    "hour_sin": "Hour-of-day\nsine",
    "spread_asof_roll_std24": "24-h spread\nSD",
    "spread_asof_roll_mad24": "24-h spread\nMAD",
    "is_evening_peak": "Evening-peak\nindicator",
    "solar_pvgrpp_system_mw": "System solar\npotential",
    "gas_price": "Henry Hub\ngas price",
    "spread_asof_lag168": "Spread lag\n(168 h)",
}
AGGRESSIVE_NAMES = {
    **WRAPPED_NAMES,
    "net_load_z30_same_hour": "Net-load\nanomaly\n(same hour)",
    "load_system_z30_same_hour": "System-load\nanomaly\n(same hour)",
    "net_load_ramp_1h_mw": "1-h net-load\nramp",
    "net_load_ramp_3h_mw": "3-h net-load\nramp",
    "cloud_cover_dfw_mean_pct": "DFW cloud\ncover",
}

LAYOUTS = {
    "A_balanced": {
        "height": 3.28,
        "title_size": 8.1,
        "tick_size": 7.0,
        "feature_size": 7.1,
        "label_size": 7.2,
        "colorbar_size": 6.7,
        "name_style": "moderate",
        "axes": {
            "a": [0.045, 0.18, 0.150, 0.67],
            "a_colorbar": [0.204, 0.18, 0.010, 0.67],
            "b": [0.385, 0.18, 0.100, 0.67],
            "c": [0.620, 0.18, 0.090, 0.67],
            "d": [0.820, 0.18, 0.095, 0.67],
            "d_colorbar": [0.925, 0.18, 0.009, 0.67],
        },
    },
    "B_readability": {
        "height": 3.68,
        "title_size": 8.5,
        "tick_size": 7.4,
        "feature_size": 7.5,
        "label_size": 7.6,
        "colorbar_size": 7.0,
        "name_style": "aggressive",
        "axes": {
            "a": [0.045, 0.16, 0.150, 0.72],
            "a_colorbar": [0.204, 0.16, 0.010, 0.72],
            "b": [0.385, 0.16, 0.100, 0.72],
            "c": [0.620, 0.16, 0.090, 0.72],
            "d": [0.820, 0.16, 0.095, 0.72],
            "d_colorbar": [0.925, 0.16, 0.009, 0.72],
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def robust_feature_color(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, 0.5, dtype=float)
    if finite.sum() == 0:
        return result
    low, high = np.nanquantile(values[finite], [0.05, 0.95])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return result
    result[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
    return result


def deterministic_beeswarm_jitter(values: np.ndarray, width: float = 0.27, bins: int = 80) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(len(values), dtype=float)
    bin_id = np.clip(((values - low) / (high - low) * bins).astype(int), 0, bins - 1)
    jitter = np.zeros(len(values), dtype=float)
    unique_bins, bin_counts = np.unique(bin_id, return_counts=True)
    max_count = int(bin_counts.max())
    for current_bin, bin_count in zip(unique_bins, bin_counts):
        indices = np.flatnonzero(bin_id == current_bin)
        count = len(indices)
        if count <= 1:
            continue
        offsets = np.arange(count, dtype=float) - (count - 1) / 2
        max_abs = np.max(np.abs(offsets))
        density_width = width * np.sqrt(bin_count / max_count)
        jitter[indices] = offsets / max_abs * density_width if max_abs else 0.0
    return jitter


def boxed_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def clean_feature_name(name: str) -> str:
    return name.replace("_mw", "").replace("_pct", " (%)").replace("_", " ")


def load_inputs(package: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    data = package / "data"
    per_feature = pd.read_csv(data / "per_feature_mean_abs_shap.csv")
    family = pd.read_csv(data / "family_size_normalized_importance.csv")
    feature_values = pd.read_parquet(data / "shap_feature_values_oof_2025.parquet")
    shap_paths = {head: data / filename for head, filename in SHAP_FILENAMES.items()}
    return per_feature, family, feature_values, shap_paths


def draw_variant(
    output_dir: Path,
    layout_name: str,
    cfg: dict,
    per_feature: pd.DataFrame,
    family: pd.DataFrame,
    feature_values: pd.DataFrame,
    shap_paths: dict[str, Path],
) -> list[dict]:
    configure_style()
    family_matrix = family.pivot(
        index="paper_feature_family",
        columns="output_head",
        values="family_size_normalized_mean_abs_shap",
    ).reindex(index=FAMILY_ORDER, columns=HEADS)

    cmap = LinearSegmentedColormap.from_list("feature_value", ["#168DE2", "#7B4CC2", "#F00067"])
    cmap_family = LinearSegmentedColormap.from_list(
        "shap_blue", ["#FFFFFF", "#BBDDF0", "#56B4E9", "#0072B2", "#002B5B"]
    )
    fig = plt.figure(figsize=(IEEE_TWO_COLUMN_IN, cfg["height"]))
    ax_a = fig.add_axes(cfg["axes"]["a"])
    family_cax = fig.add_axes(cfg["axes"]["a_colorbar"])

    image = ax_a.imshow(
        family_matrix.to_numpy(float),
        cmap=cmap_family,
        vmin=0,
        vmax=float(family_matrix.to_numpy(float).max()),
        aspect="auto",
    )
    ax_a.set_yticks(np.arange(len(FAMILY_ORDER)), [FAMILY_ABBR[name] for name in FAMILY_ORDER])
    ax_a.tick_params(axis="y", labelsize=cfg["feature_size"], pad=1.5)
    ax_a.set_xticks(np.arange(3), ["Spread", "C1 logit", "C5 logit"], rotation=31, ha="right")
    ax_a.tick_params(axis="x", labelsize=cfg["tick_size"], pad=1.5)
    ax_a.set_title(
        "(a) Family-normalized mean |SHAP|",
        loc="left",
        fontfamily="serif",
        fontweight="normal",
        fontsize=cfg["title_size"],
        pad=5,
    )
    cbar = fig.colorbar(image, cax=family_cax, orientation="vertical")
    cbar.set_label("Mean |SHAP|\nper feature", fontsize=cfg["colorbar_size"], labelpad=1.0)
    cbar.ax.tick_params(labelsize=cfg["colorbar_size"], pad=1.0)
    boxed_axis(ax_a)

    ranking_records: list[dict] = []
    feature_cax = None
    display_map = AGGRESSIVE_NAMES if cfg["name_style"] == "aggressive" else WRAPPED_NAMES
    for panel_index, display_head in enumerate(DISPLAY_HEADS, start=1):
        panel_letter = {1: "b", 2: "c", 3: "d"}[panel_index]
        ax = fig.add_axes(cfg["axes"][panel_letter])
        if panel_index == 3:
            feature_cax = fig.add_axes(cfg["axes"]["d_colorbar"])

        output_head = HEAD_LOOKUP[display_head]
        top = (
            per_feature[per_feature["output_head"].eq(output_head)]
            .nlargest(6, "mean_abs_shap")
            .reset_index(drop=True)
        )
        top_features = top["feature_name"].tolist()
        shap_table = pd.read_parquet(shap_paths[display_head], columns=top_features)
        all_abs = np.abs(shap_table[top_features].to_numpy(float)).ravel()
        x_limit = float(np.nanquantile(all_abs, 0.997))
        if x_limit <= 0:
            x_limit = float(np.nanmax(all_abs)) or 1.0

        for row, feature in enumerate(top_features):
            shap_values = shap_table[feature].to_numpy(float)
            raw_values = feature_values[feature].to_numpy(float)
            clipped_shap = np.clip(shap_values, -x_limit, x_limit)
            ax.scatter(
                clipped_shap,
                np.full(len(shap_values), row) + deterministic_beeswarm_jitter(clipped_shap),
                c=robust_feature_color(raw_values),
                cmap=cmap,
                vmin=0,
                vmax=1,
                s=2.1,
                alpha=0.42,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            ranking_records.append(
                {
                    "output_head": output_head,
                    "rank": row + 1,
                    "feature_name": feature,
                    "mean_abs_shap": float(top.loc[row, "mean_abs_shap"]),
                    "display_clip_abs_shap_q99_7": x_limit,
                    "sample_count": len(shap_values),
                }
            )

        ax.axvline(0, color="#8A8A8A", lw=0.8, zorder=1)
        ax.set_yticks(
            np.arange(len(top_features)),
            [display_map.get(feature, clean_feature_name(feature)) for feature in top_features],
        )
        ax.tick_params(axis="y", labelsize=cfg["feature_size"], pad=1.5)
        ax.tick_params(axis="x", labelsize=cfg["tick_size"], pad=1.5)
        for label in ax.get_yticklabels():
            label.set_linespacing(0.92)
        ax.invert_yaxis()
        ax.set_xlim(-x_limit * 1.04, x_limit * 1.04)
        ax.set_xlabel("SHAP value", fontsize=cfg["label_size"], labelpad=2.0)
        ax.set_title(
            f"({panel_letter}) {display_head}",
            loc="left",
            fontfamily="serif",
            fontweight="normal",
            fontsize=cfg["title_size"],
            pad=5,
        )
        ax.grid(axis="y", color="#D9D9D9", lw=0.4, ls=":", alpha=0.75)
        boxed_axis(ax)

    if feature_cax is None:
        raise AssertionError("Feature-value colorbar axis was not created.")
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=feature_cax)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Feature value", fontsize=cfg["colorbar_size"], labelpad=2.0)
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(labelsize=cfg["colorbar_size"], pad=1.0)
    stem = output_dir / f"Fig2_SHAP_{layout_name}"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return ranking_records


def validate_ranking(records: list[dict], baseline_csv: Path) -> None:
    actual = pd.DataFrame(records)
    expected = pd.read_csv(baseline_csv)
    key = ["output_head", "rank", "feature_name", "sample_count"]
    if not actual[key].equals(expected[key]):
        raise AssertionError("Feature ranking or sample counts differ from the submitted figure source.")
    if not np.allclose(
        actual["mean_abs_shap"], expected["mean_abs_shap"], rtol=0, atol=1e-10
    ):
        raise AssertionError("Mean absolute SHAP values differ from the submitted figure source.")
    if not np.allclose(
        actual["display_clip_abs_shap_q99_7"],
        expected["display_clip_abs_shap_q99_7"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Display clipping thresholds differ from the submitted figure source.")


def make_comparison_preview(output_dir: Path, source_png: Path) -> Path:
    entries = [
        ("Submitted Fig. 2 (5-pt labels)", source_png),
        ("Version A — balanced", output_dir / "Fig2_SHAP_A_balanced.png"),
        ("Version B — readability (recommended)", output_dir / "Fig2_SHAP_B_readability.png"),
    ]
    target_width = 1800
    label_height = 72
    gap = 28
    font = ImageFont.truetype("arial.ttf", 32)
    rendered = []
    for label, path in entries:
        image = Image.open(path).convert("RGB")
        new_height = round(image.height * target_width / image.width)
        image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        rendered.append((label, image))
    canvas_height = sum(label_height + image.height for _, image in rendered) + gap * (len(rendered) - 1)
    canvas = Image.new("RGB", (target_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for index, (label, image) in enumerate(rendered):
        draw.text((12, y + 15), label, fill="black", font=font)
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
        if index < len(rendered) - 1:
            y += gap
    output = output_dir / "Fig2_comparison_preview.png"
    canvas.save(output, dpi=(200, 200))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Fig. 2 SHAP layout optimization.")
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(r"D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project")
        / "outputs"
        / "manuscript_figures"
        / "oof_shap_v2",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    package = args.package.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_feature, family, feature_values, shap_paths = load_inputs(package)
    baseline_csv = package / "data" / "Fig4_beeswarm_top_feature_ranking.csv"
    for layout_name, cfg in LAYOUTS.items():
        records = draw_variant(
            output_dir, layout_name, cfg, per_feature, family, feature_values, shap_paths
        )
        validate_ranking(records, baseline_csv)

    source_png = package / "figures" / "Fig4_SHAP_StrictOOF_Beeswarm_v3.png"
    copied_source = output_dir / "Fig2_submitted_source.png"
    shutil.copy2(source_png, copied_source)
    preview = make_comparison_preview(output_dir, copied_source)

    source_paths = [
        package / "figures" / "Fig4_SHAP_StrictOOF_Beeswarm_v3.tiff",
        source_png,
        package / "code" / "build_strict_oof_shap_fig4.py",
        package / "data" / "per_feature_mean_abs_shap.csv",
        package / "data" / "family_size_normalized_importance.csv",
        package / "data" / "shap_feature_values_oof_2025.parquet",
        *shap_paths.values(),
        baseline_csv,
        package / "data" / "oof_shap_manifest.json",
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    optimization_script_sha256 = sha256(Path(__file__).resolve())
    stable_payload = json.dumps(
        {
            "source_sha256": source_hashes,
            "layouts": LAYOUTS,
            "optimization_script_sha256": optimization_script_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_id = "FIG2OPT-20261108-" + hashlib.sha256(stable_payload).hexdigest()[:12].upper()
    output_paths = sorted(
        [*output_dir.glob("Fig2_SHAP_*.*"), copied_source, preview], key=lambda path: path.name
    )
    manifest = {
        "run_id": run_id,
        "operation": "layout-only optimization of submitted Fig. 2",
        "source_package": str(package),
        "source_sha256": source_hashes,
        "optimization_script_sha256": optimization_script_sha256,
        "layouts": LAYOUTS,
        "invariants_verified": {
            "sample_count_per_output": 8760,
            "feature_ranking_matches_source_csv": True,
            "mean_abs_shap_matches_source_csv": True,
            "display_clip_q99_7_matches_source_csv": True,
            "color_mapping_unchanged": True,
            "model_training_performed": False,
            "source_data_modified": False,
        },
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "reproduction_command": "python revision/PESIM2027_Fig2_Optimization/optimize_fig2_shap.py",
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "preview": str(preview)}, indent=2))


if __name__ == "__main__":
    main()
