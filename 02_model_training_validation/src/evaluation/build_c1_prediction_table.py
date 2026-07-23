# -*- coding: utf-8 -*-
"""Build the C1 unified prediction table from B2A and B2B OOF outputs.

This utility does not train models. It performs a strict one-to-one merge on
`delivery_hour_utc`, validates probability definitions, and writes the compact
C1 handoff table used by trading and frontend consumers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2a", required=True, type=Path, help="B2A regression OOF predictions")
    parser.add_argument("--b2b", required=True, type=Path, help="B2B classifier OOF predictions")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    reg = load_table(args.b2a)
    clf = load_table(args.b2b)
    key = "delivery_hour_utc"
    if not reg[key].is_unique or not clf[key].is_unique:
        raise ValueError("delivery_hour_utc must be unique in both inputs")

    merged = reg[[key, "predicted_spread"]].merge(clf, on=key, how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("B2A and B2B timestamps do not align exactly; refusing to silently drop rows")
    merged = merged.drop(columns=["_merge"])

    merged["p_negative"] = merged["p_c1"] + merged["p_c2"]
    merged["p_neutral"] = merged["p_c3"]
    merged["p_positive"] = merged["p_c4"] + merged["p_c5"]
    prob_sum = merged[["p_c1", "p_c2", "p_c3", "p_c4", "p_c5"]].sum(axis=1)
    if not np.allclose(prob_sum, 1.0, atol=1e-5):
        raise ValueError("Class probabilities do not sum to 1")

    merged["signal"] = "NO_TRADE"
    merged.loc[(merged["p_positive"] >= 0.60) & (merged["p_positive"] > merged["p_negative"]), "signal"] = "DEC"
    merged.loc[(merged["p_negative"] >= 0.60) & (merged["p_negative"] > merged["p_positive"]), "signal"] = "INC"
    merged = merged.sort_values(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
