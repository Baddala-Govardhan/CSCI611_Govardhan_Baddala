from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .io_utils import read_table


def _is_night(hour: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour, errors="coerce")
    return ((h >= 22) | (h <= 5)).astype(int)


def _is_weekend(dow: pd.Series) -> pd.Series:
    d = pd.to_numeric(dow, errors="coerce")
    return (d >= 5).astype(int)


def fit_fare_model(df: pd.DataFrame) -> dict:
    """
    Fit a lightweight fare estimator from observed TLC amounts.

    Features are intentionally simple so we can also run it in the browser:
    - trip_distance (miles)
    - is_night (0/1)
    - is_weekend (0/1)
    """
    target_col = "total_amount" if "total_amount" in df.columns else ("fare_amount" if "fare_amount" in df.columns else None)
    if target_col is None:
        raise ValueError("Dataset missing fare columns: expected total_amount or fare_amount")

    dist = pd.to_numeric(df.get("trip_distance"), errors="coerce")
    hour = df.get("pickup_hour")
    dow = df.get("pickup_dow")
    y = pd.to_numeric(df.get(target_col), errors="coerce")

    X = pd.DataFrame(
        {
            "trip_distance": dist,
            "is_night": _is_night(hour),
            "is_weekend": _is_weekend(dow),
        }
    )

    # Basic filtering to avoid extreme outliers dominating a simple linear model.
    mask = (
        X["trip_distance"].between(0.1, 50)
        & y.between(2.0, 250.0)
        & X.notna().all(axis=1)
        & y.notna()
    )
    Xf = X.loc[mask]
    yf = y.loc[mask]

    if len(Xf) < 500:
        raise ValueError(f"Not enough rows to fit fare model (got {len(Xf)})")

    lr = LinearRegression()
    lr.fit(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))

    payload = {
        "target_col": target_col,
        "feature_names": list(Xf.columns),
        "intercept": float(lr.intercept_),
        "coef": [float(c) for c in lr.coef_],
        "train_rows": int(len(Xf)),
        "distance_range": [float(Xf["trip_distance"].min()), float(Xf["trip_distance"].max())],
        "y_range": [float(yf.min()), float(yf.max())],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a simple fare estimator and save to outputs/fare_model.json")
    parser.add_argument("--data", default="data/processed/labeled.parquet")
    parser.add_argument("--out", default="outputs/fare_model.json")
    args = parser.parse_args()

    df = read_table(args.data)
    model = fit_fare_model(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2))
    print(f"Wrote fare model to {out_path} (target={model['target_col']}, rows={model['train_rows']:,})")


if __name__ == "__main__":
    main()

