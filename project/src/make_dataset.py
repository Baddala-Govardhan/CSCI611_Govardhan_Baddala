from __future__ import annotations

import argparse

import pandas as pd

from .features import DEFAULT_GROUPS, build_features
from .io_utils import read_table, write_table
from .labeling import add_synthetic_cancellation_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw trips (.csv or .parquet)")
    parser.add_argument("--output", required=True, help="Path to labeled output (.parquet or .csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rate", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument(
        "--zones-csv",
        default=None,
        help="Optional path to taxi_zone_centroids.csv (defaults to data/aux/taxi_zone_centroids.csv)",
    )
    args = parser.parse_args()

    df = read_table(args.input)

    # TLC frequently uses Location IDs; drop unknown zones early if present.
    if "PULocationID" in df.columns:
        pu = pd.to_numeric(df["PULocationID"], errors="coerce").fillna(0).astype(int)
        df = df.loc[pu > 0].copy()
    if "DOLocationID" in df.columns:
        dol = pd.to_numeric(df["DOLocationID"], errors="coerce").fillna(0).astype(int)
        df = df.loc[dol > 0].copy()

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    df = build_features(df, zones_csv=args.zones_csv)

    if "cancelled" not in df.columns:
        df = add_synthetic_cancellation_label(df, seed=args.seed, target_rate=args.target_rate)

    # Keep only rows usable by all engineered features (avoids NaNs in sklearn baselines).
    required = ["cancelled", "pickup_hour", "pickup_dow", "trip_distance"]
    modeling_cols = [c for c in DEFAULT_GROUPS.all if c in df.columns]
    subset = list(dict.fromkeys([c for c in required if c in df.columns] + modeling_cols))
    df_out = df.dropna(subset=subset).copy()
    df_out["cancelled"] = pd.to_numeric(df_out["cancelled"], errors="coerce").astype(int)

    write_table(df_out, args.output)
    print(f"Wrote {len(df_out):,} rows to {args.output}")


if __name__ == "__main__":
    main()

