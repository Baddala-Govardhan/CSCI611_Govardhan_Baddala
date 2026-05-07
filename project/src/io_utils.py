from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet"}:
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() in {".csv"}:
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet.")

