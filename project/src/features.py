from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ColumnConfig, DEFAULT_COLS
from .zones_centroids import attach_coords_from_zone_ids, load_zone_centroids_csv


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


@dataclass(frozen=True)
class FeatureGroups:
    temporal: tuple[str, ...]
    spatial: tuple[str, ...]
    trip: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.temporal + self.spatial + self.trip))


DEFAULT_GROUPS = FeatureGroups(
    temporal=("pickup_hour", "pickup_dow"),
    spatial=("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "haversine_km"),
    trip=("trip_distance",),
)


def build_features(
    df: pd.DataFrame,
    *,
    cols: ColumnConfig = DEFAULT_COLS,
    zones_csv: Path | str | None = None,
) -> pd.DataFrame:
    out = df.copy()

    # Parse timestamps if present
    if cols.pickup_datetime in out.columns:
        pickup_dt = pd.to_datetime(out[cols.pickup_datetime], errors="coerce")
        out["pickup_hour"] = pickup_dt.dt.hour.astype("Int64")
        out["pickup_dow"] = pickup_dt.dt.dayofweek.astype("Int64")
    elif "pickup_hour" not in out.columns or "pickup_dow" not in out.columns:
        raise ValueError(
            f"Missing pickup datetime column '{cols.pickup_datetime}' (or precomputed pickup_hour/pickup_dow)."
        )

    coord_cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    has_coords = all(c in out.columns for c in coord_cols)
    has_zone_ids = {"PULocationID", "DOLocationID"} <= set(out.columns)

    # TLC parquet releases use Location IDs instead of coordinates — merge zone centroids.
    if has_zone_ids and not has_coords:
        zones_path = Path(zones_csv) if zones_csv else Path.cwd() / "data/aux/taxi_zone_centroids.csv"
        zones = load_zone_centroids_csv(zones_path)
        out = attach_coords_from_zone_ids(out, zones)

    # Standardize expected column names for groups (so ablation is stable)
    rename_map: dict[str, str] = {}
    for src, dst in [
        (cols.pickup_lat, "pickup_latitude"),
        (cols.pickup_lon, "pickup_longitude"),
        (cols.dropoff_lat, "dropoff_latitude"),
        (cols.dropoff_lon, "dropoff_longitude"),
        (cols.trip_distance, "trip_distance"),
    ]:
        if src in out.columns and src != dst:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)

    # Haversine distance if we have coords
    if all(c in out.columns for c in coord_cols):
        out["haversine_km"] = haversine_km(
            out["pickup_latitude"].to_numpy(dtype=float),
            out["pickup_longitude"].to_numpy(dtype=float),
            out["dropoff_latitude"].to_numpy(dtype=float),
            out["dropoff_longitude"].to_numpy(dtype=float),
        ).astype(np.float32)
    else:
        out["haversine_km"] = np.nan

    # Basic cleanup
    for c in ["pickup_hour", "pickup_dow"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ["trip_distance", "haversine_km"] + [col for col in coord_cols if col in out.columns]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

