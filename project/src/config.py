from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnConfig:
    pickup_datetime: str = "tpep_pickup_datetime"
    pickup_lat: str = "pickup_latitude"
    pickup_lon: str = "pickup_longitude"
    dropoff_lat: str = "dropoff_latitude"
    dropoff_lon: str = "dropoff_longitude"
    trip_distance: str = "trip_distance"

    # Optional if you want to compute duration-based features
    dropoff_datetime: str = "tpep_dropoff_datetime"


DEFAULT_COLS = ColumnConfig()

