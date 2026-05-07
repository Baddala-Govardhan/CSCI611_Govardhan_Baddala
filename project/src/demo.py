from __future__ import annotations

"""
Ride Cancellation Prediction Demo
----------------------------------
Usage:
    python -m src.demo                          # run examples + interactive
    python -m src.demo --examples               # show built-in example trips only
    python -m src.demo --hour 23 --dow 5 --distance 15.0 --pu-zone 132 --do-zone 138
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import haversine_km
from .zones_centroids import load_zone_centroids_csv

MODEL_PATH = Path("outputs/xgb_all_model.joblib")
ZONES_PATH = Path("data/aux/taxi_zone_centroids.csv")

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

EXAMPLE_TRIPS = [
    {"name": "Late night long ride (JFK → Midtown)",          "hour": 23, "dow": 4, "pu_zone": 132, "do_zone": 161, "distance": 18.5},
    {"name": "Morning weekday commute (Midtown → Brooklyn)",  "hour": 8,  "dow": 1, "pu_zone": 161, "do_zone": 33,  "distance": 4.2},
    {"name": "Saturday night short trip (Lower Manhattan)",   "hour": 22, "dow": 5, "pu_zone": 79,  "do_zone": 48,  "distance": 1.1},
    {"name": "Weekend afternoon (Queens → Manhattan)",        "hour": 14, "dow": 6, "pu_zone": 70,  "do_zone": 234, "distance": 7.8},
    {"name": "Early morning airport run (Newark)",            "hour": 4,  "dow": 2, "pu_zone": 1,   "do_zone": 138, "distance": 22.0},
]


def _risk_label(prob: float) -> str:
    if prob < 0.15:
        return "LOW RISK"
    if prob < 0.30:
        return "MEDIUM RISK"
    return "HIGH RISK"


def _risk_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {prob*100:.1f}%"


def load_model(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run first:  python -m src.train_xgb --data data/processed/labeled.parquet"
        )
    return joblib.load(model_path)


def predict_trip(
    *,
    hour: int,
    dow: int,
    distance: float,
    pu_zone: int,
    do_zone: int,
    model_bundle: dict,
    zones: pd.DataFrame,
) -> float:
    pmap = zones.set_index("LocationID")

    pu_lat = float(pmap.loc[pu_zone, "latitude"]) if pu_zone in pmap.index else 40.7128
    pu_lon = float(pmap.loc[pu_zone, "longitude"]) if pu_zone in pmap.index else -74.0060
    do_lat = float(pmap.loc[do_zone, "latitude"]) if do_zone in pmap.index else 40.7128
    do_lon = float(pmap.loc[do_zone, "longitude"]) if do_zone in pmap.index else -74.0060

    hav = float(haversine_km(
        np.array([pu_lat]), np.array([pu_lon]),
        np.array([do_lat]), np.array([do_lon]),
    )[0])

    feature_cols = model_bundle["feature_cols"]
    row = {
        "pickup_hour": hour,
        "pickup_dow": dow,
        "pickup_latitude": pu_lat,
        "pickup_longitude": pu_lon,
        "dropoff_latitude": do_lat,
        "dropoff_longitude": do_lon,
        "haversine_km": hav,
        "trip_distance": distance,
    }
    X = np.array([[row[c] for c in feature_cols]], dtype=float)
    return float(model_bundle["model"].predict_proba(X)[0, 1])


def print_prediction(*, name: str, hour: int, dow: int, distance: float, pu_zone: int, do_zone: int, prob: float) -> None:
    print()
    print(f"  Trip     : {name}")
    print(f"  Time     : {hour:02d}:00  |  {DOW_NAMES[dow]}")
    print(f"  Distance : {distance:.1f} miles")
    print(f"  Zones    : {pu_zone} → {do_zone}")
    print(f"  Risk     : {_risk_bar(prob)}  →  {_risk_label(prob)}")
    print()


def run_examples(model_bundle: dict, zones: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  RIDE CANCELLATION PREDICTION — EXAMPLE TRIPS")
    print("=" * 60)
    for trip in EXAMPLE_TRIPS:
        prob = predict_trip(
            hour=trip["hour"], dow=trip["dow"], distance=trip["distance"],
            pu_zone=trip["pu_zone"], do_zone=trip["do_zone"],
            model_bundle=model_bundle, zones=zones,
        )
        print_prediction(
            name=trip["name"], hour=trip["hour"], dow=trip["dow"],
            distance=trip["distance"], pu_zone=trip["pu_zone"],
            do_zone=trip["do_zone"], prob=prob,
        )
    print("=" * 60)


def run_interactive(model_bundle: dict, zones: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  RIDE CANCELLATION PREDICTION — INTERACTIVE DEMO")
    print("  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        print()
        try:
            hour_in = input("  Pickup hour (0-23): ").strip()
            if hour_in.lower() == "quit":
                break
            hour = int(hour_in)

            dow_in = input("  Day of week (0=Mon ... 6=Sun): ").strip()
            if dow_in.lower() == "quit":
                break
            dow = int(dow_in)

            dist_in = input("  Trip distance (miles): ").strip()
            if dist_in.lower() == "quit":
                break
            distance = float(dist_in)

            pu_in = input("  Pickup zone ID (1-263): ").strip()
            if pu_in.lower() == "quit":
                break
            pu_zone = int(pu_in)

            do_in = input("  Dropoff zone ID (1-263): ").strip()
            if do_in.lower() == "quit":
                break
            do_zone = int(do_in)

        except (ValueError, EOFError):
            print("  Invalid input, try again.")
            continue

        prob = predict_trip(
            hour=hour, dow=dow, distance=distance,
            pu_zone=pu_zone, do_zone=do_zone,
            model_bundle=model_bundle, zones=zones,
        )
        print_prediction(
            name="Custom Trip", hour=hour, dow=dow,
            distance=distance, pu_zone=pu_zone, do_zone=do_zone, prob=prob,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ride cancellation prediction demo.")
    parser.add_argument("--examples",  action="store_true", help="Run built-in example trips only")
    parser.add_argument("--model",     default=str(MODEL_PATH))
    parser.add_argument("--zones-csv", default=str(ZONES_PATH))
    parser.add_argument("--hour",      type=int,   help="Pickup hour (0-23)")
    parser.add_argument("--dow",       type=int,   help="Day of week (0=Mon, 6=Sun)")
    parser.add_argument("--distance",  type=float, help="Trip distance in miles")
    parser.add_argument("--pu-zone",   type=int,   help="Pickup zone ID")
    parser.add_argument("--do-zone",   type=int,   help="Dropoff zone ID")
    args = parser.parse_args()

    model_bundle = load_model(Path(args.model))
    zones = load_zone_centroids_csv(Path(args.zones_csv))

    # Single trip via CLI flags
    if all(v is not None for v in [args.hour, args.dow, args.distance, args.pu_zone, args.do_zone]):
        prob = predict_trip(
            hour=args.hour, dow=args.dow, distance=args.distance,
            pu_zone=args.pu_zone, do_zone=args.do_zone,
            model_bundle=model_bundle, zones=zones,
        )
        print_prediction(
            name="Single Trip Prediction", hour=args.hour, dow=args.dow,
            distance=args.distance, pu_zone=args.pu_zone, do_zone=args.do_zone, prob=prob,
        )
    elif args.examples:
        run_examples(model_bundle, zones)
    else:
        run_examples(model_bundle, zones)
        run_interactive(model_bundle, zones)


if __name__ == "__main__":
    main()
