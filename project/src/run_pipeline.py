from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.check_call(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end: zones → trips → labeled dataset → train → ablate.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download-trips", action="store_true")
    parser.add_argument("--skip-download-zones", action="store_true")
    parser.add_argument(
        "--boost-backend",
        choices=["auto", "xgboost", "histgb"],
        default="auto",
        help="Passes through to train_xgb/ablation (auto picks XGBoost when the native library loads).",
    )
    args = parser.parse_args()

    trips_path = Path(f"data/raw/yellow_tripdata_{args.year}-{args.month:02d}.parquet")
    labeled_path = Path("data/processed/labeled.parquet")
    zones_path = Path("data/aux/taxi_zone_centroids.csv")

    exe = sys.executable

    print("[pipeline] Starting end-to-end run")
    print(f"[pipeline] Year-Month: {args.year}-{args.month:02d}")
    print(f"[pipeline] Max rows: {args.max_rows:,}")
    print(f"[pipeline] Boost backend: {args.boost_backend}")
    print(f"[pipeline] Trips path: {trips_path}")
    print(f"[pipeline] Labeled path: {labeled_path}")

    if not args.skip_download_zones:
        run([exe, "-m", "src.download_zones", "--output", str(zones_path)])

    if not args.skip_download_trips:
        run([exe, "-m", "src.download_trips", "--year", str(args.year), "--month", str(args.month), "--output", str(trips_path)])

    run(
        [
            exe,
            "-m",
            "src.make_dataset",
            "--input",
            str(trips_path),
            "--output",
            str(labeled_path),
            "--max-rows",
            str(args.max_rows),
            "--seed",
            str(args.seed),
            "--zones-csv",
            str(zones_path),
        ]
    )

    run([exe, "-m", "src.fare_model", "--data", str(labeled_path), "--out", "outputs/fare_model.json"])

    run([exe, "-m", "src.train_baselines", "--data", str(labeled_path)])
    run(
        [
            exe,
            "-m",
            "src.train_xgb",
            "--data",
            str(labeled_path),
            "--backend",
            args.boost_backend,
        ]
    )
    run(
        [
            exe,
            "-m",
            "src.ablation",
            "--data",
            str(labeled_path),
            "--backend",
            args.boost_backend,
        ]
    )

    run([exe, "-m", "src.eda", "--data", str(labeled_path), "--out", "outputs/eda"])

    run([exe, "-m", "src.demo", "--examples"])

    print("\nDone. Metrics/plots under outputs/")


if __name__ == "__main__":
    main()
