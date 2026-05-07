from __future__ import annotations

import argparse
from pathlib import Path

from .zones_centroids import ensure_zone_centroids_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TLC taxi zones shapefile and build LocationID→lat/lon centroids CSV.")
    parser.add_argument(
        "--output",
        default="data/aux/taxi_zone_centroids.csv",
        help="Where to write centroids CSV",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    out = ensure_zone_centroids_csv(Path(args.output), force=args.force)
    print(f"Wrote zone centroids: {out}")


if __name__ == "__main__":
    main()
