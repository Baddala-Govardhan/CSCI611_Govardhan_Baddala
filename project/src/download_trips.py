from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


def yellow_tripdata_url(year: int, month: int) -> str:
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYC TLC Yellow Taxi trip records (official parquet).")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (default: data/raw/yellow_tripdata_YYYY-MM.parquet)",
    )
    args = parser.parse_args()

    url = yellow_tripdata_url(args.year, args.month)
    out = Path(args.output) if args.output else Path(f"data/raw/yellow_tripdata_{args.year}-{args.month:02d}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading:\n  {url}\n→ {out}")
    urlretrieve(url, str(out))
    print(f"Saved {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
