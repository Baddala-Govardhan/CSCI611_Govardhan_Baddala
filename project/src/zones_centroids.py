from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import shapefile
from pyproj import CRS, Transformer


def zone_centroids_from_taxi_zones_zip(zip_bytes: bytes) -> pd.DataFrame:
    """
    TLC publishes taxi zones as a shapefile zip (NY State Plane ft US).
    We approximate each zone center from its bounding box center and convert to WGS84.
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        z.extractall(td_path)
        base_dir = td_path / "taxi_zones"
        prj_text = (base_dir / "taxi_zones.prj").read_text(encoding="utf-8")
        crs_from = CRS.from_string(prj_text)
        transformer = Transformer.from_crs(crs_from, CRS.from_epsg(4326), always_xy=True)

        sf = shapefile.Reader(str(base_dir / "taxi_zones"))
        field_names = [f[0] for f in sf.fields[1:]]
        rows: list[dict[str, float | int]] = []
        for i in range(len(sf.shapes())):
            rec = sf.record(i)
            row = {field_names[j]: rec[j] for j in range(len(field_names))}
            lid = int(row["LocationID"])
            bbox = sf.shape(i).bbox
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)
            lon, lat = transformer.transform(cx, cy)
            rows.append(
                {
                    "LocationID": lid,
                    "zone": str(row.get("zone", "")),
                    "borough": str(row.get("borough", "")),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["LocationID"]).sort_values("LocationID")
    return out


def download_taxi_zones_zip(url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip") -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=180) as resp:
        return resp.read()


def ensure_zone_centroids_csv(
    output_csv: Path,
    *,
    force: bool = False,
    zones_zip_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and not force:
        return output_csv

    zip_bytes = download_taxi_zones_zip(zones_zip_url)
    df = zone_centroids_from_taxi_zones_zip(zip_bytes)
    df.to_csv(output_csv, index=False)
    return output_csv


def load_zone_centroids_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing zone centroid file: {path}. Run: python -m src.download_zones"
        )
    df = pd.read_csv(path)
    for c in ["LocationID", "latitude", "longitude"]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in {path}")
    for c in ["zone", "borough"]:
        if c not in df.columns:
            df[c] = ""
    df["LocationID"] = pd.to_numeric(df["LocationID"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.dropna(subset=["LocationID"]).drop_duplicates(subset=["LocationID"])


def attach_coords_from_zone_ids(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Adds pickup_/dropoff_ latitude/longitude from TLC Location IDs."""
    out = df.copy()
    pmap = zones.set_index("LocationID")

    def lat_lon(series_id: pd.Series) -> tuple[pd.Series, pd.Series]:
        lat = series_id.map(pmap["latitude"])
        lon = series_id.map(pmap["longitude"])
        return lat, lon

    if "PULocationID" in out.columns:
        plat, plon = lat_lon(pd.to_numeric(out["PULocationID"], errors="coerce").astype("Int64"))
        out["pickup_latitude"] = plat if "pickup_latitude" not in out.columns else out["pickup_latitude"].fillna(plat)
        out["pickup_longitude"] = plon if "pickup_longitude" not in out.columns else out["pickup_longitude"].fillna(plon)

    if "DOLocationID" in out.columns:
        dlat, dlon = lat_lon(pd.to_numeric(out["DOLocationID"], errors="coerce").astype("Int64"))
        out["dropoff_latitude"] = dlat if "dropoff_latitude" not in out.columns else out["dropoff_latitude"].fillna(dlat)
        out["dropoff_longitude"] = dlon if "dropoff_longitude" not in out.columns else out["dropoff_longitude"].fillna(dlon)

    return out
