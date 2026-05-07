from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from .features import haversine_km
from .zones_centroids import load_zone_centroids_csv

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
MODEL_PATH = OUTPUTS / "boost_all_model.joblib"
FARE_MODEL_PATH = OUTPUTS / "fare_model.json"
ZONES_PATH = ROOT / "data/aux/taxi_zone_centroids.csv"

app = Flask(__name__, template_folder=str(ROOT / "templates"))

model_bundle = None
zones = None
fare_model = None
_artifacts_lock = threading.Lock()
_artifacts_loaded = False


def _load_artifacts() -> None:
    """Load model + zone centroids once (deferred so --help does not touch disk/joblib)."""
    global model_bundle, zones, fare_model, _artifacts_loaded
    with _artifacts_lock:
        if _artifacts_loaded:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing trained model at {MODEL_PATH}. Run the training pipeline first."
            )
        model_bundle = joblib.load(MODEL_PATH)
        zones = load_zone_centroids_csv(ZONES_PATH)
        fare_model = json.loads(FARE_MODEL_PATH.read_text()) if FARE_MODEL_PATH.exists() else None
        print(f"Model loaded: {model_bundle['backend']}  |  Zones: {len(zones)}")
        _artifacts_loaded = True


@app.before_request
def _ensure_artifacts():
    _load_artifacts()


def estimate_fare(*, distance_miles: float, hour: int, dow: int) -> float | None:
    if fare_model is None:
        return None
    intercept = float(fare_model["intercept"])
    coef = [float(c) for c in fare_model["coef"]]
    # feature_names: ["trip_distance","is_night","is_weekend"]
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_weekend = 1 if dow >= 5 else 0
    x = [float(distance_miles), float(is_night), float(is_weekend)]
    return max(0.0, intercept + sum(ci * xi for ci, xi in zip(coef, x)))


@app.route("/")
def index():
    boost_payload = json.loads((OUTPUTS / "boost_all_metrics.json").read_text())
    xgb = boost_payload["metrics"]
    boost_backend = boost_payload.get("backend", model_bundle["backend"])
    logreg = json.loads((OUTPUTS / "logreg_all_metrics.json").read_text())
    ablation = json.loads((OUTPUTS / "ablation_metrics.json").read_text())["metrics"]
    fi = json.loads((OUTPUTS / "boost_all_feature_importance.json").read_text())
    fi_labels = [row[0] for row in fi]
    fi_values = [round(row[1], 4) for row in fi]
    zone_options = zones[["LocationID", "zone", "borough"]].sort_values("zone").to_dict(orient="records")
    zone_coords  = {int(k): {"latitude": float(v["latitude"]), "longitude": float(v["longitude"])}
                    for k, v in zones.set_index("LocationID")[["latitude","longitude"]].to_dict(orient="index").items()}
    return render_template(
        "index.html",
        xgb=xgb,
        boost_backend=boost_backend,
        logreg=logreg,
        ablation=ablation,
        fi_labels=fi_labels,
        fi_values=fi_values,
        zone_options=zone_options,
        zone_coords=zone_coords,
        fare_model=fare_model,
    )


@app.route("/api/data")
def api_data():
    page = int(request.args.get("page", 1))
    per_page = 50
    labeled_path = ROOT / "data/processed/labeled.parquet"
    df = pd.read_parquet(labeled_path)
    dow_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
    zone_map = zones.set_index("LocationID")["zone"].to_dict()

    display_cols = ["pickup_hour", "pickup_dow", "trip_distance", "haversine_km",
                    "pickup_latitude", "pickup_longitude", "cancelled"]
    df = df[[c for c in display_cols if c in df.columns]].copy()
    df["pickup_dow"] = df["pickup_dow"].map(dow_map)
    df["trip_distance"] = df["trip_distance"].round(2)
    df["haversine_km"] = df["haversine_km"].round(2)
    df["pickup_latitude"] = df["pickup_latitude"].round(5)
    df["pickup_longitude"] = df["pickup_longitude"].round(5)
    df["cancelled"] = df["cancelled"].map({1: "Yes", 0: "No"})

    total = len(df)
    start = (page - 1) * per_page
    chunk = df.iloc[start:start + per_page]
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "columns": list(chunk.columns),
        "rows": chunk.values.tolist(),
    })


@app.route("/image/<path:filepath>")
def serve_image(filepath):
    full_path = OUTPUTS / filepath
    if not full_path.exists():
        return "Not found", 404
    return send_file(full_path, mimetype="image/png")


DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _explain(hour: int, dow: int, distance: float, hav: float) -> list[dict]:
    reasons = []

    if hour >= 22 or hour <= 5:
        reasons.append({"factor": f"Pickup hour {hour:02d}:00 — late night / early morning", "impact": "high"})
    elif 7 <= hour <= 9 or 17 <= hour <= 19:
        reasons.append({"factor": f"Pickup hour {hour:02d}:00 — peak commute window", "impact": "medium"})
    else:
        reasons.append({"factor": f"Pickup hour {hour:02d}:00 — normal daytime hours", "impact": "low"})

    if distance > 15:
        reasons.append({"factor": f"Trip distance {distance:.1f} miles — very long trip", "impact": "high"})
    elif distance > 7:
        reasons.append({"factor": f"Trip distance {distance:.1f} miles — above average", "impact": "medium"})
    else:
        reasons.append({"factor": f"Trip distance {distance:.1f} miles — short trip", "impact": "low"})

    if dow >= 5:
        reasons.append({"factor": f"{DOW_NAMES[dow]} — weekend, higher demand and cancellations", "impact": "medium"})
    elif dow == 4:
        reasons.append({"factor": f"{DOW_NAMES[dow]} — end of week, elevated cancellation rate", "impact": "medium"})
    else:
        reasons.append({"factor": f"{DOW_NAMES[dow]} — regular weekday, lower risk", "impact": "low"})

    return reasons


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    hour = int(data["hour"])
    dow = int(data["dow"])
    distance = float(data["distance"])
    pu_zone = int(data["pu_zone"])
    do_zone = int(data["do_zone"])

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
        "pickup_hour": hour, "pickup_dow": dow,
        "pickup_latitude": pu_lat, "pickup_longitude": pu_lon,
        "dropoff_latitude": do_lat, "dropoff_longitude": do_lon,
        "haversine_km": hav, "trip_distance": distance,
    }
    X = np.array([[row[c] for c in feature_cols]], dtype=float)
    prob = float(model_bundle["model"].predict_proba(X)[0, 1])
    fare = estimate_fare(distance_miles=distance, hour=hour, dow=dow)

    if prob < 0.15:
        risk, color = "LOW RISK", "success"
    elif prob < 0.30:
        risk, color = "MEDIUM RISK", "warning"
    else:
        risk, color = "HIGH RISK", "danger"

    return jsonify({
        "probability": round(prob * 100, 1),
        "risk": risk,
        "color": color,
        "estimated_fare": None if fare is None else round(float(fare), 2),
        "reasons": _explain(hour, dow, distance, hav),
    })


@app.route("/estimate", methods=["POST"])
def estimate():
    data = request.get_json()
    hour = int(data["hour"])
    dow = int(data["dow"])
    distance = float(data["distance"])
    fare = estimate_fare(distance_miles=distance, hour=hour, dow=dow)
    return jsonify({"estimated_fare": None if fare is None else round(float(fare), 2)})


def main():
    parser = argparse.ArgumentParser(description="Ride cancellation demo website (Flask).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN / tunnels).")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on.")
    args = parser.parse_args()

    _load_artifacts()
    print("\n  Starting Ride Cancellation Prediction Web App")
    print(f"  Local URL:  http://127.0.0.1:{args.port}/")
    if args.host in {"0.0.0.0", "::"}:
        print("  Note: listening on all interfaces — use your tunnel (ngrok/cloudflared) or campus firewall rules carefully.")
    print()
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
