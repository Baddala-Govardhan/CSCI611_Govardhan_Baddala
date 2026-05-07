# Ride Cancellation Prediction (XGBoost)

This project predicts whether a ride will be cancelled (**binary classification**) using tabular trip features (temporal / spatial / trip).

NYC Taxi trip datasets typically do **not** contain true "cancellation" labels, so this repo includes a **synthetic label generator** (clearly marked) to support model training and evaluation for class projects.

## Folder layout

- `data/raw/` — place your input trips file here (CSV or Parquet)
- `data/processed/` — generated labeled dataset
- `src/` — scripts
- `outputs/` — metrics + plots

## Expected input schema (minimum)

Your raw file should include:

- Pickup timestamp: `tpep_pickup_datetime` (or `pickup_datetime`)
- Pickup location: `pickup_longitude`, `pickup_latitude` (or `PULocationID` if you use zone IDs)
- Dropoff location (optional): `dropoff_longitude`, `dropoff_latitude`
- Trip distance: `trip_distance`

If your column names differ, edit `src/config.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (recommended: official TLC data, end-to-end)

From the project root, after activating your venv:

```bash
python -m src.run_pipeline --year 2024 --month 1 --max-rows 100000
```

This will:

- download TLC **taxi zone** shapefile → build `data/aux/taxi_zone_centroids.csv`
- download TLC **Yellow Taxi** parquet → `data/raw/yellow_tripdata_YYYY-MM.parquet`
- build `data/processed/labeled.parquet` (with **synthetic** `cancelled` labels)
- train **logistic regression** + **XGBoost**, run **ablations**, write metrics/plots under `outputs/`

Use `--skip-download-zones` / `--skip-download-trips` if you already downloaded files.

Optional: force the boosted tree learner:

```bash
python -m src.train_xgb --data data/processed/labeled.parquet --backend histgb
```

### macOS: XGBoost + OpenMP (`libomp`)

Apple Silicon / Homebrew Python setups sometimes miss OpenMP and XGBoost fails to load with:

`Library not loaded: libomp.dylib`

Install runtime:

```bash
brew install libomp
```

Then restart your terminal / venv. Until then, `--backend auto` falls back to **sklearn `HistGradientBoostingClassifier`** (histogram GBDT), which is fine for coursework if you explain it in your report.

## Web dashboard

After running the pipeline (so `outputs/boost_all_*` files exist), launch the Flask UI:

```bash
python -m src.webapp
```

Open `http://localhost:9090`.

### Share a public “live” link (tunnel)

`localhost` only works on your machine. To share a URL with others, bind on all interfaces and run a tunnel:

```bash
python -m src.webapp --host 0.0.0.0 --port 9090
```

In another terminal (after installing [ngrok](https://ngrok.com/) or [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/)), for example:

```bash
ngrok http 9090
```

Use the HTTPS URL ngrok prints (or the cloudflared URL). Keep both processes running while you demo.

### Host a permanent link (always-on URL)

To give people a link that works without your laptop:

1. **Train locally** so `outputs/` and `data/processed/labeled.parquet` exist (same as the pipeline section above).
2. **Commit artifacts** — they are `.gitignore`d by default, so stage them once:

   ```bash
   ./scripts/stage_deploy_artifacts.sh
   git commit -m "Add trained artifacts for web hosting"
   git push
   ```

3. **Deploy** — easiest path: [Render](https://render.com/) → **New** → **Blueprint** (or **Web Service**) → connect your GitHub repo. This repo includes `render.yaml` and `Procfile` so Render runs **Gunicorn** and listens on `$PORT`.
4. When the deploy finishes, Render shows a URL like `https://nyc-taxi-cancellation-demo.onrender.com` — that is the shareable “live” link.

**Notes:** Free tiers may sleep after idle; the first request after sleep can take ~30–60s. The Flask **development** server (`python -m src.webapp`) is for local use only; production uses **Gunicorn** via `Procfile` / `render.yaml`.

## Run (manual steps)

1) Put your dataset at `data/raw/trips.csv` (or `.parquet`). For TLC parquet with `PULocationID` / `DOLocationID`, run `python -m src.download_zones` first so `data/aux/taxi_zone_centroids.csv` exists.

2) Create labeled dataset:

```bash
python -m src.make_dataset --input data/raw/trips.csv --output data/processed/labeled.parquet
```

3) Train + evaluate baselines + XGBoost:

```bash
python -m src.train_baselines --data data/processed/labeled.parquet
python -m src.train_xgb --data data/processed/labeled.parquet
```

4) Run feature ablations:

```bash
python -m src.ablation --data data/processed/labeled.parquet
```

## Notes / assumptions

- Labels are **synthetic**: `src/labeling.py` defines a cancellation probability based on plausible factors (night hours, longer pickup-to-dropoff distance, etc.) plus noise, then samples a Bernoulli label.
- If you have real cancellation labels, bypass the synthetic labeling step and map your real label column to `cancelled` (0/1).

