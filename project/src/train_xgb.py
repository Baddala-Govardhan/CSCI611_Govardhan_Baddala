from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .eval_utils import compute_metrics, save_classification_report, save_learning_curves, save_plots
from .features import DEFAULT_GROUPS
from .io_utils import read_table
from .models import make_gradient_booster, resolve_boost_backend


CANON_BOOST_PREFIX = "boost_all"


def _publish_boost_artifacts(out_dir: Path, prefix: str) -> None:
    """
    The Flask dashboard expects stable filenames (`boost_all_*`).
    Copy the latest boosted-tree outputs under a canonical prefix.
    """
    for suffix in [
        "_metrics.json",
        "_feature_importance.json",
        "_model.joblib",
        "_roc.png",
        "_pr.png",
        "_confusion.png",
        "_calibration.png",
        "_pred_hist.png",
        "_learning_roc_auc.png",
        "_learning_f1.png",
        "_report.txt",
    ]:
        src = out_dir / f"{prefix}{suffix}"
        dst = out_dir / f"{CANON_BOOST_PREFIX}{suffix}"
        if src.exists():
            shutil.copy2(src, dst)


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].copy()
    # XGBoost can handle NaNs; keep them. Ensure numeric.
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(df["cancelled"], errors="coerce").fillna(0).astype(int).to_numpy()
    return X.to_numpy(), y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled dataset (.parquet or .csv)")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--backend",
        choices=["auto", "xgboost", "histgb"],
        default="auto",
        help="Gradient boosting implementation. Use histgb if XGBoost fails to load on macOS without libomp.",
    )
    args = parser.parse_args()

    print(f"[boost] Loading dataset: {args.data}")
    df = read_table(args.data)
    if "cancelled" not in df.columns:
        raise ValueError("Dataset must contain a 'cancelled' column (0/1). Run make_dataset first.")

    feature_cols = list(DEFAULT_GROUPS.all)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}. Rebuild dataset with src/make_dataset.py")

    X, y = _prepare_xy(df, feature_cols)
    print(f"[boost] Rows: {len(df):,} | Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"[boost] Train samples: {len(X_train):,}")
    print(f"[boost] Test samples:  {len(X_test):,}")

    backend = resolve_boost_backend(args.backend)
    print(f"[boost] Backend: {backend}")
    if backend == "histgb" and args.backend == "auto":
        print(
            "Using sklearn HistGradientBoostingClassifier (XGBoost native library unavailable). "
            "On macOS install OpenMP with: brew install libomp",
            file=sys.stderr,
        )

    model_display = "XGBoost" if backend == "xgboost" else "HistGradientBoosting"

    model = make_gradient_booster(
        backend,
        seed=args.seed,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
    )

    print()
    print(f"========== Training {model_display} ==========")
    print(
        "[boost] (You should see booster training logs below; if training is fast, gaps may still look small.)",
        flush=True,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"[boost] model.fit finished in {time.perf_counter() - t0:.2f}s", flush=True)
    print()
    print(f"========== Testing {model_display} ==========")
    t1 = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"[boost] predict_proba on test set done in {time.perf_counter() - t1:.2f}s", flush=True)

    out_dir = Path(args.out)
    prefix = "xgb_all" if backend == "xgboost" else "histgb_all"
    metrics = compute_metrics(y_test, y_prob, threshold=args.threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "backend": backend}
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(payload, indent=2))

    print("[boost] Saving ROC / PR / confusion / calibration / histogram plots...")
    save_plots(y_true=y_test, y_prob=y_prob, out_dir=out_dir, prefix=prefix, threshold=args.threshold)
    save_classification_report(
        y_true=y_test, y_prob=y_prob, out_dir=out_dir, prefix=prefix, threshold=args.threshold
    )

    # Extra report-friendly plots (learning curves)
    print("[boost] Computing learning curves...")
    save_learning_curves(
        estimator=model,
        X=X,
        y=y,
        out_dir=out_dir,
        prefix=prefix,
        seed=args.seed,
    )

    # Feature importances (gain-based); sklearn HGBDT may omit this depending on version.
    importances_attr = getattr(model, "feature_importances_", None)
    if importances_attr is None:
        fi = [(name, float("nan")) for name in feature_cols]
    else:
        fi = sorted(
            zip(feature_cols, np.asarray(importances_attr).tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    (out_dir / f"{prefix}_feature_importance.json").write_text(json.dumps(fi, indent=2))

    model_path = out_dir / f"{prefix}_model.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols, "backend": backend}, model_path)
    print(f"[boost] Model saved to: {model_path}")

    _publish_boost_artifacts(out_dir, prefix)
    print(f"[boost] Published stable dashboard artifacts with prefix '{CANON_BOOST_PREFIX}_*'")

    print(f"[boost] Done. Outputs saved to: {out_dir}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

