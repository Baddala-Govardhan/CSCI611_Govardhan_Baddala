from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .eval_utils import compute_metrics, save_classification_report, save_learning_curves, save_plots
from .features import DEFAULT_GROUPS
from .io_utils import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"[logreg] Loading dataset: {args.data}")
    df = read_table(args.data)
    feature_cols = list(DEFAULT_GROUPS.all)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["cancelled"], errors="coerce").fillna(0).astype(int).to_numpy()
    print(f"[logreg] Rows: {len(df):,} | Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"[logreg] Train samples: {len(X_train):,}")
    print(f"[logreg] Test samples:  {len(X_test):,}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="saga",
                    penalty="l2",
                    tol=1e-3,
                    random_state=args.seed,
                    verbose=1,
                ),
            ),
        ]
    )
    print()
    print("========== Training Logistic Regression ==========")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(
        f"[logreg] model.fit finished in {time.perf_counter() - t0:.2f}s (solver progress may appear above)",
        flush=True,
    )
    print()
    print("========== Testing Logistic Regression ==========")
    t1 = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"[logreg] predict_proba on test set done in {time.perf_counter() - t1:.2f}s", flush=True)

    out_dir = Path(args.out)
    prefix = "logreg_all"
    metrics = compute_metrics(y_test, y_prob, threshold=args.threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[logreg] Saving ROC / PR / confusion / calibration / histogram plots...")
    save_plots(y_true=y_test, y_prob=y_prob, out_dir=out_dir, prefix=prefix, threshold=args.threshold)
    save_classification_report(
        y_true=y_test, y_prob=y_prob, out_dir=out_dir, prefix=prefix, threshold=args.threshold
    )

    # Extra report-friendly plots (learning curves)
    print("[logreg] Computing learning curves...")
    save_learning_curves(
        estimator=model,
        X=X,
        y=y,
        out_dir=out_dir,
        prefix=prefix,
        seed=args.seed,
    )

    print(f"[logreg] Done. Outputs saved to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
