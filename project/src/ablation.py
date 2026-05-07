from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .eval_utils import compute_metrics
from .features import DEFAULT_GROUPS, FeatureGroups
from .io_utils import read_table
from .models import make_gradient_booster, resolve_boost_backend


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(df["cancelled"], errors="coerce").fillna(0).astype(int).to_numpy()
    return X.to_numpy(), y


def _train_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    backend: str,
    seed: int,
    threshold: float,
) -> dict[str, float]:
    X, y = _prepare_xy(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = make_gradient_booster(
        backend,
        seed=seed,
        n_estimators=400,
        learning_rate=0.06,
        max_depth=5,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_prob, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", choices=["auto", "xgboost", "histgb"], default="auto")
    args = parser.parse_args()

    df = read_table(args.data)
    if "cancelled" not in df.columns:
        raise ValueError("Dataset must contain a 'cancelled' column (0/1).")

    backend = resolve_boost_backend(args.backend)
    if backend == "histgb" and args.backend == "auto":
        print(
            "Ablation using sklearn HistGradientBoostingClassifier (see train_xgb --backend note).",
            file=sys.stderr,
        )

    groups: FeatureGroups = DEFAULT_GROUPS
    configs: dict[str, tuple[str, ...]] = {
        "all": groups.all,
        "temporal": groups.temporal,
        "spatial": groups.spatial,
        "trip": groups.trip,
        "no_temporal": tuple([c for c in groups.all if c not in set(groups.temporal)]),
        "no_spatial": tuple([c for c in groups.all if c not in set(groups.spatial)]),
        "no_trip": tuple([c for c in groups.all if c not in set(groups.trip)]),
    }

    results: dict[str, dict[str, float]] = {}
    for name, cols in configs.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Ablation '{name}' missing columns {missing}. Rebuild dataset.")
        results[name] = _train_eval(
            df,
            list(cols),
            backend=backend,
            seed=args.seed,
            threshold=args.threshold,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"backend": backend, "metrics": results}
    (out_dir / "ablation_metrics.json").write_text(json.dumps(payload, indent=2))

    # Print a compact view
    summary = {k: {"roc_auc": v["roc_auc"], "f1": v["f1"]} for k, v in results.items()}
    print(json.dumps(summary, indent=2))
    print("Saved", out_dir / "ablation_metrics.json")


if __name__ == "__main__":
    main()

