from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def add_synthetic_cancellation_label(
    df: pd.DataFrame,
    *,
    hour_col: str = "pickup_hour",
    dow_col: str = "pickup_dow",
    distance_col: str = "trip_distance",
    seed: int = 42,
    target_rate: float = 0.20,
    min_prob: float = 0.02,
    max_prob: float = 0.70,
) -> pd.DataFrame:
    """
    Create a synthetic cancellation label.

    This is ONLY for datasets that do not include cancellations. The signal is
    intentionally plausible-but-simple:
    - higher cancellation at late night
    - higher cancellation for longer trips (proxy for higher cost/time)
    - mild day-of-week effect
    plus random noise.
    """
    rng = np.random.default_rng(seed)

    hour = df[hour_col].to_numpy(dtype=float)
    dow = df[dow_col].to_numpy(dtype=float)
    dist = df[distance_col].to_numpy(dtype=float)

    # Normalize distance robustly
    dist_clip = np.nan_to_num(np.clip(dist, 0.0, np.nanpercentile(dist, 99)))
    dist_z = (dist_clip - np.nanmedian(dist_clip)) / (np.nanmedian(np.abs(dist_clip - np.nanmedian(dist_clip))) + 1e-6)

    night = ((hour >= 22) | (hour <= 5)).astype(float)
    weekend = (dow >= 5).astype(float)

    linear = (
        -1.0
        + 1.0 * night
        + 0.35 * weekend
        + 0.55 * dist_z
        + rng.normal(0.0, 0.6, size=len(df))
    )
    p = _sigmoid(linear)

    # Calibrate to roughly match a desired overall cancellation rate.
    # We do this by shifting logits by a constant, and also *cap* the
    # synthetic probabilities to avoid unrealistic 95–99% cancellation
    # predictions in the demo UI.
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    # Binary search shift
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        p_mid = min_prob + (max_prob - min_prob) * _sigmoid(logits + mid)
        if float(np.mean(p_mid)) > target_rate:
            hi = mid
        else:
            lo = mid
    p_cal = min_prob + (max_prob - min_prob) * _sigmoid(logits + (lo + hi) / 2.0)

    y = rng.binomial(1, p_cal).astype(np.int8)
    out = df.copy()
    out["cancelled"] = y
    out["cancel_prob_synth"] = p_cal.astype(np.float32)
    return out

